
import torch
import torch
from typing import List, Dict, Optional, Tuple
import time
from urllib.parse import urlparse
import redis
from io import BytesIO
import lz4.frame
from torch.utils.data import IterableDataset
from disdl.client.minibatch_client import MiniBatchClient
import logging
from disdl.client.s3_loader_factory import BaseS3Loader # your factory class
from dataclasses import dataclass

@dataclass
class BatchMetadata:
    batch_id: str
    data_fetch_time: float
    preprocess_time: float
    cache_time: float
    cache_hit: bool
    metadata_overhead: float
    report_overhead: float


class DISDLDataset(IterableDataset):
    def __init__(self, job_id, dataset_name: str, grpc_address: str, s3_loader:BaseS3Loader, redis_host="localhost", redis_port=6379):
        self.job_id = job_id
        self.dataset_name = dataset_name
        self.grpc_address = grpc_address
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.s3_loader = s3_loader  # function or object that loads batch from S3
        if self.redis_host is None or self.redis_port is None:
            self.use_cache = False
        else:
            self.use_cache = True
        self.ssl = False
        self.redis_client = None

    def _initialize_cache_client(self):
        """Initialize Redis cache client if not already connected."""
        if self.redis_client is None:
            if self.ssl:
                self.redis_client = redis.StrictRedis(host=self.redis_host, port=self.redis_port, ssl=True)
            else:
                self.redis_client = redis.StrictRedis(host=self.redis_host, port=self.redis_port)


    def get_cached_minibatch_with_retries(self, batch_id, max_retries=3, retry_interval=0.1):
        """Attempts to load a batch from cache, retrying if it fails."""
        self._initialize_cache_client()
        attempt = 0
        minibatch = None
        while attempt <= max_retries:
            try:
                minibatch = self.redis_client.get(batch_id)
                if minibatch:
                    break
            except Exception as e:
                logging.warning(f"[REDIS ERROR] Attempt {attempt + 1} failed for batch {batch_id}: {e}")
            attempt += 1
            time.sleep(retry_interval * (2 ** attempt))  # exponential backoff
        if minibatch is None:
            logging.warning(f"[CACHE MISS] Batch {batch_id} not found after {max_retries} retries.")
        return minibatch
    
    def deserialize_batch_tensor(self, batch_bytes: bytes, use_compression=False) -> Tuple[torch.Tensor, torch.Tensor]:
        if use_compression:
            data = lz4.frame.decompress(batch_bytes)
        
        with BytesIO(batch_bytes) as buffer:
            batch_data, batch_labels = torch.load(buffer)
        
        return batch_data, batch_labels
    
    def serialize_batch_tensor(batch_data: torch.Tensor, batch_labels: torch.Tensor, use_compression: bool = False) -> bytes:
        with BytesIO() as buffer:
            torch.save((batch_data, batch_labels), buffer)
            raw_bytes = buffer.getvalue()
        if use_compression:
            raw_bytes = lz4.frame.compress(raw_bytes)
        return raw_bytes

    def cache_minibatch_with_retries(self,
                                  batch_id: str,
                                  minibatch: bytes,
                                  max_retries: int = 2,
                                  retry_interval: float = 0.0,
                                  eviction_candidate_key: str = None) -> bool:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")

        self._initialize_cache_client()
        retries = 0
        eviction_attempted = False
        evicted_key = None

        while retries <= max_retries:
            try:
                self.redis_client.set(batch_id, minibatch)
                return True, evicted_key
            except Exception as e:
                logging.debug(f"[RETRY {retries}] Failed to cache batch '{batch_id}': {e}")

                # Attempt eviction once
                if eviction_candidate_key and not eviction_attempted:
                    try:
                        deleted = self.redis_client.delete(eviction_candidate_key)
                        if deleted == 1:
                            evicted_key = eviction_candidate_key
                            logging.debug(f"Evicted candidate batch '{evicted_key}' from Redis.")
                        else:
                            logging.warning(f"Eviction candidate '{eviction_candidate_key}' did not exist.")
                    except Exception as evict_err:
                        logging.warning(f"Failed to evict candidate '{eviction_candidate_key}': {evict_err}")
                    eviction_attempted = True

                retries += 1
                time.sleep(retry_interval)

        logging.warning(f"Failed to cache batch '{batch_id}' after {max_retries} retries.")
        return False, evicted_key

    def __iter__(self):
        mini_batch_client = MiniBatchClient(address=self.grpc_address)

        while True:
            iter_start_time = time.perf_counter()
            grpc_start = time.perf_counter()
            batch_id, samples, should_cache, eviction_candidate = mini_batch_client.get_next_batch_metadata(self.job_id)
            metadata_overhead = time.perf_counter() - grpc_start

            # batch_id = batch_metadata["batch_id"]
            # should_cache = batch_metadata["should_cache"]
            # eviction_candidate = batch_metadata.get("evict_batch_id")
            # samples = batch_metadata["samples"]

            cache_hit = False
            preprocess_time = cache_time = data_fetch_time = 0.0
            evicted_key = None
            batch_is_cached = False
            next_minibatch = None
            if self.use_cache:
                next_minibatch = self.get_cached_minibatch_with_retries(batch_id, max_retries=0, retry_interval=0.25)

            if next_minibatch  is not None and (isinstance(next_minibatch , bytes) or isinstance(next_minibatch , str)):
                decode_start = time.perf_counter()
                batch_data, batch_labels = self.deserialize_batch_tensor(next_minibatch)
                preprocess_time = time.perf_counter() - decode_start
                cache_hit = True
                batch_is_cached = True
            else:
                batch_data, batch_labels, _, preprocess_time = self.s3_loader.load_batch(samples)
                cache_hit = False
                batch_data = torch.stack(batch_data)
                batch_labels = torch.tensor(batch_labels, dtype=torch.long)

                if should_cache and self.use_cache:
                    cache_start = time.perf_counter()
                    serialized = self.serialize_batch_tensor(batch_data, batch_labels)
                    cache_success, evicted_key = self.cache_minibatch_with_retries(
                        batch_id, serialized, max_retries=1, eviction_candidate_key=eviction_candidate
                    )
                    if cache_success:
                        batch_is_cached = True
                    cache_time = time.perf_counter() - cache_start

            report_start = time.perf_counter()
            mini_batch_client.report_job_update(
                job_id=self.job_id,
                batch_is_cached=batch_is_cached,
                evicted_batch_id=evicted_key )
            report_overhead = time.perf_counter() - report_start

            total_elapsed = time.perf_counter() - iter_start_time
            data_fetch_time = max(total_elapsed - (preprocess_time + cache_time), 0.0)

            yield (batch_data, batch_labels), BatchMetadata(
                batch_id=batch_id,
                data_fetch_time=data_fetch_time,
                preprocess_time=preprocess_time,
                cache_time=cache_time,
                cache_hit=cache_hit,
                metadata_overhead=metadata_overhead,
                report_overhead=report_overhead
            )
