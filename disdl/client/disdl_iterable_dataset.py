
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
from torch.utils.data import get_worker_info

@dataclass
class BatchMetadata:
    batch_id: str
    data_fetch_time: float
    preprocess_time: float
    cache_hit: bool
    grpc_get_overhead: float
    grpc_report_overhead: float
    other: float
    worker_id: Optional[int] = None


class DISDLDataset(IterableDataset):
    def __init__(self, job_id, dataset_name: str, grpc_address: str, s3_loader:BaseS3Loader, redis_host="localhost", redis_port=6379,
                 num_batches_per_epoch: Optional[int] = None,
                 use_compression: bool = True,
                 ):
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
        # self.use_cache = False
        self.num_batches_per_epoch = num_batches_per_epoch
        self.use_compression = use_compression
    def __len__(self):
        return self.num_batches_per_epoch


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
                # logging.warning(f"[REDIS ERROR] Attempt {attempt + 1} failed for batch {batch_id}: {e}")
                pass
            attempt += 1
            if attempt <= max_retries:
                time.sleep(retry_interval)  # exponential backoff
            # logging.warning(f"[CACHE MISS] Batch {batch_id} not found after {max_retries} retries.")
        return minibatch
    
    def deserialize_batch_tensor(self, batch_bytes: bytes) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_compression:
            batch_bytes = lz4.frame.decompress(batch_bytes)
        
        with BytesIO(batch_bytes) as buffer:
            batch_data, batch_labels = torch.load(buffer)
        
        return batch_data, batch_labels
    
    def serialize_batch_tensor(self, batch_data: torch.Tensor, batch_labels: torch.Tensor) -> bytes:
        with BytesIO() as buffer:
            torch.save((batch_data, batch_labels), buffer)
            raw_bytes = buffer.getvalue()
        if self.use_compression:
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
                if retries <= max_retries:
                    time.sleep(retry_interval)

        logging.warning(f"Failed to cache batch '{batch_id}' after {max_retries} retries. Eviction candidate: '{eviction_candidate_key}'. Evicted: '{evicted_key}'")
        return False, evicted_key
    
    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            # Single-process loader, yield all batches
            start = 0
            end = self.num_batches_per_epoch
        else:
            # In a worker process
            per_worker = int(self.num_batches_per_epoch / worker_info.num_workers)
            worker_id = worker_info.id
            start = worker_id * per_worker
            # Last worker takes the rest (in case of remainder)
            if worker_id == worker_info.num_workers - 1:
                end = self.num_batches_per_epoch
            else:
                end = start + per_worker

        mini_batch_client = MiniBatchClient(address=self.grpc_address)

        for _ in range(start, end):
            # Your existing batch fetch and yield logic here
            yield from self._generate_batch(mini_batch_client, worker_id=worker_info.id if worker_info else None)

    def _generate_batch(self, mini_batch_client, worker_id):
        iter_start = time.perf_counter()
        batch_id, sample_list, should_cache, eviction_candidate = mini_batch_client.get_next_batch_metadata(self.job_id)
        grpc_metadata_fetch_time = time.perf_counter() - iter_start
      
        cache_hit = False
        fetch_time = transform_time = 0.0
        evicted_key = None
        batch_is_cached = False
        cached_bytes = None

        if self.use_cache:
            fetch_start = time.perf_counter()
            cached_bytes = self.get_cached_minibatch_with_retries(batch_id, max_retries=10)
            fetch_time = time.perf_counter() - fetch_start
        if cached_bytes:
            decode_start = time.perf_counter()
            batch_data, batch_labels = self.deserialize_batch_tensor(cached_bytes)
            transform_time = time.perf_counter() - decode_start
            cache_hit = True
            batch_is_cached = True
        else:
            batch_data, batch_labels, s3_fetch_time, transform_time = self.s3_loader.load_batch(sample_list)
            fetch_time = s3_fetch_time

            convert_start = time.perf_counter()
            batch_data = torch.stack(batch_data)
            batch_labels = torch.tensor(batch_labels, dtype=torch.long)
            transform_time += time.perf_counter() - convert_start

            if should_cache and self.use_cache:
                encode_start = time.perf_counter()
                serialized = self.serialize_batch_tensor(batch_data, batch_labels)
                success, evicted_key = self.cache_minibatch_with_retries(
                    batch_id, serialized, max_retries=1, eviction_candidate_key=eviction_candidate
                )
                transform_time += time.perf_counter() - encode_start
                if success:
                    batch_is_cached = True
                if evicted_key:
                    logging.info(f"Batch {batch_id} evicted key: {evicted_key}")

        report_start = time.perf_counter()
     
        mini_batch_client.report_job_update(
            job_id=self.job_id,
            processed_batch_id=batch_id,
            batch_is_cached=batch_is_cached,
            eviction_candidate_batch_id=eviction_candidate,
            evicted_batch_id=evicted_key
        )
        grpc_report_time = time.perf_counter() - report_start

        total_time = time.perf_counter() - iter_start
        yield (batch_data, batch_labels), BatchMetadata(
            batch_id=batch_id,
            data_fetch_time=fetch_time,
            preprocess_time=transform_time,
            cache_hit=cache_hit,
            grpc_get_overhead=grpc_metadata_fetch_time,
            grpc_report_overhead=grpc_report_time,
            other=total_time - (fetch_time + transform_time + grpc_metadata_fetch_time + grpc_report_time),
            worker_id=worker_id
        )

    # def __iter__(self):
    #     mini_batch_client = MiniBatchClient(address=self.grpc_address)
        

    #     while True:
    #         iter_start = time.perf_counter()
    #         # 1. Fetch metadata
    #         grpc_start = time.perf_counter()
    #         batch_id, sample_list, should_cache, eviction_candidate = mini_batch_client.get_next_batch_metadata(self.job_id)
    #         grpc_metadata_fetch_time = time.perf_counter() - grpc_start

    #         # 2. Attempt to load from Redis
    #         cache_hit = False
    #         fetch_time = transform_time = 0.0
    #         evicted_key = None
    #         batch_is_cached = False
    #         cached_bytes = None

    #         if self.use_cache:
    #             fetch_start = time.perf_counter()
    #             cached_bytes = self.get_cached_minibatch_with_retries(batch_id, max_retries=0)
    #             fetch_time = time.perf_counter() - fetch_start
    #         if cached_bytes:
    #             # 3a. Cache hit: decode directly
    #             decode_start = time.perf_counter()
    #             batch_data, batch_labels = self.deserialize_batch_tensor(cached_bytes)
    #             transform_time = time.perf_counter() - decode_start
    #             cache_hit = True
    #             batch_is_cached = True
    #         else:
    #             # 3b. Load from S3
    #             batch_data, batch_labels, s3_fetch_time, transform_time = self.s3_loader.load_batch(sample_list)
    #             fetch_time = s3_fetch_time

    #             convert_start = time.perf_counter()
    #             batch_data = torch.stack(batch_data)
    #             batch_labels = torch.tensor(batch_labels, dtype=torch.long)
    #             transform_time += time.perf_counter() - convert_start

    #             # 4. Optionally cache
    #             if should_cache and self.use_cache:
    #                 encode_start = time.perf_counter()
    #                 serialized = self.serialize_batch_tensor(batch_data, batch_labels)
    #                 success, evicted_key = self.cache_minibatch_with_retries(
    #                     batch_id, serialized, max_retries=1, eviction_candidate_key=eviction_candidate
    #                 )
    #                 transform_time += time.perf_counter() - encode_start
    #                 if success:
    #                     batch_is_cached = True
            
    #         # 5. Report metadata
    #         report_start = time.perf_counter()
    #         mini_batch_client.report_job_update(
    #             job_id=self.job_id,
    #             processed_batch_id=batch_id,
    #             batch_is_cached=batch_is_cached,
    #             evicted_batch_id=evicted_key
    #         )
    #         grpc_report_time = time.perf_counter() - report_start

    #         # 6. Yield batch + timings
    #         total_time = time.perf_counter() - iter_start
    #         yield (batch_data, batch_labels), BatchMetadata(
    #             batch_id=batch_id,
    #             data_fetch_time=fetch_time,
    #             preprocess_time=transform_time,
    #             cache_hit=cache_hit,
    #             grpc_get_overhead=grpc_metadata_fetch_time,
    #             grpc_report_overhead=grpc_report_time,
    #             other=total_time - (fetch_time + transform_time + grpc_metadata_fetch_time + grpc_report_time)
    #         )