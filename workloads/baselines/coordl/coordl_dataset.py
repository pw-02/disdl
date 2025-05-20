import torch
import time
import redis
import lz4.frame
import numpy as np
import functools
import logging
from io import BytesIO
from typing import List, Tuple
from dataclasses import dataclass
# from disdl.s3_loader_factory import BaseS3Loader  # your factory class

logger = logging.getLogger(__name__)

@dataclass
class BatchMetadata:
    batch_id: str
    data_fetch_time: float
    preprocess_time: float
    cache_hit: bool
    grpc_get_overhead: float
    grpc_report_overhead: float
    other: float

    
class CoorDLDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 job_id, 
                 total_jobs,
                 s3_loader, 
                 redis_host="localhost", 
                 redis_port=6379,
                 ssl=False, 
                 use_compression=True,
                 syncronized_mode=True):
        
        self.job_id = job_id
        self.use_compression = use_compression
        self.syncronized_mode = syncronized_mode
        self.s3_loader = s3_loader
        self.total_jobs = total_jobs

        self.redis_host = redis_host
        self.redis_port = redis_port
        self.ssl = ssl
        self.redis_client = None
        self.samples = self.s3_loader.load_sample_list()
        pass
        
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['redis_client']  # Remove the Redis connection before pickling
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.redis_client = redis.StrictRedis(host=self.redis_host, port=self.redis_port, ssl=self.ssl)
    
    def _initialize_cache_client(self):
        """Initialize Redis cache client if not already connected."""
        if self.redis_client is None:
            self.redis_client = redis.StrictRedis(host=self.redis_host, port=self.redis_port, ssl=self.ssl)

    def get_cached_minibatch(self, batch_id, max_retries=0, retry_interval=0.1):
        self._initialize_cache_client()
        attempt = 0
        while attempt <= max_retries:
            try:
                minibatch = self.redis_client.get(batch_id)
                if minibatch:
                    return minibatch
            except Exception as e:
                logger.debug(f"[REDIS ERROR] Attempt {attempt + 1} failed for batch {batch_id}: {e}")
            attempt += 1
            if attempt <= max_retries:
                time.sleep(retry_interval * (2 ** attempt))  # Exponential backoff
        logger.debug(f"[CACHE MISS] Batch {batch_id} not found after {max_retries} retries.")
        return None

    
    def deserialize_batch_tensor(self, batch_bytes: bytes) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_compression:
            batch_bytes = lz4.frame.decompress(batch_bytes)
        with BytesIO(batch_bytes) as buffer:
            return torch.load(buffer)
            
    def serialize_batch_tensor(self, batch_data: torch.Tensor, batch_labels: torch.Tensor) -> bytes:
        with BytesIO() as buffer:
            torch.save((batch_data, batch_labels), buffer)
            raw_bytes = buffer.getvalue()
        return lz4.frame.compress(raw_bytes) if self.use_compression else raw_bytes
    
    def cache_minibatch(self, batch_id: str, minibatch: bytes, max_retries: int = 0, retry_interval: float = 0.1) -> bool:
        self._initialize_cache_client()
        retries = 0
        while retries <= max_retries:
            try:
                self.redis_client.set(batch_id, minibatch)
                return True
            except Exception as e:
                logger.debug(f"[RETRY {retries}] Failed to cache batch '{batch_id}': {e}")
                retries += 1
                if retries <= max_retries:
                    time.sleep(retry_interval)
        logger.warning(f"Failed to cache batch '{batch_id}' after {max_retries} retries.")
        return False
    
    def __len__(self) -> int:
        return sum(len(class_items) for class_items in self.samples.values())
    
    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [(blob, class_index)
            for class_index, blob_class in enumerate(self.samples)
            for blob in self.samples[blob_class]]
    
    def __getitem__(self, batch_metadata):
        iter_start = time.perf_counter()

        batch_data = batch_labels = None
        is_cache_hit = False
        fetch_time = transform_time = 0.0

        batch_id, batch_indices, is_batch_owner = batch_metadata

        # Try fetching batch from cache
        fetch_start = time.perf_counter()
        cached_bytes = self.get_cached_minibatch(batch_id)
        fetch_time = time.perf_counter() - fetch_start

        if cached_bytes:
            # Cache hit — decode and done
            decode_start = time.perf_counter()
            batch_data, batch_labels = self.deserialize_batch_tensor(cached_bytes)
            transform_time = time.perf_counter() - decode_start
            is_cache_hit = True

        elif is_batch_owner or not self.syncronized_mode:
            # Load raw samples
            samples = [self._classed_items[idx] for idx in batch_indices]
            batch_data, batch_labels, s3_fetch_time, transform_time = self.s3_loader.load_batch(samples)
            fetch_time = s3_fetch_time

            convert_start = time.perf_counter()
            batch_data = torch.stack(batch_data)
            batch_labels = torch.tensor(batch_labels, dtype=torch.long)
            transform_time += time.perf_counter() - convert_start
    

            # If owner, encode + cache it
            if is_batch_owner:
                encode_start = time.perf_counter()
                serialized = self.serialize_batch_tensor(batch_data, batch_labels)
                self.cache_minibatch(batch_id, serialized, max_retries=2)
                transform_time += time.perf_counter() - encode_start
                # redis_write_time = 0.0  # only count Redis insert if you want to isolate it
        else:
            # Wait until peer caches it
            wait_start = time.perf_counter()
            cached_bytes = self.get_cached_minibatch(batch_id, max_retries=np.inf, retry_interval=0.25)
            fetch_time = time.perf_counter() - wait_start
            is_cache_hit = True #need to decided whether this is a cache hit or not, keep it as true for now
            decode_start = time.perf_counter()
            batch_data, batch_labels = self.deserialize_batch_tensor(cached_bytes)
            transform_time = time.perf_counter() - decode_start

        # Redis access tracking
        access_key = f"access_tracker:{batch_id}"
        # access_count = int(self.redis_client.get(access_key) or 0) + 1
        # self.redis_client.set(access_key, access_count)
        access_count = self.redis_client.incr(access_key)

        if self.total_jobs > 0 and access_count >= self.total_jobs:
            logger.info(f"Evicting batch {batch_id} after {access_count} accesses (expected {self.total_jobs})")
            self.redis_client.delete(batch_id)
            self.redis_client.delete(access_key)
            
        
        total_time = time.perf_counter() - iter_start

        return (batch_data, batch_labels), BatchMetadata(
            batch_id=batch_id,
            data_fetch_time=fetch_time,
            preprocess_time=transform_time,
            cache_hit=is_cache_hit,
            grpc_get_overhead=0.0,
            grpc_report_overhead=0.0,
            other=total_time - (fetch_time + transform_time)
        )
