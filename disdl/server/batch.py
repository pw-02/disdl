
import hashlib
from typing import List, Set
import threading
import time
from typing import List, Optional, Dict, Tuple
from collections import deque, OrderedDict
from enum import Enum
from cache_status import CacheStatus
class Batch:
    def __init__(self, batch_indicies, epoch_idx, partition_idx, batch_idx):
        self.indices: List[int] = batch_indicies
        self.epoch_idx:int = epoch_idx
        self.partition_idx:int = partition_idx
        self.batch_idx:int = batch_idx
        self.batch_id:str = self._gen_batch_id(epoch_idx, partition_idx, batch_idx)
        self.set_id = f"{epoch_idx}_{partition_idx}"
        self.cache_status:CacheStatus = CacheStatus.NOT_CACHED
        self.last_accessed_time:float = 0 #None #float('inf')
        self.is_first_access = True
        self.lock = threading.Lock()  # Lock for accessing shared resources
        self.reuse_score: float = 0.0
        self.awaiting_to_be_seen_by: Dict[str, float] = {}

    def compute_weighted_reuse_score(self):
        """Compute the reuse score based on the number of jobs that have seen this batch."""
        with self.lock:
            self.reuse_score = sum(self.awaiting_to_be_seen_by.values())
    
    def mark_seen_by(self, job_id: str):
        # with self.lock:
            if job_id in self.awaiting_to_be_seen_by:
                del self.awaiting_to_be_seen_by[job_id]
            self.compute_weighted_reuse_score()

    def mark_awaiting_to_be_seen_by(self, job_id: str, weight: float):
        # with self.lock:
            if job_id not in self.awaiting_to_be_seen_by:
                self.awaiting_to_be_seen_by[job_id] = weight
            self.compute_weighted_reuse_score()


    def _gen_batch_id(self, epoch_idx:int, partition_idx:int, batch_idx:int) -> str:
        # Convert integers to strings and concatenate them
        id_string = ''.join(str(x) for x in self.indices)
        unique_id = hashlib.md5(id_string.encode()).hexdigest()
        unique_id = unique_id[:16]
        batch_id = f"{epoch_idx}_{partition_idx}_{batch_idx}_{unique_id}"
        return batch_id


    def time_since_last_access(self):
        """Calculate time elapsed since last access."""
        if self.is_first_access:
            return 0
        with self.lock:
            return time.perf_counter() - self.last_accessed_time
        
    def set_last_accessed_time(self):
        """Set the last accessed time to the current time."""
        with self.lock:
            self.last_accessed_time = time.perf_counter()
    
    def set_cache_status(self, cache_status:CacheStatus):
        with self.lock:
            self.cache_status = cache_status

    def is_cached(self):
        with self.lock:
            return self.cache_status == CacheStatus.CACHED or self.cache_status == CacheStatus.CACHING_IN_PROGRESS

class BatchSet:
    def __init__(self, set_id: str, num_batches: int):
        self.id = set_id
        self.batches: Dict[str, Batch] = OrderedDict()
        self.num_batches = num_batches
        self.marked_for_eviction = False
        self.lock = threading.Lock()
        self.reuse_score = 0.0
    
    def is_finalized(self):
        if len(self.batches) >= self.num_batches:
            return True
        return False
    
    def score_batch_set(self, alpha=1.0, beta=1.0):
        total_reuse_score = 0.0
        cached_count = 0
        batches = self.batches.values()
        total_batches = 0

        for b in batches:
            total_reuse_score += b.reuse_score
            if b.cache_status == CacheStatus.CACHED:
                cached_count += 1
            total_batches += 1

        if total_batches == 0:
            return float('-inf')  # discourage empty sets

        cached_fraction = cached_count / total_batches
        return alpha * total_reuse_score + beta * cached_fraction

    
    # def compute_reuse_score(self):
    #     """Compute the total reuse score for this batch set."""
    #     #count number of batches in the set whose cache status is CACHED
    #     # num_cached_batches = sum(batch.cache_status == CacheStatus.CACHED for batch in self.batches.values())
    #     if not self.batches:
    #         return 0.0
    #     total_score = sum(batch.reuse_score for batch in self.batches.values())
    #     return total_score / len(self.batches)
       
       
        # reuse_score_of_all_batches = sum(batch.reuse_score for batch in self.batches.values())
        # self.reuse_score = reuse_score_of_all_batches
        # return self.reuse_score