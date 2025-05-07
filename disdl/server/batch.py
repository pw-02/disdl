
import hashlib
from typing import List, Set
import threading
import time
from typing import List, Optional, Dict, Tuple
from collections import deque, OrderedDict
from enum import Enum

class CacheStatus(Enum):
    CACHED = "CACHED"
    CACHING_IN_PROGRESS = "CACHING_IN_PROGRESS"
    NOT_CACHED = "NOT_CACHED"


class BatchSet:
    def __init__(self, set_id: str):
        self.id = set_id
        self.batches: Dict[str, Batch] = OrderedDict()
        self.is_finalized = False
        self.marked_for_eviction = False
        self.lock = threading.Lock()
        self.reuse_score = 0.0
    
    def compute_reuse_score(self):
        """Compute the total reuse score for this batch set."""
        with self.lock:
            self.reuse_score = sum(batch.reuse_score for batch in self.batches.values())

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

