
import hashlib
from typing import List
import threading
# from utils.utils import create_unique_id
import time
from typing import List, Optional, Dict, Tuple
from collections import deque, OrderedDict
from logger_config import logger
from enum import Enum

class CacheStatus(Enum):
    CACHED = "CACHED"
    CACHING_IN_PROGRESS = "CACHING_IN_PROGRESS"
    NOT_CACHED = "NOT_CACHED"

class BatchSet:
    def __init__(self, id:str):
        self.id = id
        self.batches: Dict[str, Batch] = OrderedDict()
        self.batches_finalized = False
        self.mark_for_eviction = False

class Batch:
    def __init__(self, batch_indicies, epoch_idx, partition_idx, batch_idx):
        self.indices: List[int] = batch_indicies
        self.epoch_idx:int = epoch_idx
        self.partition_idx:int = partition_idx
        self.batch_idx:int = batch_idx
        self.batch_id:str = self.gen_batch_id()
        self.epoch_partition_id = f"{self.epoch_idx}_{self.partition_idx}"
        self.cache_status:CacheStatus = CacheStatus.NOT_CACHED
        self.last_accessed_time:float = 0 #None #float('inf')
        self.is_first_access = True
        self.lock = threading.Lock()  # Lock for accessing shared resources
           
    def gen_batch_id(self):
        # Convert integers to strings and concatenate them
        id_string = ''.join(str(x) for x in self.indices)
        unique_id = hashlib.md5(id_string.encode()).hexdigest()
        unique_id = unique_id[:16]
        batch_id = f"{self.epoch_idx}_{self.partition_idx}_{self.batch_idx}_{unique_id}"
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