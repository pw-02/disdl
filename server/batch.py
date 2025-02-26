
from typing import List
import threading
# from utils.utils import create_unique_id
import time
from typing import List, Optional, Dict, Tuple
from collections import deque, OrderedDict
from logger_config import logger

class BatchSet:
    def __init__(self, id:str):
        self.id = id
        self.batches: Dict[str, Batch] = OrderedDict()
        self.batches_finalized = False
        self.mark_for_eviction = False

class Batch:
    def __init__(self, batch_indicies, batch_id, epoch_idx, partition_idx):
        self.indicies: List[int] = batch_indicies
        self.epoch_idx:int = epoch_idx
        self.partition_id:int = partition_idx
        self.batch_id:str = batch_id
        # self.epoch_seed:int = epoch_seed
        self.is_cached:bool = False
        self.caching_in_progress:bool = False
        self.next_access_time:float = None
        self.last_accessed_time:float = 0 #None #float('inf')
        self.has_been_accessed_before = False
        self.lock = threading.Lock()  # Lock for accessing shared resources
        self.batch_partition_id = f"{self.epoch_idx}_{self.partition_id}"
        self.evict_from_cache_simulation_time: Optional[float] = None
        self.ttl_timer: Optional[threading.Timer] = None  # Initialize timer
        self.prefetched_time_utc = None

    def time_since_last_access(self):
        """Calculate time elapsed since last access."""
        with self.lock:
            return time.time() - self.last_accessed_time
        
    def set_last_accessed_time(self):
        """Set the last accessed time to the current time."""
        with self.lock:
            self.last_accessed_time = time.time()
    
    def is_first_access(self):
        """Check if this is the first time the batch is being accessed."""
        with self.lock:
            if self.has_been_acessed_before:
                return False
            else:
                self.has_been_acessed_before = True
                return True

    def set_cache_status(self, is_cached:bool):
        """Set the cache status and handle cache eviction timer."""
        with self.lock:
            self.is_cached = is_cached

            if self.is_cached and self.evict_from_cache_simulation_time:
                # Cancel the existing timer if it exists
                if self.ttl_timer:
                    self.ttl_timer.cancel()
                    self.ttl_timer = None
                self.ttl_timer = threading.Timer(self.evict_from_cache_simulation_time, self._evict_cache)
                self.ttl_timer.start()
    
    """Evict the batch from cache due to TTL expiration."""
    def _evict_cache(self):
        with self.lock:
            self.is_cached = False
            self.ttl_timer = None
            # logger.info("Cache evicted due to TTL expiration.")

    
    
    def set_caching_in_progress(self, in_progress:bool):
        with self.lock:
            self.caching_in_progress = in_progress
            
    def set_has_been_accessed_before(self, has_been_accessed_before:bool):
        with self.lock:
            self.has_been_accessed_before = has_been_accessed_before

