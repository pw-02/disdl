import collections
import random
import time
from typing import Any, Optional, Tuple

class SharedCache:
    def __init__(self, capacity: int, eviction_policy: str = "noevict"): #"lru", "fifo", "mru", "random", "noevict"
        
        self.cache = collections.OrderedDict()  # batch_id -> set of seen jobs
        self._timestamps = {}
        self.capacity = capacity
        self.policy = eviction_policy
        self.max_size_used = 0
    
    def batch_exists(self, batch_id) -> bool:
        return batch_id in self.cache

    def get_batch(self, batch_id) -> bool:
        if batch_id in self.cache:
            if self.policy in ("lru", "mru"):
                self.cache.move_to_end(batch_id)
            return True
        else:
            return False
            
    def put_batch(self, batch_id) -> Tuple[bool, Optional[str]]:
        
        if batch_id in self.cache:
            return True, None  # Already cached, no eviction

        if self.cache_is_full():
            if self.policy == "noevict":
                return False, None  # Cannot insert and cannot evict
            evicted_batch_id = self._evict_one()
        else:
            evicted_batch_id = None

        self.cache[batch_id] = True
        self._timestamps[batch_id] = time.time()
        self.max_size_used = max(self.max_size_used, len(self.cache))
        return True, evicted_batch_id
    
    def cache_is_full(self) -> bool:
        """Determine if adding one more batch would exceed capacity."""
        return (len(self.cache) + 1) > self.capacity

        
    def remove_batch(self, batch_id: Any) -> bool:
        """Remove a specific batch from the cache if it exists.

        Returns:
            True if the batch was removed, False if it was not in the cache.
        """
        if batch_id in self.cache:
            self.cache.pop(batch_id, None)
            self._timestamps.pop(batch_id, None)
            return True
        return False
    

    def _evict_one(self):
        if self.policy == "lru":
            batch_id = self.cache.popitem(last=False)
            # return batch_id        
        if self.policy == "fifo":
            batch_id = min(self._timestamps, key=self._timestamps.get)
            self.cache.pop(batch_id, None)
        
        if self.policy == "mru":
            batch_id, _ = self.cache.popitem(last=True)

        if self.policy == "random":
            batch_id = random.choice(list(self.cache.keys()))
            self.cache.pop(batch_id, None)
        self._timestamps.pop(batch_id, None)
        return batch_id
     
    def current_usage_gb(self, size_per_batch_gb: float) -> float:
        return len(self.cache) * size_per_batch_gb