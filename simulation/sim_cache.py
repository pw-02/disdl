import heapq
import numpy as np
import logging
from typing import List, Tuple, Dict, Set, Any
import sys
sys.path.append(".")
from simulation.sim_workloads import workloads
import os
import csv
import time
import collections
import random

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log',         # Log file name
    filemode='w'                # Overwrite the file each run; use 'a' to append
)
logger = logging.getLogger(__name__)


class SharedCache:
    def __init__(self, capacity: int, eviction_policy: str = "noevict"): #"lru", "fifo", "mru", "random", "noevict"
        self.cache = collections.OrderedDict()  # batch_id -> set of seen jobs
        self._timestamps = {}
        self.capacity = capacity
        self.policy = eviction_policy
        self.max_size_used = 0

    def get_batch(self, batch_id) -> bool:
        if batch_id in self.cache:
            if self.policy in ("lru", "mru"):
                self.cache.move_to_end(batch_id)
            return True
        else:
            return False
        
    def put_batch(self, batch_id) -> None:
        if batch_id in self.cache:
            return
        
        if self.cache_is_full():
            if self.policy == "noevict":
                # Do nothing; can't insert and can't evict
                logger.debug(f"Cache full and noevict policy: Skipping insert for batch {batch_id}")
                return
            else:
                self._evict_one()
        # Now safe to insert
        self.cache[batch_id] = True
        self._timestamps[batch_id] = time.time()
        logger.debug(f"Added batch {batch_id} to cache")
        self.max_size_used = max(self.max_size_used, len(self.cache))

    def cache_is_full(self):
        return (len(self.cache) + 1) > self.capacity
    
    def _remove(self, batch_id: Any):
        self.cache.pop(batch_id, None)
        self._timestamps.pop(batch_id, None)
        logger.debug(f"Removed batch {batch_id} from cache")

    def _evict_one(self):
        if self.policy == "lru":
            return self.cache.popitem(last=False)
        if self.policy == "fifo":
            oldest = min(self._timestamps, key=self._timestamps.get)
            return self._remove(oldest)
        if self.policy == "mru":
            return self.cache.popitem(last=True)
        if self.policy == "random":
            victim = random.choice(list(self.cache.keys()))
            return self._remove(victim)
     
    def current_usage_gb(self, size_per_batch_gb: float) -> float:
        return len(self.cache) * size_per_batch_gb
