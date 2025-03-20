import heapq
import numpy as np
import random
from collections import deque, OrderedDict
import logging

# Set up logging
format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

class Cache:
    def __init__(self, total_batches, num_jobs, max_cache_size = np.inf, eviction_policy="LRU"):
        self.cache = {}  # {batch_id: count of jobs that processed it}
        self.access_order = deque()  # Track order for FIFO
        self.usage_count = {}  # Track usage for LFU
        self.lru_cache = OrderedDict()  # Track LRU access order
        self.total_batches = total_batches
        self.num_jobs = num_jobs
        self.max_cache_size = max_cache_size
        self.eviction_policy = eviction_policy
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_size_log = []

    def calculate_hit_ratio(self):
        total_accesses = self.cache_hits + self.cache_misses
        return self.cache_hits / total_accesses if total_accesses > 0 else 0
    
    def check_cache(self, batch, current_time):
        self.cache_size_log.append((current_time, len(self.cache)))  # Track cache size over time
        if batch in self.cache:
            self.cache_hits += 1
            self.cache[batch] += 1
            if self.eviction_policy == "CoorDL":
                if self.cache[batch] == self.num_jobs:
                    logger.debug(f"Batch {batch} evicted from cache (CoorDL).")
                    del self.cache[batch]
            elif self.eviction_policy == "LRU":
                self.lru_cache.move_to_end(batch)  # Mark batch as recently used
            elif self.eviction_policy == "LFU":
                self.usage_count[batch] += 1  # Increment usage count
            return True
        else:            
            self.cache_misses += 1
            return False
        
    def evict_batch(self):
        """Evicts a batch based on the selected policy when the cache is full."""
        if self.eviction_policy == "CoorDL":
            # Evict a batch that has been processed once by all jobs, if possible
            for batch in self.cache:
                if self.cache[batch] == self.num_jobs:
                    logger.debug(f"Batch {batch} evicted from cache (CoorDL).")
                    del self.cache[batch]
                    return
        elif self.eviction_policy == "NoEviction":
            return
        elif self.eviction_policy == "FIFO":
            if self.access_order:
                oldest_batch = self.access_order.popleft()  # Remove the oldest batch
                logger.debug(f"Batch {oldest_batch} evicted from cache (FIFO).")
                del self.cache[oldest_batch]  # Remove from cache
        elif self.eviction_policy == "LRU":
            if self.lru_cache:
                lru_batch, _ = self.lru_cache.popitem(last=False)  # Remove least recently used batch
                logger.debug(f"Batch {lru_batch} evicted from cache (LRU).")
                del self.cache[lru_batch]  # Remove from cache
        elif self.eviction_policy == "LFU":
            if self.usage_count:
                least_frequent_batch = min(self.usage_count, key=self.usage_count.get)
                logger.debug(f"Batch {least_frequent_batch} evicted from cache (LFU).")
                del self.cache[least_frequent_batch]
                del self.usage_count[least_frequent_batch]
        elif self.eviction_policy == "RR":
            if self.cache:
                random_batch = random.choice(list(self.cache.keys()))
                logger.debug(f"Batch {random_batch} evicted from cache (RR).")
                del self.cache[random_batch]
                
    def add_batch(self, batch):
        
        if len(self.cache) >= self.max_cache_size:
            #try evicting a batch
            self.evict_batch()

        if len(self.cache) >= self.max_cache_size:
            #eviction failed, cache is full, return
            logger.debug(f"Cache is full with {len(self.cache)} objects! Batch {batch} cannot be added. Eviction policy: {self.eviction_policy}")
            return
        else:
            self.cache[batch] = 1  # Add batch
            if self.eviction_policy == "FIFO":
                self.access_order.append(batch)  # Track FIFO order
            elif self.eviction_policy == "LRU":
                self.lru_cache[batch] = None  # Track access order
            elif self.eviction_policy == "LFU":
                self.usage_count[batch] = 1  # Start usage count

    def mark_processed(self, batch, current_time):
        if batch in self.cache:
            self.cache[batch] += 1
            if self.cache[batch] == self.num_jobs:
                del self.cache[batch]  # Evict when processed by all jobs
                logger.debug(f"Time {current_time}s: Batch {batch} evicted from cache.")
        self.cache_size_log.append((current_time, len(self.cache)))  # Track cache size over time


class Job:
    def __init__(self, id, speed, cache, event_queue):
        self.id = id
        self.speed = speed  # Time taken per batch (in seconds)
        self.cache:Cache = cache
        self.current_batch = 1  # Start processing from batch 1
        self.event_queue = event_queue

    def process_batch(self, current_time):
        if self.current_batch > self.cache.total_batches:
            return  # Stop processing if all batches are done

        if self.cache.check_cache(self.current_batch, current_time):
            logger.debug(f"Time {current_time}s: Job {self.id} processed batch {self.current_batch} from cache.")

        else:
            logger.debug(f"Time {current_time}s: Job {self.id} added batch {self.current_batch} to cache.")
            self.cache.add_batch(self.current_batch)

        # Schedule the next batch event
        next_event_time = current_time + self.speed
        heapq.heappush( self.event_queue, (next_event_time, self.id))
        self.current_batch += 1  # Move to the next batch


def run_sim(num_batches, max_cache_size, eviction_policy, job_speeds, case_name='run', use_prefetch=False):
    shared_cache = Cache(num_batches, num_jobs=len(job_speeds), max_cache_size=max_cache_size, eviction_policy=eviction_policy)
    jobs = []
    event_queue = []
    
    for i, speed in enumerate(job_speeds):
        new_job = Job(id=i+1, speed=speed, cache=shared_cache,event_queue=event_queue)
        jobs.append(new_job)
        heapq.heappush(event_queue, (new_job.speed, new_job.id))

    while event_queue:
        current_time, job_id = heapq.heappop(event_queue)
        job:Job = next(j for j in jobs if j.id == job_id)
        job.process_batch(current_time)

    logger.info(f"{case_name}:")
    logger.info(f"  Duration of simulation: {current_time}s")
    if shared_cache.cache_size_log:
        max_cache_size = max(shared_cache.cache_size_log, key=lambda x: x[1])
        logger.info(f"  Max cache size reached: {max_cache_size[1]} at time {max_cache_size[0]}s")
    else:
        logger.info(f"   No batches were stored in the cache.")
    logger.info(f"  Cache Hits: {shared_cache.cache_hits}, Cache Misses: {shared_cache.cache_misses},  Cache hit ratio: {shared_cache.calculate_hit_ratio() * 100:.2f}%")
    logger.info("-" * 40)
    return shared_cache.cache_hits, shared_cache.cache_misses, shared_cache.calculate_hit_ratio()

# Initialize Simulation
num_batches = 10000

#set cache size to be a percentage of the total number of batches
max_cache_size = int(num_batches * 0.5) # 10% of total batches

# max_cache_size = 500 # Limit the cache size
eviction_policies = ["FIFO", "LRU", "LFU", "RR", "NoEviction", "CoorDL"]
#eviction_policies = ["RR"]

job_speeds = [1, 10]  # Speeds of the two jobs
for policy in eviction_policies:
    case_name = policy
    run_sim(num_batches, max_cache_size, policy, job_speeds, case_name)
