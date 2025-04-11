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
    def __init__(self, total_batches, num_jobs, max_cache_size=np.inf, eviction_policy="LRU", lookahead=400):
        self.cache = {}  # {batch_id: count of jobs that processed it}
        self.access_order = deque()  # FIFO tracking
        self.usage_count = {}  # LFU tracking
        self.lru_cache = OrderedDict()  # LRU tracking
        self.total_batches = total_batches
        self.num_jobs = num_jobs
        self.max_cache_size = max_cache_size
        self.eviction_policy = eviction_policy
        self.lookahead = lookahead
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_size_log = []

    def calculate_hit_ratio(self):
        total_accesses = self.cache_hits + self.cache_misses
        return self.cache_hits / total_accesses if total_accesses > 0 else 0

    def check_cache(self, batch, current_time):
        self.cache_size_log.append((current_time, len(self.cache)))
        if batch in self.cache:
            self.cache_hits += 1
            self.cache[batch] += 1
            if self.eviction_policy == "CoorDL" and self.cache[batch] == self.num_jobs:
                logger.debug(f"Batch {batch} evicted from cache (CoorDL).")
                del self.cache[batch]
            elif self.eviction_policy == "LRU":
                self.lru_cache.move_to_end(batch)
            elif self.eviction_policy == "LFU":
                self.usage_count[batch] += 1
            return True
        else:
            self.cache_misses += 1
            return False

    def evict_batch(self, future_batches=None):
        if self.eviction_policy == "BeadlyOptimal" and future_batches is not None:
            future_accesses = {batch: min(future_batches.get(batch, [float('inf')])) for batch in self.cache}
            farthest_batch = max(future_accesses, key=future_accesses.get)
            logger.debug(f"Batch {farthest_batch} evicted from cache (Beadly Optimal).")
            del self.cache[farthest_batch]
        elif self.eviction_policy == "FIFO" and self.access_order:
            oldest_batch = self.access_order.popleft()
            logger.debug(f"Batch {oldest_batch} evicted from cache (FIFO).")
            del self.cache[oldest_batch]
        elif self.eviction_policy == "LRU" and self.lru_cache:
            lru_batch, _ = self.lru_cache.popitem(last=False)
            logger.debug(f"Batch {lru_batch} evicted from cache (LRU).")
            del self.cache[lru_batch]
        elif self.eviction_policy == "LFU" and self.usage_count:
            least_frequent_batch = min(self.usage_count, key=self.usage_count.get)
            logger.debug(f"Batch {least_frequent_batch} evicted from cache (LFU).")
            del self.cache[least_frequent_batch]
            del self.usage_count[least_frequent_batch]
        elif self.eviction_policy == "RR" and self.cache:
            random_batch = random.choice(list(self.cache.keys()))
            logger.debug(f"Batch {random_batch} evicted from cache (RR).")
            del self.cache[random_batch]

    def add_batch(self, batch):
        if len(self.cache) >= self.max_cache_size:
            self.evict_batch()
        if len(self.cache) < self.max_cache_size:
            self.cache[batch] = 1
            if self.eviction_policy == "FIFO":
                self.access_order.append(batch)
            elif self.eviction_policy == "LRU":
                self.lru_cache[batch] = None
            elif self.eviction_policy == "LFU":
                self.usage_count[batch] = 1
        else:
            logger.debug(f"Cache is full! Batch {batch} cannot be added. Eviction policy: {self.eviction_policy}")

    def mark_processed(self, batch, current_time):
        if batch in self.cache:
            self.cache[batch] += 1
            if self.cache[batch] == self.num_jobs:
                del self.cache[batch]
                logger.debug(f"Time {current_time}s: Batch {batch} evicted from cache.")
        self.cache_size_log.append((current_time, len(self.cache)))

class Job:
    def __init__(self, id, speed, cache, event_queue, lookahead=30):
        self.id = id
        self.speed = speed
        self.cache = cache
        self.current_batch = 1
        self.event_queue = event_queue
        self.lookahead = lookahead
        self.future_batches = []

    def predict_future_batches(self):
        self.future_batches = list(range(self.current_batch + 1, self.current_batch + self.lookahead + 1))

    def process_batch(self, current_time):
        if self.current_batch > self.cache.total_batches:
            return

        if self.cache.check_cache(self.current_batch, current_time):
            logger.debug(f"Time {current_time}s: Job {self.id} processed batch {self.current_batch} from cache.")
        else:
            logger.debug(f"Time {current_time}s: Job {self.id} added batch {self.current_batch} to cache.")
            self.cache.add_batch(self.current_batch)

        self.predict_future_batches()
        next_event_time = current_time + self.speed
        heapq.heappush(self.event_queue, (next_event_time, self.id))
        self.current_batch += 1

def run_sim(num_batches, max_cache_size, eviction_policy, job_speeds, case_name='run', lookahead=30):
    shared_cache = Cache(num_batches, num_jobs=len(job_speeds), max_cache_size=max_cache_size, eviction_policy=eviction_policy, lookahead=lookahead)
    jobs = []
    event_queue = []

    for i, speed in enumerate(job_speeds):
        job = Job(id=i + 1, speed=speed, cache=shared_cache, event_queue=event_queue, lookahead=lookahead)
        jobs.append(job)
        heapq.heappush(event_queue, (speed, job.id))

    while event_queue:
        current_time, job_id = heapq.heappop(event_queue)
        job = next(j for j in jobs if j.id == job_id)
        job.process_batch(current_time)

    logger.info(f"{case_name}:")
    logger.info(f"  Duration of simulation: {current_time:.2f}s")
    if shared_cache.cache_size_log:
        max_size = max(shared_cache.cache_size_log, key=lambda x: x[1])
        logger.info(f"  Max cache size reached: {max_size[1]} at time {max_size[0]:.2f}s")
    else:
        logger.info("  No batches were stored in the cache.")
    logger.info(f"  Cache Hits: {shared_cache.cache_hits}, Cache Misses: {shared_cache.cache_misses}, Cache hit ratio: {shared_cache.calculate_hit_ratio() * 100:.2f}%")
    logger.info("-" * 40)
    return shared_cache.cache_hits, shared_cache.cache_misses, shared_cache.calculate_hit_ratio()

# Initialize Simulation
if __name__ == "__main__":
    num_batches = 10000
    max_cache_size = int(num_batches * 0.5)
    eviction_policies = ["CoorDL", "BeadlyOptimal"]
    job_speeds = [0.137222914, 0.14272167, 0.351509787, 0.519805225]
    lookahead = 1000

    for policy in eviction_policies:
        run_sim(num_batches, max_cache_size, policy, job_speeds, case_name=policy, lookahead=lookahead)
