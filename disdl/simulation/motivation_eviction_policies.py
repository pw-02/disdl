import heapq
import numpy as np
import random
from collections import deque, OrderedDict

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
                    del self.cache[batch]
                    return
        elif self.eviction_policy == "NoEviction":
            return
        elif self.eviction_policy == "FIFO":
            self.cache.popitem(last=False)  # Remove oldest batch
        elif self.eviction_policy == "LRU":
            self.cache.popitem(last=False)  # LRU is maintained in OrderedDict
        elif self.eviction_policy == "LFU":
            least_frequent_batch = min(self.usage_count, key=self.usage_count.get)
            del self.cache[least_frequent_batch]
            del self.usage_count[least_frequent_batch]
        elif self.eviction_policy == "RR":
            random_batch = random.choice(list(self.cache.keys()))
            del self.cache[random_batch]
            
    def add_batch(self, batch):
        
        if len(self.cache) >= self.max_cache_size:
            #try evicting a batch
            self.evict_batch()

        if len(self.cache) >= self.max_cache_size:
            #eviction failed, cache is full, return
            print(f"Cache is full with {len(self.cache)} objects! Batch {batch} cannot be added. Eviction policy: {self.eviction_policy}")
            return
        else:
            self.cache[batch] = 1  # Add batch
            if self.eviction_policy == "FIFO":
                self.access_order.append(batch)
            elif self.eviction_policy == "LRU":
                self.lru_cache[batch] = None  # Track access order
            elif self.eviction_policy == "LFU":
                self.usage_count[batch] = 1  # Start usage count

    def mark_processed(self, batch, current_time):
        if batch in self.cache:
            self.cache[batch] += 1
            if self.cache[batch] == self.num_jobs:
                del self.cache[batch]  # Evict when processed by all jobs
                print(f"Time {current_time}s: Batch {batch} evicted from cache.")
        self.cache_size_log.append((current_time, len(self.cache)))  # Track cache size over time


class Job:
    def __init__(self, id, speed, cache):
        self.id = id
        self.speed = speed  # Time taken per batch (in seconds)
        self.cache:Cache = cache
        self.current_batch = 1  # Start processing from batch 1

    def process_batch(self, current_time):
        if self.current_batch > self.cache.total_batches:
            return  # Stop processing if all batches are done

        if self.cache.check_cache(self.current_batch, current_time):
            print(f"Time {current_time}s: Job {self.id} processed batch {self.current_batch} from cache.")

        else:
            print(f"Time {current_time}s: Job {self.id} added batch {self.current_batch} to cache.")
            self.cache.add_batch(self.current_batch)

        # Schedule the next batch event
        next_event_time = current_time + self.speed
        heapq.heappush(event_queue, (next_event_time, self.id))
        self.current_batch += 1  # Move to the next batch

# Initialize Simulation
num_batches = 100
max_cache_size = 91 # Limit the cache size
eviction_policy = "CoorDL"  # Change to "FIFO", "LFU",  "RR", "NoEviction", "CoorDL"

shared_cache = Cache(num_batches, num_jobs=2, max_cache_size=max_cache_size, eviction_policy=eviction_policy)
job_speeds = [1, 10]  # Speeds of the two jobs
jobs = []
event_queue = []

for i, speed in enumerate(job_speeds):
    new_job = Job(id=i+1, speed=speed, cache=shared_cache)
    jobs.append(new_job)
    heapq.heappush(event_queue, (new_job.speed, new_job.id))

# Run the simulation
while event_queue:
    current_time, job_id = heapq.heappop(event_queue)
    job:Job = next(j for j in jobs if j.id == job_id)
    job.process_batch(current_time)

# Compute final statistics
print(f"Total duration of simulation: {current_time}s")
print(f"Cache Hits: {shared_cache.cache_hits}, Cache Misses: {shared_cache.cache_misses},  Cache hit ratio: {shared_cache.calculate_hit_ratio() * 100:.2f}%")

if shared_cache.cache_size_log:
    max_cache_size = max(shared_cache.cache_size_log, key=lambda x: x[1])
    print(f"Max cache size reached: {max_cache_size[1]} at time {max_cache_size[0]}s")
else:
    print("No batches were stored in the cache.")