import heapq
import numpy as np
import random
from collections import deque, OrderedDict
import logging
import matplotlib.pyplot as plt
import csv
import os

# Set up logging
format_str = "%(asctime)s: %(message)s"
logging.basicConfig(format=format_str, level=logging.INFO, datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

class Cache:
    def __init__(self, total_batches, num_jobs, max_cache_size=np.inf, eviction_policy="LRU"):
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
        self.cache_size_log = []  # List of tuples (time, cache_size)
        self.evictions = 0      # Count total evictions

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
                    self.evictions += 1
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
        evicted = None
        if self.eviction_policy == "CoorDL":
            for batch in list(self.cache.keys()):
                if self.cache[batch] == self.num_jobs:
                    evicted = batch
                    logger.debug(f"Batch {batch} evicted from cache (CoorDL).")
                    del self.cache[batch]
                    self.evictions += 1
                    break
        elif self.eviction_policy == "NoEviction":
            return
        elif self.eviction_policy == "FIFO":
            if self.access_order:
                evicted = self.access_order.popleft()  # Remove the oldest batch
                logger.debug(f"Batch {evicted} evicted from cache (FIFO).")
                del self.cache[evicted]  # Remove from cache
                self.evictions += 1
        elif self.eviction_policy == "LRU":
            if self.lru_cache:
                evicted, _ = self.lru_cache.popitem(last=False)  # Remove least recently used batch
                logger.debug(f"Batch {evicted} evicted from cache (LRU).")
                del self.cache[evicted]  # Remove from cache
                self.evictions += 1
        elif self.eviction_policy == "LFU":
            if self.usage_count:
                evicted = min(self.usage_count, key=self.usage_count.get)
                logger.debug(f"Batch {evicted} evicted from cache (LFU).")
                del self.cache[evicted]
                del self.usage_count[evicted]
                self.evictions += 1
        elif self.eviction_policy == "RR":
            if self.cache:
                evicted = random.choice(list(self.cache.keys()))
                logger.debug(f"Batch {evicted} evicted from cache (RR).")
                del self.cache[evicted]
                self.evictions += 1
        # Also remove it from FIFO or LRU data structures if it still exists
        if evicted is not None:
            try:
                self.access_order.remove(evicted)
            except ValueError:
                pass
            self.lru_cache.pop(evicted, None)

    def add_batch(self, batch):
        if len(self.cache) >= self.max_cache_size:
            # try evicting a batch
            self.evict_batch()

        if len(self.cache) >= self.max_cache_size:
            # eviction failed, cache is full, return without adding
            logger.debug(f"Cache is full with {len(self.cache)} objects! Batch {batch} cannot be added. Eviction policy: {self.eviction_policy}")
            return
        else:
            self.cache[batch] = 1  # Add batch with initial processing count 1
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
                logger.debug(f"Time {current_time}s: Batch {batch} evicted from cache after all jobs processed it.")
                self.evictions += 1
        self.cache_size_log.append((current_time, len(self.cache)))  # Track cache size over time

class Job:
    def __init__(self, id, speed, cache, event_queue):
        self.id = id
        self.speed = speed  # Time taken per batch (in seconds)
        self.cache: Cache = cache
        self.current_batch = 1  # Start processing from batch 1
        self.event_queue = event_queue
        self.cache_hits = 0  # Track cache hits per job

    def process_batch(self, current_time):
        if self.current_batch > self.cache.total_batches:
            return  # Stop processing if all batches are done

        if self.cache.check_cache(self.current_batch, current_time):
            self.cache_hits += 1
            logger.debug(f"Time {current_time}s: Job {self.id} processed batch {self.current_batch} from cache.")
        else:
            logger.debug(f"Time {current_time}s: Job {self.id} added batch {self.current_batch} to cache.")
            self.cache.add_batch(self.current_batch)

        # Schedule the next batch event if batches remain
        next_event_time = current_time + self.speed
        heapq.heappush(self.event_queue, (next_event_time, self.id))
        self.current_batch += 1  # Move to the next batch

def plot_cache_history(cache_history, policy, case_name):
    # Unzip time and cache_size
    times, sizes = zip(*cache_history) if cache_history else ([], [])
    plt.figure(figsize=(10, 5))
    plt.plot(times, sizes, label="Cache Size")
    plt.xlabel("Time (s)")
    plt.ylabel("Cache Size (Num of Batches)")
    plt.title(f"Cache Occupancy Over Time - {policy} ({case_name})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_filename = f"cache_history_{policy}_{case_name}.png"
    plt.savefig(plot_filename)
    logger.info(f"Cache occupancy plot saved as {plot_filename}")
    plt.close()

def export_results_to_csv(results, filename="cache_simulation_results.csv"):
    # If file doesn't exist, write header
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as csv_file:
        fieldnames = [
            "Policy", "Case", "Total Simulation Time (s)", "Max Cache Size Reached", "Time of Max Cache",
            "Average Cache Occupancy", "Cache Hits", "Cache Misses", "Hit Ratio", "Total Evictions", "Job-specific Hits"
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)

def run_sim(num_batches, max_cache_size, eviction_policy, job_speeds, case_name='run', use_prefetch=False):
    # Initialize shared cache and jobs
    shared_cache = Cache(num_batches, num_jobs=len(job_speeds), max_cache_size=max_cache_size, eviction_policy=eviction_policy)
    jobs = []
    event_queue = []
    
    for i, speed in enumerate(job_speeds):
        new_job = Job(id=i+1, speed=speed, cache=shared_cache, event_queue=event_queue)
        jobs.append(new_job)
        heapq.heappush(event_queue, (new_job.speed, new_job.id))
    
    current_time = 0
    # Process events until all are finished
    while event_queue:
        current_time, job_id = heapq.heappop(event_queue)
        job = next(j for j in jobs if j.id == job_id)
        if job.current_batch <= num_batches:
            job.process_batch(current_time)
    
    # Compute additional metrics
    hit_ratio = shared_cache.calculate_hit_ratio()
    avg_cache_occupancy = np.mean([size for _, size in shared_cache.cache_size_log]) if shared_cache.cache_size_log else 0
    if shared_cache.cache_size_log:
        max_cache_entry = max(shared_cache.cache_size_log, key=lambda x: x[1])
        max_cache_size_reached = max_cache_entry[1]
        time_of_max_cache = max_cache_entry[0]
    else:
        max_cache_size_reached = 0
        time_of_max_cache = 0
    job_hits_info = {f"Job_{job.id}_hits": job.cache_hits for job in jobs}

    logger.info(f"{case_name} ({eviction_policy}):")
    logger.info(f"  Duration of simulation: {current_time}s")
    logger.info(f"  Max cache size reached: {max_cache_size_reached} at time {time_of_max_cache}s")
    logger.info(f"  Average cache occupancy: {avg_cache_occupancy:.2f}")
    logger.info(f"  Total Cache Hits: {shared_cache.cache_hits}, Total Cache Misses: {shared_cache.cache_misses}, Hit Ratio: {hit_ratio * 100:.2f}%")
    logger.info(f"  Total Evictions: {shared_cache.evictions}")
    logger.info(f"  Per-Job Cache Hits: {job_hits_info}")
    logger.info("-" * 40)
    
    # Plot cache occupancy over time
    plot_cache_history(shared_cache.cache_size_log, eviction_policy, case_name)

    # Prepare results for CSV export
    results = {
        "Policy": eviction_policy,
        "Case": case_name,
        "Total Simulation Time (s)": current_time,
        "Max Cache Size Reached": max_cache_size_reached,
        "Time of Max Cache": time_of_max_cache,
        "Average Cache Occupancy": f"{avg_cache_occupancy:.2f}",
        "Cache Hits": shared_cache.cache_hits,
        "Cache Misses": shared_cache.cache_misses,
        "Hit Ratio": f"{hit_ratio * 100:.2f}%",
        "Total Evictions": shared_cache.evictions,
        "Job-specific Hits": job_hits_info
    }
    export_results_to_csv(results)
    
    return shared_cache.cache_hits, shared_cache.cache_misses, hit_ratio

# ---------------------------
# Main Simulation Execution
# ---------------------------
if __name__ == "__main__":
    num_batches = 10000
    # set cache size to be a percentage of the total number of batches
    max_cache_size = int(num_batches * 0.5)  # e.g., 50% of total batches
    
    job_speeds = [0.137222914, 0.14272167, 0.351509787, 0.519805225]  # Speeds in batches per second

    eviction_policies = ["FIFO", "LRU", "LFU", "RR", "NoEviction", "CoorDL"]
    # job_speeds = [1, 10]  # Speeds of the two jobs

    # Clear or create results CSV file
    results_csv = "cache_simulation_results.csv"
    if os.path.exists(results_csv):
        os.remove(results_csv)
    
    for policy in eviction_policies:
        case_name = f"{policy}_test"
        run_sim(num_batches, max_cache_size, policy, job_speeds, case_name)
