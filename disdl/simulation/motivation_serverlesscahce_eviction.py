import heapq
import logging
import numpy as np
# Set up logging
format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.DEBUG, datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

class TTLCache:
    def __init__(self, max_cache_size, num_jobs, ttl_seconds=60):
        self.cache = {}  # {batch_id: (expiration_time, usage_count)}
        self.expiry_queue = []  # Min-heap of (expiration_time, batch_id)
        self.max_cache_size = max_cache_size
        self.ttl_seconds = ttl_seconds
        self.cache_misses = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.num_jobs = num_jobs
        self.usage_counter = {}
        self.cache_requests = {} #request type, counter
        self.cache_size_log = []

    def calculate_hit_ratio(self):
        total_accesses = self.cache_hits + self.cache_misses
        return self.cache_hits / total_accesses if total_accesses > 0 else 0

    
    def check_cache(self, batch_id, current_time):
        self.cache_size_log.append((current_time, len(self.cache)))  # Track cache size over time
        self.cache_requests["get/put"] = self.cache_requests.get("get/put", 0) + 1
        self._evict_expired(current_time)
        if batch_id in self.cache:
            self.cache_hits += 1
            #check usage count and remove if all job have used it
            #update usage count
            self.usage_counter[batch_id] = self.usage_counter[batch_id] + 1
            if self.usage_counter[batch_id] == self.num_jobs:
                del self.cache[batch_id]
                logger.debug(f"Time {current_time}s: Batch {batch_id} evicted due to usage count = {self.num_jobs}.")
            else:
                # Reset TTL and update expiration time
                new_expiry = current_time + self.ttl_seconds
                self.cache[batch_id] = new_expiry
                heapq.heappush(self.expiry_queue, (new_expiry, batch_id))
            return True
        else:
            self.cache_misses += 1
            return False
    
    def add_batch(self, batch_id, current_time):
        self._evict_expired(current_time)
        if batch_id in self.usage_counter and self.usage_counter[batch_id] + 1 == self.num_jobs:
            self.usage_counter[batch_id] = self.usage_counter[batch_id] + 1
            return False #dont add bacth as it wontbe used again
        
        # Add new batch to cache
        if len(self.cache) >= self.max_cache_size:
            self._evict_oldest()
        
        self.cache_requests["get/put"] = self.cache_requests.get("get/put", 0) + 1
        expiry_time = current_time + self.ttl_seconds
        self.cache[batch_id] = expiry_time
        self.usage_counter[batch_id] = 1
        heapq.heappush(self.expiry_queue, (expiry_time, batch_id))
        return True
    
    def refresh_all_ttls(self, current_time):
        """Refresh TTL for all cached batches by extending their expiration."""
        for batch_id in list(self.cache.keys()):
            #check usage count and remove if all job have not used it reset ttl
            if self.usage_counter[batch_id] < self.num_jobs:
                self.cache_requests["warmup"] = self.cache_requests.get("warmup", 0) + 1
                new_expiry = current_time + self.ttl_seconds
                self.cache[batch_id] = new_expiry
                logger.debug(f"Time {current_time}s: Batch {batch_id} TTL refreshed. New expiry: {new_expiry}s.")
                heapq.heappush(self.expiry_queue, (new_expiry, batch_id))

    def _evict_expired(self, current_time):
        while self.expiry_queue and self.expiry_queue[0][0] <= current_time:
            _, batch_id = heapq.heappop(self.expiry_queue)
            if batch_id in self.cache and self.cache[batch_id] <= current_time:
                del self.cache[batch_id]
                logger.debug(f"Time {current_time}s: Batch {batch_id} evicted due to TTL expiration.")
    
    def _evict_oldest(self):
        if self.cache:
            oldest_batch = min(self.cache, key=lambda k: self.cache[k][0])
            del self.cache[oldest_batch]
            logger.debug(f"Batch {oldest_batch} evicted due to cache limit.")

class Job:
    def __init__(self, job_id, speed, cache, event_queue, num_batches):
        self.job_id = job_id
        self.speed = speed
        self.cache: TTLCache = cache
        self.current_batch = 1
        self.event_queue = event_queue
        self.total_batches = num_batches
        self.job_complete = False

    def process_batch(self, current_time):
   
        if self.current_batch > self.total_batches:
            self.job_complete = True
            return  # Stop processing if all batches are done
        if self.cache.check_cache(self.current_batch, current_time):
            logger.debug(f"Time {current_time}s: Job {self.job_id} processed batch {self.current_batch} from cache. Hits = {self.cache.cache_hits}, Misses = {self.cache.cache_misses}")
        else:
            logger.debug(f"Time {current_time}s: Job {self.job_id} processed batch {self.current_batch} after cache miss. Hits = {self.cache.cache_hits}, Misses = {self.cache.cache_misses}")
            if self.cache.add_batch(self.current_batch, current_time):
                logger.debug(f"Time {current_time}s: Job {self.job_id} added batch {self.current_batch} to cache.")

        next_event_time = current_time + self.speed
        heapq.heappush(self.event_queue, (next_event_time, self.job_id))
        self.current_batch += 1


def compute_aws_lmabda_cost(num_requests, memory_size=2048, execution_time=0.12):
    """Compute the cost of running AWS Lambda function."""
    # Cost per 1M requests
    cost_per_million = 0.20
    # Memory allocation in GB-seconds
    memory_gb_seconds = memory_size * execution_time
    # Total cost
    total_cost = cost_per_million / 1e6 * num_requests + memory_gb_seconds / 1e9
    return total_cost

def run_sim(num_batches, max_cache_size, ttl_seconds, job_speeds, keep_alive_interval):
    cache = TTLCache(max_cache_size, len(job_speeds), ttl_seconds)
    jobs = []
    event_queue = []

    # Initialize jobs
    for i, speed in enumerate(job_speeds):
        job = Job(str(i + 1), speed, cache, event_queue, num_batches)
        jobs.append(job)
        heapq.heappush(event_queue, (0, job.job_id))

    # Schedule the first keep-alive event
    heapq.heappush(event_queue, (keep_alive_interval, "keep-alive")) 
    current_time = 0

    #while all jobs are not complete
    while not all(job.job_complete for job in jobs):
        current_time, event_type = heapq.heappop(event_queue)
        if event_type == "keep-alive":
            # Trigger keep-alive and schedule next one
            cache.refresh_all_ttls(current_time)
            heapq.heappush(event_queue, (current_time + keep_alive_interval, "keep-alive"))
        else:
            # Process job event
            job:Job = next(j for j in jobs if j.job_id == event_type)
            job.process_batch(current_time)
    
    logger.info(f"Duration of simulation: {current_time:.2f}s")
    if cache.cache_size_log:
        max_cache_size = max(cache.cache_size_log, key=lambda x: x[1])
        logger.info(f"Max cache size reached: {max_cache_size[1]} at time {max_cache_size[0]:.2f}s")
    else:
        logger.info(f"No batches were stored in the cache.")
    logger.info(f"Cache Hits: {cache.cache_hits}, Cache Misses: {cache.cache_misses},  Cache hit ratio: {cache.calculate_hit_ratio() * 100:.2f}%")
     
    total_requests = sum(cache.cache_requests.values())
    logger.info(f"Cache Requests: {total_requests}")
    for key, value in cache.cache_requests.items():
        logger.info(f"  {key}: {value} ({value / total_requests * 100:.2f}%)")
   
    logger.info(f"Cache Cost: {compute_aws_lmabda_cost(total_requests):.2f}")
    logger.info("-" * 40)


# Run the simulation
num_batches = 9000 * 60
max_cache_size = np.inf
job_speeds = [1, 1.03]
ttl_seconds = 120
keep_alive_interval = np.inf  # Keep-alive runs every 30 seconds
run_sim(num_batches, max_cache_size, ttl_seconds, job_speeds, keep_alive_interval)