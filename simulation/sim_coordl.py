import heapq
import numpy as np
import logging
from typing import List, Tuple, Dict, Set, Any
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RedisCache:
    def __init__(self, 
                 max_cache_size_gb,
                 size_per_batch_gb, 
                 num_jobs, 
                 eviction_policy="LRU"):
        
        self.data:Dict[Any,Set] = {}  # Stores cached batches
        self.size_per_batch_gb = size_per_batch_gb
        self.num_jobs = num_jobs
        self.max_cache_size_gb = max_cache_size_gb
        self.eviction_policy = eviction_policy
        self.num_jobs = num_jobs
        self.batch_ref_count = {} #counter for the number of times a batch has been accessed
        self.cache_size_over_time = []

    def handle_request(self, key, job_id):
        cache_hit = False
        if key in self.data:
            logger.debug(f"Cache hit: Job {job_id} accessed {key}")
            self.batch_ref_count[key] = self.batch_ref_count.get(key, 0) + 1
            if self.batch_ref_count[key] >= self.num_jobs:
                self.data.pop(key, None)                 #evict the batch from the cache
            cache_hit = True
        else:
            # Check if the cache is full and apply eviction policy
            if (len(self.data) + 1) * self.size_per_batch_gb >= self.max_cache_size_gb:
                    pass
                # if self.eviction_policy == "LRU":
            else:
                # Add the new batch to the cache
                self.data[key] = set()
                logger.debug(f"Cache miss: Job {job_id} added {key} to cache")
                self.data[key].add(job_id)

                #Check if its the first time the batch is added to the cache
                if self.batch_ref_count.get(key, 0) == 0:
                    cache_hit = True
                self.batch_ref_count[key] = self.batch_ref_count.get(key, 0) + 1 #cache get(key, 1) for now to assume prefetching, but it should be 0
        self.cache_size_over_time.append(len(self.data))
        return cache_hit

def calculate_elasticache_serverless_cost(
    average_gb_usage: float,
    duration_hours: float = 1,  # default to one hour
    price_per_gb_hour: float = 0.125,
    ecpu_cost: float = 0.0
) -> dict:
    gb_hours = average_gb_usage * duration_hours
    storage_cost = gb_hours * price_per_gb_hour
    total_cost = storage_cost + ecpu_cost
    return total_cost
   
def run_coordl_simulation(
        job_speeds,
        max_cache_size_gb,
        size_per_batch_gb,
        cache_miss_penalty,
        hourly_cache_cost,
        hourly_ec2_cost=12.24,
        simulation_time_sec=3600,
        batches_per_job=np.inf,
        use_elasticache_severless_pricing=False):
    
    num_jobs = len(job_speeds)
    job_progress = [0] * num_jobs  # Tracks how many batches each job has completed
    event_queue = []  # Priority queue for next event times
    cache_miss_count = 0
    cache_hit_count = 0
    
    cache = RedisCache(
        max_cache_size_gb = max_cache_size_gb,
        size_per_batch_gb=size_per_batch_gb,
        num_jobs = num_jobs
    )
   
    for job_id, speed in enumerate(job_speeds):
        heapq.heappush(event_queue, (speed, job_id))  # (time to complete first batch, job_id)

    time_elapsed = 0  # Global simulation time

    while True:
        if simulation_time_sec is not None and time_elapsed >= simulation_time_sec:
            break
        #check all jobs have completed their batches
        if batches_per_job is not None and all(job_progress[job_id] >= batches_per_job for job_id in range(num_jobs)):
            break
        
        time_elapsed, job_id = heapq.heappop(event_queue)  # Get next job event
        current_batch = job_progress[job_id]  # Current batch for the job
        cache_hit = cache.handle_request(current_batch, job_id)  # Handle cache request
        if cache_hit:
            cache_hit_count +=1
            next_event_time = time_elapsed + (job_speeds[job_id])
        else:
            cache_miss_count += 1
            next_event_time = time_elapsed + (job_speeds[job_id]) + cache_miss_penalty
        
        job_progress[job_id] += 1
        if batches_per_job is not None and job_progress[job_id] < batches_per_job:
            heapq.heappush(event_queue, (next_event_time, job_id))
        else:
            heapq.heappush(event_queue, (next_event_time, job_id))

    
    total_batches_processed = sum(job_progress)
    throughput = total_batches_processed / time_elapsed  # Batches per second
    compute_cost = (hourly_ec2_cost / 3600) * time_elapsed


    # cache_cost = hourly_cache_cost
    cache_hit_percent = (cache_hit_count / (cache_hit_count + cache_miss_count)) * 100 if (cache_hit_count + cache_miss_count) > 0 else 0
    max_cached_bacthes = max(cache.cache_size_over_time)
    max_cache_capacity_used = max_cached_bacthes * size_per_batch_gb
    if use_elasticache_severless_pricing:
        cache_cost = calculate_elasticache_serverless_cost(average_gb_usage=max_cache_capacity_used)
    else:
        cache_cost = (hourly_cache_cost / 3600) * time_elapsed
    total_cost = compute_cost + cache_cost  # No additional costs in this simulation

    #resote these results in a dict to be used in the report
    results = {
        'cache_miss_count': cache_miss_count,
        'cache_hit_count': cache_hit_count,
        'cache_hit_percent': cache_hit_percent,
        'total_batches_processed': total_batches_processed,
        'time_elapsed': time_elapsed,
        'throughput': throughput,
        'compute_cost': compute_cost,
        'cache_cost': cache_cost,
        'total_cost': total_cost,
        'max_cache_capacity_used_gb': max_cache_capacity_used,
        'max_cached_bacthes': max_cached_bacthes
    }
    
    #print some results
    print(f"CoorDL:")
    print(f"  Time: {time_elapsed:.2f} seconds")
    print(f"  Cache Size: {max_cache_size_gb} GB")
    print(f"  Cache Used: {max_cache_capacity_used:.4f} GB")
    # print(f"  Max Cached Batches: {max_cached_bacthes} GB")
    print(f"  Cache Miss Count: {cache_miss_count}")
    print(f"  Cache Hit Count: {cache_hit_count}") 
    print(f"  Cache Hit Percentage: {cache_hit_percent:.2f}%")
    print(f"  Total Batches Processed: {total_batches_processed}")
    print(f"  Elapsed Time: {time_elapsed:.2f}s, {time_elapsed/60:.2f} min")
    print(f"  Overall Throughput: {throughput:.2f} batches/sec")
    print(f"  Cache Cost: ${cache_cost:.2f}")
    print(f"  Compute Cost: ${compute_cost:.2f}")
    print(f"  Total Cost : ${total_cost:.2f}")
    print("-" * 40)
    return results
   
if __name__ == "__main__":
    from simulation.models_v100_times import RESNET18, RESNET50, SHUFFLENETV, VGG16
    models = [RESNET18, RESNET50, SHUFFLENETV, VGG16]
    simulation_time_sec =  None #3600 * 5 # Simulate 1 hour
    max_batches_per_job = 8500 #np.inf
    hourly_ec2_cost = 12.24 
    hourly_cache_cost = 3.25
    redis_cache_size_gb = 150 #np.inf
    size_per_batch_gb = 20 / 1024
    cache_miss_penalty = 0
    use_elasticache_severless_pricing = False

    run_coordl_simulation(
        job_speeds = models,
        max_cache_size_gb=redis_cache_size_gb,
        size_per_batch_gb = size_per_batch_gb,
        cache_miss_penalty = cache_miss_penalty,
        hourly_cache_cost = hourly_cache_cost,
        simulation_time_sec=simulation_time_sec,
        batches_per_job=max_batches_per_job,
        use_elasticache_severless_pricing = use_elasticache_severless_pricing
    )

