import heapq
import numpy as np
import logging
from typing import List, Tuple, Dict, Set, Any
import sys
sys.path.append(".")
from simulation.sim_utils import workloads, calculate_elasticache_serverless_cost, save_dict_list_to_csv, gen_report_data
import os
import csv
import time
import collections
from typing import Dict, Any, Set
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


import collections
import random
import time
from typing import Any

class SharedCache:
    """
    Tracks which jobs have seen each batch and evicts according to a pluggable policy.
    Supported policies: "lru", "fifo", "mru", "random", "noevict"
    """
    def __init__(
        self,
        cache_capacity_gb: float,
        size_per_batch_gb: float,
        num_jobs: int,
        eviction_policy: str = "noevict",
    ):
        # batch_id -> set of job_ids that have consumed it
        # OrderedDict so we can do LRU/FIFO/MRU easily
        self.cache = collections.OrderedDict()
        # track insertion time for FIFO (and TTL if you add it later)
        self._timestamps = {}
        self.size_per_batch_gb = size_per_batch_gb
        self.cache_capacity_gb = cache_capacity_gb
        # self.max_batches = int(cache_capacity_gb / size_per_batch_gb)
        self.num_jobs = num_jobs

        allowed = {"lru", "fifo", "mru", "random", "noevict"}
        if eviction_policy not in allowed:
            raise ValueError(f"Unknown policy {eviction_policy!r}")
        self.policy = eviction_policy

    def access(self, batch_id: Any, job_id: int) -> bool:
        """
        Job `job_id` requests `batch_id`.  
        Returns True on hit (and first time this job sees it), False on miss.
        """
        if batch_id in self.cache:
            self.cache[batch_id] += 1

            # LRU and MRU need to reorder on access
            if self.policy in ("lru", "mru"):
                self.cache.move_to_end(batch_id)

            # Check if the batch has been accessed by all jobs, and evict if yes
            if self.cache[batch_id] >= self.num_jobs:
                logger.debug(f"Evicting batch {batch_id} from cache")
                self.cache.pop(batch_id, None)
            return True
        else:
            if self.cache_is_full():
                # Cache is full, evict according to policy
                if self.policy != "noevict":
                    self._evict_one()
                else:
                    return False
                    
            # insert new batch
            if not self.cache_is_full():
                self.cache[batch_id] = 1
                self._timestamps[batch_id] = time.time()
            return False
    
    def cache_is_full(self):
        # Check if the cache is full based on the size of the batches and the maximum cache size
        return (len(self.cache) + 1) * self.size_per_batch_gb >= self.cache_capacity_gb
    
    def _remove(self, batch_id: Any):
        """Helper to delete from both cache and timestamp dict."""
        self.cache.pop(batch_id, None)
        self._timestamps.pop(batch_id, None)

    def _evict_one(self):
        """Evict exactly one batch according to self.policy."""
        if self.policy == "lru":
            return self.cache.popitem(last=False)

        if self.policy == "fifo":
            # FIFO = oldest insertion
            oldest = min(self._timestamps, key=self._timestamps.get)
            return self._remove(oldest)

        if self.policy == "mru":
            # MRU = most recently used
            return self.cache.popitem(last=True)

        if self.policy == "random":
            victim = random.choice(list(self.cache.keys()))
            return self._remove(victim)

        # noevict should never reach here, but just in case
        return

    def current_usage_gb(self, size_per_batch_gb: float) -> float:
        return len(self.cache) * size_per_batch_gb

class DLTJOB():
    def __init__(self, job_id, speed, cache:SharedCache):
        self.job_id = job_id
        self.cache = cache  # Reference to the shared cache
        self.speed = speed  # Speed in GPU time per batch (seconds)
        self.job_progress = 0  # Tracks how many batches the job has completed
        self.cache_hit_count = 0  # Tracks cache hits for this job
        self.cache_miss_count = 0  # Tracks cache misses for this job
        self.elapased_time_sec = 0  # Tracks the current time for this job
        self.throughput = 0  # Tracks the throughput for this job
        self.compute_cost = 0  # Tracks the compute cost for this job
    
    def next_training_step(self, current_time_sec):
        self.elapased_time_sec = current_time_sec        
        cache_hit = self.cache.access(self.job_progress+1, self.job_id)  # Simulate cache access
        if cache_hit:
            self.cache_hit_count +=1
        else:
            self.cache_miss_count += 1
        self.job_progress += 1
        return cache_hit

    
    def get_job_progress(self, ):
        return self.job_progress

    def __lt__(self, other):
        return self.speed < other.speed  # Compare based on speed
    
    def get_performance(self, sim_id, ec2_cost=12.24, cache_cost=3.25):
        # Calculate performance metrics for this job
        cache_hit_rate = self.cache_hit_count / (self.cache_hit_count + self.cache_miss_count) if (self.cache_hit_count + self.cache_miss_count) > 0 else 0
        throughput = self.job_progress / self.elapased_time_sec if self.elapased_time_sec > 0 else 0
        # Calculate costs
        self.compute_cost = (ec2_cost / 3600) * self.elapased_time_sec
        cache_cost = (cache_cost / 3600) * self.elapased_time_sec
        total_cost = self.compute_cost + cache_cost
        return {
            'sim_id': sim_id,
            'dataloader': 'coordl',
            'job_id': self.job_id,
            'job_speed': self.speed,
            'cache_capacity_gb': self.cache.cache_capacity_gb,
            'bacthes_processed': self.job_progress,
            'cache_hit_count': self.cache_hit_count,
            'cache_miss_count': self.cache_miss_count,
            'cache_hit_%': cache_hit_rate,
            'elapsed_time': self.elapased_time_sec,
            'throughput(batches/s)': throughput,
            'compute_cost': self.compute_cost,
            'cache_cost': cache_cost,
            'total_cost': total_cost,
        }

def run_simulation(
        sim_id,
        workload_name,
        workload_jobs,
        cache_capacity_gb,
        size_per_batch_gb,
        cache_miss_penalty,
        hourly_cache_cost,
        hourly_ec2_cost=12.24,
        simulation_time_sec=3600,
        batches_per_job=np.inf,
        eviction_policy="uniform",
        use_elasticache_severless_pricing=False):
    
    shared_cache = SharedCache(
        cache_capacity_gb = cache_capacity_gb,
        size_per_batch_gb=size_per_batch_gb,
        num_jobs = len(workload_jobs),
        eviction_policy=eviction_policy)
    
    jobs:List[DLTJOB] = [DLTJOB(model_name, speed, shared_cache) for model_name, speed in workload_jobs]
    cache_size_over_time = []  # Store cache size over time for plotting

    event_queue = []  # Priority queue for next event times
    for job in jobs:
        heapq.heappush(event_queue, (job.speed, job))
    
    time_elapsed = 0  # Global simulation time

    while True:
        if simulation_time_sec is not None and time_elapsed >= simulation_time_sec:
            break
        #check all jobs have completed their batches
        if batches_per_job is not None and all(job.job_progress >= batches_per_job for job in jobs):
            break

        time_elapsed, job = heapq.heappop(event_queue)
        job:DLTJOB = job  # Get next job event
        cache_hit = job.next_training_step(time_elapsed)  # Simulate the next training step for this job
        #schudle the next event for this job if it has not completed its batches
        if cache_hit:
            next_event_time = time_elapsed + job.speed
        else:
            next_event_time = time_elapsed + job.speed + cache_miss_penalty
        
        if batches_per_job is None or job.job_progress < batches_per_job: #job has not completed its batches
            heapq.heappush(event_queue, (next_event_time, job))

        cache_size = shared_cache.current_usage_gb(size_per_batch_gb)

        cache_size_over_time.append(cache_size)  # Store cache size over time
    
    job_performances = [job.get_performance(sim_id, hourly_ec2_cost/len(jobs), hourly_cache_cost/len(jobs)) for job in jobs]

    coordl_overall_results = gen_report_data(
            dataloader_name = 'baseline',
            job_performances = job_performances,
            cache_size_over_time = cache_size_over_time,
            eviction_policy = eviction_policy,
            size_per_batch_gb = size_per_batch_gb,
            cache_capacity_gb = cache_capacity_gb,
            cache_miss_penalty = cache_miss_penalty,
            hourly_ec2_cost = hourly_ec2_cost,
            hourly_cache_cost = hourly_cache_cost,
            sim_id = str(int(time.time())),
            workload_name = workload_name,
            use_elasticache_severless_pricing = use_elasticache_severless_pricing
        )
    return job_performances, coordl_overall_results
   
if __name__ == "__main__":

    #print name variable name 'imagenet_128_batch_size'
    workload_name = 'imagenet_128'
    workload =  workloads[workload_name]
    simulation_time_sec = None #3600 # None  #3600 * 1 # Simulate 1 hour
    max_batches_per_job = 8500 # 8500 #np.inf
    hourly_ec2_cost = 12.24 
    hourly_cache_cost = 3.25
    cache_capacity_gb = np.inf# 100 #np.inf
    size_per_batch_gb = 20 / 1024
    cache_miss_penalty = 0
    use_elasticache_severless_pricing = False
    eviction_policy = "noevict" # "lru", "fifo", "mru", "random", "noevict"

    job_performances, coordl_overall_results = run_simulation(
        sim_id = str(int(time.time())),
        workload_name = workload_name,
        workload_jobs = workload.items(),
        cache_capacity_gb=cache_capacity_gb,
        size_per_batch_gb = size_per_batch_gb,
        cache_miss_penalty = cache_miss_penalty,
        hourly_cache_cost = hourly_cache_cost,
        simulation_time_sec=simulation_time_sec,
        batches_per_job=max_batches_per_job,
        eviction_policy=eviction_policy,
        use_elasticache_severless_pricing = use_elasticache_severless_pricing
    )
    
    #save overall results to a file
    report_folder = os.path.join(os.getcwd(), "simulation", "reports", workload_name)
    os.makedirs(report_folder, exist_ok=True)
    
    overall_report_file = os.path.join(report_folder, "overall_results.csv")
    save_dict_list_to_csv([coordl_overall_results], overall_report_file)

    job_performance_file = os.path.join(report_folder, "job_results.csv")
    save_dict_list_to_csv(job_performances, job_performance_file)


