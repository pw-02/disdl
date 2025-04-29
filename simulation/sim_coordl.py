import heapq
import numpy as np
import logging
from typing import List, Tuple, Dict, Set, Any
import sys
sys.path.append(".")
from simulation.workloads import workloads, calculate_elasticache_serverless_cost, save_dict_list_to_csv, gen_report_data
import os
import csv
import time
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CoorDLCache:
    def __init__(self, 
                 cache_capacity_gb,
                 size_per_batch_gb, 
                 num_jobs, 
                 eviction_policy="uniform"):
        
        self.data:Dict[Any,int] = {}  # Stores cached batches, and reference counts or the number of times a batch has been accessed
        self.size_per_batch_gb = size_per_batch_gb
        self.num_jobs = num_jobs
        self.cache_capacity_gb = cache_capacity_gb
        self.eviction_policy = eviction_policy
        self.num_jobs = num_jobs
    
    def get(self, key, job_id):
        # Simulate cache access and return the batch if it exists
        if key in self.data:
            logger.debug(f"Cache hit: Job {job_id} accessed {key}")
            self.data[key] += 1 # Increment the reference count for the batch
            # Check if the batch has been accessed by all jobs, and evict if cache policy is uniform
            if self.data[key] >= self.num_jobs and self.eviction_policy == "uniform":
                logger.debug(f"Evicting batch {key} from cache")
                self.data.pop(key, None)
            return True
        else:
            logger.debug(f"Cache miss: Job {job_id} could not access {key}")
            # Check if the cache is full and apply eviction policy if necessary
            if self.cache_is_full() and self.eviction_policy not in ["None", "uniform"]:
                self.run_evition_policy()
            #Add the new batch to the cache
            if not self.cache_is_full():
                self.data[key] = 1
            
            return False
    
    def cache_is_full(self):
        # Check if the cache is full based on the size of the batches and the maximum cache size
        return (len(self.data) + 1) * self.size_per_batch_gb >= self.cache_capacity_gb
    
    def run_evition_policy(self):
        # Implement the eviction policy here (e.g., LRU, LFU, etc.)
        if self.eviction_policy == "LRU":
            # Find the least recently used batch and evict it
            lru_key = min(self.data, key=self.data.get)
            logger.debug(f"Evicting batch {lru_key} from cache (LRU)")
            self.data.pop(lru_key, None)
    
    def get_cache_size(self):
        # Calculate the current cache size in GB
        return len(self.data) * self.size_per_batch_gb
    
    # def get_cache_length(self):
    #     # Calculate the current cache size in GB
    #     return len(self.data) * self.size_per_batch_gb

class DLTJOB():
    def __init__(self, job_id, speed, cache:CoorDLCache):
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
        cache_hit = self.cache.get(self.job_progress+1, self.job_id)  # Simulate cache access
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

def run_coordl_simulation(
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
        use_elasticache_severless_pricing=False):
    
    shared_cache = CoorDLCache(cache_capacity_gb = cache_capacity_gb,
                       size_per_batch_gb=size_per_batch_gb,
                       num_jobs = len(workload_jobs))
    
    jobs:List[DLTJOB] = [DLTJOB(model_name, speed, shared_cache) for model_name, speed in workload_jobs]
    cache_size_over_time = []  # Store cache size over time for plotting
    
    #set tying for evenet queue
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
        
        if batches_per_job is None or job.job_progress < batches_per_job:
            heapq.heappush(event_queue, (next_event_time, job))

        cache_size = shared_cache.get_cache_size()

        cache_size_over_time.append(cache_size)  # Store cache size over time
    
    job_performances = [job.get_performance(sim_id, hourly_ec2_cost/len(jobs), hourly_cache_cost/len(jobs)) for job in jobs]

    coordl_overall_results = gen_report_data(
            dataloader_name = 'coordl',
            job_performances = job_performances,
            cache_size_over_time = cache_size_over_time,
            eviction_policy = "coordl",
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
    cache_capacity_gb = 100 #np.inf
    size_per_batch_gb = 20 / 1024
    cache_miss_penalty = 0
    use_elasticache_severless_pricing = False

    job_performances, coordl_overall_results = run_coordl_simulation(
        sim_id = str(int(time.time())),
        workload_name = workload_name,
        workload_jobs = workload.items(),
        cache_capacity_gb=cache_capacity_gb,
        size_per_batch_gb = size_per_batch_gb,
        cache_miss_penalty = cache_miss_penalty,
        hourly_cache_cost = hourly_cache_cost,
        simulation_time_sec=simulation_time_sec,
        batches_per_job=max_batches_per_job,
        use_elasticache_severless_pricing = use_elasticache_severless_pricing
    )
    
    #save overall results to a file
    report_folder = os.path.join(os.getcwd(), "simulation", "reports", workload_name)
    os.makedirs(report_folder, exist_ok=True)
    
    overall_report_file = os.path.join(report_folder, "overall_results.csv")
    save_dict_list_to_csv([coordl_overall_results], overall_report_file)

    job_performance_file = os.path.join(report_folder, "job_results.csv")
    save_dict_list_to_csv(job_performances, job_performance_file)


