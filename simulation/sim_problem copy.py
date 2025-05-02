import heapq
import numpy as np
import logging
from typing import List, Tuple, Dict, Set, Any
import sys
sys.path.append(".")
from simulation.sim_utils import workloads, save_dict_list_to_csv, gen_report_data
import os
import csv
import time
import collections
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SharedCache:
    """
    Tracks which jobs have seen each batch and evicts according to a pluggable policy.
    Supported policies: "lru", "fifo", "mru", "random", "noevict", "cus"
    """
    def __init__(
        self,
        cache_capacity_gb: float,
        size_per_batch_gb: float,
        num_jobs: int,
        eviction_policy: str = "noevict",
    ):
        self.cache = collections.OrderedDict()  # batch_id -> set of seen jobs
        self._timestamps = {}
        self._seen_by = collections.defaultdict(set)  # batch_id -> job_ids
        self.size_per_batch_gb = size_per_batch_gb
        self.cache_capacity_gb = cache_capacity_gb
        self.num_jobs = num_jobs

        allowed = {"lru", "fifo", "mru", "random", "noevict", "cus"}
        if eviction_policy not in allowed:
            raise ValueError(f"Unknown policy {eviction_policy!r}")
        self.policy = eviction_policy

        # job iteration time estimates for CUS
        self.job_weights = {}  # set externally

    def access(self, batch_id: Any, job_id: int) -> bool:
        """
        Job `job_id` requests `batch_id`.  
        Returns True on hit (and first time this job sees it), False on miss.
        """
        if batch_id in self.cache:
            self._seen_by[batch_id].add(job_id)
            if self.policy in ("lru", "mru"):
                self.cache.move_to_end(batch_id)
            if len(self._seen_by[batch_id]) >= self.num_jobs:
                logger.debug(f"Evicting batch {batch_id} from cache")
                self._remove(batch_id)
            return True
        else:
            if self.cache_is_full():
                if self.policy != "noevict":
                    self._evict_one()
                else:
                    return False
            if not self.cache_is_full():
                self.cache[batch_id] = True
                self._seen_by[batch_id].add(job_id)
                self._timestamps[batch_id] = time.time()
            return False

    def cache_is_full(self):
        return (len(self.cache) + 1) * self.size_per_batch_gb >= self.cache_capacity_gb

    def _remove(self, batch_id: Any):
        self.cache.pop(batch_id, None)
        self._timestamps.pop(batch_id, None)
        self._seen_by.pop(batch_id, None)

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
        if self.policy == "cus":
            def cus_score(batch_id):
                unseen = [j for j in self.job_weights if j not in self._seen_by[batch_id]]
                return sum(self.job_weights[j] for j in unseen)
            victim = min(self.cache.keys(), key=cus_score)
            logger.debug(f"[CUS] Evicting {victim} with utility {cus_score(victim):.2f}")
            return self._remove(victim)

    def current_usage_gb(self, size_per_batch_gb: float) -> float:
        return len(self.cache) * size_per_batch_gb

    def prefetch(self, batch_id: Any) -> None:
        if batch_id in self.cache:
            return
        if self.cache_is_full() and self.policy != "noevict":
            self._evict_one()
        if not self.cache_is_full():
            self.cache[batch_id] = True
            self._timestamps[batch_id] = time.time()

class DLTJOB:
    def __init__(self, job_id, speed, cache: SharedCache):
        self.job_id = job_id
        self.cache = cache
        self.speed = speed
        self.job_progress = 0
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.elapased_time_sec = 0
        self.throughput = 0
        self.compute_cost = 0

    def next_training_step(self, current_time_sec):
        self.elapased_time_sec = current_time_sec
        cache_hit = self.cache.access(self.job_progress + 1, self.job_id)
        if cache_hit:
            self.cache_hit_count += 1
        else:
            self.cache_miss_count += 1
        self.job_progress += 1
        return cache_hit

    def get_performance(self, sim_id, ec2_cost=12.24, cache_cost=3.25):
        hit_rate = self.cache_hit_count / (self.cache_hit_count + self.cache_miss_count) if (self.cache_hit_count + self.cache_miss_count) > 0 else 0
        throughput = self.job_progress / self.elapased_time_sec if self.elapased_time_sec > 0 else 0
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
            'cache_hit_%': hit_rate,
            'elapsed_time': self.elapased_time_sec,
            'throughput(batches/s)': throughput,
            'compute_cost': self.compute_cost,
            'cache_cost': cache_cost,
            'total_cost': total_cost,
        }

def run_simulation(
        workload_name,
        workload_jobs,
        cache_capacity,
        eviction_policy,
        cache_miss_penalty,
        hourly_cache_cost,
        hourly_ec2_cost
        cache_miss_penalty,
        simulation_time_sec,
        batches_per_job=,
        prefetch_distance=1, 
        prefetch_latency=0.01):
    
    shared_cache = SharedCache(
        cache_capacity_gb = cache_capacity_gb,
        size_per_batch_gb=size_per_batch_gb,
        num_jobs = len(workload_jobs),
        eviction_policy=eviction_policy)
    
    jobs:List[DLTJOB] = [DLTJOB(model_name, speed, shared_cache) for model_name, speed in workload_jobs]


    cache_size_over_time = []  # Store cache size over time for plotting
    #create dict of jobid:speed
    job_speeds = {job.job_id: job.speed for job in jobs}    
    job_weights = {j: 1.0 / job_speeds[j] for j in job_speeds}
    shared_cache.job_weights = job_weights
    event_queue = []  # Priority queue for next event times
    for job in jobs:
        heapq.heappush(event_queue, (job.speed, "step", job))
    
    time_elapsed = 0  # Global simulation time

    while True:
        if simulation_time_sec is not None and time_elapsed >= simulation_time_sec:
            break
        #check all jobs have completed their batches
        if batches_per_job is not None and all(job.job_progress >= batches_per_job for job in jobs):
            break
        time_elapsed, event_type, payload = heapq.heappop(event_queue)
        if event_type == "step":
            job:DLTJOB = payload
            cache_hit = job.next_training_step(time_elapsed)  # Simulate the next training step for this job
            delay = job.speed + (0 if cache_hit else cache_miss_penalty)
            if batches_per_job is None or job.job_progress < batches_per_job: #job has not completed its batches
                heapq.heappush(event_queue, (time_elapsed + delay, "step", job))
            
            if prefetch_distance >0:
                # schedule a prefetch for the K-th future batch
                next_batch_id = job.job_progress + prefetch_distance
                heapq.heappush(event_queue, (
                    time_elapsed + prefetch_latency,
                    "prefetch",
                    next_batch_id
                ))
        elif event_type == "prefetch":
            batch_id = payload
            shared_cache.prefetch(batch_id)

            # Schedule the next event for this job if it has not completed its batches

        cache_size_over_time.append(shared_cache.current_usage_gb(size_per_batch_gb))  # Store cache size over time
    
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
    workload_name = 'imagenet_128_nas'
    jobs =  workloads[workload_name].items()
    simulation_time_sec = None #3600 # None  #3600 * 1 # Simulate 1 hour
    batches_per_job = 3000 * 1 # 8500 #np.inf
    cache_capacity = 0.1 * batches_per_job #number of batches as a % of the total number of batches
    eviction_policy = "noevict" # "lru", "fifo", "mru", "random", "noevict"
    hourly_ec2_cost = 12.24 
    hourly_cache_cost = 3.25
    cache_miss_penalty = 0.2


    
    job_performances, coordl_overall_results = run_simulation(
        sim_id = str(int(time.time())),
        workload_name = workload_name,
        workload_jobs = jobs,
        cache_capacity = cache_capacity,
        cache_miss_penalty = cache_miss_penalty,
        hourly_cache_cost = hourly_cache_cost,
        simulation_time_sec=simulation_time_sec,
        batches_per_job=max_batches_per_job,
        eviction_policy=eviction_policy,
        use_elasticache_severless_pricing = use_elasticache_severless_pricing,
        prefetch_distance=prefetch_distance,
        prefetch_latency=prefetch_latency
    )
    
    #save overall results to a file
    report_folder = os.path.join(os.getcwd(), "simulation", "reports", workload_name)
    os.makedirs(report_folder, exist_ok=True)
    
    overall_report_file = os.path.join(report_folder, "overall_results.csv")
    save_dict_list_to_csv([coordl_overall_results], overall_report_file)

    job_performance_file = os.path.join(report_folder, "job_results.csv")
    save_dict_list_to_csv(job_performances, job_performance_file)


