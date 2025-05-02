import heapq
import numpy as np
import logging
from typing import List, Tuple, Dict, Set, Any
import sys
sys.path.append(".")
from simulation.sim_workloads import workloads, save_dict_list_to_csv, gen_report_data
import os
import csv
import time
import collections
import random

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log',         # Log file name
    filemode='w'                # Overwrite the file each run; use 'a' to append
)

logger = logging.getLogger(__name__)

class SharedCache:
    """
    Tracks which jobs have seen each batch and evicts according to a pluggable policy.
    Supported policies: "lru", "fifo", "mru", "random", "noevict", "cus"
    """
    def __init__(
        self,
        cache_capacity: int, #number of batches
        num_jobs: int,
        eviction_policy: str = "noevict",
    ):
        self.cache = collections.OrderedDict()  # batch_id -> set of seen jobs
        self._timestamps = {}
        self.cache_capacity = cache_capacity
        self.num_jobs = num_jobs

        allowed = {"lru", "fifo", "mru", "random", "noevict", "cus"}
        if eviction_policy not in allowed:
            raise ValueError(f"Unknown policy {eviction_policy!r}")
        self.policy = eviction_policy

        # job iteration time estimates for CUS
        self.job_weights = {}  # set externally

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

        if self.cache_is_full() and self.policy != "noevict":
            logger.debug(f"Cache is full: {len(self.cache)} >= {self.cache_capacity}")
            self._evict_one()

        if not self.cache_is_full():
            self.cache[batch_id] = True
            self._timestamps[batch_id] = time.time()
            logger.debug(f"Added batch {batch_id} to cache")

    def cache_is_full(self):
        #check if cache is full based on max acpacity and current cache size

        if (len(self.cache) + 1) > self.cache_capacity:
            logger.debug(f"Cache is full: {len(self.cache)} >= {self.cache_capacity}")
            return True
        else:
            return False
        
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
        # if self.policy == "cus":
        #     def cus_score(batch_id):
        #         unseen = [j for j in self.job_weights if j not in self._seen_by[batch_id]]
        #         return sum(self.job_weights[j] for j in unseen)
        #     victim = min(self.cache.keys(), key=cus_score)
        #     logger.debug(f"[CUS] Evicting {victim} with utility {cus_score(victim):.2f}")
        #     return self._remove(victim)

    def current_usage_gb(self, size_per_batch_gb: float) -> float:
        return len(self.cache) * size_per_batch_gb
    
    def current_length(self) -> int:
        return len(self.cache)

    def prefetch(self, batch_id: Any) -> None:
        if batch_id in self.cache:
            return
        if self.cache_is_full() and self.policy != "noevict":
            self._evict_one()
        if not self.cache_is_full():
            self.cache[batch_id] = True
            self._timestamps[batch_id] = time.time()



class CoorDLSampler:
    def __init__(self, total_batches: int, job_ids: List[str], cache: Any, assignment_strategy: str = "coordinated"):
        self.total_batches = total_batches
        self.job_ids = job_ids
        self.assignment_strategy = assignment_strategy
        self.cache = cache

        self.job_to_batches = self._assign_batches()
        self.batch_seen_by = collections.defaultdict(set)

        self.global_sequence = self._build_global_schedule()  # List of (batch_id, owner)
        self.job_pointers = {jid: 0 for jid in job_ids}

    def _assign_batches(self) -> Dict[str, List[int]]:
        """
        Round-robin batch assignment.
        """
        job_batches = {jid: [] for jid in self.job_ids}
        for i, batch_id in enumerate(range(1, self.total_batches + 1)):
            jid = self.job_ids[i % len(self.job_ids)]
            job_batches[jid].append(batch_id)
        return job_batches

    def _build_global_schedule(self) -> List[tuple]:
        """
        Interleaved sequence of all batches with their owner.
        """
        max_len = max(len(b) for b in self.job_to_batches.values())
        schedule = []
        for i in range(max_len):
            for jid in self.job_ids:
                if i < len(self.job_to_batches[jid]):
                    schedule.append((self.job_to_batches[jid][i], jid))
        return schedule

    def get_batch_for_job(self, job_id: str, job_progress: int) -> tuple:
        """
        Returns the next batch in the global schedule for this job.
        Skips over batches not in the job's assignment, but still includes them for coordination.
        Returns a tuple: (batch_id, self_load_flag)
        """
        ptr = self.job_pointers[job_id]
        while ptr < len(self.global_sequence):
            batch_id, owner = self.global_sequence[ptr]
            self.job_pointers[job_id] += 1
            self_load = (owner == job_id)
            return batch_id, self_load

        raise IndexError(f"Job {job_id} has no more batches to process.")

    def mark_seen(self, job_id: str, batch_id: int) -> bool:
        """
        Track when each job sees a batch. Evict when all have seen it.
        """
        self.batch_seen_by[batch_id].add(job_id)
        if len(self.batch_seen_by[batch_id]) >= self.cache.num_jobs:
            self.cache._remove(batch_id)
            return True
        return False

class DLTJOB:
    def __init__(self, job_id, speed, cache: SharedCache,
                 sampler:CoorDLSampler):
        self.job_id = job_id
        self.cache = cache
        self.speed = speed
        # self.job_progress = 0
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.elapased_time_sec = 0
        self.throughput = 0
        self.compute_cost = 0
        self.sampler:CoorDLSampler = sampler
        self.current_batch_id = None
        self.job_progress = 0

    def next_training_step(self, current_time_sec):
        self.elapased_time_sec = current_time_sec
        self.job_progress += 1
        batch_id, self_load = self.sampler.get_batch_for_job(self.job_id, self.job_progress)

        if self_load:
            cache_hit = self.cache.get_batch(batch_id)

        if cache_hit:
            logger.debug(f"Cache hit: Job {self.job_id} accessed {batch_id}")
            self.cache_hit_count += 1
        else:
            logger.debug(f"Cache miss: Job {self.job_id} could not access {batch_id}")
            self.cache_miss_count += 1
        self.sampler.mark_seen(self.job_id, batch_id)  # Mark this batch as seen by this job
        return cache_hit, batch_id

    def get_performance(self, horurly_ec2_cost=12.24, hourly_cache_cost=3.25):
        hit_rate = self.cache_hit_count / (self.cache_hit_count + self.cache_miss_count) if (self.cache_hit_count + self.cache_miss_count) > 0 else 0
        throughput = self.job_progress / self.elapased_time_sec if self.elapased_time_sec > 0 else 0
        self.compute_cost = (horurly_ec2_cost / 3600) * self.elapased_time_sec
        cache_cost = (hourly_cache_cost / 3600) * self.elapased_time_sec
        total_cost = self.compute_cost + hourly_cache_cost
        return {
            'job_id': self.job_id,
            'job_speed': self.speed,
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
        hourly_ec2_cost,
        simulation_time_sec,
        batches_per_job,
        batch_assignment_strategy):
    
    shared_cache = SharedCache(
        cache_capacity = cache_capacity,
        num_jobs = len(workload_jobs),
        eviction_policy=eviction_policy)
    
    sampler = CoorDLSampler(total_batches=batches_per_job, job_ids=[job[0] for job in workload_jobs], cache=shared_cache, assignment_strategy=batch_assignment_strategy)

    jobs:List[DLTJOB] = [DLTJOB(model_name, speed, shared_cache, sampler=sampler) for model_name, speed in workload_jobs]

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
        
        if event_type == "cacheinsert":
            batch_id = payload
            shared_cache.put_batch(batch_id)  # Simulate cache insert for this job

        elif event_type == "step":
            job:DLTJOB = payload
            cache_hit, batch_id = job.next_training_step(time_elapsed)  # Simulate the next training step for this job

            if cache_hit:
                job_delay = job.speed
            else: # cache miss
                job_delay = job.speed + cache_miss_penalty
                heapq.heappush(event_queue, (time_elapsed + cache_miss_penalty, "cacheinsert", batch_id))
   
            if batches_per_job is None or job.job_progress < batches_per_job: #job has not completed its batches
                heapq.heappush(event_queue, (time_elapsed + job_delay, "step", job))
       
        elif event_type == "prefetch":
            batch_id = payload
            shared_cache.prefetch(batch_id)

        cache_size_over_time.append(shared_cache.current_length())  # Store cache size over time
    
    job_performances = [job.get_performance(hourly_ec2_cost/len(jobs), hourly_cache_cost/len(jobs)) for job in jobs]

    aggregated_batches_processed = sum(job['bacthes_processed'] for job in job_performances)
    aggregated_cache_hits = sum(job['cache_hit_count'] for job in job_performances)
    aggregated_cache_misses = sum(job['cache_miss_count'] for job in job_performances)
    aggregated_cache_hit_percent = (aggregated_cache_hits / (aggregated_cache_hits + aggregated_cache_misses)) * 100 if (aggregated_cache_hits + aggregated_cache_misses) > 0 else 0
    aggregated_compute_cost = sum(job['compute_cost'] for job in job_performances)
    aggregated_throuhgput = sum(job['throughput(batches/s)'] for job in job_performances)
    aggregated_time_sec = sum(job['elapsed_time'] for job in job_performances)
    elapsed_time_sec = max(job['elapsed_time'] for job in job_performances) if job_performances else 0

    max_cache_capacity_used = max(cache_size_over_time) if cache_size_over_time else 0
    average_cache_capacity_used = np.mean(cache_size_over_time) if cache_size_over_time else 0

    # if use_elasticache_severless_pricing:
    #     cache_cost = calculate_elasticache_serverless_cost(average_gb_usage=average_cache_capacity_used)
    # else:
    cache_cost = (hourly_cache_cost / 3600) * elapsed_time_sec

    # if dataloader_name == 'tensorsocket':
    #     cache_cost = 0

    total_cost = aggregated_compute_cost + cache_cost  # No additional costs in this simulation
    job_speeds = {job['job_id']: job['job_speed'] for job in job_performances}

    overall_results = {
        'workload_name': workload_name,
        'job_speeds': job_speeds,
        'batches_per_job': batches_per_job,
        'cache_capacity': cache_capacity,
        'cache_eviction_policy': eviction_policy,
        'num_jobs': len(job_performances),
        'cache_miss_penalty': cache_miss_penalty,
        'hourly_ec2_cost': hourly_ec2_cost,
        'hourly_cache_cost': hourly_cache_cost,
        'max_cache_capacity': max_cache_capacity_used,
        'average_cache_capacity_used': average_cache_capacity_used,
        'cache_hit_count': aggregated_cache_hits,
        'cache_miss_count': aggregated_cache_misses,
        'cache_hit_percent': aggregated_cache_hit_percent,
        'total_batches_processed': aggregated_batches_processed,
        'time_elapsed': elapsed_time_sec,
        'total_job_time': aggregated_time_sec,
        'throughput(batches/s)': aggregated_throuhgput,
        'compute_cost': aggregated_compute_cost,
        'cache_cost': cache_cost,
        'total_cost': total_cost,
    }
    print(f"{eviction_policy}:")
    print(f"  Jobs: {job_speeds}"),
    print(f"  Batches per Job: {batches_per_job}")
    print(f"  Cache Eviction Policy: {eviction_policy}")
    print(f"  Cache Miss Penalty: {cache_miss_penalty:.2f}s")
    print(f"  Total Time: {elapsed_time_sec:.2f}s")
    print(f"  Total Throughput: {aggregated_throuhgput:.2f} batches/sec")
    print(f"  Cache Size: {cache_capacity} batche")
    print(f"  Cache Used: {max_cache_capacity_used:} batches")
    print(f"  Cache Miss Penalty: {cache_miss_penalty:.2f}s")
    # print(f"  Cache Eviction Policy: {eviction_policy}")
    print(f"  Cache Hit %: {aggregated_cache_hit_percent:.2f}%")
    # print(f"  Total Batches Processed: {aggregated_batches_processed}")
    # print(f"  Elapsed Time: {elapsed_time_sec:.2f}s, {elapsed_time_sec/60:.2f} min")
    # print(f"  Overall Throughput: {aggregated_throuhgput:.2f} batches/sec")
    print(f"  Cache Cost: ${cache_cost:.2f}")
    print(f"  Compute Cost: ${aggregated_compute_cost:.2f}")
    print(f"  Total Cost : ${total_cost:.2f}")
    print("-" * 40)
    return overall_results

if __name__ == "__main__":
    workload_name = 'imagenet_128_nas'
    jobs =  workloads[workload_name].items()
    simulation_time_sec = None #3600 # None  #3600 * 1 # Simulate 1 hour
    batches_per_job = 100 * 1 # 8500 #np.inf
    cache_capacity = 1 * batches_per_job #number of batches as a % of the total number of batches
    eviction_policy = "noevict" # "lru", "fifo", "mru", "random", "noevict"
    hourly_ec2_cost = 12.24 
    hourly_cache_cost = 3.25
    cache_miss_penalty = 0.2
    batch_assignment_strategy = "coordinated" # "sequential", "shuffle", "coordinated"
    run_simulation(
        workload_name = workload_name,
        workload_jobs = jobs,
        cache_capacity = cache_capacity,
        eviction_policy = eviction_policy,
        cache_miss_penalty=cache_miss_penalty,
        hourly_cache_cost = hourly_cache_cost,
        hourly_ec2_cost = hourly_ec2_cost,
        simulation_time_sec=simulation_time_sec,
        batches_per_job=batches_per_job,
        batch_assignment_strategy=batch_assignment_strategy,
    )