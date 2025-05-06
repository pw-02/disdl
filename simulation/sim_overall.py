import heapq
import numpy as np
import logging
from typing import List, Tuple, Dict, Set, Any
import sys
sys.path.append(".")
from simulation.sim_workloads import workloads
from simulation.sim_cache import SharedCache
from simulation.sim_job import Job
import os
import csv
import time
import collections
import random

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log',         # Log file name
    filemode='w'                # Overwrite the file each run; use 'a' to append
)
logger = logging.getLogger(__name__)

class TensorSockerProducer:
    def __init__(self, batches_per_job):
        self.batches_per_job = batches_per_job
        self.current_batch = 0


class CoorDLSampler:
    def __init__(self, total_batches: int, job_ids: List[str], cache: SharedCache):
        self.total_batches = total_batches
        self.job_ids = job_ids
        self.cache = cache
        self.job_to_batches = self._assign_batches()
        self.batch_seen_by = collections.defaultdict(set)
        self.global_sequence = self._build_global_schedule()  # List of (batch_id, owner)
        self.job_pointers = {jid: 0 for jid in job_ids}
        self.num_jobs = len(job_ids)

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

    def get_batch_for_job(self, job_id: str) -> tuple:
        """
        Returns the next batch in the global schedule for this job.
        Skips over batches not in the job's assignment, but still includes them for coordination.
        Returns a tuple: (batch_id, self_load_flag)
        """
        ptr = self.job_pointers[job_id]
        while ptr < len(self.global_sequence):
            batch_id, owner = self.global_sequence[ptr]
            self_load = (owner == job_id)
            logger.info(f"Job {job_id} assigned batch {batch_id} (owner: {owner})")
            return batch_id, self_load

        raise IndexError(f"Job {job_id} has no more batches to process.")

    def mark_seen(self, job_id: str, batch_id: int) -> bool:
        self.job_pointers[job_id] += 1
        self.batch_seen_by[batch_id].add(job_id)
        if len(self.batch_seen_by[batch_id]) >= self.num_jobs:
            self.cache._remove(batch_id)
            return True
        return False


def run_simulation(
    dataloader_system: str,
    workload_name: str,
    workload_jobs: List[Tuple[str, float]],
    cache_capacity: float,
    eviction_policy: str,
    load_from_s3_time: float,
    hourly_cache_cost: float,
    hourly_ec2_cost: float,
    simulation_time_sec: int = None,
    batches_per_job: int = 1000,
):
    logger.info(f"Starting simulation for {workload_name} with {len(workload_jobs)} jobs")
    
    # Initialize cache and jobs
    cache = SharedCache(capacity=cache_capacity, eviction_policy=eviction_policy)
    jobs = [Job(job_id, speed) for job_id, speed in workload_jobs]
    event_queue = []  # Priority queue for next event times

    for job in jobs:
        heapq.heappush(event_queue, (job.speed, "step", job))
    
    time_elapsed = 0  # Global simulation time
    
    if dataloader_system == "CoorDL":
        sampler = CoorDLSampler(total_batches=batches_per_job, job_ids=[job[0] for job in workload_jobs], cache=cache)
    
    elif dataloader_system == "TensorSocket":
        sampler = None  # No sampler needed for TensorSocket
        ts_producer = TensorSockerProducer(batches_per_job=batches_per_job)
        heapq.heappush(event_queue, (0.001, "producer_step", ts_producer))

    while True:
        if simulation_time_sec is not None and time_elapsed >= simulation_time_sec:
            break
        #check all jobs have completed their batches
        if batches_per_job is not None and all(job.num_batches_processed >= batches_per_job for job in jobs):
            break
        
        time_elapsed, event_type, payload = heapq.heappop(event_queue)
        
        if event_type == "cache_insert":
            batch_id = payload
            cache.put_batch(batch_id)  # Simulate cache insert for this batch
        
        if isinstance(sampler, CoorDLSampler):
            if event_type == "step":
                job:Job = payload
                job.elapased_time_sec = time_elapsed
                batch_id, job_is_owner = sampler.get_batch_for_job(job.job_id)
                if job_is_owner:
                    # Job is the owner of this batch; process it
                    if cache.get_batch(batch_id):
                        job.cache_hit_count += 1
                        delay = job.speed
                    else:
                        job.cache_miss_count += 1
                        heapq.heappush(event_queue, (time_elapsed + load_from_s3_time, "cache_insert", batch_id))
                        delay = job.speed + load_from_s3_time
                    job.num_batches_processed += 1
                    sampler.mark_seen(job.job_id, batch_id)
                else:
                    # Job is not the owner; just mark it as seen
                    cache_hit = cache.get_batch(batch_id)
                    if cache_hit:
                        job.cache_hit_count += 1
                        job.num_batches_processed += 1
                        delay = job.speed
                        sampler.mark_seen(job.job_id, batch_id)
                    else:
                        delay = 0.05  # check cache again in a small time
                        logger.debug(f"[consumer miss] Job {job.job_id} retrying for {batch_id}")
                
                # if batches_per_job is None or job.num_batches_processed < batches_per_job:
                if batches_per_job is None or job.num_batches_processed < batches_per_job:
                    heapq.heappush(event_queue, (time_elapsed + delay, "step", job))
        
        elif isinstance(ts_producer, TensorSockerProducer):
            #tensorsocket producer
            if event_type == "producer_step":
                if all(len(job.local_cache) < cache_capacity for job in jobs):
                    ts_producer.current_batch += 1
                    batch_id = ts_producer.current_batch
                    logger.info(f"Producer generated batch {batch_id}")
                    for job in jobs:
                        job.local_cache[batch_id] = batch_id
                    if ts_producer.current_batch < batches_per_job:
                        heapq.heappush(event_queue, (time_elapsed + load_from_s3_time, "producer_step", ts_producer))
                else:
                    #wait for all jobs to be ready
                    heapq.heappush(event_queue, (time_elapsed + 0.05, "producer_step", ts_producer))
            
            if event_type == "step":
                job:Job = payload
                job.elapased_time_sec = time_elapsed
                batch_id = job.num_batches_processed + 1
                if batch_id in job.local_cache:
                    job.cache_hit_count += 1
                    job.num_batches_processed += 1
                    delay = job.speed
                    job.local_cache.pop(batch_id)
                    logger.info(f"Job {job.job_id} processed batch {batch_id} from local cache")
                else:
                    delay = 0.1  # check cache again in a small time
                    logger.debug(f"[consumer miss] Job {job.job_id} retrying for {batch_id}")
                
                if batches_per_job is None or job.num_batches_processed < batches_per_job:
                    heapq.heappush(event_queue, (time_elapsed + delay, "step", job))

    job_performances = [job.perf_stats(hourly_ec2_cost/len(jobs), hourly_cache_cost/len(jobs)) for job in jobs]
    agg_batches_processed = sum(job['batches_processed'] for job in job_performances)
    agg_cache_hits = sum(job['cache_hit_count'] for job in job_performances)
    agg_cache_misses = sum(job['cache_miss_count'] for job in job_performances)
    agg_cache_hit_percent = (agg_cache_hits / (agg_cache_hits + agg_cache_misses)) * 100 if (agg_cache_hits + agg_cache_misses) > 0 else 0
    agg_compute_cost = sum(job['compute_cost'] for job in job_performances)
    agg_throuhgput = sum(job['throughput(batches/s)'] for job in job_performances)
    agg_time_sec = sum(job['elapsed_time'] for job in job_performances)
    elapsed_time_sec = max(job['elapsed_time'] for job in job_performances) if job_performances else 0
    max_cache_capacity_used = cache.max_size_used
    cache_cost = (hourly_cache_cost / 3600) * elapsed_time_sec
    total_cost = agg_compute_cost + cache_cost  # No additional costs in this simulation
    job_speeds = {job['job_id']: job['job_speed'] for job in job_performances}
    throuhgputs_for_jobs = {job['job_id']: job['throughput(batches/s)'] for job in job_performances}
    print(f"{dataloader_system}")
    print(f"  Jobs: {job_speeds}"),
    print(f"  Batches per Job: {batches_per_job}")
    print(f"  S3_load_time: {load_from_s3_time:.2f}s")
    print(f"  Cache Capacity: {cache_capacity:.0f} batches")
    print(f"  Cache Used: {max_cache_capacity_used:} batches")
    print(f"  Cache Hit %: {agg_cache_hit_percent:.2f}%")
    print(f"  Cache Cost: ${cache_cost:.2f}")
    print(f"  Compute Cost: ${agg_compute_cost:.2f}")
    print(f"  Total Cost : ${total_cost:.2f}")
    print(f"  Total Batches: {agg_batches_processed}")
    print(f"  Total Time: {elapsed_time_sec:.2f}s")
    print(f"  Job Throughputs: {throuhgputs_for_jobs}")
    print(f"  Total Throughput: {agg_throuhgput:.2f} batches/s")
    print("-" * 40)
    # return overall_results


if __name__ == "__main__":
    dataloader_system  = 'CoorDL' #'CoorDL', TensorSocket, DisDL
    workload_name = 'imagenet_128_nas' #'imagenet_128_hpo', 'imagenet_128_resnet50', imagenet_128_nas
    jobs =  workloads[workload_name].items()
    simulation_time_sec = None #3600 # None  #3600 * 1 # Simulate 1 hour
    batches_per_job = 8500 * 1 # 8500 #np.inf
    cache_capacity = 500 #number of batches as a % of the total number of batches
    eviction_policy = "noevict" # "lru", "fifo", "mru", "random", "noevict"
    hourly_ec2_cost = 12.24 
    hourly_cache_cost = 3.25
    load_from_s3_time = 0.2

    run_simulation(
        dataloader_system = dataloader_system,
        workload_name = workload_name,
        workload_jobs = jobs,
        cache_capacity = cache_capacity,
        eviction_policy = eviction_policy,
        load_from_s3_time=load_from_s3_time,
        hourly_cache_cost = hourly_cache_cost,
        hourly_ec2_cost = hourly_ec2_cost,
        simulation_time_sec=simulation_time_sec,
        batches_per_job=batches_per_job,
    )