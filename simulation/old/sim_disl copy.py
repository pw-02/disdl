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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log',         # Log file name
    filemode='w'                # Overwrite the file each run; use 'a' to append
)
logger = logging.getLogger(__name__)

class BatchManager:
    def __init__(self, total_batches: int, cache: SharedCache,  jobs:List[Job] ):
        self.cache = cache
        self.total_batches = total_batches
        self.batch_seen_by = collections.defaultdict(set)
        self.batch_in_progress = set()
        self.jobs: List[Job]  = jobs
        self.base_sequence = list(range(1, total_batches + 1))
        self.in_progress = False
        self.num_jobs = len(jobs)
        self.remaining_batches = {
            job.job_id: list(self.base_sequence) for job in jobs
        }
        self.job_thetas = {job.job_id: 1 / job.speed for job in jobs}
        pass

    def _mark_batch_in_progress(self, batch_id: int, job_id: str = None) -> None:
        self.batch_in_progress.add(batch_id)
        self.remaining_batches[job_id].remove(batch_id)

   
    def get_next_batch_for_job(self, job_id: str) -> Optional[Batch]:
        with self.lock:
            if self.prefetch_service and self.prefetch_service.prefetch_stop_event.is_set():
                self.prefetch_service.start()

            job = self._get_or_register_job(job_id)

            if not job.future_batches:
                self.assign_batch_set_to_job(job)

            reuse_score = {}  # Placeholder: plug in real reuse scores
            batch_seen_by = {}  # Placeholder: plug in actual seen tracking
            batch_in_progress = set()  # Placeholder: plug in actual shared set

            # Step 1: Prefer cached + unseen
            eligible_cached = [
                batch for batch_id, batch in job.future_batches.items()
                if batch.cache_status != CacheStatus.NOT_CACHED and
                   job.job_id not in batch_seen_by.get(batch_id, set())
            ]

            if eligible_cached:
                next_batch = min(eligible_cached, key=lambda b: reuse_score.get(b.batch_id, float('inf')))
            else:
                next_batch = None
                for batch_id, batch in job.future_batches.items():
                    if batch_id not in batch_in_progress:
                        next_batch = batch
                        break

                if not next_batch and job.future_batches:
                    next_batch = next(iter(job.future_batches.values()))

            if next_batch:
                batch_id = next_batch.batch_id
                batch_in_progress.add(batch_id)
                next_batch.mark_seen_by(job.job_id)
                job.future_batches.pop(batch_id, None)

                self._tag_batch_for_caching(next_batch)
                self._maybe_trigger_batch_generation(next_batch)

            return next_batch


    # def get_batch_for_job(self, job_id: str) -> int:
    #     # Step 1: Prefer a cached batch the job hasn't seen
    #     #find all batches that the job has not seen and are in the cache
    #     next_batch = None
    #     eligible_cached = [
    #         batch_id for batch_id in self.remaining_batches[job_id]
    #         if self.cache.get_batch(batch_id) and job_id not in self.batch_seen_by[batch_id]
    #         ]
    #     # Prefer cached batches, sele   cting the one with the lowest reuse score
    #     if eligible_cached:
    #         next_batch = min(eligible_cached, key=self.reuse_score)
    #     else:
    #         # If no cached batches are available, select the next unseen batch
    #         for batch_id in self.remaining_batches[job_id]:
    #             if batch_id not in self.batch_in_progress:
    #                 next_batch = batch_id
    #                 break
        
    #     if next_batch is None:
    #         next_batch = self.remaining_batches[job_id][0]  # Fallback to the first batch in the list
    #     self._mark_batch_in_progress(next_batch, job_id)
    #     # logger.info(f"Assigned batch {next_batch} to job {job_id}")

    #     return next_batch

    def reuse_score(self, batch_id: int) -> float:
        unseen = [j.job_id for j in self.jobs if j.job_id not in self.batch_seen_by[batch_id]]
        score = sum(self.job_thetas[j] for j in unseen)
        return score
    
    def normalized_reuse_score(self, batch_id: int) -> float:
        score = self.reuse_score(batch_id)
        max_score = sum(self.job_thetas[j.job_id] for j in self.jobs)
        # max_score = sum(self.job_thetas[j] for j in self.job_ids)
        return score / max_score if max_score > 0 else 0.0

    def mark_seen(self, job_id: str, batch_id: int) -> bool:
        self.batch_seen_by[batch_id].add(job_id)
        if batch_id in self.batch_in_progress:
            self.batch_in_progress.remove(batch_id)
        if len(self.batch_seen_by[batch_id]) >= self.num_jobs:
            self.cache._remove(batch_id)
            return True
        return False
    
    def top_prefetch_candidates(self, N: int) -> List[int]:
        candidates = [
            batch_id for batch_id in self.base_sequence
            if batch_id not in self.cache.cache and batch_id not in self.batch_in_progress
        ]
        candidates.sort(key=self.reuse_score, reverse=True)
        return candidates[:N]

    
    def evict_batch_with_lowest_reuse_score(self) -> int:
        # Find the batch with the lowest reuse score
        #find all batches in cache first
        eligible_cached = [batch_id for batch_id in self.cache.cache.keys()]
        #of all the cached bacthes find the one with the lowest reuse score
        best_batch_to_evict = min(eligible_cached, key=self.reuse_score)
        self.cache._remove(best_batch_to_evict)
        logger.info(f"Evicted batch {best_batch_to_evict} from cache. Reuse score: {self.reuse_score(best_batch_to_evict)}")


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
    use_prefetcher: bool = False,
    prefetch_delay: float = 0.1,):
    
    cache = SharedCache(capacity=cache_capacity, eviction_policy=eviction_policy)
    jobs = [Job(job_id, speed) for job_id, speed in workload_jobs]
    sampler = BatchManager(batches_per_job, cache,jobs)
    event_queue = []  # Priority queue for next event times
    time_elapsed = 0  # Global simulation time

    for job in jobs:
        heapq.heappush(event_queue, (job.speed, "step", job))

    if use_prefetcher:
        # Optional: prefetch top-N reuse-worthy batches
        for batch_id in sampler.top_prefetch_candidates(N=2):  # N can be tuned
            heapq.heappush(event_queue, (time_elapsed + prefetch_delay, "prefetch", batch_id))

    while True:
        if simulation_time_sec is not None and time_elapsed >= simulation_time_sec:
            break
        #check all jobs have completed their batches
        if batches_per_job is not None and all(job.num_batches_processed >= batches_per_job for job in jobs):
            break
        
        time_elapsed, event_type, payload = heapq.heappop(event_queue)
        
        if event_type == "cache_insert":
            batch_id = payload
            if cache.cache_is_full():
                if eviction_policy == "reuse_score":
                # Evict a batch with the lowest reuse score
                    sampler.evict_batch_with_lowest_reuse_score()
            cache.put_batch(batch_id)  # Simulate cache insert for this batch
        
        elif event_type == "step":
            job:Job = payload
            batch_id = sampler.get_batch_for_job(job.job_id)
            job.elapased_time_sec = time_elapsed
            cache_hit = cache.get_batch(batch_id)
            logger.info(f"Job {job.job_id} processing batch {batch_id} at time {time_elapsed:.2f}s. Cache hit: {cache_hit}")
            if not cache_hit:
                job.cache_miss_count += 1
                logger.debug(f"[self-load miss] Job {job.job_id} will insert {batch_id}")
                heapq.heappush(event_queue, (time_elapsed + load_from_s3_time, "cache_insert", batch_id))
                delay = job.speed + load_from_s3_time
            else:
                job.cache_hit_count += 1
                delay = job.speed
            job.num_batches_processed += 1
            
            if job.num_batches_processed > 1:
                sampler.mark_seen(job.job_id, job.num_batches_processed -1)

            if batches_per_job is None or job.num_batches_processed < batches_per_job:
                heapq.heappush(event_queue, (time_elapsed + delay, "step", job))
        
        elif event_type == "prefetch":
            batch_id = payload
            if not cache.get_batch(batch_id):  # Only prefetch if still not present
                if cache.cache_is_full():
                    if eviction_policy == "reuse_score":
                        sampler.evict_batch_with_lowest_reuse_score()
                cache.put_batch(batch_id)
                logger.debug(f"Prefetched batch {batch_id} at time {time_elapsed:.2f}")
            
            for batch_id in sampler.top_prefetch_candidates(N=2):  # N can be tuned
                    heapq.heappush(event_queue, (time_elapsed + prefetch_delay, "prefetch", batch_id))


    
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
    print(f"  Eviction Policy: {eviction_policy}")
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
    dataloader_system  = 'DisDL' #'CoorDL', TensorSocket, DisDL
    workload_name = 'imagenet_128_nas'
    jobs =  workloads[workload_name].items()
    simulation_time_sec = None #3600 # None  #3600 * 1 # Simulate 1 hour
    batches_per_job = 100 * 1 # 8500 #np.inf
    cache_capacity = 0.2 * batches_per_job #number of batches as a % of the total number of batches
    eviction_policy = "reuse_score" # "lru", "fifo", "mru", "random", "noevict", "reuse_score"
    hourly_ec2_cost = 12.24 
    hourly_cache_cost = 3.25
    load_from_s3_time = 0.02
    prefetcher_speed = load_from_s3_time /2
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
        use_prefetcher=False,
        prefetch_delay=prefetcher_speed,
    )