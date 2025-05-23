import heapq
import numpy as np
import logging
from typing import List, Tuple, Dict, Set, Any
import sys
sys.path.append(".")
sys.path.append("disdl\server")
from simulation.sim_workloads import workloads, save_dict_list_to_csv
from disdl.core.job import DLTJob
from disdl.core.sampler import PartitionedBatchSampler
from disdl.core.batch import Batch, CacheStatus, BatchSet
from simulation.sim_cache import SharedCache
import collections


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import random
class Dataset:
    def __init__(self, num_samples: str, batch_size:int ,  num_partitions: int = 1):
        self.num_samples = num_samples
        self.num_partitions = num_partitions
        self.batch_size = batch_size
    def __len__(self) -> int:
        return self.num_samples



class CoorDLBatchScheduler:
    def __init__(self, batches_to_process: List[Batch], job_ids: List[str], cache: SharedCache, assignment_strategy="rotate"):
        self.batches_to_process = batches_to_process
        self.job_ids = job_ids
        self.cache = cache
        self.assignment_strategy = assignment_strategy
        self.num_jobs = len(job_ids)
        self.total_batches = len(batches_to_process)
        self.batches_per_job = self.total_batches // self.num_jobs

        self.job_owns = {}
        self._assign_batches()

        self.batch_seen_by = collections.defaultdict(set)

        # Track which batch index each job has seen from each producer
        self.job_producer_ptrs = {
            jid: [0] * self.num_jobs for jid in job_ids
        }

        # Each job rotates over producer indices (starting from its own ID)
        self.job_to_next_producer = {
            jid: self.job_ids.index(jid) for jid in job_ids
        }

        # Map: producer_id -> list of owned batch_ids
        self.producer_to_batch_ids = {
            jid: [b.batch_id for b in self.job_owns[jid]] for jid in job_ids
        }
    def _assign_batches(self):
        base = self.total_batches // self.num_jobs
        extra = self.total_batches % self.num_jobs
        cursor = 0

        for i, job_id in enumerate(self.job_ids):
            count = base + (1 if i < extra else 0)
            self.job_owns[job_id] = self.batches_to_process[cursor:cursor + count]
            cursor += count

    # def _assign_batches(self):
    #     for i, job_id in enumerate(self.job_ids):
    #         start = i * self.batches_per_job
    #         end = start + self.batches_per_job
    #         self.job_owns[job_id] = self.batches_to_process[start:end]

    def get_batch_for_job(self, job_id: str) -> tuple:
        ptrs = self.job_producer_ptrs[job_id]
        producer_idx = self.job_to_next_producer[job_id]

        for _ in range(self.num_jobs):  # Try each producer
            producer = self.job_ids[producer_idx]
            batch_list = self.producer_to_batch_ids[producer]
            producer_ptr = ptrs[producer_idx]

            if producer_ptr < len(batch_list):
                batch_id = batch_list[producer_ptr]

                # If not yet seen by this job
                if job_id not in self.batch_seen_by[batch_id]:
                    is_owner = (producer == job_id)
                    # print(f"Job {job_id} assigned batch {batch_id} (owner: {producer})")
                    return batch_id, is_owner

            # Try next producer
            producer_idx = (producer_idx + 1) % self.num_jobs

        raise StopIteration(f"Job {job_id} has no more unseen batches.")

    def mark_seen(self, job_id: str, batch_id: int) -> bool:
        # Determine which producer owns this batch
        for producer_idx, producer in enumerate(self.job_ids):
            if batch_id in self.producer_to_batch_ids[producer]:
                break
        else:
            raise ValueError(f"Batch {batch_id} not found in any producer list.")

        # Advance the pointer for this job's view of that producer
        self.job_producer_ptrs[job_id][producer_idx] += 1

        # Mark as seen
        self.batch_seen_by[batch_id].add(job_id)

        # Rotate to next producer
        self.job_to_next_producer[job_id] = (self.job_to_next_producer[job_id] + 1) % self.num_jobs

        # Evict if all jobs have seen it
        if len(self.batch_seen_by[batch_id]) >= self.num_jobs:
            self.cache.remove_batch(batch_id)
            return True

        return False


    

def run_simulation(
        dataloader_system,
        workload_name,
        workload_jobs,
        cache_capacity,
        eviction_policy,
        cache_hit_delay,
        cache_miss_delay,
        hourly_cache_cost,
        hourly_ec2_cost,
        simulation_time_sec,
        batches_per_epoch,
        epochs_per_job,
        batch_size,
        syncronized_mode):
    logger.info(f"Starting simulation for {workload_name} with {len(workload_jobs)} jobs")
    cache = SharedCache(capacity=cache_capacity, eviction_policy=eviction_policy)
    batches_per_job = batches_per_epoch * epochs_per_job
    sampler = PartitionedBatchSampler(
        num_files=len(Dataset(num_samples=batches_per_epoch, batch_size=batch_size)),
        batch_size=batch_size,
        num_partitions=1,
        drop_last=False,
        shuffle=False)
    
    batches_to_process =[]
    for _ in range(batches_per_job):
        next_batch:Batch = next(sampler)
        batches_to_process.append(next_batch)
    
    jobs:List[DLTJob] = [DLTJob(job_id) for job_id in workload_jobs]
    for job in jobs:
        job.set_job_processing_speed(workload_jobs[job.job_id])
    
    scheduler = CoorDLBatchScheduler(batches_to_process=batches_to_process, job_ids=[job.job_id for job in jobs], cache=cache)
    time_elapsed = 0  # Global simulation time
    event_queue = []  # Priority queue for next event times    
    time_between_job_starts = 0.1
    next_job_start_time = time_elapsed

    for job in jobs:
        heapq.heappush(event_queue, (next_job_start_time, "start_training_step", job))
        next_job_start_time += time_between_job_starts

    while True:
        if simulation_time_sec is not None and time_elapsed >= simulation_time_sec:
            break
        #check all jobs have completed their batches
        if batches_per_job is not None and all(job.num_batches_processed >= batches_per_job for job in jobs):
            break

        time_elapsed, event_type, payload = heapq.heappop(event_queue)
        
        if event_type == "start_training_step":
            job: DLTJob = payload
            job.elapased_time_sec = time_elapsed
            batch_id, job_is_owner = scheduler.get_batch_for_job(job.job_id)
            cache_hit = cache.get_batch(batch_id)
            if cache_hit:
                job.cache_hit_count += 1
                delay = cache_hit_delay + job.processing_speed
                heapq.heappush(event_queue, (time_elapsed + delay, "end_training_step", (job, batch_id)))
                # logger.info(f"[job_step] Job {job.job_id} processing batch {batch_id} (owner={job_is_owner})")
            else:
                if job_is_owner:
                    # Schedule prefetch
                    job.cache_miss_count += 1
                    delay = cache_miss_delay + job.processing_speed
                    heapq.heappush(event_queue, (time_elapsed + cache_miss_delay, "cache_insert", (job, batch_id)))
                    heapq.heappush(event_queue, (time_elapsed + delay, "end_training_step", (job, batch_id)))
                    logger.debug(f"[job_step] Job {job.job_id} processing batch {batch_id} (owner={job_is_owner})")

                elif not syncronized_mode:
                    # Non-owner but allowed to proceed
                    job.cache_miss_count += 1
                    delay = cache_miss_delay + job.processing_speed
                    heapq.heappush(event_queue, (time_elapsed + delay, "end_training_step", (job, batch_id)))
                    # logger.info(f"[job_step] Job {job.job_id} processing batch {batch_id} (owner={job_is_owner})")
                else:
                    # Non-owner must wait and retry
                    delay = 0.05
                    heapq.heappush(event_queue, (time_elapsed + delay, "start_training_step", job))
        
        elif event_type == "end_training_step":
            job, batch_id = payload
            job.elapased_time_sec = time_elapsed
            job.num_batches_processed += 1
            scheduler.mark_seen(job.job_id, batch_id)

            if batches_per_job is None or job.num_batches_processed == batches_per_job:
                print(f"Job {job.job_id} has completed its batches after {time_elapsed:.2f}s. Throughput: {job.num_batches_processed / time_elapsed:.2f} batches/s")
            else:
                heapq.heappush(event_queue, (time_elapsed, "start_training_step", job))

        elif event_type == "cache_insert":
            job, batch_id = payload
            cache.put_batch(batch_id)  # Simulate cache insert for this batch

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
    optimal_throughputs = {job['job_id']: job['optimal_throughput(batches/s)'] for job in job_performances}
    job_cache_hit_percent = {job['job_id']: job['cache_hit_%'] for job in job_performances}

    print(f"{dataloader_system}")
    print(f"  Jobs: {job_speeds}"),
    print(f"  Batches per Job: {batches_per_job}")
    print(f"  Cache Hit Delay: {cache_hit_delay:.2f}s")
    print(f"  Cache Miss Delay: {cache_miss_delay:.2f}s")
    print(f"  Eviction Policy: {eviction_policy}")
    print(f"  Cache Capacity: {cache_capacity:.0f} batches")
    print(f"  Cache Used: {max_cache_capacity_used:} batches")
    print(f"  Cache Hit %: {agg_cache_hit_percent:.2f}%, Hits: {agg_cache_hits}, Misses: {agg_cache_misses}")
    print(f"  Cache Hit % per Job: {job_cache_hit_percent}")
    print(f"  Cache Cost: ${cache_cost:.2f}")
    print(f"  Compute Cost: ${agg_compute_cost:.2f}")
    print(f"  Total Cost : ${total_cost:.2f}")
    print(f"  Total Batches: {agg_batches_processed}")
    print(f"  Total Time: {elapsed_time_sec:.2f}s")
    print(f"  Optimal Job Throughputs: {optimal_throughputs}")
    print(f"  Actual Job Throughputs: {throuhgputs_for_jobs}")
    print(f"  Total Throughput: {agg_throuhgput:.2f} batches/s")
    print("-" * 40)
    # for job in jobs:
    #     print(f"  Job {job.job_id}: {list(job.used_batch_set_ids.keys())}")



if __name__ == "__main__":

    dataloader_system  = 'CoorDL' 
    workload_name = 'imagenet_128_nas'
    workload_jobs = dict(workloads[workload_name])
    batches_per_epoch = 1173 # batches
    epochs_per_job = 1 #np.inf
    batches_per_job = batches_per_epoch * epochs_per_job
    cache_capacity = 250 #0.5 * batches_per_job
    eviction_policy = "noevict" # "lru", "fifo", "mru", "random", "noevict"
    hourly_ec2_cost = 12.24 
    hourly_cache_cost = 3.25
    # load_from_s3_time = 0.29187976675
    # preprocesssing_time = 0.19094351
    cache_hit_delay = 0.070791547
    cache_miss_delay = 0.29187976675 + 0.19094351
    batch_size = 1
    simulation_time_sec = None #3600 # None  #3600 * 1 # Simulate 1 hour
    syncronized_mode = True
    run_simulation(
        dataloader_system = dataloader_system,
        workload_name = workload_name,
        workload_jobs = workload_jobs,
        cache_capacity = cache_capacity,
        eviction_policy = eviction_policy,
        cache_hit_delay=cache_hit_delay,
        cache_miss_delay=cache_miss_delay,
        hourly_cache_cost = hourly_cache_cost,
        hourly_ec2_cost = hourly_ec2_cost,
        simulation_time_sec=simulation_time_sec,
        batches_per_epoch=batches_per_epoch,
        epochs_per_job=epochs_per_job,
        batch_size= batch_size,
        syncronized_mode=syncronized_mode
    )