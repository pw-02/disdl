import heapq
import numpy as np
import logging
from typing import List, Tuple, Dict, Set, Any
import sys
sys.path.append(".")
sys.path.append("disdl\server")
from simulation.sim_workloads import workloads, save_dict_list_to_csv
from simulation.sim_job import DLTJob
from disdl.server.sampler import PartitionedBatchSampler
from disdl.server.batch import Batch, CacheStatus, BatchSet
import copy
import os
import csv
import time
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Dataset:
    def __init__(self, num_samples: str, batch_size:int ,  num_partitions: int = 1):
        self.num_samples = num_samples
        self.num_partitions = num_partitions
        self.batch_size = batch_size
    def __len__(self) -> int:
        return self.num_samples


class TensorSockerProducer:
    def __init__(self, speed, batches_to_process: List[Batch]):
        self.batches_to_process = batches_to_process
        self.speed = speed
        self.delays_caused_by_cache = 0
      

def run_simulation(
        dataloader_system,
        workload_name,
        workload_jobs,
        buffer_size,
        load_from_s3_time,
        hourly_cache_cost,
        hourly_ec2_cost,
        simulation_time_sec,
        batches_per_epoch,
        epochs_per_job,
        preprocesssing_time,
        batch_size):
    
    batches_per_job = batches_per_epoch * epochs_per_job
    sampler = PartitionedBatchSampler(
            num_files=len(Dataset(num_samples=batches_per_epoch, batch_size=batch_size)),
            batch_size=batch_size,
            num_partitions=1,
            drop_last=False,
            shuffle=False)
    
    batches_to_process =[]
    for _ in range(batches_per_job):
        batch_indices, epoch_idx, partition_idx, batch_idx = next(sampler)
        batches_to_process.append(Batch(batch_indices, epoch_idx, partition_idx, batch_idx))
    
    jobs:List[DLTJob] = [DLTJob(job_id) for job_id in workload_jobs]
    for job in jobs:
        job.set_job_processing_speed(workload_jobs[job.job_id])

    producer = TensorSockerProducer(speed=load_from_s3_time, batches_to_process=list(batches_to_process))
    job_batches = {job.job_id: list(batches_to_process) for job in jobs}
    cache_size_over_time = []  # Store cache size over time
    event_queue = []  # Priority queue for next event times
   
    heapq.heappush(event_queue, (producer.speed, "producer_step", producer, True))
    
    for job in jobs:
        heapq.heappush(event_queue, (job.processing_speed, "job_step", job, True))

    time_elapsed = 0  # Global simulation time
    while True:
        if simulation_time_sec is not None and time_elapsed >= simulation_time_sec:
            break
        #check all jobs have completed their batches
        if batches_per_job is not None and all(job.num_batches_processed >= batches_per_job for job in jobs):
            break

        time_elapsed, event_type, payload, is_first_attempt = heapq.heappop(event_queue)

        if event_type == "job_step":
            job:DLTJob = payload
            job.elapased_time_sec = time_elapsed
            #get first batch for the job
            batch:Batch = job_batches[job.job_id][0]
            if batch.batch_id in job.local_cache:
                job.cache_hit_count += 1 if is_first_attempt else 0
                job.num_batches_processed += 1
                job_batches[job.job_id].pop(0)
                #remove the batch from the cache
                del job.local_cache[batch.batch_id]

                if batches_per_job is None or job.num_batches_processed == batches_per_job:
                    print(f"Job {job.job_id} has completed its batches after {time_elapsed:.2f}s. Throughput: {job.num_batches_processed / time_elapsed:.2f} batches/s")
                else:
                    heapq.heappush(event_queue, (time_elapsed + job.processing_speed, "job_step", job, True))
            else:
                if is_first_attempt:
                    job.cache_miss_count += 1
                recheck_in = 0.1  # retry later
                logger.debug(f"[consumer miss] Job {job.job_id} retrying for {batch.batch_id}")
                heapq.heappush(event_queue, (time_elapsed + recheck_in, "job_step", job, False))
        elif event_type == "producer_step":
            producer:TensorSockerProducer = payload
            #check that the cache for all jobs is not full
            if all(len(job.local_cache) < buffer_size for job in jobs) and len(producer.batches_to_process) > 0:
                #pop of the next batch from the producer
                batch:Batch = producer.batches_to_process.pop(0)
                batch_id = batch.batch_id
                for job in jobs:
                    job.local_cache[batch_id] = batch_id
                heapq.heappush(event_queue, (time_elapsed + producer.speed, "producer_step", producer, True))
            else:
                producer.delays_caused_by_cache += 0.05 + producer.speed
                heapq.heappush(event_queue, (time_elapsed + 0.05, "producer_step", producer, False)) #try again later

        # Update cache size over time
        cache_size_over_time.append(max(len(job.local_cache) for job in jobs))


    job_performances = [job.perf_stats(hourly_ec2_cost/len(jobs), hourly_cache_cost/len(jobs)) for job in jobs]
    agg_batches_processed = sum(job['batches_processed'] for job in job_performances)
    agg_cache_hits = sum(job['cache_hit_count'] for job in job_performances)
    agg_cache_misses = sum(job['cache_miss_count'] for job in job_performances)
    agg_cache_hit_percent = (agg_cache_hits / (agg_cache_hits + agg_cache_misses)) * 100 if (agg_cache_hits + agg_cache_misses) > 0 else 0
    agg_compute_cost = sum(job['compute_cost'] for job in job_performances)
    agg_throuhgput = sum(job['throughput(batches/s)'] for job in job_performances)
    agg_time_sec = sum(job['elapsed_time'] for job in job_performances)
    elapsed_time_sec = max(job['elapsed_time'] for job in job_performances) if job_performances else 0
    max_cache_capacity_used = max(cache_size_over_time)
    cache_cost = (hourly_cache_cost / 3600) * elapsed_time_sec
    total_cost = agg_compute_cost + cache_cost  # No additional costs in this simulation
    job_speeds = {job['job_id']: job['job_speed'] for job in job_performances}
    throuhgputs_for_jobs = {job['job_id']: job['throughput(batches/s)'] for job in job_performances}
    optimal_throughputs = {job['job_id']: job['optimal_throughput(batches/s)'] for job in job_performances}
    print(f"{dataloader_system}")
    print(f"  Jobs: {job_speeds}"),
    print(f"  Batches per Job: {batches_per_job}")
    print(f"  S3_load_time: {load_from_s3_time:.2f}s")
    print(f"  Buffer Size: {buffer_size:.0f} batches")
    print(f"  Buffer Used: {max_cache_capacity_used:} batches")
    print(f"  Cache Hit %: {agg_cache_hit_percent:.2f}%")
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

    dataloader_system  = 'tensorsocket' 
    workload_name = 'imagenet_128_nas'
    workload_jobs = dict(workloads[workload_name])
    batches_per_epoch = 100 # batches
    epochs_per_job = 1 #np.inf
    buffer_size = 1
    hourly_ec2_cost = 12.24 
    hourly_cache_cost = 3.25
    load_from_s3_time = 0.1
    preprocesssing_time = 0.00
    batch_size = 1
    simulation_time_sec = None #3600 # None  #3600 * 1 # Simulate 1 hour
    run_simulation(
        dataloader_system = dataloader_system,
        workload_name = workload_name,
        workload_jobs = workload_jobs,
        buffer_size = buffer_size,
        load_from_s3_time=load_from_s3_time,
        hourly_cache_cost = hourly_cache_cost,
        hourly_ec2_cost = hourly_ec2_cost,
        simulation_time_sec=simulation_time_sec,
        batches_per_epoch=batches_per_epoch,
        epochs_per_job=epochs_per_job,
        preprocesssing_time=preprocesssing_time,
        batch_size= batch_size
    )