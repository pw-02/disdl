import heapq
import numpy as np
import logging
from typing import List, Tuple, Dict, Set, Any
import sys
sys.path.append(".")
from simulation.sim_workloads import workloads, save_dict_list_to_csv
import os
import csv
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TensorSockerProducer:
    def __init__(self, speed, total_batches):
        self.speed = speed
        self.total_batches = total_batches
        self.current_batch = 0


class DLTJOB:
    def __init__(self, job_id, speed):
        self.job_id = job_id
        self.speed = speed
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.elapased_time_sec = 0
        self.throughput = 0
        self.compute_cost = 0
        self.current_batch_id = None
        self.job_progress = 0
        self.local_cache = {}

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
    def __lt__(self, other):
        return self.speed < other.speed  # Compare based on speed


def run_simulation(
        workload_name,
        workload_jobs,
        cache_capacity,
        producer_step_time,
        hourly_cache_cost,
        hourly_ec2_cost,
        simulation_time_sec,
        batches_per_job,):
    
    jobs:List[DLTJOB] = [DLTJOB(model_name, speed) for model_name, speed in workload_jobs]
    producer = TensorSockerProducer(speed=producer_step_time, total_batches=batches_per_job)
    cache_size_over_time = []  # Store cache size over time
    event_queue = []  # Priority queue for next event times
    heapq.heappush(event_queue, (producer.speed, "producer_step", producer))

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

        if event_type == "producer_step":
            #check that the cache for all jobs is not full
            if all(len(job.local_cache) < cache_capacity for job in jobs):
                producer.current_batch += 1
                batch_id = producer.current_batch
                heapq.heappush(event_queue, (time_elapsed + producer.speed, "producer_step", producer))
                for job in jobs:
                    job.local_cache[batch_id] = batch_id
            else:
                heapq.heappush(event_queue, (time_elapsed + 0.05, "producer_step", producer))

        elif event_type == "step":
            job:DLTJOB = payload
            batch_id = job.job_progress + 1
            job.elapased_time_sec = time_elapsed
            if batch_id in job.local_cache:
                job.cache_hit_count += 1
                job.job_progress += 1
                delay = job.speed
                job.local_cache.pop(batch_id)
            else:
                delay = 0.1  # retry later
                logger.debug(f"[consumer miss] Job {job.job_id} retrying for {batch_id}")
            if batches_per_job is None or job.job_progress < batches_per_job:
                # Simulate the next step for this job
                heapq.heappush(event_queue, (time_elapsed + delay, "step", job))
    
        # Update cache size over time
        # cache_size = max(len(job.local_cache) for job in jobs)
        cache_size_over_time.append(max(len(job.local_cache) for job in jobs))


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
    cache_cost = (hourly_cache_cost / 3600) * elapsed_time_sec
    total_cost = aggregated_compute_cost + cache_cost  # No additional costs in this simulation
    job_speeds = {job['job_id']: job['job_speed'] for job in job_performances}
    
    overall_results = {
        'workload_name': workload_name,
        'job_speeds': job_speeds,
        'batches_per_job': batches_per_job,
        'cache_capacity': cache_capacity,
        'cache_eviction_policy': 'uniform',
        'num_jobs': len(job_performances),
        'producer_step_time': producer_step_time,
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

    throuhgputs_for_jobs = {job['job_id']: job['throughput(batches/s)'] for job in job_performances}
    print(f"{'TensorSocket'}:")
    print(f"  Jobs: {job_speeds}"),
    print(f"  Batches per Job: {batches_per_job}")
    print(f"  producer_step_time: {producer_step_time:.2f}s")
    print(f"  Total Batches Processed: {aggregated_batches_processed}")
    print(f"  Total Time: {elapsed_time_sec:.2f}s")
    print(f"  Total Throughput: {aggregated_throuhgput:.2f} batches/s")
    print(f"  Job Throughputs: {throuhgputs_for_jobs}")
    print(f"  Cache Size: {cache_capacity} batches")
    print(f"  Cache Used: {max_cache_capacity_used:} batches")
    print(f"  Cache Hit %: {aggregated_cache_hit_percent:.2f}%")
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
    cache_capacity = 1000 #1 * batches_per_job #number of batches as a % of the total number of batches
    hourly_ec2_cost = 12.24 
    hourly_cache_cost = 3.25
    producer_step_time = 0.2 #time to load data and create batches in seconds
    run_simulation(
        workload_name = workload_name,
        workload_jobs = jobs,
        cache_capacity = cache_capacity,
        producer_step_time=producer_step_time,
        hourly_cache_cost = hourly_cache_cost,
        hourly_ec2_cost = hourly_ec2_cost,
        simulation_time_sec=simulation_time_sec,
        batches_per_job=batches_per_job,
    )



