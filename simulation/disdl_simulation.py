import heapq
import logging
from typing import List, Sized, Tuple, Dict, Set, Any
import sys
sys.path.append(".")
sys.path.append("disdl\server")
from sim_workloads import workloads
from sim_cache import SharedCache
from typing import List, Optional, Dict, Tuple
from collections import OrderedDict
# from disdl.server.batch import Batch, CacheStatus, BatchSet
from disdl.server.batch_manager import BatchManager, Batch, CacheStatus, DLTJob
from disdl.server.utils import AverageMeter
import threading
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='simulation.log',         # Log file name
    filemode='w'                # Overwrite the file each run; use 'a' to append
)
logger = logging.getLogger(__name__)
# logger = configure_simulation_logger()

class Dataset:
    def __init__(self, num_samples: str, batch_size:int , num_partitions: int = 1):
        self.num_samples = num_samples
        self.num_partitions = num_partitions
        self.batch_size = batch_size

    def __len__(self) -> int:
        return self.num_samples
    
def run_simulation(
    dataloader_system: str,
    workload_name: str,
    workload_jobs: Dict[str, float],
    cache_capacity: float,
    cache_policy: str,
    load_from_s3_time: float,
    hourly_cache_cost: float,
    hourly_ec2_cost: float,
    simulation_time_sec: int = None,
    batches_per_epoch: int = 1000,
    epochs_per_job: int = 1,
    use_prefetcher: bool = False,
    prefetch_delay: float = 3,
    num_partitions: int = 1,
    preprocesssing_time: float = 0.0001,
    batch_size: int = 1):

    cache = SharedCache(capacity=cache_capacity, eviction_policy=cache_policy)
    manager = BatchManager(
        dataset=Dataset(num_samples=batches_per_epoch, batch_size=batch_size, num_partitions=num_partitions),
        drop_last=False,
        shuffle=False,
        prefetch_lookahead_steps=90,
        use_prefetching=False,
        prefetch_lambda_name=None,
        prefetch_simulation_time=None,
        cache_address=None,
        shared_cache=cache)

    for job_id in workload_jobs:
        job_speed = workload_jobs[job_id]
        manager.register_job(job_id, job_speed)
    
    event_queue = []  # Priority queue for next event times
    time_elapsed = 0  # Global simulation time
    batches_per_job = batches_per_epoch * epochs_per_job
    time_between_job_starts = 0
    next_job_start_time = time_elapsed

    for job in manager.job_registry.all():
        heapq.heappush(event_queue, (next_job_start_time, "start_training_step", job))
        next_job_start_time += time_between_job_starts

    if use_prefetcher:
        heapq.heappush(event_queue, (prefetch_delay, "prefetcher_step", (None)))

    while True:
        if simulation_time_sec is not None and time_elapsed >= simulation_time_sec:
            break
        
        #check if all jobs have completed their batches
        if batches_per_job is not None and all(job.num_batches_processed >= batches_per_job for job in manager.job_registry.all()):
            break
        
        time_elapsed, event_type, payload = heapq.heappop(event_queue)

        if event_type == "start_training_step":
            job:DLTJob = payload
            job.elapased_time_sec = time_elapsed
            next_bacth, should_cache_on_miss, eviction_candidate = manager.get_next_batch_for_job(job.job_id)
            if next_bacth.batch_id == '1_1_69_14bfa6bb14875e45':
                pass
            next_batch:Batch = next_bacth
            logger.debug(f"Job {job.job_id} assigned batch {next_batch.batch_id} at time {time_elapsed:.2f}s")

            #load the next batch
            cache_hit = cache.get_batch(next_batch.batch_id)
            if cache_hit:
                batch_is_cached = True
                job.cache_hit_count += 1
                delay = preprocesssing_time
                heapq.heappush(event_queue, (time_elapsed + delay, "end_training_step", (job, batch_is_cached, None)))
            else:
                # if job.job_id == "RESNET18":
                #     pass
                job.cache_miss_count += 1
                delay = load_from_s3_time + preprocesssing_time
                if should_cache_on_miss:
                    cache_delay = load_from_s3_time + preprocesssing_time
                    heapq.heappush(event_queue, (time_elapsed + cache_delay, "cache_insert", (job, next_batch, eviction_candidate)))
                else:
                    delay = load_from_s3_time + preprocesssing_time
                    heapq.heappush(event_queue, (time_elapsed + delay, "end_training_step", (job, False, None)))
            logger.info(f"Job {job.job_id} processing batch {job.current_batch.batch_id}. Time {time_elapsed:.2f}s. Cache hit: {cache_hit}.")

        
        elif event_type == "end_training_step":
            job, batch_is_cached, evicited_batch_id = payload
            job.elapased_time_sec = time_elapsed
            job.num_batches_processed += 1

            manager.processed_batch_update(
                job_id=job.job_id,
                batch_is_cached=batch_is_cached,
                evicited_batch_id=evicited_batch_id)
            
            if batches_per_job is None or job.num_batches_processed == batches_per_job:
                job.elapased_time_sec += job.processing_speed
                print(f"Job {job.job_id} has completed its batches after {job.elapased_time_sec:.2f}s. Throughput: {job.num_batches_processed / job.elapased_time_sec:.2f} batches/s")
            else:
                heapq.heappush(event_queue, (time_elapsed + job.processing_speed, "start_training_step", job))
        
        elif event_type == "cache_insert":
            job, next_batch, eviction_candidate_batch_id = payload
            batch_id = next_batch.batch_id
            job.elapased_time_sec = time_elapsed
            #check if the batch is already in the cache
            if cache.batch_exists(batch_id):
                logger.debug(f"Batch {batch_id} already in cache, skipping insertion.")
                heapq.heappush(event_queue, (time_elapsed, "end_training_step", (job, True, None)))
            else:
                # Try inserting into the cache
                batch_is_cached, evicted_batch_id = cache.put_batch(batch_id)
                if not batch_is_cached and eviction_candidate_batch_id is not None:
                    # Cache is full and eviction policy is 'noevict' – try manual eviction
                    evicted = cache.remove_batch(eviction_candidate_batch_id)
                    if evicted:
                        evicted_batch_id = eviction_candidate_batch_id
                        batch_is_cached, _ = cache.put_batch(batch_id)
                    
                    if not batch_is_cached:
                        logger.error(f"Failed to insert batch {batch_id} even after manual attempt eviction of {eviction_candidate_batch_id}.")

                heapq.heappush(event_queue, (time_elapsed, "end_training_step", (job, batch_is_cached, evicted_batch_id)))

    #do a sanity check that batches in cache mactch all batch in manager.cache
    for batch_id in cache.cache:
        if batch_id not in manager.cache.cached_batches:
            logger.error(f"Batch {batch_id} in cache but not in manager cache.")
    for batch_id in manager.cache.cached_batches:
        if batch_id not in cache.cache:
            logger.error(f"Batch {batch_id} in manager cache but not in cache.")
    if len(cache.cache) != len(manager.cache.cached_batches):
        logger.error(f"Cache size mismatch: {len(cache.cache)} in cache but {len(manager.cache.cached_batches)} in manager cache.")
        

    jobs = manager.job_registry.all()
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
    print(f"  S3_load_time: {load_from_s3_time:.2f}s")
    print(f"  Eviction Policy: {cache_policy}")
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
    for job in jobs:
        print(f"  Job {job.job_id}: {list(job.used_batch_set_ids.keys())}")

    # return overall_results
    # print(sampler.assigned_eviction_candidates)

if __name__ == "__main__":
    dataloader_system  = 'DisDL' #'CoorDL', TensorSocket, DisDL
    workload_name = 'imagenet_slowfast' #'imagenet_128_hpo', 'imagenet_128_resnet50', imagenet_128_nas, imagenet_slowfast
    workload_jobs = dict(workloads[workload_name])

    simulation_time_sec = None #3600 # None  #3600 * 1 # Simulate 1 hour
    batches_per_epoch = 1000 # batches
    epochs_per_job = 1 #np.inf
    batches_per_job = batches_per_epoch * epochs_per_job
    cache_capacity = 0.5 * batches_per_job #0.5 * batches_per_epoch  #np.inf #5.0 * batches_per_epoch #number of batches as a % of the total number of batches
    cache_policy = "noevict" # "lru", "fifo", "mru", "random", "noevict", "reuse_score"
    hourly_ec2_cost = 12.24
    hourly_cache_cost = 3.25
    load_from_s3_time = 0.1
    prefetcher_speed = 3
    preprocesssing_time = 0.00
    num_partitions = 1
    batch_size = 1

    run_simulation(
        dataloader_system = dataloader_system,
        workload_name = workload_name,
        workload_jobs = workload_jobs,
        cache_capacity = cache_capacity,
        cache_policy = cache_policy,
        load_from_s3_time=load_from_s3_time,
        hourly_cache_cost = hourly_cache_cost,
        hourly_ec2_cost = hourly_ec2_cost,
        simulation_time_sec=simulation_time_sec,
        batches_per_epoch=batches_per_epoch,
        epochs_per_job=epochs_per_job,
        use_prefetcher=False,
        prefetch_delay=prefetcher_speed,
        num_partitions=num_partitions,
        preprocesssing_time=preprocesssing_time,
        batch_size= batch_size)