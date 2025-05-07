import heapq
import logging
from typing import List, Sized, Tuple, Dict, Set, Any
import sys
sys.path.append(".")
from simulation.sim_workloads import workloads
from simulation.sim_cache import SharedCache
from typing import List, Optional, Dict, Tuple
from batch_manager import BatchManager
from job import DLTJob
from logger_config import configure_simulation_logger

logger = configure_simulation_logger()

class Dataset:
    def __init__(self, num_samples: str, batch_size:int ,  num_partitions: int = 1):
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
    eviction_policy: str,
    load_from_s3_time: float,
    hourly_cache_cost: float,
    hourly_ec2_cost: float,
    simulation_time_sec: int = None,
    batches_per_job: int = 1000,
    use_prefetcher: bool = False,
    prefetch_delay: float = 0.1,
    num_partitions: int = 1,
    preprocesssing_time: float = 0.0001,
    batch_size: int = 1):
    
    cache = SharedCache(capacity=cache_capacity, eviction_policy=eviction_policy)
    jobs:List[DLTJob] = [DLTJob(job_id) for job_id in workload_jobs]
    for job in jobs:
        job.set_job_processing_speed(workload_jobs[job.job_id])

    sampler = BatchManager(
        dataset=Dataset(num_samples=batches_per_job, batch_size=batch_size, num_partitions=num_partitions),
        drop_last=False,
        shuffle=False,
        min_lookahead_steps=40,
        use_prefetching=use_prefetcher)
    sampler.jobs = {job.job_id: job for job in jobs}

    # for job in jobs:
    #     sampler.register_job(job)
    
    event_queue = []  # Priority queue for next event times
    time_elapsed = 0  # Global simulation time
    time_between_job_starts = 0.25
    next_job_start_time = 0.1
    for job in jobs:
        heapq.heappush(event_queue, (next_job_start_time, "dataloader_step", job))
        next_job_start_time += time_between_job_starts

    while True:
        if simulation_time_sec is not None and time_elapsed >= simulation_time_sec:
            break
        #check all jobs have completed their batches
        if batches_per_job is not None and all(job.num_batches_processed >= batches_per_job for job in jobs):
            break
        
        time_elapsed, event_type, payload = heapq.heappop(event_queue)

        if event_type == "dataloader_step":
            job:DLTJob = payload
            job.elapased_time_sec = time_elapsed
            next_batch, cache_on_miss, eviction_candidate = sampler.get_next_batch_for_job(job.job_id)
            cache_hit = cache.get_batch(next_batch.batch_id)
            logger.debug(f"Job {job.job_id} assigned batch {next_batch.batch_id} at time {time_elapsed:.2f}s")
            if cache_hit:
                job.cache_hit_count += 1
                delay = preprocesssing_time
                heapq.heappush(event_queue, (time_elapsed + delay, "training_step", (job, cache_hit, eviction_candidate, False)))
            else:
                job.cache_miss_count += 1
                if cache_on_miss:
                    heapq.heappush(event_queue, (time_elapsed + 0.001, "cache_insert", (job, next_batch, eviction_candidate)))
                else:
                    delay = load_from_s3_time + preprocesssing_time
                    heapq.heappush(event_queue, (time_elapsed + delay, "training_step", (job, cache_hit, eviction_candidate, False)))
        
        elif event_type == "cache_insert":
            job, next_batch, eviction_candidate = payload
            delay = load_from_s3_time + preprocesssing_time
            batch_is_cached  = False
            did_evict = False
            batch_id = next_batch.batch_id
            batch_reuse_score = next_batch.reuse_score
            canditdate_batch_reuse_score = sampler.get_batch_reuse_score(eviction_candidate) if eviction_candidate else None

            if cache.cache_is_full():
                if eviction_candidate is None:
                    heapq.heappush(event_queue, (time_elapsed + delay, "training_step", (job, batch_is_cached, eviction_candidate, did_evict)))
                    logger.debug(f"Cache is full, but no eviction candidate found for {batch_id}.")
                else:
                    # Evict the batch with the lowest reuse score
                    if eviction_policy not in ["noevict", "reuse_score"]:
                        evicted_batchid, did_evict = cache._evict_one()
                    else:
                        evicted_batchid, did_evict  = cache._remove(eviction_candidate)
                    if did_evict:
                        batch_is_cached = cache.put_batch(batch_id)
                    else:
                        logger.error(f"Batch {eviction_candidate} not found in cache when trying to evict.")
                    
                    heapq.heappush(event_queue, (time_elapsed + delay, "training_step", (job, batch_is_cached, evicted_batchid, did_evict)))
                    logger.debug(f"Evicted {evicted_batchid} ({sampler.get_batch_reuse_score(evicted_batchid)}) for {batch_id} ({sampler.get_batch_reuse_score(evicted_batchid)}).")
            else:
                logger.debug(f"Cache is not full, inserting {batch_id}.")
                batch_is_cached = cache.put_batch(batch_id)
                heapq.heappush(event_queue, (time_elapsed + delay, "training_step", (job, batch_is_cached, eviction_candidate, did_evict)))


        elif event_type == "training_step":
            job, batch_is_cached, eviction_candidate_batch_id, did_evict  = payload
            logger.info(f"Job {job.job_id} processing batch {job.current_batch.batch_id} at time {time_elapsed:.2f}s. Cache hit: {batch_is_cached}.")
            job.elapased_time_sec = time_elapsed
            sampler.processed_batch_update(
                job.job_id,
                batch_is_cached=batch_is_cached,
                eviction_candidate_batch_id=eviction_candidate_batch_id,
                did_evict=did_evict
                )
            # logger.debug(f"Job {job.job_id} finished processing batch {job.current_batch.batch_id} at time {time_elapsed + job.processing_speed:.2f}s")
            
            job.num_batches_processed += 1
            if batches_per_job is None or job.num_batches_processed == batches_per_job:
                print(f"Job {job.job_id} has completed its batches after {time_elapsed:.2f}s. Throughput: {job.num_batches_processed / time_elapsed:.2f} batches/s")
                continue
            heapq.heappush(event_queue, (time_elapsed + job.processing_speed, "dataloader_step", job))

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
    print(f"  Optimal Job Throughputs: {optimal_throughputs}")
    print(f"  Actual Job Throughputs: {throuhgputs_for_jobs}")
    print(f"  Total Throughput: {agg_throuhgput:.2f} batches/s")
    print("-" * 40)
    # return overall_results
    print(sampler.assigned_eviction_candidates)

if __name__ == "__main__":
    dataloader_system  = 'DisDL' #'CoorDL', TensorSocket, DisDL
    workload_name = 'imagenet_128_nas' #'imagenet_128_hpo', 'imagenet_128_resnet50', imagenet_128_nas
    workload_jobs = dict(workloads[workload_name])

    simulation_time_sec = None #3600 # None  #3600 * 1 # Simulate 1 hour
    batches_per_job = 100 * 1 # 8500 #np.inf
    cache_capacity = 0.25 * batches_per_job #* batches_per_job #number of batches as a % of the total number of batches
    eviction_policy = "reuse_score" # "lru", "fifo", "mru", "random", "noevict", "reuse_score"
    hourly_ec2_cost = 12.24 
    hourly_cache_cost = 3.25
    load_from_s3_time = 0.2 
    prefetcher_speed = load_from_s3_time /2
    preprocesssing_time = 0.01
    num_partitions = 2
    batch_size = 1

    run_simulation(
        dataloader_system = dataloader_system,
        workload_name = workload_name,
        workload_jobs = workload_jobs,
        cache_capacity = cache_capacity,
        eviction_policy = eviction_policy,
        load_from_s3_time=load_from_s3_time,
        hourly_cache_cost = hourly_cache_cost,
        hourly_ec2_cost = hourly_ec2_cost,
        simulation_time_sec=simulation_time_sec,
        batches_per_job=batches_per_job,
        use_prefetcher=False,
        prefetch_delay=prefetcher_speed,
        num_partitions=num_partitions,
        preprocesssing_time=preprocesssing_time,
        batch_size= batch_size)