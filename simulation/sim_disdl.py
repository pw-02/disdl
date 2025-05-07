import hashlib
import heapq
from itertools import cycle
import numpy as np
import logging
from typing import List, Sized, Tuple, Dict, Set, Any
import sys
from sortedcontainers import SortedList
sys.path.append(".")
from simulation.sim_workloads import workloads
from simulation.sim_cache import SharedCache
import time
from typing import List, Optional, Dict, Tuple
from collections import deque, OrderedDict
from disdl.server.sampler import PartitionedBatchSampler
from disdl.server.batch import Batch, BatchSet, CacheStatus
from disdl.server.utils import AverageMeter
import threading

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='disdlapp.log',         # Log file name
    filemode='w'                # Overwrite the file each run; use 'a' to append
)

logger = logging.getLogger(__name__)

class Dataset:
    def __init__(self, num_samples: str, batch_size:int ,  num_partitions: int = 1):
        self.num_samples = num_samples
        self.num_partitions = num_partitions
        self.batch_size = batch_size
    def __len__(self) -> int:
        return self.num_samples
    
class DLTJob:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.current_epoch = 0
        # For reuse logic
        # self.used_epoch_partition_pairs: Set[Tuple[int, int]] = set()
        self.used_batch_set_ids: Set[str] = set()
        self.partitions_covered_this_epoch: Set[int] = set()
        # Active state
        self.current_batch = None
        self.current_batch_set_id  = None
        self.future_batches: OrderedDict[str, Batch] = OrderedDict()
        self.processing_speed = 1.0
        self.optimal_throughput = 1/self.processing_speed #batches/sec
        self.weight = self.optimal_throughput
        self.num_batches_processed = 0
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.elapased_time_sec = 0
        self.dataload_delay = AverageMeter('Dataload Delay')
        self.lock = threading.Lock()

    def set_job_processing_speed(self, speed: float):
        self.processing_speed = speed
        self.optimal_throughput = 1 / speed
        self.weight = self.optimal_throughput
   
    def reset_for_new_epoch(self):
        self.current_epoch += 1
        self.partitions_covered_this_epoch.clear()
    
    def next_batch(self) -> Optional[Batch]:
        # with self.lock:
        next_batch = None
        best_score = float('inf')
        fallback_batch = None
        for batch in self.future_batches.values():
            if batch.cache_status == CacheStatus.CACHED:
                if batch.reuse_score < best_score:
                    next_batch = batch
                    best_score = batch.reuse_score
            elif batch.cache_status != CacheStatus.CACHING_IN_PROGRESS and fallback_batch is None:
                fallback_batch = batch
        if not next_batch:
            next_batch = fallback_batch

        if not next_batch:
            #just get the next batch in the future batches
            next_batch = next(iter(self.future_batches.values()), None)
        
        if next_batch:
            next_batch.set_last_accessed_time()
            self.future_batches.pop(next_batch.batch_id, None)
        self.current_batch = next_batch
        return next_batch
    
    def perf_stats(self, horurly_ec2_cost=12.24, hourly_cache_cost=3.25):
            hit_rate = self.cache_hit_count / (self.cache_hit_count + self.cache_miss_count) if (self.cache_hit_count + self.cache_miss_count) > 0 else 0
            throughput = self.num_batches_processed / self.elapased_time_sec if self.elapased_time_sec > 0 else 0
            self.compute_cost = (horurly_ec2_cost / 3600) * self.elapased_time_sec
            cache_cost = (hourly_cache_cost / 3600) * self.elapased_time_sec
            total_cost = self.compute_cost + hourly_cache_cost
            return {
                'job_id': self.job_id,
                'job_speed': self.processing_speed,
                'batches_processed': self.num_batches_processed,
                'cache_hit_count': self.cache_hit_count,
                'cache_miss_count': self.cache_miss_count,
                'cache_hit_%': hit_rate,
                'elapsed_time': self.elapased_time_sec,
                'throughput(batches/s)': throughput,
                'optimal_throughput(batches/s)': self.optimal_throughput,
                'compute_cost': self.compute_cost,
                'cache_cost': cache_cost,
                'total_cost': total_cost
                }
    def __lt__(self, other):
        return self.processing_speed < other.processing_speed  # Compare based on speed

class BatchManager:
    def __init__(self, 
                 dataset:Dataset, 
                 drop_last: bool = False,
                 shuffle: bool = False,
                 min_lookahead_steps: int = 40,
                 use_prefetching: bool = False,
                 prefetch_lambda_name: str = None,
                 prefetch_simulation_time: int = None
                ):
        
        self.dataset = dataset
        self.sampler = PartitionedBatchSampler(
            num_files=len(dataset),
            batch_size=dataset.batch_size,
            num_partitions=dataset.num_partitions,
            drop_last=drop_last,
            shuffle=shuffle)
        
        self.lock = threading.Lock()
        self.batch_sets: Dict[int, Dict[int, BatchSet]] = OrderedDict()
        self.cached_batches: Dict[str, Batch] = {}
        self.eviction_index: SortedList[Tuple[float, float, str]] = SortedList() # Sorted by (reuse_score, timestamp)
        self.eviction_index_lookup: Dict[str, Tuple[float, float, str]] = {} #delete batches from eviction_index efficiently
        self.assigned_eviction_candidates: Dict[str, Batch] = {}
        self.lookahead_distance = min(self.sampler.calc_num_batchs_per_partition() - 1, 40)
        # self.jobs: Dict[str, DLTJob] = {job.job_id: job for job in jobs}
        self.jobs: Dict[str, DLTJob] = {}

        for _ in range(self.lookahead_distance):
            self._generate_new_batch()
    
    def _generate_new_batch(self):
        batch_indices, epoch_idx, partition_idx, batch_idx = next(self.sampler)
        next_batch = Batch(batch_indices, epoch_idx, partition_idx, batch_idx)

        epoch_map = self.batch_sets.setdefault(next_batch.epoch_idx, OrderedDict())
        batch_set = epoch_map.setdefault(next_batch.partition_idx, BatchSet(set_id=next_batch.set_id))
        batch_set.batches[next_batch.batch_id] = next_batch

        # Check if any job is currently assigned to this BatchSet
        for job in self.jobs.values():
            if job.current_batch_set_id  == batch_set.id:
                next_batch.mark_awaiting_to_be_seen_by(job.job_id, job.weight)
                job.future_batches[next_batch.batch_id] = next_batch
        return next_batch
    
    def _score_batch_set(self, batch_set: BatchSet, epoch_idx, partition_idx) -> float:
        return float(f"{epoch_idx}.{partition_idx:02d}")
    
    def _get_or_register_job(self, job_id: str) -> DLTJob:
        if job_id not in self.jobs:
            logger.info(f"Registering new job '{job_id}'")
            self.jobs[job_id] = DLTJob(job_id)
        return self.jobs[job_id]
    
    def get_next_batch_for_job(self, job_id: str) -> Optional[Batch]:    
        job:DLTJob  = self._get_or_register_job(job_id)

        if not job.future_batches:
            self.assign_batch_set_to_job(job)
        
        next_batch = job.next_batch()
        if next_batch is None:
            logger.error(f"Job {job.job_id} has no future batches.")
            return None, False, None
        
        should_cache, eviction_candidate = self._maybe_cache_batch(next_batch)
        
        self._maybe_trigger_sample_next_batch(next_batch)
        
        return next_batch, should_cache, eviction_candidate
    
    def assign_batch_set_to_job(self, job: DLTJob):
        if len(job.partitions_covered_this_epoch) == self.dataset.num_partitions:
            job.reset_for_new_epoch()
        
        best_candidate = None
        best_score = float('-inf')

        for epoch_idx, partition_map in self.batch_sets.items():
            for partition_idx, batch_set in partition_map.items():
                if batch_set.id in job.used_batch_set_ids:
                    continue
                if partition_idx in job.partitions_covered_this_epoch:
                    continue

                score = self._score_batch_set(batch_set, epoch_idx, partition_idx)
                if score > best_score:
                    best_candidate = (partition_idx, batch_set)
                    best_score = score
        if best_candidate is None:
            return

        partition_idx, batch_set = best_candidate
        job.used_batch_set_ids.add(batch_set.id)
        job.partitions_covered_this_epoch.add(partition_idx)
        job.current_batch_set_id = batch_set.id

        for batch in batch_set.batches.values():
            batch.mark_awaiting_to_be_seen_by(job.job_id, job.weight)
            job.future_batches[batch.batch_id] = batch

    def _maybe_cache_batch(self, batch: Batch):
        min_reuse_score_to_cache = 0.01

        if batch.cache_status in (CacheStatus.CACHED, CacheStatus.CACHING_IN_PROGRESS):
            return False, None
        
        # Apply minimum score cutoff
        if batch.reuse_score <= min_reuse_score_to_cache:
            logger.debug(f"Skipped caching {batch.batch_id}: reuse_score {batch.reuse_score:.2f} below threshold {min_reuse_score_to_cache}")
            return False, None

        # Mark as eligible for caching
        batch.set_cache_status(CacheStatus.CACHING_IN_PROGRESS)
        eviction_candidate = None
        if self.eviction_index:
            for score, ts, batch_id in self.eviction_index:
                if batch.reuse_score > score and batch_id not in self.assigned_eviction_candidates:
                    self.assigned_eviction_candidates[batch_id] = batch
                    eviction_candidate = batch_id
                    break
        return True, eviction_candidate


    
    def processed_batch_update(self,
                               job_id: int,
                               batch_is_cached: bool,
                               eviction_candidate_batch_id: Optional[str],
                               did_evict: bool = False):
        
        job:DLTJob = self.jobs[job_id]
        batch:Batch  = job.current_batch
        batch.mark_seen_by(job.job_id)

        if batch_is_cached:
            batch.set_cache_status(CacheStatus.CACHED)
            self.cached_batches[batch.batch_id] = batch
            #Update eviction index entry
            old_entry = self.eviction_index_lookup.pop(batch.batch_id, None)
            if old_entry:
                self.eviction_index.discard(old_entry)

            # Re-insert updated entry
            new_entry = (batch.reuse_score, time.time(), batch.batch_id)
            self.eviction_index.add(new_entry)
            self.eviction_index_lookup[batch.batch_id] = new_entry
        else:
            batch.set_cache_status(CacheStatus.NOT_CACHED)
            self.cached_batches.pop(batch.batch_id, None)

        if eviction_candidate_batch_id:
            self.assigned_eviction_candidates.pop(eviction_candidate_batch_id, None)
            if did_evict:
                self.cached_batches.pop(eviction_candidate_batch_id, None)
                evicted_entry = self.eviction_index_lookup.pop(eviction_candidate_batch_id, None)
                if evicted_entry:
                    self.eviction_index.discard(evicted_entry)

    def _maybe_trigger_sample_next_batch(self, batch: Batch):
        if batch.is_first_access:
            batch.is_first_access = False
            self._generate_new_batch()

    def get_batch_reuse_score(self, batch_id: str) -> float:
        #find batch somehher across all the batches in the batch_sets
        for partition_map in self.batch_sets.values():
            for batch_set in partition_map.values():
                if batch_id in batch_set.batches:
                    return batch_set.batches[batch_id].reuse_score

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
    batches_per_job = 8500 * 10 # 8500 #np.inf
    cache_capacity = 0.25 * batches_per_job #* batches_per_job #number of batches as a % of the total number of batches
    eviction_policy = "lru" # "lru", "fifo", "mru", "random", "noevict", "reuse_score"
    hourly_ec2_cost = 12.24 
    hourly_cache_cost = 3.25
    load_from_s3_time = 0.00002 
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