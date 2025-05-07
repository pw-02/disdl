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
import random
from enum import Enum
from typing import List, Optional, Dict, Tuple
from collections import deque, OrderedDict

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='disdlapp.log',         # Log file name
    filemode='w'                # Overwrite the file each run; use 'a' to append
)


logger = logging.getLogger(__name__)

class CacheStatus(Enum):
    CACHED = "CACHED"
    CACHING_IN_PROGRESS = "CACHING_IN_PROGRESS"
    NOT_CACHED = "NOT_CACHED"


class BatchSet:
    def __init__(self, batch_set_id:str):
        self.id = batch_set_id
        self.batches: Dict[str, Batch] = OrderedDict()
        self.batches_finalized = False
        self.mark_for_eviction = False
        self.bacth_set_reuse_score = 0.0
    
    def compute_batch_set_reuse_score(self):
        """Compute the reuse score based on the number of jobs that have seen this batch."""
        with self.lock:
            score = 0.0
            for batch in self.batches.values():
                score += batch.reuse_score
            self.bacth_set_reuse_score = score
    

class Batch:
    def __init__(self, batch_indicies, epoch_idx, partition_idx, batch_idx):
        self.indices: List[int] = batch_indicies
        self.epoch_idx:int = epoch_idx
        self.partition_idx:int = partition_idx
        self.batch_idx:int = batch_idx
        self.batch_id:str = self._gen_batch_id(epoch_idx, partition_idx, batch_idx)
        self.batch_set_id = f"{epoch_idx}_{partition_idx}"
        self.cache_status:CacheStatus = CacheStatus.NOT_CACHED
        self.last_accessed_time:float = 0 #None #float('inf')
        self.is_first_access = True
        self.reuse_score: float = 0.0
        self.awaiting_to_be_seen_by: Dict[str, float] = {}

    def compute_weighted_reuse_score(self):
        """Compute the reuse score based on the number of jobs that have seen this batch."""
        self.reuse_score = sum(self.awaiting_to_be_seen_by.values())
    
    def mark_seen_by(self, job_id: str):
        # with self.lock:
            if job_id in self.awaiting_to_be_seen_by:
                del self.awaiting_to_be_seen_by[job_id]
            self.compute_weighted_reuse_score()

    def mark_awaiting_to_be_seen_by(self, job_id: str, weight: float):
        # with self.lock:
            if job_id not in self.awaiting_to_be_seen_by:
                self.awaiting_to_be_seen_by[job_id] = weight
            self.compute_weighted_reuse_score()


    def _gen_batch_id(self, epoch_idx:int, partition_idx:int, batch_idx:int) -> str:
        # Convert integers to strings and concatenate them
        id_string = ''.join(str(x) for x in self.indices)
        unique_id = hashlib.md5(id_string.encode()).hexdigest()
        unique_id = unique_id[:16]
        batch_id = f"{epoch_idx}_{partition_idx}_{batch_idx}_{unique_id}"
        return batch_id

    def time_since_last_access(self):
        """Calculate time elapsed since last access."""
        if self.is_first_access:
            return 0
        return time.perf_counter() - self.last_accessed_time
        
    def set_last_accessed_time(self):
        """Set the last accessed time to the current time."""
        self.last_accessed_time = time.perf_counter()
    
    def set_cache_status(self, cache_status:CacheStatus):
        self.cache_status = cache_status


class Dataset:
    def __init__(self, num_samples: str):
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples
    
class PartitionedBatchSampler():
    def __init__(self, num_files:Sized, batch_size, num_partitions = 10,  drop_last=False, shuffle=True):
        self.num_files = num_files
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last
        partitions = self._partition_indices(num_partitions) # List of partitions (each a list of indices)
        self.partitions_cycle = cycle(enumerate(partitions))  # Track partition index
        self.num_partitions = len(partitions)
        self.batches_per_epoch = self.calc_num_batchs_per_epoch()

        # Initialize epoch tracking
        self.current_epoch = 1
        self.current_idx = 1
        self.processed_partitions = 0  # Track processed partitions in the current epoch
        # Start with the first partition
        self.active_partition_idx, self.active_partition = next(self.partitions_cycle)
        self.sampler = self._create_sampler(self.active_partition)

    def _create_sampler(self, partition):
        """Create a new sampler based on the shuffle setting."""
        if self.shuffle:
            return iter(random.sample(partition, len(partition)))  # Random order
        else:
            return iter(partition)  # Sequential order
    
    def __iter__(self):
        return self
    
    def __next__(self):
        """Generate a mini-batch from the current partition, switching partitions when needed."""
        sampled_indices = []

        while len(sampled_indices) < self.batch_size:
            try:
                sampled_indices.append(next(self.sampler))  # Get an index from the partition
            except StopIteration:
                if not self.drop_last and sampled_indices:
                    return self.generate_batch(sampled_indices)  # Return smaller batch if drop_last=False
                
                # Move to the next partition
                self.processed_partitions += 1
                if self.processed_partitions == self.num_partitions:
                    self.current_epoch += 1  # Full epoch completed
                    self.processed_partitions = 0  # Reset for the next epoch
                    self.current_idx = 1  # Reset for the next epoch
                    # print(f"Epoch {self.current_epoch} completed!")  # Notify when an epoch ends

                self.active_partition_idx, self.active_partition = next(self.partitions_cycle)
                self.sampler = self._create_sampler(self.active_partition)
                continue  # Restart batch sampling from new partition

        return self.generate_batch(sampled_indices)
    
    def generate_batch(self, batch_indices):
        next_batch = Batch(batch_indices, self.current_epoch, self.active_partition_idx+1, self.current_idx)
        self.current_idx += 1  # Increment the batch index for the next batch
        return next_batch

    def _partition_indices(self, num_partitions):
    # Initialize a list to hold the partitions
        indices = list(range(self.num_files))  # Create a list of indices [0, 1, ..., num_files - 1]
        if self.shuffle:
            random.shuffle(indices)  # Shuffle the indices once

        # Split into roughly equal partitions
        partition_size = self.num_files // num_partitions
        partitions = [indices[i * partition_size : (i + 1) * partition_size] for i in range(num_partitions)]

        # Add remaining indices to the last partition (if num_files is not evenly divisible)
        remainder = self.num_files % num_partitions
        for i in range(remainder):
            partitions[i].append(indices[num_partitions * partition_size + i])

        total_files = sum(len(samples) for samples in partitions)
        # #print number files in each partition
        # for i, partition in enumerate(partitions):
        #     print(f"Partition {i}: {len(partition)} files")
        assert total_files == self.num_files

        return partitions
    
    def calc_num_batchs_per_partition(self):
        # Calculate the number of batches
        if self.drop_last:
            return len(self.active_partition) // self.batch_size
        else:
            return (len(self.active_partition) + self.batch_size - 1) // self.batch_size
    
    def calc_num_batchs_per_epoch(self):
        # Calculate the number of batches
        if self.drop_last:
            return self.num_files // self.batch_size
        else:
            return (self.num_files + self.batch_size - 1) // self.batch_size
    

class DLTJob:
    def __init__(self, job_id: str, num_partitions: int, speed: float):
        self.job_id = job_id
        self.num_partitions = num_partitions
        self.current_epoch_idx = 0
        # For reuse logic
        self.used_epoch_partition_pairs: Set[Tuple[int, int]] = set()
        self.partitions_covered_this_epoch: Set[int] = set()
        # Active state
        self.active_bacth_set_id = None
        self.future_batches: OrderedDict[str, Batch] = OrderedDict()
        self.processing_speed = speed
        self.current_batch = None
        self.num_batches_processed = 0
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.elapased_time_sec = 0
        self.speed = speed
        self.weight = 1/speed
        self.optimal_throughput = 1/speed #batches/sec

    def has_completed_epoch(self) -> bool:
        return len(self.partitions_covered_this_epoch) == self.num_partitions

    def reset_for_new_epoch(self):
        self.current_epoch_idx += 1
        self.partitions_covered_this_epoch.clear()
    
    def next_batch(self) -> Optional[Batch]:
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
                'job_speed': self.speed,
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
        return self.speed < other.speed  # Compare based on speed

class BatchManager:
    def __init__(self,
                 dataset,
                 batch_size: int,
                 num_partitions: int,
                 jobs: List[DLTJob]):
        self.dataset = dataset
        self.sampler = PartitionedBatchSampler(num_files=len(dataset),
                batch_size=batch_size,
                num_partitions=num_partitions,
                drop_last=False,
                shuffle=False)
        self.epoch_partition_batches: Dict[int, Dict[int, BatchSet]] = OrderedDict()
        self.cached_batches: Dict[str, Batch] = {}
        self.eviction_index: SortedList[Tuple[float, float, str]] = SortedList() # Sorted by (reuse_score, timestamp)
        self.eviction_index_lookup: Dict[str, Tuple[float, float, str]] = {} #delete batches from eviction_index efficiently
        self.assigned_eviction_candidates: Dict[str, Batch] = {}
        self.lookahead_distance = min(self.sampler.calc_num_batchs_per_partition() - 1, 40)
        self.jobs: Dict[str, DLTJob] = {job.job_id: job for job in jobs}

        for _ in range(self.lookahead_distance):
            self._generate_new_batch()
    
    def _generate_new_batch(self):
        next_batch: Batch = next(self.sampler)
        partition_batches = self.epoch_partition_batches.setdefault(
            next_batch.epoch_idx, OrderedDict()
        )
        batch_set = partition_batches.setdefault(
            next_batch.partition_idx, BatchSet(next_batch.batch_set_id)
        )
        batch_set.batches[next_batch.batch_id] = next_batch
        # Check if any job is currently assigned to this BatchSet
        for job in self.jobs.values():
            if job.active_bacth_set_id == batch_set.id:
                next_batch.mark_awaiting_to_be_seen_by(job.job_id, job.weight)
                job.future_batches[next_batch.batch_id] = next_batch
        return next_batch
    
    def _score_batch_set(self, batch_set: BatchSet, epoch_idx, partition_idx) -> float:
        return float(f"{epoch_idx}.{partition_idx:02d}")
    
    def assign_batch_set_to_job(self, job: DLTJob):
        if job.has_completed_epoch():
            job.reset_for_new_epoch()

        best_candidate = None
        best_score = float('-inf')

        for epoch_idx, partition_map in self.epoch_partition_batches.items():
            for partition_idx, batch_set in partition_map.items():
                if (epoch_idx, partition_idx) in job.used_epoch_partition_pairs:
                    continue
                if partition_idx in job.partitions_covered_this_epoch:
                    continue

                score = self._score_batch_set(batch_set, epoch_idx, partition_idx)
                if score > best_score:
                    best_candidate = (epoch_idx, partition_idx, batch_set)
                    best_score = score
        if best_candidate is None:
            return

        epoch_idx, partition_idx, batch_set = best_candidate
        job.used_epoch_partition_pairs.add((epoch_idx, partition_idx))
        job.partitions_covered_this_epoch.add(partition_idx)
        job.active_bacth_set_id = batch_set.id

        for batch in batch_set.batches.values():
            batch.mark_awaiting_to_be_seen_by(job.job_id, job.weight)
            job.future_batches[batch.batch_id] = batch
    
    def job_processed_batch_update(self, 
                                   job_id: int,
                                   batch_is_cached: bool, 
                                   job_cached_batch: bool, 
                                   eviction_candidate_batch_id: Optional[str],
                                   evicted: bool = False):
        
        job:DLTJob = self.jobs[job_id]
        batch:Batch  = job.current_batch
        batch.mark_seen_by(job.job_id)    

        if batch_is_cached:
            batch.set_cache_status(CacheStatus.CACHED)

            if job_cached_batch:
                self.cached_batches[batch.batch_id] = batch

            # Remove outdated reuse score from eviction index
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
        
        if evicted:
            evicted_batch:Batch = self.cached_batches.get(eviction_candidate_batch_id, None)
            if evicted_batch:
                evicted_batch.set_cache_status(CacheStatus.NOT_CACHED)
            self.cached_batches.pop(eviction_candidate_batch_id, None)
            self.assigned_eviction_candidates.pop(eviction_candidate_batch_id, None)
            entry = self.eviction_index_lookup.pop(eviction_candidate_batch_id, None)
            if entry:
                self.eviction_index.discard(entry)
        else:
            if eviction_candidate_batch_id in self.assigned_eviction_candidates:
                self.assigned_eviction_candidates.pop(eviction_candidate_batch_id, None)

    def _maybe_cache_batch(self, batch: Batch):
        should_cache = False
        eviction_candidate = None
        min_reuse_score_to_cache = 0.01

        if batch.cache_status in (CacheStatus.CACHED, CacheStatus.CACHING_IN_PROGRESS):
            return should_cache, eviction_candidate

        # Apply minimum score cutoff
        if (batch.reuse_score) <= min_reuse_score_to_cache:
            logger.debug(f"Skipped caching {batch.batch_id}: reuse_score {batch.reuse_score:.2f} below threshold {min_reuse_score_to_cache}")
            return should_cache, eviction_candidate

        should_cache = True
        batch.set_cache_status(CacheStatus.CACHING_IN_PROGRESS)

        if self.eviction_index:
            for score, ts, batch_id in self.eviction_index:
                if batch.reuse_score > score and batch_id not in self.assigned_eviction_candidates:
                    self.assigned_eviction_candidates[batch_id] = batch
                    eviction_candidate = batch_id
                    break
        return should_cache, eviction_candidate
    
    def _maybe_trigger_smaple_next_bacth(self, batch: Batch):
        if batch.is_first_access:
            batch.is_first_access = False
            self._generate_new_batch()

    def get_next_batch_for_job(self, job_id: str) -> Optional[Batch]:
        job:DLTJob = self.jobs[job_id]
        if not job.future_batches:
            self.assign_batch_set_to_job(job)
        next_batch = job.next_batch()
        if next_batch is None:
            logger.debug(f"Job {job.job_id} has no future batches.")
            return None, False, None
        should_cache, eviction_candidate = self._maybe_cache_batch(next_batch)
        self._maybe_trigger_smaple_next_bacth(next_batch)
        return next_batch, should_cache, eviction_candidate
    
    # def register_job(self, job: DLTJob):
    #     self.jobs[job.job_id] = job
    #     self.assign_batch_set_to_job(job)
    
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
    prefetch_delay: float = 0.1,
    num_partitions: int = 1,
    preprocesssing_time: float = 0.0001,
    batch_size: int = 1):
    
    cache = SharedCache(capacity=cache_capacity, eviction_policy=eviction_policy)
    jobs:List[DLTJob] = [DLTJob(job_id, num_partitions, speed) for job_id, speed in workload_jobs]

    sampler = BatchManager(
        dataset=Dataset(num_samples=batches_per_job),
        batch_size=batch_size,
        num_partitions=num_partitions,
        jobs=jobs)
    # for job in jobs:
    #     sampler.register_job(job)
    
    event_queue = []  # Priority queue for next event times
    time_elapsed = 0  # Global simulation time

    for job in jobs:
        heapq.heappush(event_queue, (0, "step", job))

    while True:
        if simulation_time_sec is not None and time_elapsed >= simulation_time_sec:
            break
        #check all jobs have completed their batches
        if batches_per_job is not None and all(job.num_batches_processed >= batches_per_job for job in jobs):
            break
        
        time_elapsed, event_type, payload = heapq.heappop(event_queue)
        
        if event_type == "step":
            job:DLTJob = payload
            next_batch, should_cache, eviction_candidate = sampler.get_next_batch_for_job(job.job_id)
            batch_id = next_batch.batch_id
            job.elapased_time_sec = time_elapsed
            cache_hit = cache.get_batch(batch_id)
            logger.info(f"Job {job.job_id} processing batch {batch_id} at time {time_elapsed:.2f}s. Cache hit: {cache_hit}")
            if cache_hit:
                job.cache_hit_count += 1
                job_delay = job.speed + preprocesssing_time
                heapq.heappush(event_queue, (time_elapsed + preprocesssing_time, "batch_fetch_complete", (job, True, False, eviction_candidate, False)))
            else:
                job.cache_miss_count += 1
                if should_cache:
                    heapq.heappush(event_queue, (time_elapsed + load_from_s3_time, "cache_insert", (job, next_batch, eviction_candidate)))
                job_delay = job.speed + load_from_s3_time + preprocesssing_time
            
            job.num_batches_processed += 1
            if batches_per_job is None or job.num_batches_processed < batches_per_job:
                heapq.heappush(event_queue, (time_elapsed + job_delay, "step", job))
            else:
                print(f"Job {job.job_id} has completed its batches after {time_elapsed:.2f}s. Throughput: {job.num_batches_processed / time_elapsed:.2f} batches/s")
                pass
    
        elif event_type == "cache_insert":
            job, next_batch, eviction_candidate = payload
            batch_id = next_batch.batch_id
            if cache.cache_is_full():
                if eviction_candidate is None:
                    heapq.heappush(event_queue, (time_elapsed + preprocesssing_time, "batch_fetch_complete", (job, False, False, eviction_candidate, False)))
                    logger.debug(f"Cache is full, but no eviction candidate found for {batch_id}.")
                else:
                    logger.debug(f"Cache is full, evicting {eviction_candidate} for {batch_id}.")
                    # Evict the batch with the lowest reuse score
                    if eviction_policy not in ["noevict", "reuse_score"]:
                        evicted_batchid, evicted = cache._evict_one()
                    else:
                        evicted_batchid, evicted  = cache._remove(eviction_candidate)
                    if evicted:
                        cache.put_batch(batch_id)
                        # print(f"Evicted {evicted_batchid} for {batch_id}.")
                    else:
                        logger.error(f"Batch {eviction_candidate} not found in cache when trying to evict.")

                    heapq.heappush(event_queue, (time_elapsed + preprocesssing_time, "batch_fetch_complete", (job, True, True, eviction_candidate, evicted)))
                    logger.debug(f"Evicted {evicted_batchid} for {batch_id}.")
            else:
                logger.debug(f"Cache is not full, inserting {batch_id}.")
                cache.put_batch(batch_id)
                heapq.heappush(event_queue, (time_elapsed + preprocesssing_time, "batch_fetch_complete", (job, True, True, eviction_candidate, False)))

        elif event_type == "batch_fetch_complete":
            job, batch_is_cached, job_cached_batch, eviction_candidate_batch_id, evicted  = payload
            job.elapased_time_sec = time_elapsed
            sampler.job_processed_batch_update(
                job.job_id,
                batch_is_cached=batch_is_cached,
                job_cached_batch=job_cached_batch,
                eviction_candidate_batch_id=eviction_candidate_batch_id,
                evicted=evicted
            )


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
    jobs =  workloads[workload_name].items()
    simulation_time_sec = None #3600 # None  #3600 * 1 # Simulate 1 hour
    batches_per_job = 100 * 1 # 8500 #np.inf
    cache_capacity = 0.25 * batches_per_job #* batches_per_job #number of batches as a % of the total number of batches
    eviction_policy = "reuse_score" # "lru", "fifo", "mru", "random", "noevict", "reuse_score"
    hourly_ec2_cost = 12.24 
    hourly_cache_cost = 3.25
    load_from_s3_time = 0.2
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
        num_partitions=1,
        preprocesssing_time=0.01,
        batch_size=1
    )