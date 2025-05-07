import random
import threading
import hydra
from sampler import PartitionedBatchSampler
from args import DisDLArgs
from collections import deque, OrderedDict
from typing import List, Optional, Dict, Set, Tuple
from dataset import MSCOCODataset, LibSpeechDataset, ImageNetDataset
from batch import Batch, BatchSet, CacheStatus
import time
from logger_config import logger
import json
from datetime import datetime, timezone
from itertools import cycle  # Import cycle from itertools
from typing import OrderedDict as TypingOrderedDict
import csv
import os
from cache_prefetching import PrefetchServiceAsync
# minibatch_service.py
from sortedcontainers import SortedList

import threading
from collections import OrderedDict
from typing import Dict, Set, Tuple, Optional
from batch import Batch, BatchSet, CacheStatus
from sampler import PartitionedBatchSampler
from cache_prefetching import PrefetchServiceAsync
from logger_config import logger
from cache_tracking import CacheManager
from job import DLTJob

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
        
        # if use_prefetching:
        #     self.prefetch_service = PrefetchServiceAsync(
        #         lambda_name=prefetch_lambda_name,
        #         cache_address=cache_address,
        #         simulate_time=prefetch_simulation_time
        #     )
        #     self.prefetch_service.start()

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

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: DisDLArgs):
    # Initialize the dataset based on the workload specified in the args
    if args.workload == 'mscoco':
        dataset = MSCOCODataset(dataset_location=args.dataset_location)
    elif args.workload.name == 'librispeech':
        dataset = LibSpeechDataset(dataset_location=args.dataset_location)
    elif args.workload.name == 'imagenet':
        dataset = ImageNetDataset(dataset_location='ss3://imagenet1k-sdl/val/')
    else:
        raise ValueError(f"Unsupported workload: {args.workload}")

    # Create the CentralBatchManager instance
    batch_manager = BatchManager(dataset=dataset,
                                        cache_address=args.cache_address,
                                        batch_size=args.workload.batch_size,
                                        num_partitions=args.workload.num_partitions,
                                        drop_last=args.workload.drop_last,
                                        shuffle=args.workload.shuffle,
                                        min_lookahead_steps=args.workload.min_lookahead_steps,
                                        use_prefetching=args.enable_prefetching,
                                        prefetch_lambda_name=args.workload.prefetch_lambda_name,
                                        prefetch_simulation_time=args.prefetch_simulation_time)
    # print(batch_manager.dataset_info())
    # job_id = batch_manager.add_job()
    for step in range(200):
        job_id = 1
        batch, should_cache, eivction_candidate = batch_manager.get_next_batch_for_job(job_id)
        if batch:
                # Simulate time spent on training step
                time.sleep(random.uniform(0.001, 0.005))  # Fast-forwarded for sim
                print(f"Step {step:03} | Job {job_id} got batch {batch.batch_id} | Cached: {batch.cache_status.name} | Reuse: {batch.reuse_score:.2f}")
        else:
                print(f"Step {step:03} | Job {job_id} got no batch")
        
        batch_manager.job_processed_batch_update(job_id, batch_is_cached=True, job_cached_batch=True, job_evicted_batch_id=None)


def simulate_training_loop(batch_manager:BatchManager, num_jobs: int, steps_per_job: int = 100):
    job_ids = [str(i) for i in range(1, num_jobs + 1)]
    
    for job_id in job_ids:
        batch_manager._get_or_register_job(job_id)

    for step in range(steps_per_job):
        for job_id in job_ids:
            batch, should_cache, eivction_candidate = batch_manager.get_next_batch_for_job(job_id)
            if batch:
                # Simulate time spent on training step
                time.sleep(random.uniform(0.001, 0.005))  # Fast-forwarded for sim
                print(f"Step {step:03} | Job {job_id} got batch {batch.batch_id} | Cached: {batch.cache_status.name} | Reuse: {batch.reuse_score:.2f}")
            else:
                print(f"Step {step:03} | Job {job_id} got no batch")
            

            batch_manager.job_processed_batch_update(job_id, batch_is_cached=True, job_cached_batch=True, job_evicted_batch_id=None)
            

    print("\nSimulation complete.")


if __name__ == "__main__":
    
    main()
