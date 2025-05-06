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

import threading
from collections import OrderedDict
from typing import Dict, Set, Tuple, Optional
from batch import Batch, BatchSet, CacheStatus
from sampler import PartitionedBatchSampler
from cache_prefetching import PrefetchServiceAsync
from logger_config import logger
from cache_tracking import CacheManager
from job import DLTJob

class CentralBatchManager:
    def __init__(
        self,
        dataset,
        cache_address: str,
        batch_size: int,
        num_partitions: int,
        drop_last: bool,
        shuffle: bool,
        min_lookahead_steps: int,
        use_prefetching: bool,
        prefetch_lambda_name: str,
        prefetch_simulation_time: int = None
    ):
        self.dataset = dataset
        self.sampler = PartitionedBatchSampler(
            num_files=len(dataset),
            batch_size=batch_size,
            num_partitions=num_partitions,
            drop_last=drop_last,
            shuffle=shuffle
        )
        self.num_partitions = num_partitions
        self.lookahead_distance = min(self.sampler.calc_num_batchs_per_partition() - 1, min_lookahead_steps)
        self.jobs: Dict[str, DLTJob] = {}
        self.epoch_partition_batches: Dict[int, Dict[int, BatchSet]] = OrderedDict()
        self.lock = threading.Lock()

        self.prefetch_service: Optional[PrefetchServiceAsync] = None
        if use_prefetching:
            self.prefetch_service = PrefetchServiceAsync(
                lambda_name=prefetch_lambda_name,
                cache_address=cache_address,
                simulate_time=prefetch_simulation_time
            )
            self.prefetch_service.start()
        if False:
            self.cache_manager = CacheManager(
            redis_url=cache_address,
            epoch_partition_batches=self.epoch_partition_batches,
            jobs=self.jobs)
        # self.cache_manager.start()

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
                job.future_batches[next_batch.batch_id] = next_batch
        return next_batch

    def _get_or_register_job(self, job_id: str) -> DLTJob:
        if job_id not in self.jobs:
            logger.info(f"Registering new job '{job_id}' with dataset '{self.dataset.s3_bucket}'")
            self.jobs[job_id] = DLTJob(job_id, self.num_partitions)
        return self.jobs[job_id]

    def _tag_batch_for_caching(self, batch: Batch):
        if batch.cache_status == CacheStatus.NOT_CACHED:
            batch.cache_status = CacheStatus.CACHING_IN_PROGRESS

    def _maybe_trigger_batch_generation(self, batch: Batch):
        if batch.is_first_access:
            batch.is_first_access = False
            self._generate_new_batch()
     
    def _score_batch_set(self, batch_set: BatchSet, epoch_idx, partition_idx) -> float:
        return float(f"{epoch_idx}.{partition_idx:02d}")

        # num_cached = sum(1 for batch in batch_set.batches.values()
        #                  if batch.cache_status != CacheStatus.NOT_CACHED)
        # num_active_jobs = sum(1 for job in self.jobs.values()
        #                       if job.active_bacth_set_id == batch_set.id)
        # return num_cached + 2.0 * num_active_jobs  # prioritize reuse and concurrency

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
            job.future_batches[batch.batch_id] = batch
    
    def get_next_batch_for_job(self, job_id: str) -> Optional[Batch]:
        with self.lock:
            if self.prefetch_service and self.prefetch_service.prefetch_stop_event.is_set():
                self.prefetch_service.start()

            job = self._get_or_register_job(job_id)

            if not job.future_batches:
                self.assign_batch_set_to_job(job)

            # Step 1: Prefer cached batches with highest reuse score
            eligible_cached = [
                batch for batch in job.future_batches.values()
                if batch.cache_status != CacheStatus.NOT_CACHED
            ]
            if eligible_cached:
                next_batch = min(eligible_cached, key=lambda b: b.reuse_score)
            else:
                # Step 2: Fallback to any non-CACHING_IN_PROGRESS batch
                next_batch = None
                for batch_id, batch in job.future_batches.items():
                    if batch.cache_status != CacheStatus.CACHING_IN_PROGRESS:
                        next_batch = batch
                        break

                # Step 3: Fallback to any available batch
                if not next_batch and job.future_batches:
                    next_batch = next(iter(job.future_batches.values()))

            if next_batch:
                next_batch.set_last_accessed_time()
                job.future_batches.pop(next_batch.batch_id, None)

                self._tag_batch_for_caching(next_batch)
                self._maybe_trigger_batch_generation(next_batch)

            return next_batch

    # def get_next_batch_for_job(self, job_id: str) -> Optional[Batch]:
    #     with self.lock:
    #         if self.prefetch_service and self.prefetch_service.prefetch_stop_event.is_set():
    #             self.prefetch_service.start()

    #         job = self._get_or_register_job(job_id)

    #         if not job.future_batches:
    #             self.assign_batch_set_to_job(job)

    #         next_batch = job.next_batch()

    #         if next_batch:
    #             self._tag_batch_for_caching(next_batch)
    #             self._maybe_trigger_batch_generation(next_batch)

    #         return next_batch


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
    batch_manager = CentralBatchManager(dataset=dataset,
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
    for i in range(200):
        batch = batch_manager.get_next_batch_for_job(job_id=1)
        if batch is not None:
            print(f"Batch {batch.batch_id} with {len(batch.indices)} samples")
        else:
            print("No batch available")
    


if __name__ == "__main__":
    main()