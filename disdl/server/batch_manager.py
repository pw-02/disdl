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


class DLTJob:
    def __init__(self, job_id: str, num_partitions: int):
        self.job_id = job_id
        self.num_partitions = num_partitions
        self.current_epoch_idx = 0

        # For reuse logic
        self.used_epoch_partition_pairs: Set[Tuple[int, int]] = set()
        self.partitions_covered_this_epoch: Set[int] = set()

        # Active state
        self.active_bacth_set_id = None
        self.future_batches: OrderedDict[str, Batch] = OrderedDict()

    def has_completed_epoch(self) -> bool:
        return len(self.partitions_covered_this_epoch) == self.num_partitions

    def reset_for_new_epoch(self):
        self.current_epoch_idx += 1
        self.partitions_covered_this_epoch.clear()

    def next_batch(self) -> Optional[Batch]:
        if self.future_batches:
            batch = self.future_batches.popitem(last=False)[1]
            batch.mark_seen_by(self.job_id)
            return batch
        return None


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

    def assign_batches_to_job(self, job: DLTJob):
        if job.has_completed_epoch():
            job.reset_for_new_epoch()

        for epoch_idx, partition_map in self.epoch_partition_batches.items():
            for partition_idx, batch_set in partition_map.items():
                if (epoch_idx, partition_idx) in job.used_epoch_partition_pairs:
                    continue
                if partition_idx in job.partitions_covered_this_epoch:
                    continue

                job.used_epoch_partition_pairs.add((epoch_idx, partition_idx))
                job.partitions_covered_this_epoch.add(partition_idx)
                job.active_bacth_set_id = batch_set.id

                for batch in batch_set.batches.values():
                    job.future_batches[batch.batch_id] = batch

                return  # One assignment per call

    def get_next_batch_for_job(self, job_id: str) -> Optional[Batch]:
        with self.lock:
            if self.prefetch_service and self.prefetch_service.prefetch_stop_event.is_set():
                self.prefetch_service.start()
                
            job = self._get_or_register_job(job_id)

            if not job.future_batches:
                self.assign_batches_to_job(job)

            next_batch = job.next_batch()

            if next_batch:
                self._tag_batch_for_caching(next_batch)
                self._maybe_trigger_batch_generation(next_batch)

            return next_batch


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