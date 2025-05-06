import threading
import hydra
from sampler import PartitionedBatchSampler
from job import DLTJob
from args import DisDLArgs
from collections import deque, OrderedDict
from typing import List, Optional, Dict, Tuple
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

        num_batches_per_partition = self.sampler.calc_num_batchs_per_partition()
        self.active_epoch_idx = None
        self.active_partition_idx = None
        self.lookahead_distance = min(num_batches_per_partition - 1, min_lookahead_steps)
        self.jobs: Dict[str, DLTJob] = {}
        self.epoch_partition_batches: Dict[int, Dict[int, BatchSet]] = OrderedDict()
        self.lock = threading.Lock()  # Lock for thread safety

        # Initialize prefetching service if enabled
        self.prefetch_service: Optional[PrefetchServiceAsync] = None
        if use_prefetching:
            self.prefetch_service = PrefetchServiceAsync(
                lambda_name=prefetch_lambda_name,
                cache_address=cache_address,
                simulate_time=prefetch_simulation_time
            )
            self.prefetch_service.start()
        
        # Generate initial lookahead batches
        for _ in range(self.lookahead_distance):
            self._generate_new_batch()

    def add_job(self):
        job_id = len(self.jobs) + 1
        with self.lock:
            self.jobs[job_id] = DLTJob(job_id)
            return job_id
        
    def dataset_info(self):
        return {
            'location': self.dataset.dataset_location,
            'num_samples': len(self.dataset),
            'batch_size': self.sampler.batch_size,
            'batches/epoch': self.sampler.batches_per_epoch,
            'partitions/epoch': self.sampler.num_partitions,
            'batches/partition': self.sampler.calc_num_batchs_per_partition(),

        }
    
    def _generate_new_batch(self):
        next_batch:Batch = next(self.sampler)
        self.active_epoch_idx = next_batch.epoch_idx
        self.active_partition_idx = next_batch.partition_idx
        self.active_batch_set_id = next_batch.batch_set_id

        # Get or create partition dictionary for the current epoch
        partition_batches = self.epoch_partition_batches.setdefault(
            self.active_epoch_idx, OrderedDict()
        )

        # Get or create BatchSet for the current partition
        batch_set = partition_batches.setdefault(
            self.active_partition_idx, BatchSet(self.active_batch_set_id)
        )
         # Insert the batch into the batch set
        batch_set.batches[next_batch.batch_id] = next_batch

        # self._clean_up_old_batches()

        # now lets add the batch to the future batches of all active jobs that are waiting for it
        # for job in self.active_jobs.values():
        #     if batch_set_id == job.active_bacth_set_id:
        #         job.future_batches[next_batch.batch_id] = next_batch
        
        return next_batch
                
    def _clean_up_old_batches(self):
        #clean up old batches that are no longer needed
        for partition_id, partition_batches in self.epoch_partition_batches.items():
            #check if there are more than two batch sets in the partition_batches
            if len(partition_batches) >= 2:
                #check if any job is processing the oldest batch set
                #get all the batch sets in the partition_batches except the last one
                candidate_batch_sets = list(partition_batches.keys())[:-1]
                for bacth_set_id in candidate_batch_sets:
                    not_in_use = True
                    for job in self.active_jobs.values():
                        if bacth_set_id == job.active_bacth_set_id:
                            not_in_use = False
                            break
                    if not_in_use:
                        #remove the entire batch set from the partition_batches
                        partition_batches.pop(bacth_set_id)
                
    def total_active_batches(self):
        count = 0
        for epoch in self.epoch_partition_batches.values():
            for partition_batches in epoch.values():
                count += len(partition_batches.batches)
        return count

    def total_active_partitions(self):
        return sum(len(partition_batches) for partition_batches in self.epoch_partition_batches.values())
    
    def _warm_up_cache(self, max_batches:int = 40):
        warm_up_started = time.perf_counter()
        prefetch_list: TypingOrderedDict[str, Tuple[Batch, str]] = OrderedDict()
        for batch in self.epoch_partition_batches[self.active_epoch_idx][self.active_partition_idx].batches.values():
            payload = {
                'bucket_name': self.dataset.s3_bucket,
                'batch_id': batch.batch_id,
                'batch_samples': self.dataset.get_samples(batch.indicies),
                'cache_address': self.prefetch_service.cache_address,
                'task': 'prefetch',
            }
            prefetch_list[batch.batch_id] = (batch, json.dumps(payload))
            if len(prefetch_list) >= max_batches:
                break
        logger.info(f"Warming up cache with {len(prefetch_list)} batches.")
        self.prefetch_service.prefetch_batches_from_list(prefetch_list, warm_up_started, True)
        logger.info(f"Prefetch took: {time.perf_counter()-warm_up_started:.4f}s for {len(prefetch_list)} batches.")
    
    
    def get_next_batch_for_job(self, job_id: str) -> Optional[Batch]:
        with self.lock:
            if self.prefetch_service is not None and self.prefetch_service.prefetch_stop_event.is_set():  
                self.prefetch_service.start()#prefetcher is stopped, start it

            #if this is a new job, lets add it to the active jobs
            if job_id not in self.jobs:
                logger.info(f"Registering new job '{job_id}' on dataset '{self.dataset.s3_bucket}'")

            current_job = self.jobs.setdefault(job_id, DLTJob(job_id))

            if len(current_job.future_batches) == 0: 
                self.assign_batches_to_job(current_job)

            next_batch = current_job.next_batch()

            if next_batch.cache_status == CacheStatus.NOT_CACHED:
               next_batch.cache_status == CacheStatus.CACHING_IN_PROGRESS #mark the batch as being prefetched by a job to avoid multiple prefetches

            if next_batch.is_first_access:
                next_batch.is_first_access = False
                self._generate_new_batch()
            
            next_batch.set_last_accessed_time()
            self.job_cache_request_counts[job_id] = self.job_cache_request_counts.get(job_id, 0) + 1

            # logger.debug(f"Job '{job_id}' given batch '{next_batch.batch_id}'")
            # samples = self.dataset.get_samples(next_batch.indices)
            return next_batch
        
    def reverse_cycle_mod(self, start: int, mod: int):
        for i in range(mod):
            yield (start - i) % mod

    def assign_batches_to_job(self, job: DLTJob):
        with self.lock:
            # Step 1: Check if job has completed its local epoch
            if len(job.partitions_covered_this_epoch) == self.sampler.num_partitions:
                job.current_epoch_idx += 1
                job.partitions_covered_this_epoch.clear()
            
            # Step 2: Find an available (epoch, partition) pair for an uncovered partition
            for epoch_idx, partition_map in self.epoch_partition_batches.items():
                for partition_idx, batch_set in partition_map.items():
                    # Skip if this pair was already used by the job
                    if (epoch_idx, partition_idx) in job.used_epoch_partition_pairs:
                        continue

                    # Skip if this partition is already covered in this local epoch
                    if partition_idx in job.partitions_covered_this_epoch:
                        continue

                    # Assign this partition to the job for the current local epoch
                    job.partitions_covered_this_epoch.add(partition_idx)
                    job.used_epoch_partition_pairs.add((epoch_idx, partition_idx))
                    job.active_bacth_set_id = batch_set.id
                    for batch in batch_set.batches.values():
                        job.future_batches[batch.batch_id] = batch
        
      

    def update_job_progess(self, 
                           job_id,
                           previous_step_batch_id,
                           previous_step_wait_for_data_time,
                           previous_step_was_cache_hit,
                           previous_step_gpu_time,
                           previous_batch_cached_on_miss):

     with self.lock:
        job = self.active_jobs.setdefault(job_id, DLTJob(job_id))
        parts = previous_step_batch_id.split('_')
        epoch_idx = int(parts[0])
        partition_idx = int(parts[1]) 
        batch_set_id = f'{epoch_idx}_{partition_idx}'
        if previous_step_was_cache_hit or previous_batch_cached_on_miss:
            batch = self.epoch_partition_batches[partition_idx][batch_set_id].batches[previous_step_batch_id]
            batch.set_last_accessed_time()
            batch.set_cache_status(CacheStatus.CACHED)
        # else:
        #     batch.set_cache_status(False)

        job.update_perf_metrics(previous_step_wait_for_data_time, previous_step_was_cache_hit, previous_step_gpu_time)

    
    def log_just_in_time_line(self, line):
        file_name = 'cached_bacth_duration.csv'
        file_exists = os.path.isfile(file_name)
        with open(file_name, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=line.keys())
            if not file_exists:
                writer.writeheader()
                writer.writerow(line)

        
    def handle_job_ended(self, job_id):
        with self.lock:
            if job_id in self.active_jobs:
                job = self.active_jobs[job_id]
                self.active_jobs.pop(job_id)
                logger.info(f"Job '{job_id}' ended after {job.total_lifetime():.4f}s. Current active jobs {len(self.active_jobs)}" )


            if len(self.active_jobs) == 0:
                # logger.info("All jobs have ended. Stopping prefetcher.")
                if self.prefetch_service:
                    self.prefetch_service.stop()
                if self.eviction_service:
                    self.eviction_service.stop_cache_evictor()

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
    print(batch_manager.dataset_info())
    job_id = batch_manager.add_job()
    batch_manager.get_next_batch_for_job(job_id)
    


if __name__ == "__main__":
    main()