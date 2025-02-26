import threading
from sampler import PartitionedBatchSampler
from job import DLTJob
from args import DisDLArgs
from collections import deque, OrderedDict
from typing import List, Optional, Dict, Tuple
from dataset import Dataset
from batch import Batch, BatchSet
import time
from logger_config import logger
import json
from datetime import datetime, timezone
import copy
from itertools import cycle  # Import cycle from itertools
from typing import OrderedDict as TypingOrderedDict
import csv
import os
from cache_eviction import CacheEvictionService
from cache_prefetching import PrefetchService

class CentralBatchManager:
    def __init__(self, dataset: Dataset, args: DisDLArgs):
        self.dataset = dataset
        self.sampler = PartitionedBatchSampler(
            num_files=len(dataset),
            batch_size=args.batch_size,
            num_partitions=args.num_dataset_partitions,
            drop_last=args.drop_last,
            shuffle=args.shuffle)
           
        self.active_epoch_idx = None
        self.active_partition_id = None
        self.evict_from_cache_simulation_time = args.evict_from_cache_simulation_time
        self.lookahead_distance = args.lookahead_steps

        # self.lookahead_steps = min(args.lookahead_steps, self.dataset.partitions[1].num_batches)
        self.active_jobs: Dict[str, DLTJob] = {}
        self.epoch_partition_batches: Dict[int, Dict[int, BatchSet]] = OrderedDict()  #first key is epoch id, second key is partition id, value is the batches

        # Generate initial batches
        for _ in range(self.lookahead_distance):
            self._generate_new_batch()
        
        # Initialize prefetch service
        self.prefetch_service: Optional[PrefetchService] = None
        if args.use_prefetching:
            self.prefetch_service = PrefetchService(
                prefetch_lambda_name=args.prefetch_lambda_name,
                cache_address=args.serverless_cache_address,
                jobs=self.active_jobs,
                dataset=self.dataset,
                cost_threshold_per_hour=args.prefetch_cost_cap_per_hour,
                simulate_time=args.prefetch_simulation_time)
            self._warm_up_cache()
        
        self.eviction_service: Optional[CacheEvictionService] = None
        if args.use_keep_alive:
            #Initialize cache eviction service
            self.eviction_service:CacheEvictionService = CacheEvictionService(
                cache_address=args.serverless_cache_address,
                jobs=self.active_jobs,
                keep_alive_time_threshold=args.cache_keep_alive_timeout,
                simulate_keep_alvive= True if self.evict_from_cache_simulation_time is not None else False
                # simulate_time=args.evict_from_cache_simulation_time
            )     

        self.lock = threading.Lock()  # Lock for thread safety


    def _generate_new_batch(self):
        next_batch:Batch = next(self.sampler)
        
        if self.evict_from_cache_simulation_time:
            next_batch.evict_from_cache_simulation_time = self.evict_from_cache_simulation_time

        self.active_epoch_idx = next_batch.epoch_idx
        self.active_partition_id = next_batch.partition_id

        # Ensure epoch exists, initializing with an OrderedDict for partitions
        partition_batches = self.epoch_partition_batches.setdefault(next_batch.epoch_idx, OrderedDict())
        
         # Ensure partition exists, initializing with a new BatchSet if needed
        partition_batch_set = partition_batches.setdefault(
            next_batch.partition_id, BatchSet(f'{next_batch.epoch_idx}_{next_batch.partition_id}')
        )
        # Store the batch in the BatchSet
        partition_batch_set.batches[next_batch.batch_id] = next_batch

        # now lets add the batch to the future batches of all active jobs that are waiting for it
        for job in self.active_jobs.values():
            if partition_batch_set.id in job.active_batch_set_ids:
                job.future_batches[next_batch.batch_id] = next_batch
    
    def _warm_up_cache(self, max_batches:int = 40):
        warm_up_started = time.perf_counter()
        prefetch_list: TypingOrderedDict[str, Tuple[Batch, str]] = OrderedDict()
        for batch in self.epoch_partition_batches[self.active_epoch_idx][self.active_partition_id].batches.values():
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
                self.prefetch_service.start_prefetcher() #prefetcher is stopped, start it
            
            if self.eviction_service and self.eviction_service.cache_eviction_stop_event.is_set():
                self.eviction_service.start_cache_evictor() #cache evictor is stopped, start it

            #if this is a new job, lets add it to the active jobs
            if job_id not in self.active_jobs:
                logger.info(f"Registering new job '{job_id}' on dataset '{self.dataset.s3_bucket}'")
            
            current_job = self.active_jobs.setdefault(job_id, DLTJob(job_id))

            if not current_job.future_batches or len(current_job.future_batches) < self.lookahead_distance:
                # if not job.future_batches or len(job.future_batches) == 0: #reached end of partition so start preparing for next one
                self.allocate_batches_to_job(current_job)

            next_batch = current_job.next_batch()

            if not next_batch.is_cached and not next_batch.caching_in_progress:
                next_batch.set_caching_in_progress(True) #mark the batch as being prefetched by a job to avoid multiple prefetches

            if next_batch.is_first_access:
                next_batch.is_first_access = False
                self._generate_new_batch()

            logger.debug(f"Job '{job_id}' given batch '{next_batch.batch_id}' from partition '{next_batch.partition_id}' in epoch '{next_batch.epoch_idx}'")
            return next_batch

    def allocate_batches_to_job(self, job: DLTJob):
        is_new_job: bool = False
        if job.partition_id_cycle is None: #new job, lets start cycling partitons at the currently active partition
            partition_ids = list(self.dataset.partitions.keys())
            start_index = partition_ids.index(self.active_partition_id)
            reordered_ids = partition_ids[start_index:] + partition_ids[:start_index]
            job.partition_id_cycle = cycle(reordered_ids)
            job.started_partition_index = copy.deepcopy(self.active_partition_id)
            is_new_job = True

        next_partition_id = next(job.partition_id_cycle)
        if next_partition_id == job.started_partition_index:
            job.epochs_completed_count += 1
        
        for epoch_id in reversed(self.epoch_partition_batches.keys()):
                if next_partition_id in self.epoch_partition_batches[epoch_id]:
                    batch_set = self.epoch_partition_batches[epoch_id][next_partition_id]
                    if batch_set not in job.active_batch_set_ids:
                        job.active_batch_set_ids.add(batch_set.id)
                        for batch_id, batch in reversed(list(batch_set.batches.items())):
                            job.future_batches[batch_id] = batch  # Add the batch
                            if is_new_job:
                                job.future_batches.move_to_end(batch_id, last=False)
                        
                        if len(job.future_batches) >= self.look_ahead:
                            break
                    if len(job.future_batches) >= self.look_ahead:
                            break
        pass
   
    def update_job_progess(self, job_id,
                           previous_step_batch_id,
                           previous_step_wait_for_data_time,
                           previous_step_is_cache_hit,
                           previous_step_gpu_time,
                           previous_batch_cached_on_miss):

     with self.lock:
        parts = previous_step_batch_id.split('_')
        epoch_id = int(parts[0])
        partition_id = int(parts[1])
        batch = self.epoch_partition_batches[epoch_id][partition_id].batches[previous_step_batch_id]
        if previous_step_is_cache_hit or previous_batch_cached_on_miss:
            batch.set_last_accessed_time()
            batch.set_cache_status(True)
        # else:
        #     batch.set_cache_status(False)

        self.active_jobs[job_id].update_perf_metrics(previous_step_wait_for_data_time, previous_step_is_cache_hit, previous_step_gpu_time)

    
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
                logger.info(f"Job '{job_id}' ended. Total time: {job.total_training_time():.4f}s, Steps: {job.total_training_steps()}, Hits: {job.training_step_times_on_hit.count}, Misses: {job.training_step_times_on_miss.count}, Rate: {job.training_step_times_on_hit.count / job.total_training_steps():.4f}" )
                self.active_jobs.pop(job_id)

            if len(self.active_jobs) == 0:
                # logger.info("All jobs have ended. Stopping prefetcher.")
                if self.prefetch_service:
                    self.prefetch_service.stop_prefetcher()
                if self.eviction_service:
                    self.eviction_service.stop_cache_evictor()
    
if __name__ == "__main__":
    pass
    # # Constants
    # MISS_WAIT_FOR_DATA_TIME = 1.4
    # HIT_WAIT_FOR_DATA_TIME = 0.001
    # PREFETCH_TIME = 3
    # NUM_JOBS = 1 # Number of parallel jobs to simulate
    # DELAY_BETWEEN_JOBS = 0.1  # Delay in seconds between the start of each job
    # BATCHES_PER_JOB = 2346  # Number of batches each job will process
    # GPU_TIME = 0.01
    # super_args:SUPERArgs = SUPERArgs(
    #         batch_size = 128,
    #         partitions_per_dataset = 1,
    #         lookahead_steps = 1000,
    #         serverless_cache_address = '',
    #         use_prefetching = True,
    #         use_keep_alive = False,
    #         prefetch_lambda_name = 'CreateVisionTrainingBatch',
    #         prefetch_cost_cap_per_hour=None,
    #         cache_evition_ttl_threshold = 1000,
    #         prefetch_simulation_time = PREFETCH_TIME,
    #         evict_from_cache_simulation_time = None,
    #         shuffle = False,
    #         drop_last = False,
    #         workload_kind = 'vision')

    # dataset = Dataset(data_dir='s3://sdl-cifar10/test/', batch_size=super_args.batch_size, drop_last=super_args.drop_last, num_partitions=super_args.partitions_per_dataset)
    # batch_manager = CentralBatchManager(dataset=dataset, args=super_args)
    
    # job_id = '1'
    # cache_hits = 0
    # cache_misses = 0
    # previous_step_total_time = 0
    # previous_step_is_cache_hit = False
    # cached_previous_batch = False
    # BATCHES_PER_JOB = 1173  # Number of batches each job will process

    # end = time.perf_counter()

    # for i in range(BATCHES_PER_JOB):
    #     batch:Batch = batch_manager.get_next_batch(job_id=job_id)

    #     if batch.is_cached:
    #         previous_step_wait_for_data_time = HIT_WAIT_FOR_DATA_TIME
    #         previous_step_is_cache_hit = True
    #         cache_hits += 1
    #         time.sleep(previous_step_wait_for_data_time + GPU_TIME)
    #         cached_missed_batch = False
    #     else:
    #         previous_step_wait_for_data_time = MISS_WAIT_FOR_DATA_TIME
    #         previous_step_is_cache_hit = False
    #         cache_misses += 1
    #         cached_missed_batch = False
    #         time.sleep(previous_step_wait_for_data_time + GPU_TIME)
        
    #     batch_manager.update_job_progess(job_id, batch.batch_id, previous_step_wait_for_data_time, previous_step_is_cache_hit, GPU_TIME, cached_missed_batch)
    #     hit_rate = cache_hits / (i + 1) if (i + 1) > 0 else 0
    #     if i % 100== 0 or not previous_step_is_cache_hit:
    #         logger.info(f'Setp {i+1}, Job {job_id}, {batch.batch_id}, Hits: {cache_hits}, Misses: {cache_misses}, Rate: {hit_rate:.2f}')
  
    # batch_manager.job_ended(job_id)

    # time.sleep(5)