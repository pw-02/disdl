import threading
from sampler import PartitionedBatchSampler
from job import DLTJob
from args import DisDLArgs
from collections import deque, OrderedDict
from typing import List, Optional, Dict, Tuple
from dataset import MSCOCODataset, LibSpeechDataset
from batch import Batch, BatchSet, CacheStatus
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
from cache_prefetching import PrefetchServiceAsync, PrefetchServiceEvent

class CentralBatchManager:
    def __init__(self, dataset, args: DisDLArgs, prefetch_workers:int = 10):
        self.dataset = dataset
        self.job_counter = 0
        self.sampler = PartitionedBatchSampler(
            num_files=len(dataset),
            batch_size=args.batch_size,
            num_partitions=args.num_dataset_partitions,
            drop_last=args.drop_last,
            shuffle=args.shuffle)
        self.active_epoch_idx = None
        self.active_partition_idx = None
        self.evict_from_cache_simulation_time = args.evict_from_cache_simulation_time

        print(f"Number of batches per partition: {self.sampler.calc_num_batchs_per_partition()}")
        # self.lookahead_distance = min((self.sampler.calc_num_batchs_per_partition() -1),args.lookahead_steps)
        self.lookahead_distance = args.lookahead_steps

        # self.lookahead_steps = min(args.lookahead_steps, self.dataset.partitions[1].num_batches)
        self.active_jobs: Dict[str, DLTJob] = {}
        # self.epoch_partition_batches: Dict[int, Dict[int, BatchSet]] = OrderedDict()  #first key is epoch id, second key is partition id, value is the batches

        self.epoch_partition_batches: Dict[int, Dict[str, BatchSet]] = OrderedDict()  #first partition id, value list of bacthes for that partition

        # Initialize prefetch service
        self.prefetch_service: Optional[PrefetchServiceAsync] = None
        if args.use_prefetching:
            self.prefetch_service = PrefetchServiceAsync(
                lambda_name=args.prefetch_lambda_name,
                cache_address=args.serverless_cache_address,
                simulate_time=args.prefetch_simulation_time,
                num_workers=prefetch_workers)
            self.prefetch_service.start()
            # self._warm_up_cache()
        
        self.eviction_service: Optional[CacheEvictionService] = None
        if args.use_keep_alive:
            #Initialize cache eviction service
            self.eviction_service:CacheEvictionService = CacheEvictionService(
                cache_address=args.serverless_cache_address,
                jobs=self.active_jobs,
                keep_alive_time_threshold=args.cache_keep_alive_timeout,
                simulate_keep_alvive=True
            )     

        self.lock = threading.Lock()  # Lock for thread safety
        self.job_cache_request_counts = {}
        for _ in range(self.lookahead_distance):
            self._generate_new_batch()

        #wait a few seconds for the cache to be warmed up
        time.sleep(5)

        # #wait for cache to be warmed up
        # if self.prefetch_service:
        #     while self.prefetch_service.queue.qsize() > 0:
        #         time.sleep(1)


    def add_job(self):
        #generate a job id including the dataset_name and the curren_job_count + 1
        self.job_counter += 1
        # job_id = f'{self.dataset.dataset_location}_{self.job_counter}'
        job_id = str(self.job_counter)
        with self.lock:
            self.active_jobs[job_id] = DLTJob(job_id)
            return job_id
        
    def dataset_info(self):
        return {
            'dataset_location': self.dataset.dataset_location,
            'num_samples': len(self.dataset),
            'num_batches': self.sampler.batches_per_epoch,
            'num_partitions': self.sampler.num_partitions,
        }
    
    def cacl_lamda_invocation_counts(self):
        prefetch_counts = 0
        warm_counts = 0
        get_set_counts = 0
        
        if self.prefetch_service:
            prefetch_counts = self.prefetch_service.lambda_invocations_count
            get_set_counts = self.prefetch_service.lambda_invocations_count

        if self.eviction_service:
            eviction_counts = self.eviction_service.lambda_invocations_count
        
        get_set_counts = sum(self.job_cache_request_counts.values())

        total_counts = prefetch_counts + warm_counts + get_set_counts
        return {
            'num_prefetches': prefetch_counts,
            'num_warmup_requests': warm_counts,
            'num_getset_requests': get_set_counts,
            'total': total_counts
        }
 
    def _generate_new_batch(self):
        next_batch:Batch = next(self.sampler)

        if self.evict_from_cache_simulation_time:
            next_batch.evict_from_cache_simulation_time = self.evict_from_cache_simulation_time

        self.active_epoch_idx = next_batch.epoch_idx
        self.active_partition_idx = next_batch.partition_idx
        # Ensure epoch exists, initializing with an OrderedDict for partitions
        partition_batches = self.epoch_partition_batches.setdefault(next_batch.partition_idx, OrderedDict())
         # Ensure partition exists, initializing with a new BatchSet if needed
        batch_set_id = f'{next_batch.epoch_idx}_{next_batch.partition_idx}'
        if batch_set_id in partition_batches:
            partition_batches[batch_set_id].batches[next_batch.batch_id] = next_batch
        else:
            partition_batches[batch_set_id] = BatchSet(batch_set_id)
            partition_batches[batch_set_id].batches[next_batch.batch_id] = next_batch
        
        if self.prefetch_service:
            payload = {
                'bucket_name': self.dataset.s3_bucket,
                'batch_id': next_batch.batch_id,
                'batch_samples': self.dataset.get_samples(next_batch.indices),
                'cache_address': self.prefetch_service.cache_address,
                'task': 'prefetch',
            }
            self.prefetch_service.enqueue_batch(next_batch, json.dumps(payload))
            #print queue size

        self._clean_up_old_batches()

        # now lets add the batch to the future batches of all active jobs that are waiting for it
        for job in self.active_jobs.values():
            if batch_set_id == job.active_bacth_set_id:
                job.future_batches[next_batch.batch_id] = next_batch
        
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
            
            if self.eviction_service is not None and self.eviction_service.cache_eviction_stop_event.is_set():
                self.eviction_service.start_cache_evictor() #cache evictor is stopped, start it

            #if this is a new job, lets add it to the active jobs
            if job_id not in self.active_jobs:
                logger.info(f"Registering new job '{job_id}' on dataset '{self.dataset.s3_bucket}'")
            
            current_job = self.active_jobs.setdefault(job_id, DLTJob(job_id))

            # if not current_job.future_batches or len(current_job.future_batches) < self.lookahead_distance:
            if len(current_job.future_batches) == 0: #reached end of partition so start preparing for next one
                self.allocate_batches_to_job(current_job)

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

    def allocate_batches_to_job(self, job: DLTJob):
        if not job.partitions_remaining_in_current_epoch:
            #start a new epoch at the current and reset the partitions
            job.epochs_completed_count += 1
            job.partitions_remaining_in_current_epoch = list(range(self.sampler.num_partitions))
            # self._clean_up_old_batches()

        #get the next partition for the job by popping the first one
        job.active_partition_idx = job.partitions_remaining_in_current_epoch.pop(0)
        #find the most recent bacth set for the active partition in self.epoch_partition_batche
        batch_set = self.epoch_partition_batches[job.active_partition_idx][next(reversed(self.epoch_partition_batches[job.active_partition_idx]))]
        job.active_bacth_set_id = batch_set.id
        for batch in batch_set.batches.values():
            job.future_batches[batch.batch_id] = batch

        # logger.info(f"Job '{job.job_id}' assigned batch set '{job.active_bacth_set_id}'")
        logger.info(f"Job '{job.job_id}' assigned partion '{job.active_partition_idx}' epoch '{self.active_epoch_idx}'")
                    

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


if __name__ == "__main__":
    # Constants    
    PREFETCH_TIME = 0.1

    CACHE_MISS_DELAY = 0.1
    CACHE_HIT_DELAY = 0.1
    DELAY_BETWEEN_JOBS = 1  # Delay in seconds between the start of each job
    BATCHES_PER_JOB = 20  # Number of batches each job will process
    JOB_SPEEDS = [0.1]

    # Initialize dataset and batch manager
    args = DisDLArgs(
        batch_size=10,
        num_dataset_partitions=1,
        lookahead_steps=20,
        shuffle=False,
        drop_last=False,
        workload='speech',
        serverless_cache_address=None,
        use_prefetching=False,
        use_keep_alive=True,
        prefetch_lambda_name='CreateVisionTrainingBatch',
        prefetch_cost_cap_per_hour=None,
        cache_keep_alive_timeout=60,  # 3 minutes
        prefetch_simulation_time=PREFETCH_TIME,
        evict_from_cache_simulation_time=None
    )

    # dataset = MSCOCODataset(dataset_location='s3://coco-dataset/coco_train.json')
    # batch_manager = CentralBatchManager(dataset=dataset, args=args, prefetch_workers=10)

    dataset = LibSpeechDataset(dataset_location="s3://disdlspeech/test-clean")
    batch_manager = CentralBatchManager(dataset=dataset, args=args, prefetch_workers=10)

    def run_job(job_id, job_speed):
        """Function to simulate a job processing batches."""
        cache_hits = 0
        cache_misses = 0

        for i in range(BATCHES_PER_JOB):
            batch = batch_manager.get_next_batch_for_job(job_id=job_id)

            if batch.cache_status == CacheStatus.CACHED:
                time.sleep(CACHE_HIT_DELAY + job_speed)
                cache_hits += 1
            else:
                time.sleep(CACHE_MISS_DELAY + job_speed)
                cache_misses += 1

            batch_manager.update_job_progess(
                job_id, batch.batch_id, CACHE_MISS_DELAY, batch.cache_status == CacheStatus.CACHED, job_speed, True
            )

            hit_rate = cache_hits / (i + 1)
            if i % 1 == 0 or batch.cache_status != CacheStatus.CACHED:
                logger.info(f'Step {i+1}, Job {job_id}, {batch.batch_id}, Hits: {cache_hits}, Misses: {cache_misses}, Rate: {hit_rate:.2f}')

        batch_manager.handle_job_ended(job_id)


    if __name__ == "__main__":
        from dataset import ImageNetDataset
        run_job(1,1)

    
# if __name__ == "__main__":
#     # Constants    
#     PREFETCH_TIME = 0.1

#     CACHE_MISS_DELAY = 0.1
#     CACHE_HIT_DELAY = 0.1
#     DELAY_BETWEEN_JOBS = 1  # Delay in seconds between the start of each job
#     BATCHES_PER_JOB = 20  # Number of batches each job will process
#     JOB_SPEEDS = [0.1]

#     # Initialize dataset and batch manager
#     args = DisDLArgs(
#         batch_size=10,
#         num_dataset_partitions=1,
#         lookahead_steps=20,
#         shuffle=False,
#         drop_last=False,
#         workload_kind='vision',
#         serverless_cache_address=None,
#         use_prefetching=False,
#         use_keep_alive=True,
#         prefetch_lambda_name='CreateVisionTrainingBatch',
#         prefetch_cost_cap_per_hour=None,
#         cache_keep_alive_timeout=60,  # 3 minutes
#         prefetch_simulation_time=PREFETCH_TIME,
#         evict_from_cache_simulation_time=60
#     )

#     dataset = ImageNetDataset(dataset_location='s3://imagenet-dataset/train/')
#     batch_manager = CentralBatchManager(dataset=dataset, args=args, prefetch_workers=0)

#     def run_job(job_id, job_speed):
#         """Function to simulate a job processing batches."""
#         cache_hits = 0
#         cache_misses = 0

#         for i in range(BATCHES_PER_JOB):
#             batch = batch_manager.get_next_batch_for_job(job_id=job_id)

#             if batch.cache_status == CacheStatus.CACHED:
#                 time.sleep(CACHE_HIT_DELAY + job_speed)
#                 cache_hits += 1
#             else:
#                 time.sleep(CACHE_MISS_DELAY + job_speed)
#                 cache_misses += 1

#             batch_manager.update_job_progess(
#                 job_id, batch.batch_id, CACHE_MISS_DELAY, batch.cache_status == CacheStatus.CACHED, job_speed, True
#             )

#             hit_rate = cache_hits / (i + 1)
#             if i % 1 == 0 or batch.cache_status != CacheStatus.CACHED:
#                 logger.info(f'Step {i+1}, Job {job_id}, {batch.batch_id}, Hits: {cache_hits}, Misses: {cache_misses}, Rate: {hit_rate:.2f}')

#         batch_manager.handle_job_ended(job_id)


#     if __name__ == "__main__":
#         threads = []

#         for job_id, speed in enumerate(JOB_SPEEDS):
#             thread = threading.Thread(target=run_job, args=(job_id, speed,))
#             threads.append(thread)
#             thread.start()
#             time.sleep(DELAY_BETWEEN_JOBS)  # Stagger job start times

#         for thread in threads:
#             thread.join()  # Wait for all jobs to complete