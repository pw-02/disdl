
from collections import deque, OrderedDict
import time
from utils import AverageMeter
from typing import Iterator, Optional
from batch import Batch
from dataset import Dataset
import threading
from logger_config import logger



class DLTJob:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.partition_id_cycle: Optional[Iterator[int]] = None
        self.epochs_completed_count = -1
        self.started_partition_index = None
        self.active_batch_set_ids = set()
        # self.active_batch_set_id = None

        # self.active_epoch = initial_epoch
        self.total_steps = 0
        self.future_batches: OrderedDict[str, Batch] = OrderedDict()
        self.time_waiting_on_data = AverageMeter('Time Waiting on Data')
        # self.cache_hit_window = deque(maxlen=50)  # Sliding window for recent hits
        self.training_step_times_on_hit = AverageMeter('Training Step Time on Hit')
        self.training_step_times_on_miss =  AverageMeter('Training Step Time on Miss')
        self.training_step_gpu_times =  AverageMeter('training_step_gpu_times')
        self.dataload_time_on_miss  = AverageMeter('Dataload Time on Miss')
        self.dataload_time_on_hit = AverageMeter('Dataload Time on Hit')    
        self.current_batch:Batch = None
        self.cycle_bacthes = []
        self.lock = threading.Lock()
        self.step_idx = None

        # self.future_batches: OrderedDict[int,OrderedDict[str, Batch]] = OrderedDict()
    
    def get_total_batches_assigned_to_job(self):
        return len(self.future_batches)

    def __repr__(self):
        return (f"Job(job_id={self.job_id}, current_epoch={self.active_epoch}, "
                f"current_index={self.total_steps})")
    
    def total_training_time(self):
        return self.training_step_times_on_hit.sum + self.training_step_times_on_miss.sum
    
    def total_training_steps (self):
        return self.total_steps
    
    def update_perf_metrics(self, previous_step_wait_for_data_time:float, previous_step_is_cache_hit:float, previous_step_gpu_time:float):
        with self.lock:
            self.total_steps += 1
            if self.total_steps > 1: #skip the first step for recording gpu times
                self.training_step_gpu_times.update(previous_step_gpu_time)
                if previous_step_is_cache_hit:
                    self.dataload_time_on_hit.update(previous_step_wait_for_data_time)
                    self.training_step_times_on_hit.update(previous_step_wait_for_data_time + previous_step_gpu_time)
                else:
                    self.dataload_time_on_miss.update(previous_step_wait_for_data_time)
                    self.training_step_times_on_miss.update(previous_step_wait_for_data_time + previous_step_gpu_time)

   
    def next_batch(self):
        with self.lock:
            next_training_batch = None
            not_in_progress_batch = None  # To hold the first batch found that's in progress

            active_batch_set_id = next(iter(self.future_batches.items()))[1].batch_partition_id
            if active_batch_set_id not in self.active_batch_set_ids or active_batch_set_id != '1_1':
                pass

           # First pass: Find the first cached batch
            for batch_id, batch in list(self.future_batches.items()):
                if batch.batch_partition_id == active_batch_set_id:
                    if batch.is_cached:
                        next_training_batch = self.future_batches.pop(batch_id)  # Cached batch found
                        break
                    elif  not batch.caching_in_progress:
                        # Store the first available batch in progress if no cached batch is found
                        if not not_in_progress_batch:
                            not_in_progress_batch = batch_id
            # Second pass: If no cached batch was found, use the first in-progress batch
            if not next_training_batch and not_in_progress_batch:
                next_training_batch = self.future_batches.pop(not_in_progress_batch)
            
            if not next_training_batch:
                logger.debug("No cached or in-progress batch found. Returning the first available batch")
                next_training_batch = self.future_batches.pop(next(iter(self.future_batches)))
                
            self.current_batch = next_training_batch
            return next_training_batch

        

         
    
    # def next_training_step_batch(
    #     self,
    #     dataset: Dataset,
    #     prefetch_service: PrefetchManager = None,
    #     keep_alive_service: CacheEvicitonService = None,
    #     prefetch_delay: float = 0,
    # ) -> Batch:
    #     self.total_steps += 1

    #     # Prefetching Logic
    #     if self.total_steps >= 1 and prefetch_service and not self.cycle_bacthes:
    #         prefetch_list = []
    #         prefetch_cycle_duration = prefetch_service.prefetch_execution_times.avg + prefetch_delay

    #         training_step_time_on_hit = (
    #             self.training_step_gpu_times.avg
    #             if self.training_step_times_on_hit.count == 0
    #             else self.training_step_times_on_hit.avg
    #         )
    #         training_step_time_on_miss = (
    #             prefetch_cycle_duration
    #             if self.training_step_times_on_miss.count == 0
    #             else self.training_step_times_on_miss.avg
    #         )
    #         optimal_prefetch_concurrency = find_optimal_prefetch_conncurrency(
    #             prefetch_cycle_duration, training_step_time_on_hit
    #         )

    #         # Prefetch and Cache Management
    #         prefetch_counter, time_counter = 0, 0
    #         for batch in self.future_batches.values():
    #             if time_counter <= prefetch_cycle_duration:
    #                 self.cycle_bacthes.append(batch.batch_id)
    #                 time_counter += (
    #                     training_step_time_on_hit if batch.is_cached else training_step_time_on_miss
    #                 )
                
    #             elif prefetch_counter < optimal_prefetch_concurrency:
    #                 prefetch_counter += 1
    #                 if not batch.is_cached and not batch.caching_in_progress:
    #                     batch.set_caching_in_progress(True)
    #                     payload = {
    #                         'bucket_name': dataset.bucket_name,
    #                         'batch_id': batch.batch_id,
    #                         'batch_samples': dataset.get_samples(batch.indicies),
    #                         'cache_address': prefetch_service.cache_address,
    #                         'task': 'prefetch',
    #                     }
    #                     prefetch_list.append((batch, payload))
    #             else:
    #                 break

    #         if len(prefetch_list) > 0:
    #             prefetch_service.prefetch_queue.put((time.perf_counter(), prefetch_list))

    #     # Keep-Alive Logic
    #     if keep_alive_service:
    #         keep_alive_list = [
    #             batch for batch in self.future_batches.values()
    #             if batch.is_cached and batch.time_since_last_access() > 1000
    #         ]
    #         if keep_alive_list:
    #             keep_alive_service.keep_alive_queue.put((time.perf_counter(), keep_alive_list))

    #     # Find the next training batch to process
    #     next_training_batch = None
    #     for batch_id, batch in list(self.future_batches.items()):
    #         if batch.is_cached or not batch.caching_in_progress:
    #             next_training_batch = self.future_batches.pop(batch_id)
    #             break

    #     # If no suitable batch found, get the first available one
    #     if not next_training_batch:
    #         next_training_batch = self.future_batches.pop(next(iter(self.future_batches)))

    #     # Update cycle batches
    #     if next_training_batch.batch_id in self.cycle_bacthes:
    #         self.cycle_bacthes.remove(next_training_batch.batch_id)

    #     return next_training_batch