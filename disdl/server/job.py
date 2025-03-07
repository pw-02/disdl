
from collections import deque, OrderedDict
from utils import AverageMeter
from batch import Batch, CacheStatus
import threading
from logger_config import logger
import math
import time
class DLTJob:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.epochs_completed_count = -1
        self.partitions_remaining_in_current_epoch = []
        self.active_partition_idx = None
        self.active_bacth_set_id = None
        self.total_steps = 0
        self.job_registered_time = time.perf_counter()
        self.future_batches: OrderedDict[str, Batch] = OrderedDict()
        self.time_waiting_on_data = AverageMeter('Time Waiting on Data')
        # self.training_step_times_on_hit = AverageMeter('Training Step Time on Hit')
        # self.training_step_times_on_miss =  AverageMeter('Training Step Time on Miss')
        self.training_step_gpu_times =  AverageMeter('training_step_gpu_times')
        # self.dataload_time_on_miss  = AverageMeter('Dataload Time on Miss')
        # self.dataload_time_on_hit = AverageMeter('Dataload Time on Hit')
        self.dataload_time = AverageMeter('Dataload Time')    
    
        self.lock = threading.Lock()

    def compute_required_prefetch_concurrency(self, lambda_processing_time, buffer = 0):
        gpu_batch_rate = 1 / self.training_step_gpu_times.avg
        job_load_rate = 1 / self.dataload_time.avg

        required_prefetch_rate = gpu_batch_rate - job_load_rate
        if required_prefetch_rate <= 0:
            return 0 # No additional concurrency needed, no delay
        concurrnecy = math.ceil(required_prefetch_rate * lambda_processing_time) + buffer
        return concurrnecy

    def get_total_batches_assigned_to_job(self):
        return len(self.future_batches)

    def __repr__(self):
        return (f"Job(job_id={self.job_id}, current_epoch={self.active_epoch}, "
                f"current_index={self.total_steps})")
    
    def get_data_loading_delay(self):
        return self.time_waiting_on_data.avg
    
    def get_gpu_batch_rate(self):
        #batches per second
        return 1 / self.training_step_gpu_times.avg
    
    def get_required_prefetching_rate(self, prefetching_cost_per_hour:float):
        if self.training_step_gpu_times.count < 2: #ignore first two steps for GPU warm up
                return 0
        pass


    def total_lifetime(self):
        return time.perf_counter() - self.job_registered_time
    
    def total_training_steps (self):
        return self.total_steps
    
    def update_perf_metrics(self, 
                            previous_step_wait_for_data_time:float, 
                            previous_step_is_cache_hit:float, 
                            previous_step_gpu_time:float):
         with self.lock:
            self.total_steps += 1
            if self.total_steps > 1: #skip the first step for recording gpu times
                self.training_step_gpu_times.update(previous_step_gpu_time)
                # if previous_step_is_cache_hit:
                #     self.dataload_time_on_hit.update(previous_step_wait_for_data_time)
                #     self.training_step_times_on_hit.update(previous_step_wait_for_data_time + previous_step_gpu_time)
                # else:
                #     self.dataload_time_on_miss.update(previous_step_wait_for_data_time)
                #     self.training_step_times_on_miss.update(previous_step_wait_for_data_time + previous_step_gpu_time)

   
    def next_batch(self):
        with self.lock:
            next_training_batch = None
            first_available_batch_id  = None  # First batch that is not already being processed

            for batch_id, batch in list(self.future_batches.items()):
                if batch.cache_status == CacheStatus.CACHED: 
                    next_training_batch = self.future_batches.pop(batch_id)  # Cached batch found
                    break
                elif not batch.cache_status == CacheStatus.CACHING_IN_PROGRESS and not first_available_batch_id:
                    first_available_batch_id = batch_id
            
            if not next_training_batch and first_available_batch_id:
                next_training_batch = self.future_batches.pop(first_available_batch_id)

            if not next_training_batch:
                # logger.debug("No cached or in-progress batch found. Returning the next batch")
                next_training_batch = self.future_batches.pop(next(iter(self.future_batches)))
            
            # self.current_batch = next_training_batch
            return next_training_batch

    
   