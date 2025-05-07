
from collections import defaultdict, deque, OrderedDict
from typing import Dict, List, Optional, Set, Tuple
from utils import AverageMeter
from batch import Batch, CacheStatus
import threading

# from sortedcontainers import SortedList

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
# class DLTJob:
#     def __init__(self, job_id: str):
#         self.job_id = job_id
#         self.num_partitions = 1
#         self.current_epoch_idx = 0
#         # Tracks which (global epoch, partition) pairs have been used by the job
#         self.used_epoch_partition_pairs: Set[Tuple[int, int]] = set()
#         # For current job-local epoch, track which partitions have been covered
#         self.partitions_covered_this_epoch: Set[int] = set()



#         # Keeps track of which epochs the job has completed
#         self.epoch_history:Set[int] = set()
#          # The epoch currently assigned to this job (None until assigned)
#         self.current_epoch: Optional[int] = None
#          # Tracks which batch_ids this job has processed in the current epoch
#         self.seen_batches: Set[str] = set()

#         self.epochs_completed_count = -1
#         self.active_partition_idx = None
#         self.active_bacth_set_id = None
#         self.total_steps = 0
#         self.job_registered_time = time.perf_counter()
#         self.future_batches: OrderedDict[str, Batch] = OrderedDict()
#         self.time_waiting_on_data = AverageMeter('Time Waiting on Data')
#         # self.training_step_times_on_hit = AverageMeter('Training Step Time on Hit')
#         # self.training_step_times_on_miss =  AverageMeter('Training Step Time on Miss')
#         self.training_step_gpu_times =  AverageMeter('training_step_gpu_times')
#         # self.dataload_time_on_miss  = AverageMeter('Dataload Time on Miss')
#         # self.dataload_time_on_hit = AverageMeter('Dataload Time on Hit')
#         self.dataload_time = AverageMeter('Dataload Time')    

#         self.lock = threading.Lock()

#     def assign_epoch(self, epoch_idx: int):
#         """Assigns a new epoch to this job and clears current progress."""
#         self.current_epoch = epoch_idx
#         self.seen_batches.clear()
#         self.epoch_history.add(epoch_idx)
    
#     def has_seen_epoch(self, epoch_idx: int) -> bool:
#         return epoch_idx in self.epoch_history
    
#     def has_seen_batch(self, batch_id: str) -> bool:
#         return batch_id in self.seen_batches
    
#     def mark_batch_seen(self, batch_id: str):
#         self.seen_batches.add(batch_id)
    
#     def epoch_complete(self, total_batches: int) -> bool:
#         return len(self.seen_batches) >= total_batches






#     def compute_required_prefetch_concurrency(self, lambda_processing_time, buffer = 0):
#         gpu_batch_rate = 1 / self.training_step_gpu_times.avg
#         job_load_rate = 1 / self.dataload_time.avg

#         required_prefetch_rate = gpu_batch_rate - job_load_rate
#         if required_prefetch_rate <= 0:
#             return 0 # No additional concurrency needed, no delay
#         concurrnecy = math.ceil(required_prefetch_rate * lambda_processing_time) + buffer
#         return concurrnecy

#     def get_total_batches_assigned_to_job(self):
#         return len(self.future_batches)

#     def __repr__(self):
#         return (f"Job(job_id={self.job_id}, current_epoch={self.active_epoch}, "
#                 f"current_index={self.total_steps})")
    
#     def get_data_loading_delay(self):
#         return self.time_waiting_on_data.avg
    
#     def get_gpu_batch_rate(self):
#         #batches per second
#         return 1 / self.training_step_gpu_times.avg
    
#     def get_required_prefetching_rate(self, prefetching_cost_per_hour:float):
#         if self.training_step_gpu_times.count < 2: #ignore first two steps for GPU warm up
#                 return 0
#         pass


#     def total_lifetime(self):
#         return time.perf_counter() - self.job_registered_time
    
#     def total_training_steps (self):
#         return self.total_steps
    
#     def update_perf_metrics(self, 
#                             previous_step_wait_for_data_time:float, 
#                             previous_step_is_cache_hit:float, 
#                             previous_step_gpu_time:float):
#          with self.lock:
#             self.total_steps += 1
#             if self.total_steps > 1: #skip the first step for recording gpu times
#                 self.training_step_gpu_times.update(previous_step_gpu_time)
#                 # if previous_step_is_cache_hit:
#                 #     self.dataload_time_on_hit.update(previous_step_wait_for_data_time)
#                 #     self.training_step_times_on_hit.update(previous_step_wait_for_data_time + previous_step_gpu_time)
#                 # else:
#                 #     self.dataload_time_on_miss.update(previous_step_wait_for_data_time)
#                 #     self.training_step_times_on_miss.update(previous_step_wait_for_data_time + previous_step_gpu_time)

#     def next_batch(self):
#         with self.lock:
#             next_training_batch = None
#             first_available_batch_id  = None  # First batch that is not already being processed

#             for batch_id, batch in list(self.future_batches.items()):
#                 if batch.cache_status == CacheStatus.CACHED or batch.cache_status == CacheStatus.CACHING_IN_PROGRESS: 
#                     next_training_batch = self.future_batches.pop(batch_id)  # Cached batch found
#                     break
#                 elif not first_available_batch_id:
#                     first_available_batch_id = batch_id

#             if not next_training_batch and first_available_batch_id:
#                 next_training_batch = self.future_batches.pop(first_available_batch_id)
#             # self.current_batch = next_training_batch
#             return next_training_batch


    # def next_batch(self):
    #     with self.lock:
    #         next_training_batch = None
    #         first_available_batch_id  = None  # First batch that is not already being processed

    #         for batch_id, batch in list(self.future_batches.items()):
    #             if batch.cache_status == CacheStatus.CACHED: 
    #                 next_training_batch = self.future_batches.pop(batch_id)  # Cached batch found
    #                 break
    #             elif not batch.cache_status == CacheStatus.CACHING_IN_PROGRESS and not first_available_batch_id:
    #                 first_available_batch_id = batch_id
            
    #         if not next_training_batch and first_available_batch_id:
    #             next_training_batch = self.future_batches.pop(first_available_batch_id)

    #         if not next_training_batch:
    #             # logger.debug("No cached or in-progress batch found. Returning the next batch")
    #             next_training_batch = self.future_batches.pop(next(iter(self.future_batches)))
            
    #         # self.current_batch = next_training_batch
    #         return next_training_batch

    
   