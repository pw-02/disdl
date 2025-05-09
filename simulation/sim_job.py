import heapq
import logging
from typing import List, Sized, Tuple, Dict, Set, Any
import sys
# sys.path.append(".")
sys.path.append("disdl\server")
from sim_workloads import workloads
from typing import List, Optional, Dict, Tuple
# from disdl.server.batch_manager import BatchManager
# from job import DLTJob
from disdl.server.logger_config import configure_simulation_logger
from collections import OrderedDict
from disdl.server.batch import Batch, CacheStatus, BatchSet
from disdl.server.batch_manager import BatchManager
from disdl.server.utils import AverageMeter
from disdl.server.dataset import S3DatasetBase
from disdl.server.sampler import PartitionedBatchSampler
import threading
import numpy as np
from sortedcontainers import SortedList
import time

class DLTJob:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.current_epoch = 0
        # For reuse logic
        # self.used_epoch_partition_pairs: Set[Tuple[int, int]] = set()
        self.used_batch_set_ids:  Dict[int, None] = {}
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
        self.reset_for_new_epoch()
        self.local_cache = {} #used by tensorocket producer to store batches
        
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