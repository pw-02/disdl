
# batch_manager/manager.py
import threading
from collections import OrderedDict
from typing import Dict, Optional, Tuple
from cache_prefetching import PrefetchServiceAsync
from job_registry import JobRegistry
from cache_manager import CacheManager
from sampler import PartitionedBatchSampler
from batch import Batch, BatchSet
from cache_status import CacheStatus
from job import DLTJob
# from logger_config import configure_logger
from dataset import S3DatasetBase
# logger = configure_logger()

import time
from utils import AverageMeter


class CoorDLBatchScheduler:
    def __init__(self, dataset:S3DatasetBase, job_registry:JobRegistry):        
        self.dataset:S3DatasetBase = dataset
        self.job_registry = job_registry
        self.num_jobs = len(job_registry.jobs)
        #compute how many that needs to be processed per job




        self.sampler = PartitionedBatchSampler(
            num_files=len(dataset),
            batch_size=dataset.batch_size,
            num_partitions=dataset.num_partitions,
            drop_last=dataset.drop_last,
            shuffle=dataset.shuffle)

        self.lock = threading.Lock()
        self.batch_sets: Dict[int, Dict[int, BatchSet]] = OrderedDict()
        # self.lookahead_distance = min(self.sampler.calc_num_batchs_per_partition() - 1, min_lookahead_steps)
        self.lookahead_distance = min(self.sampler.calc_num_batchs_per_partition() - 1, dataset.min_lookahead_steps)
        self.shared_cache = shared_cache

        if use_prefetching:
            self.prefetch_service = PrefetchServiceAsync(
                lambda_name=prefetch_lambda_name,
                cache_address=cache_address,
                simulate_time=prefetch_simulation_time)
            self.prefetch_service.start()

        self.sample_next_lookahead_batches()