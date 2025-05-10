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
from logger_config import configure_logger
from dataset import S3DatasetBase
logger = configure_logger()

class BatchManager:
    def __init__(self, 
                 dataset:S3DatasetBase,
                 drop_last=False,
                 shuffle=False,
                 prefetch_lookahead_steps=40,
                 use_prefetching=False,
                 prefetch_lambda_name=None,
                 prefetch_simulation_time=None,
                 cache_address=None,
                 shared_cache=None):

        self.dataset:S3DatasetBase = dataset
        self.registry = JobRegistry()
        self.cache = CacheManager()
        self.sampler = PartitionedBatchSampler(
            num_files=len(dataset),
            batch_size=dataset.batch_size,
            num_partitions=dataset.num_partitions,
            drop_last=drop_last,
            shuffle=shuffle)

        self.lock = threading.Lock()
        self.batch_sets: Dict[int, Dict[int, BatchSet]] = OrderedDict()
        # self.lookahead_distance = min(self.sampler.calc_num_batchs_per_partition() - 1, min_lookahead_steps)
        self.lookahead_distance = max(self.sampler.calc_num_batchs_per_partition() - 1, prefetch_lookahead_steps)
        self.shared_cache = shared_cache

        if use_prefetching:
            self.prefetch_service = PrefetchServiceAsync(
                lambda_name=prefetch_lambda_name,
                cache_address=cache_address,
                simulate_time=prefetch_simulation_time)
            self.prefetch_service.start()

        self.sample_next_lookahead_batches()
    
    def sampple_next_batch_set(self):
        pass
    
    def sample_next_lookahead_batches(self):
        for _ in range(self.lookahead_distance):
            self._generate_new_batch()

    def register_job(self, job_id: str):
        job = self.registry.register(job_id)
        if not job.future_batches:
            self.assign_batch_set_to_job(job)

    def get_next_batch_for_job(self, job_id: str) -> Tuple[Optional[Batch], bool, Optional[str]]:
        job = self.registry.get(job_id)
        if not job.future_batches:
            self.assign_batch_set_to_job(job)

        next_batch = job.next_batch()
        if next_batch is None:
            return None, False, None

        # job_weights = self.registry.job_weights()
        # next_batch.update_reuse_score(job_weights)

        should_cache, eviction_candidate = self.cache.maybe_cache(next_batch)
        job.set_eviction_candidate(eviction_candidate)
        self._maybe_trigger_sample_next_batch(next_batch)

        return next_batch, should_cache, eviction_candidate

    def assign_batch_set_to_job(self, job: DLTJob):
        self.registry.reset_if_new_epoch(job, self.dataset.num_partitions)

        candidate = self._find_best_batch_set_for_job(job)
        if candidate is None:
            self.sample_next_lookahead_batches()
            candidate = self._find_best_batch_set_for_job(job)

        if candidate is None:
            logger.warning(f"[assign_batch_set_to_job] No batch set available for job {job.job_id}")
            return

        partition_idx, batch_set = candidate
        self.registry.update_assignment(job, batch_set, job.elapased_time_sec)
    
    def processed_batch_update(self,
                               job_id: str,
                               did_cache: bool,
                               evicited_batch_id: Optional[str]):
        
        job = self.registry.get(job_id)
        batch = job.current_batch
        batch.mark_seen_by(job.job_id) # Mark the batch as seen by the job and update the reuse score
        eviction_candidate_batch = job.current_eviction_candidate
    
        if did_cache:
            self.cache.mark_cached(batch)
        else:
            self.cache.mark_not_cached(batch)
        
        self.cache._remove_eviction_candidate(eviction_candidate_batch.batch_id)

        if evicited_batch_id is not None:
            #find the batch somehow from the batch set dict
            epoxh_id = int(evicited_batch_id.split("_")[0])
            partition_id = int(evicited_batch_id.split("_")[1])
            evicted_batch = self.batch_sets[epoxh_id][partition_id].batches.get(evicited_batch_id)
            self.cache.mark_evicted(evicted_batch) 

    # def processed_batch_update(self, 
    #                            job_id: str,
    #                            batch_is_cached: bool,
    #                            eviction_candidate_batch_id: Optional[str],
    #                            did_evict: bool = False):
        
    #     job = self.registry.get(job_id)
    #     batch = job.current_batch
    #     batch.mark_seen_by(job.job_id)

    #     if batch_is_cached:
    #         self.cache.mark_cached(batch)
    #     else:
    #         batch.set_cache_status(CacheStatus.NOT_CACHED)
    #         self.cache.cached_batches.pop(batch.batch_id, None)

    #     if eviction_candidate_batch_id:
    #         self.cache.assigned_eviction_candidates.pop(eviction_candidate_batch_id, None)
    #         if did_evict:
    #             evicted_batch = self.cache.cached_batches.get(eviction_candidate_batch_id)
    #             if evicted_batch:
    #                 self.cache.mark_evicted(evicted_batch)

    def _generate_new_batch(self):
        next_batch:Batch = next(self.sampler)

        epoch_map = self.batch_sets.setdefault(next_batch.epoch_idx, OrderedDict())
        batch_set = epoch_map.setdefault(next_batch.partition_idx, BatchSet(set_id=next_batch.set_id,
                                                                 num_batches=self.sampler.calc_num_batchs_per_partition()))
        batch_set.batches[next_batch.batch_id] = next_batch

        for job in self.registry.all():
            if job.current_batch_set_id == batch_set.id:
                next_batch.mark_awaiting_to_be_seen_by(job.job_id, job.weight)
                job.future_batches[next_batch.batch_id] = next_batch

        return next_batch

    def _find_best_batch_set_for_job(self, job: DLTJob) -> Optional[Tuple[int, BatchSet]]:
        best_candidate = None
        best_score = float('-inf')

        for epoch_idx, partition_map in self.batch_sets.items():
            for partition_idx, batch_set in partition_map.items():
                if batch_set.id in job.used_batch_set_ids:
                    continue  # Already processed
                if partition_idx in job.partitions_covered_this_epoch:
                    continue  # Already covered in this epoch

                reuse_score = batch_set.compute_reuse_score()

                if reuse_score > best_score:
                    best_score = reuse_score
                    best_candidate = (partition_idx, batch_set)

        return best_candidate


    # def _find_best_batch_set_for_job(self, job:DLTJob) -> Optional[Tuple[int, BatchSet]]:
    #     for epoch_idx, partition_map in self.batch_sets.items():
    #         for partition_idx, batch_set in partition_map.items():
    #             if batch_set.id in job.used_batch_set_ids:
    #                 continue
    #             if partition_idx in job.partitions_covered_this_epoch:
    #                 continue
    #             return (partition_idx, batch_set)
    #     return None

    def _maybe_trigger_sample_next_batch(self, batch: Batch):
        if batch.is_first_access:
            batch.is_first_access = False
            self._generate_new_batch()
