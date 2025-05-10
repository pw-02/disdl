import random
import threading
import time
from collections import OrderedDict, deque
from typing import Dict, List, Optional, Set, Tuple, Any

import hydra
from sortedcontainers import SortedList


from batch import Batch, BatchSet, CacheStatus
from cache_prefetching import PrefetchServiceAsync
from dataset import *  # Consider listing explicit imports instead of *
from job import DLTJob
from logger_config import configure_logger
from sampler import PartitionedBatchSampler

logger = configure_logger()

class BatchManager:
    def __init__(self, 
                 dataset:S3DatasetBase, 
                 drop_last: bool = False,
                 shuffle: bool = False,
                 min_lookahead_steps: int = 40,
                 use_prefetching: bool = False,
                 prefetch_lambda_name: str = None,
                 prefetch_simulation_time: int = None,
                 cache_address: str = None,
                 shared_cache=None
                 ):
        
        self.dataset = dataset
        self.sampler = PartitionedBatchSampler(
            num_files=len(dataset),
            batch_size=dataset.batch_size,
            num_partitions=dataset.num_partitions,
            drop_last=drop_last,
            shuffle=shuffle
            )
        
        self.lock = threading.Lock()
        self.batch_sets: Dict[int, Dict[int, BatchSet]] = OrderedDict()
        self.cached_batches: Dict[str, Batch] = {}
        self.eviction_index: SortedList[Tuple[float, float, str]] = SortedList() # Sorted by (reuse_score, timestamp)
        self.eviction_index_lookup: Dict[str, Tuple[float, float, str]] = {} #delete batches from eviction_index efficiently
        self.assigned_eviction_candidates: Dict[str, Batch] = {}
        self.lookahead_distance = min(self.sampler.calc_num_batchs_per_partition() - 1, 40)
        # self.jobs: Dict[str, DLTJob] = {job.job_id: job for job in jobs}
        self.jobs: Dict[str, DLTJob] = {}
        self.shared_cache = shared_cache
        if use_prefetching:
            self.prefetch_service = PrefetchServiceAsync(
                lambda_name=prefetch_lambda_name,
                cache_address=cache_address,
                simulate_time=prefetch_simulation_time
            )
            self.prefetch_service.start()
        self.sample_next_lookahead_batches()

    def _generate_new_batch(self):
        batch_indices, epoch_idx, partition_idx, batch_idx = next(self.sampler)
        next_batch = Batch(batch_indices, epoch_idx, partition_idx, batch_idx)

        epoch_map = self.batch_sets.setdefault(next_batch.epoch_idx, OrderedDict())
        batch_set = epoch_map.setdefault(next_batch.partition_idx, BatchSet(set_id=next_batch.set_id, num_batches=self.sampler.calc_num_batchs_per_partition()))
        batch_set.batches[next_batch.batch_id] = next_batch

        # Check if any job is currently assigned to this BatchSet
        for job in self.jobs.values():
            if job.current_batch_set_id  == batch_set.id:
                next_batch.mark_awaiting_to_be_seen_by(job.job_id, job.weight)
                job.future_batches[next_batch.batch_id] = next_batch
        # self.clean_up_old_batch_sets()
        return next_batch
    
    def _score_batch_set(self, batch_set: BatchSet, epoch_idx, partition_idx) -> float: #used for baseline 
        return float(f"{epoch_idx}.{partition_idx:02d}")
    
    def _get_or_register_job(self, job: DLTJob) -> DLTJob:
        if job.job_id not in self.jobs:
            logger.info(f"Registering new job '{job.job_id}'")
            self.assign_batch_set_to_job(job)
            self.jobs[job.job_id] = job
        return self.jobs[job.job_id]
    
    def get_next_batch_for_job(self, job_id: str) -> Optional[Batch]:    
        
        # job:DLTJob  = self._get_or_register_job(job_id)
        job:DLTJob = self.jobs[job_id]
        if not job.future_batches:
            self.assign_batch_set_to_job(job)
        
        next_batch = job.next_batch()
        if next_batch is None:
            logger.error(f"Job {job.job_id} has no future batches.")
            return None, False, None
        
        should_cache, eviction_candidate = self._maybe_cache_batch(next_batch)
        
        self._maybe_trigger_sample_next_batch(next_batch)
        
        return next_batch, should_cache, eviction_candidate
    
    def sample_next_lookahead_batches(self):
        for _ in range(self.lookahead_distance):
            self._generate_new_batch()

    def _find_best_batch_set_for_job(self, job: DLTJob, baseline=True) -> Optional[Tuple[int, Any]]:
        best_candidate = None
        best_score = float('-inf')

        for epoch_idx, partition_map in self.batch_sets.items():
            for partition_idx, batch_set in partition_map.items():
                if batch_set.id in job.used_batch_set_ids:
                    continue
                if partition_idx in job.partitions_covered_this_epoch:
                    continue
                if baseline:
                    # score = self._score_batch_set(batch_set, epoch_idx, partition_idx)
                    best_candidate = (partition_idx, batch_set)
                    return best_candidate
                else:
                    score = batch_set.compute_reuse_score()
                    if score > best_score:
                        best_candidate = (partition_idx, batch_set)
                        best_score = score

        return best_candidate
    
    def _mark_batch_evicted(self, batch: Batch):
        # Remove the batch from the eviction index
        batch_id = batch.batch_id
        self.assigned_eviction_candidates.pop(batch_id, None)
        self.cached_batches.pop(batch_id, None)
        evicted_entry = self.eviction_index_lookup.pop(batch_id, None)
        if evicted_entry:
            self.eviction_index.discard(evicted_entry)
        batch.set_cache_status(CacheStatus.NOT_CACHED)

        # evicted_entry = self.eviction_index_lookup.pop(batch.batch_id, None)
        # if evicted_entry:
        #     self.eviction_index.discard(evicted_entry)
        # # Remove the batch from the cache
        # # self.shared_cache._remove(batch.batch_id)
        # # Remove the batch from the cached batches
        # self.cached_batches.pop(batch.batch_id, None)
        # batch.set_cache_status(CacheStatus.NOT_CACHED)
    
    def _mark_batch_cached(self, batch: Batch):
        batch.set_cache_status(CacheStatus.CACHED)
        self.cached_batches[batch.batch_id] = batch
        #Update eviction index entry
        old_entry = self.eviction_index_lookup.pop(batch.batch_id, None)
        if old_entry:
            self.eviction_index.discard(old_entry)

        # Re-insert updated entry
        new_entry = (batch.reuse_score, time.time(), batch.batch_id)
        self.eviction_index.add(new_entry)
        self.eviction_index_lookup[batch.batch_id] = new_entry

    
    def assign_batch_set_to_job(self, job: DLTJob):

        if len(job.partitions_covered_this_epoch) == self.dataset.num_partitions:
            job.reset_for_new_epoch()

        # First attempt
        candidate = self._find_best_batch_set_for_job(job)

        if candidate is None:
            self.sample_next_lookahead_batches()
            candidate = self._find_best_batch_set_for_job(job)

        if candidate is None:
            logger.warning(f"[assign_batch_set_to_job] No batch set available for job {job.job_id}")
            return

        partition_idx, batch_set = candidate
        job.used_batch_set_ids[batch_set.id] = job.elapased_time_sec
        job.partitions_covered_this_epoch.add(partition_idx)
        job.current_batch_set_id = batch_set.id

        for batch in batch_set.batches.values():
            batch.mark_awaiting_to_be_seen_by(job.job_id, job.weight)
            job.future_batches[batch.batch_id] = batch


    def _maybe_cache_batch(self, batch: Batch):
        min_reuse_score_to_cache = 0.00

        if batch.cache_status in (CacheStatus.CACHED, CacheStatus.CACHING_IN_PROGRESS):
            return False, None
        
        # Apply minimum score cutoff
        if batch.reuse_score <= min_reuse_score_to_cache:
            logger.debug(f"Skipped caching {batch.batch_id}: reuse_score {batch.reuse_score:.2f} below threshold {min_reuse_score_to_cache}")
            return False, None

        # Mark as eligible for caching
        batch.set_cache_status(CacheStatus.CACHING_IN_PROGRESS)
        eviction_candidate = self._get_eviction_candidate(batch)
        # eviction_candidate = None
        # if self.eviction_index:
        #     for score, ts, batch_id in self.eviction_index:
        #         if batch.reuse_score > score and batch_id not in self.assigned_eviction_candidates:
        #             self.assigned_eviction_candidates[batch_id] = batch
        #             eviction_candidate = batch_id
        #             break
        return True, eviction_candidate
    
    # 'For each (epoch_id, partition_id) in self.batch_sets, if a newer epoch exists that also contains the same partition, '
    # 'and no jobs are still working on the old one, then the old one can be safely deleted.'

    def clean_up_old_batch_sets(self):
         #clean up old batches that are no longer needed
        epochs = list(self.batch_sets.keys())
        for i, epoch_id in enumerate(epochs[:-1]):  # Skip last epoch
            partitions = self.batch_sets[epoch_id]
            to_delete = []
            for partition_id, batch_set in partitions.items():
                newer_found = False

                # Look for newer epochs that have the same partition, and that parition is finalized
                for later_epoch_id in epochs[i + 1:]:
                    later_partitions = self.batch_sets[later_epoch_id]
                    if partition_id in later_partitions and later_partitions[partition_id].is_finalized():
                        #check if any job is using this later partition

                        in_use = any( 
                            job.current_batch_set_id == f"{later_epoch_id}_{partition_id}"
                            for job in self.jobs.values())
                        if in_use:
                            newer_found = True
                        break
                if not newer_found:
                    continue  # Keep if no newer version of this partition exists

                # Now check if any job is still using this old batch_set
                batch_set_id = batch_set.id  # Should be something like "1_3"
                still_in_use = any(
                    job.current_batch_set_id == batch_set_id
                    for job in self.jobs.values())
                
                
                if not still_in_use:
                    # if batch_set.id == '2_1':
                    #     pass
                    # Remove all batches in the cache from the batch set
                    for batch in batch_set.batches.values():
                        self.shared_cache._remove(batch.batch_id)
                        if batch.batch_id in self.cached_batches:
                            self.cached_batches.pop(batch.batch_id, None)
                            # Remove the batch from the eviction index
                            evicted_entry = self.eviction_index_lookup.pop(batch.batch_id, None)
                            if evicted_entry:
                                self.eviction_index.discard(evicted_entry)

                    to_delete.append(partition_id)
            
            for pid in to_delete:
                logger.info(f"Evicting batch set {epoch_id}_{pid}")
                del self.batch_sets[epoch_id][pid]
            if not self.batch_sets[epoch_id]:
                del self.batch_sets[epoch_id]


    
    def processed_batch_update(self,
                               job_id: int,
                               batch_is_cached: bool,
                               eviction_candidate_batch_id: Optional[str],
                               did_evict: bool = False):
        
        job:DLTJob = self.jobs[job_id]
        batch:Batch  = job.current_batch
        batch.mark_seen_by(job.job_id)

        if batch_is_cached:
            self._mark_batch_cached(batch)
            # batch.set_cache_status(CacheStatus.CACHED)
            # self.cached_batches[batch.batch_id] = batch
            # #Update eviction index entry
            # old_entry = self.eviction_index_lookup.pop(batch.batch_id, None)
            # if old_entry:
            #     self.eviction_index.discard(old_entry)

            # # Re-insert updated entry
            # new_entry = (batch.reuse_score, time.time(), batch.batch_id)
            # self.eviction_index.add(new_entry)
            # self.eviction_index_lookup[batch.batch_id] = new_entry
        else:
            batch.set_cache_status(CacheStatus.NOT_CACHED)
            self.cached_batches.pop(batch.batch_id, None)

        if eviction_candidate_batch_id:
            self.assigned_eviction_candidates.pop(eviction_candidate_batch_id, None)
            if did_evict:
                self._mark_batch_evicted(self.cached_batches[eviction_candidate_batch_id])
                # self.cached_batches.pop(eviction_candidate_batch_id, None)
                # evicted_entry = self.eviction_index_lookup.pop(eviction_candidate_batch_id, None)
                # if evicted_entry:
                #     self.eviction_index.discard(evicted_entry)


    def _get_eviction_candidate(self, batch: Batch) -> List[Batch]:
        # Find batches that are eligible for eviction
        eviction_candidate = None
        if self.eviction_index:
            for score, ts, batch_id in self.eviction_index:
                if batch.reuse_score > score and batch_id not in self.assigned_eviction_candidates:
                    self.assigned_eviction_candidates[batch_id] = batch
                    eviction_candidate = batch_id
                    break
        return self.assigned_eviction_candidates[batch_id]

    def _maybe_trigger_sample_next_batch(self, batch: Batch):
        if batch.is_first_access:
            batch.is_first_access = False
            self._generate_new_batch()

    def get_batch_reuse_score(self, batch_id: str) -> float:
        #find batch somehher across all the batches in the batch_sets
        for partition_map in self.batch_sets.values():
            for batch_set in partition_map.values():
                if batch_id in batch_set.batches:
                    return batch_set.batches[batch_id].reuse_score
                

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args):
    # Initialize the dataset based on the workload specified in the args
    if args.workload == 'mscoco':
        dataset = MSCOCODataset(dataset_location=args.dataset_location)
    # elif args.workload.name == 'librispeech':
    #     dataset = LibSpeechDataset(dataset_location=args.dataset_location)
    elif args.workload.name == 'imagenet':
        dataset = ImageNetDataset(dataset_location='ss3://imagenet1k-sdl/val/')
    else:
        raise ValueError(f"Unsupported workload: {args.workload}")

    # Create the CentralBatchManager instance
    batch_manager = BatchManager(dataset=dataset,
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
    for step in range(200):
        job_id = 1
        batch, should_cache, eivction_candidate = batch_manager.get_next_batch_for_job(job_id)
        if batch:
                # Simulate time spent on training step
                time.sleep(random.uniform(0.001, 0.005))  # Fast-forwarded for sim
                print(f"Step {step:03} | Job {job_id} got batch {batch.batch_id} | Cached: {batch.cache_status.name} | Reuse: {batch.reuse_score:.2f}")
        else:
                print(f"Step {step:03} | Job {job_id} got no batch")
        
        batch_manager.processed_batch_update(job_id, batch_is_cached=True, job_cached_batch=True, job_evicted_batch_id=None)


def simulate_training_loop(batch_manager:BatchManager, num_jobs: int, steps_per_job: int = 100):
    job_ids = [str(i) for i in range(1, num_jobs + 1)]
    
    for job_id in job_ids:
        batch_manager._get_or_register_job(job_id)

    for step in range(steps_per_job):
        for job_id in job_ids:
            batch, should_cache, eivction_candidate = batch_manager.get_next_batch_for_job(job_id)
            if batch:
                # Simulate time spent on training step
                time.sleep(random.uniform(0.001, 0.005))  # Fast-forwarded for sim
                print(f"Step {step:03} | Job {job_id} got batch {batch.batch_id} | Cached: {batch.cache_status.name} | Reuse: {batch.reuse_score:.2f}")
            else:
                print(f"Step {step:03} | Job {job_id} got no batch")
            

            batch_manager.job_processed_batch_update(job_id, batch_is_cached=True, job_cached_batch=True, job_evicted_batch_id=None)
            

    print("\nSimulation complete.")


if __name__ == "__main__":
    
    main()
