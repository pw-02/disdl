import time
from typing import Dict, Optional, Tuple
from sortedcontainers import SortedList
from batch import Batch  # Ensure these are properly defined
from cache_status import CacheStatus


class CacheManager:
    def __init__(self):
        self.cached_batches: Dict[str, Batch] = {}
        self.eviction_index: SortedList[Tuple[float, float, str]] = SortedList()
        self.eviction_index_lookup: Dict[str, Tuple[float, float, str]] = {}
        self.assigned_eviction_candidates: Dict[str, Batch] = {}

    def maybe_cache(self, batch: Batch, job_weight:float = 0) -> Tuple[bool, Optional[str]]:
        """Decide whether to cache a batch and suggest an eviction candidate if needed."""
        if batch.batch_id  == '1_1_835_4d5b995358e7798b':
            pass
        if batch.cache_status in (CacheStatus.CACHED, CacheStatus.CACHING_IN_PROGRESS):
            return False, None

        if batch.reuse_score - job_weight <= 0.0:
            return True, None #only cache if there is some cache space
        batch.set_cache_status(CacheStatus.CACHING_IN_PROGRESS)
        eviction_candidate = self._find_eviction_candidate(batch)
        return True, eviction_candidate

    def _find_eviction_candidate(self, incoming_batch: Batch) -> Optional[str]:
        for score, ts, batch_id in self.eviction_index:
            if incoming_batch.reuse_score > score and batch_id not in self.assigned_eviction_candidates:
                self.assigned_eviction_candidates[batch_id] = incoming_batch
                return batch_id
        return None
    
    def _remove_eviction_candidate(self, eviction_candidate_id: str):
        """Remove a batch from the eviction candidates."""
        self.assigned_eviction_candidates.pop(eviction_candidate_id, None)

    def mark_cached(self, batch: Batch):
        """Insert or update a batch in the eviction index and mark as cached."""
        batch.set_cache_status(CacheStatus.CACHED)
        self.cached_batches[batch.batch_id] = batch

        old_entry = self.eviction_index_lookup.pop(batch.batch_id, None)
        if old_entry:
            self.eviction_index.discard(old_entry)

        new_entry = (batch.reuse_score, time.time(), batch.batch_id)
        self.eviction_index.add(new_entry)
        self.eviction_index_lookup[batch.batch_id] = new_entry
    
    def mark_not_cached(self, batch: Batch):
        """Remove a batch from the eviction index and mark as not cached."""
        self.mark_evicted(batch)

    def mark_evicted(self, batch: Batch):
        batch.set_cache_status(CacheStatus.NOT_CACHED)

        """Remove a batch from the cache and eviction index."""
        self.assigned_eviction_candidates.pop(batch.batch_id, None)
        self.cached_batches.pop(batch.batch_id, None)

        evicted_entry = self.eviction_index_lookup.pop(batch.batch_id, None)
        if evicted_entry:
            self.eviction_index.discard(evicted_entry)


    def is_cached(self, batch_id: str) -> bool:
        return batch_id in self.cached_batches
    
    def get_batch(self, batch_id: str) -> Optional[Batch]:
        """Retrieve a batch from the cache."""
        if batch_id not in self.cached_batches:
            return None
        return self.cached_batches.get(batch_id)

    def clear(self):
        """Clear all cached state (useful for testing or reset)."""
        self.cached_batches.clear()
        self.eviction_index.clear()
        self.eviction_index_lookup.clear()
        self.assigned_eviction_candidates.clear()
