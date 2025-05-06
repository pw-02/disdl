import threading
import time
from typing import Dict
import redis
from batch import BatchSet, CacheStatus
from job import DLTJob

class CacheManager:
    def __init__(self, redis_url, epoch_partition_batches, jobs):
        self.redis = redis.Redis.from_url(redis_url)
        self.epoch_partition_batches: Dict[int, Dict[int, BatchSet]] = epoch_partition_batches
        self.jobs:Dict[str, DLTJob] = jobs
        self.lock = threading.Lock()

    def start(self):
        t = threading.Thread(target=self._refresh_loop, daemon=True)
        t.start()

    def _refresh_loop(self):
        while True:
            with self.lock:
                self._refresh_cache_status_and_reuse_scores()
            time.sleep(2)

    def _get_cached_keys(self) -> set:
        cursor = 0
        cached_keys = set()
        while True:
            cursor, keys = self.redis.scan(cursor=cursor, match="*", count=1000)
            cached_keys.update(keys)
            if cursor == 0:
                break
        return cached_keys

    def _refresh_cache_status_and_reuse_scores(self):
        cached_keys = self._get_cached_keys()
        job_speeds = {
            job.job_id: max(getattr(job, 'processing_speed', 1.0), 1e-6)
            for job in self.jobs.values()
        }

        for epoch in self.epoch_partition_batches.values():
            for partition in epoch.values():
                for batch in partition.batches.values():
                    is_cached = batch.batch_id.encode() in cached_keys
                    if is_cached:
                        batch.cache_status = CacheStatus.CACHED
                        batch.reuse_score = 0.0
                        for job_id, job in self.jobs.items():
                            if batch.batch_id in job.future_batches:
                                batch.reuse_score += 1.0 / job_speeds[job_id]
                    else:
                        batch.cache_status = CacheStatus.NOT_CACHED
                        batch.reuse_score = 0.0
       