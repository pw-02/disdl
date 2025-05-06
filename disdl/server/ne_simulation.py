import heapq
from typing import Dict, List, Tuple
from batch import CacheStatus, Batch
from batch_manager import CentralBatchManager

class NextEventSimulator:
    MAX_CACHE_SIZE = 100  # Max number of batches that can be held in cache
    CACHE_DELAY = 1.0        # Seconds until a batch becomes fully cached
    REFRESH_INTERVAL = 2.0   # Seconds between reuse score updates

    def __init__(self, batch_manager:CentralBatchManager, job_speeds: Dict[str, float], max_steps: int):
        self.batch_manager:CentralBatchManager = batch_manager
        self.job_speeds = job_speeds  # job_id -> steps/sec
        self.max_steps = max_steps
        self.event_queue: List[Tuple[float, str, str, dict]] = []
        self.current_time = 0.0
        self.steps_done: Dict[str, int] = {job_id: 0 for job_id in job_speeds}
        self.cached_batches: List[str] = []  # Holds batch_ids currently in cache
        # self.pending_cache: Dict[str, float] = {}  # batch_id -> cache ready time

        for job_id in job_speeds:
            self._schedule_event(self.current_time, job_id, "REQUEST_BATCH", {})

        # Trigger first reuse score refresh
        self._schedule_event(self.current_time + self.REFRESH_INTERVAL, "SYSTEM", "REFRESH_CACHE", {})

    def _schedule_event(self, time: float, job_id: str, event_type: str, payload: dict):
        heapq.heappush(self.event_queue, (time, job_id, event_type, payload))

    def _handle_event(self, event: Tuple[float, str, str, dict]):
        time, job_id, event_type, payload = event
        self.current_time = time

        if event_type == "REQUEST_BATCH":
            batch = self.batch_manager.get_next_batch_for_job(job_id)
            if batch:
                print(f"{time:.3f} | Job {job_id} received batch {batch.batch_id} | Cached: {batch.cache_status.name} | Reuse: {batch.reuse_score:.2f}")
                if batch.cache_status == CacheStatus.CACHING_IN_PROGRESS and batch.batch_id not in self.pending_cache:
                    ready_time = self.current_time + self.CACHE_DELAY
                    self._schedule_event(ready_time, job_id, "CACHE_COMPLETE", {"batch_id": batch.batch_id})
                process_time = 1.0 / self.job_speeds[job_id]
                self._schedule_event(self.current_time + process_time, job_id, "PROCESS_BATCH", {"batch_id": batch.batch_id})
            else:
                print(f"{time:.3f} | Job {job_id} no batch available")
                self._schedule_event(self.current_time + 0.01, job_id, "REQUEST_BATCH", {})

        elif event_type == "PROCESS_BATCH":
            self.steps_done[job_id] += 1
            if self.steps_done[job_id] < self.max_steps:
                self._schedule_event(self.current_time, job_id, "REQUEST_BATCH", {})
            else:
                print(f"{time:.3f} | Job {job_id} completed all steps.")

        elif event_type == "CACHE_COMPLETE":
            batch_id = payload["batch_id"]
            for epoch in self.batch_manager.epoch_partition_batches.values():
                for partition in epoch.values():
                    if batch_id in partition.batches:
                        partition.batches[batch_id].cache_status = CacheStatus.CACHED
                        self._insert_into_cache(batch_id)
                        print(f"{time:.3f} | Batch {batch_id} marked as CACHED")
                        break

        elif event_type == "REFRESH_CACHE":
            if hasattr(self.batch_manager, "cache_manager"):
                self.batch_manager.cache_manager._refresh_cache_status_and_reuse_scores()
                print(f"{time:.3f} | Refreshed reuse scores")
            self._schedule_event(self.current_time + self.REFRESH_INTERVAL, "SYSTEM", "REFRESH_CACHE", {})

    def _insert_into_cache(self, batch_id: str):
        self.cached_batches.append(batch_id)
        if len(self.cached_batches) > self.MAX_CACHE_SIZE:
            evicted = self.cached_batches.pop(0)
            for epoch in self.batch_manager.epoch_partition_batches.values():
                for partition in epoch.values():
                    if evicted in partition.batches:
                        partition.batches[evicted].cache_status = CacheStatus.NOT_CACHED
                        print(f"{self.current_time:.3f} | Evicted batch {evicted} from cache")
                        return


        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            self._handle_event(event)
        print("\nSimulation finished.")
