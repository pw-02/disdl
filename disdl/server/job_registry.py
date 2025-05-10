from typing import Dict, Iterable
from collections import OrderedDict
from batch import Batch, BatchSet
from job import DLTJob  # Assumes each job tracks weights, future batches, etc.

class JobRegistry:
    def __init__(self):
        self.jobs: Dict[str, DLTJob] = {}

    def register(self, job_id: str) -> DLTJob:
        if job_id not in self.jobs:
            self.jobs[job_id] = DLTJob(job_id)
        return self.jobs[job_id]

    def get(self, job_id: str) -> DLTJob:
        return self.jobs[job_id]

    def all(self) -> Iterable[DLTJob]:
        return self.jobs.values()

    def has(self, job_id: str) -> bool:
        return job_id in self.jobs

    def job_weights(self) -> Dict[str, float]:
        return {job.job_id: job.weight for job in self.jobs.values()}

    def update_assignment(self, job: DLTJob, batch_set: BatchSet, elapsed_time: float):
        """Assigns a new batch set to a job and updates job tracking state."""
        job.used_batch_set_ids[batch_set.id] = elapsed_time
        partition_id = int(batch_set.id.split("_")[1])
        job.partitions_covered_this_epoch.add(partition_id)
        job.current_batch_set_id = batch_set.id

        for batch in batch_set.batches.values():
            batch.mark_awaiting_to_be_seen_by(job.job_id, job.weight)
            job.future_batches[batch.batch_id] = batch

    def reset_if_new_epoch(self, job: DLTJob, num_partitions: int):
        if len(job.partitions_covered_this_epoch) == num_partitions:
            job.reset_for_new_epoch()
