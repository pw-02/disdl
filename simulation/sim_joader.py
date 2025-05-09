# Refactoring Joader-style simulation into a modular class-based simulator (like the user's example)

from dataclasses import dataclass
import heapq
import random
from typing import List, Dict, Tuple
from collections import defaultdict

@dataclass
class JoaderJob:
    job_id: str
    speed: float
    queue: List[int]
    processed: int = 0
    shared_hits: int = 0
    total_hits: int = 0
    total_misses: int = 0
    next_time: float = 0.0

class JoaderCache:
    def __init__(self):
        self.cache = set()

    def get(self, idx):
        return idx in self.cache

    def insert(self, idx):
        self.cache.add(idx)

    def size(self):
        return len(self.cache)

class JoaderSimulator:
    def __init__(
        self,
        dataset_size: int,
        num_leaf_blocks: int,
        num_jobs: int,
        job_speeds: List[float],
        batch_size: int,
        shared_fraction: float,
        batches_per_job: int,
        shared_seed: int = 42
    ):
        self.dataset_size = dataset_size
        self.num_leaf_blocks = num_leaf_blocks
        self.samples_per_block = dataset_size // num_leaf_blocks
        self.ds_tree_blocks = {
            i: list(range(i * self.samples_per_block, (i + 1) * self.samples_per_block))
            for i in range(num_leaf_blocks)
        }
        self.jobs = [
            JoaderJob(f"job_{i}", job_speeds[i], list(range(batches_per_job)))
            for i in range(num_jobs)
        ]
        self.batch_size = batch_size
        self.shared_fraction = shared_fraction
        self.batches_per_job = batches_per_job
        self.shared_seed = shared_seed
        self.cache = JoaderCache()
        self.event_queue = [(0.0, job.job_id) for job in self.jobs]
        heapq.heapify(self.event_queue)

    def get_shared_block(self, epoch, batch_idx):
        combined_seed = f"{self.shared_seed}_{epoch}_{batch_idx}"
        random.seed(combined_seed)
        return random.randint(0, self.num_leaf_blocks - 1)

    def get_difference_set(self, shared_block_id):
        all_indices = set(range(self.dataset_size))
        shared_block = set(self.ds_tree_blocks[shared_block_id])
        return list(all_indices - shared_block)

    def run(self):
        time = 0.0
        while self.event_queue:
            current_time, job_id = heapq.heappop(self.event_queue)
            job = next(j for j in self.jobs if j.job_id == job_id)
            job.next_time = current_time

            if not job.queue:
                continue

            batch_idx = job.queue.pop(0)
            shared_block_id = self.get_shared_block(epoch=0, batch_idx=batch_idx)
            shared_indices = random.sample(
                self.ds_tree_blocks[shared_block_id],
                int(self.batch_size * self.shared_fraction)
            )
            diff_indices = random.sample(
                self.get_difference_set(shared_block_id),
                self.batch_size - len(shared_indices)
            )
            batch = shared_indices + diff_indices

            for idx in shared_indices:
                if self.cache.get(idx):
                    job.shared_hits += 1
                    job.total_hits += 1
                else:
                    job.total_misses += 1
                    self.cache.insert(idx)

            job.processed += 1
            heapq.heappush(self.event_queue, (current_time + job.speed, job_id))

        # Aggregate results
        summary = {
            "job_stats": [{
                "job_id": job.job_id,
                "batches": job.processed,
                "shared_cache_hits": job.shared_hits,
                "total_cache_hits": job.total_hits,
                "total_cache_misses": job.total_misses,
                "cache_hit_rate": job.total_hits / (job.total_hits + job.total_misses) * 100 if (job.total_hits + job.total_misses) > 0 else 0
            } for job in self.jobs],
            "final_cache_size": self.cache.size(),
            "total_time": max(job.next_time for job in self.jobs),
            "total_shared_hit_samples": sum(job.total_hits for job in self.jobs)
        }

        return summary

# Initialize and run simulation
simulator = JoaderSimulator(
    dataset_size=128,
    num_leaf_blocks=16,
    num_jobs=3,
    job_speeds=[0.5, 0.8, 1.0],
    batch_size=8,
    shared_fraction=0.75,
    batches_per_job=20
)

joader_simulation_summary = simulator.run()
joader_simulation_summary
