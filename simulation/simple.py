import heapq
import random
import collections
from typing import List, Dict

class Job:
    def __init__(self, job_id: int, batch_sequence: List[int], processing_time: float):
        self.job_id = job_id
        self.batch_sequence = batch_sequence
        self.current_index = 0
        self.processing_time = processing_time
        self.start_time = None
        self.end_time = None

    def next_batch(self):
        if self.current_index >= len(self.batch_sequence):
            return None
        batch = self.batch_sequence[self.current_index]
        self.current_index += 1
        return batch

    def is_done(self):
        return self.current_index >= len(self.batch_sequence)


class SharedCacheBaseline:
    def __init__(self, capacity: int, policy: str):
        assert policy in ("lru", "fifo")
        self.cache = collections.OrderedDict()
        self.capacity = capacity
        self.policy = policy

    def get_batch(self, batch_id) -> bool:
        if batch_id in self.cache:
            if self.policy == "lru":
                self.cache.move_to_end(batch_id)
            return True
        return False

    def put_batch(self, batch_id):
        if batch_id in self.cache:
            return
        if len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[batch_id] = True


class SharedCacheCUSWeighted:
    def __init__(self, capacity: int, job_weights: Dict[int, float]):
        self.cache = collections.OrderedDict()
        self.capacity = capacity
        self.job_weights = job_weights

    def get_batch(self, batch_id) -> bool:
        if batch_id in self.cache:
            self.cache.move_to_end(batch_id)
            return True
        return False

    def put_batch(self, batch_id: int, cus_scores: Dict[int, float]):
        if batch_id in self.cache:
            return
        if len(self.cache) >= self.capacity:
            to_evict = min(self.cache, key=lambda b: cus_scores.get(b, 0))
            self.cache.pop(to_evict)
        self.cache[batch_id] = True


class BatchManager:
    def __init__(self, jobs: List[Job]):
        self.jobs = jobs

    def compute_weighted_cus_scores(self, weights: Dict[int, float]) -> Dict[int, float]:
        cus = collections.defaultdict(float)
        for job in self.jobs:
            for idx in range(job.current_index, len(job.batch_sequence)):
                batch = job.batch_sequence[idx]
                cus[batch] += weights[job.job_id]
        return cus


def run_comparative_simulation(policy: str = "lru", use_cus: bool = False):
    num_jobs = 20
    num_batches = 20
    cache_capacity = 5
    jobs = []
    job_weights = {}

    for jid in range(num_jobs):
        batch_seq = list(range(num_batches))
        random.shuffle(batch_seq)
        processing_time = random.uniform(1.0, 2.0)
        jobs.append(Job(jid, batch_seq, processing_time))
        job_weights[jid] = random.uniform(0.5, 1.5)

    if use_cus:
        cache = SharedCacheCUSWeighted(cache_capacity, job_weights)
    else:
        cache = SharedCacheBaseline(cache_capacity, policy)

    batch_manager = BatchManager(jobs)
    event_queue = []
    heapq.heapify(event_queue)

    cache_hits = 0
    cache_misses = 0
    global_time = 0.0
    job_timings = {}

    for job in jobs:
        job.start_time = 0.0
        heapq.heappush(event_queue, (0.0, job.job_id, "request_batch"))

    while event_queue:
        current_time, jid, event_type = heapq.heappop(event_queue)
        job = jobs[jid]
        global_time = max(global_time, current_time)

        if event_type == "request_batch":
            batch_id = job.next_batch()
            if batch_id is None:
                job.end_time = current_time
                continue

            if cache.get_batch(batch_id):
                cache_hits += 1
                finish_time = current_time + job.processing_time
            else:
                cache_misses += 1
                if use_cus:
                    cus_scores = batch_manager.compute_weighted_cus_scores(job_weights)
                    cache.put_batch(batch_id, cus_scores)
                else:
                    cache.put_batch(batch_id)
                finish_time = current_time + 1.5 * job.processing_time

            heapq.heappush(event_queue, (finish_time, jid, "request_batch"))

    durations = [(job.end_time - job.start_time) for job in jobs if job.end_time is not None]
    avg_duration = sum(durations) / len(durations)

    return {
        "policy": "CUS" if use_cus else policy.upper(),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "total_batches": num_jobs * num_batches,
        "hit_rate": cache_hits / (cache_hits + cache_misses),
        "avg_job_duration": avg_duration,
        "total_runtime": global_time
    }
res_fifo = run_comparative_simulation(policy="fifo")
res_lru = run_comparative_simulation(policy="lru")
res_cus = run_comparative_simulation(use_cus=True)

from pprint import pprint
pprint(res_fifo)
pprint(res_lru)
pprint(res_cus)
