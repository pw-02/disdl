import heapq
import random
from collections import defaultdict, OrderedDict
import numpy as np

# --- Job and Cache Models ---
class SharedCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.store = OrderedDict()  # batch_id -> True

    def access(self, batch_id):
        if batch_id in self.store:
            self.store.move_to_end(batch_id)
            return True
        return False

    def insert(self, batch_id):
        if batch_id in self.store:
            return
        if len(self.store) >= self.capacity:
            self.store.popitem(last=False)  # FIFO eviction
        self.store[batch_id] = True

    def contents(self):
        return list(self.store.keys())

class Job:
    def __init__(self, job_id, speed):
        self.job_id = job_id
        self.speed = speed
        self.next_time = 0
        self.cursor = 0
        self.cache_hits = 0
        self.cache_misses = 0
    def __lt__(self, other):
        return self.speed < other.speed  # Compare based on speed

# --- Simulation Engine ---
def run_sim(eviction_policy="fifo"):
    jobs = [Job("A", 0.2), Job("B", 0.4), Job("C", 0.6), Job("D", 0.8)]
    batches = list(range(1, 31))
    per_job_sequence = {j.job_id: batches[:] for j in jobs}
    reuse_map = defaultdict(set)  # batch_id -> set(job_ids)
    job_weights = {j.job_id: 1.0 / j.speed for j in jobs}

    cache = SharedCache(capacity=10)
    event_q = [(j.next_time, j) for j in jobs]
    heapq.heapify(event_q)

    def weighted_cus(batch_id):
        unseen = [j.job_id for j in jobs if j.job_id not in reuse_map[batch_id]]
        return sum(job_weights[jid] for jid in unseen)

    time = 0
    while event_q:
        time, job = heapq.heappop(event_q)
        if job.cursor >= len(per_job_sequence[job.job_id]):
            continue
        batch = per_job_sequence[job.job_id][job.cursor]

        # --- CUS assignment override ---
        if eviction_policy == "cus":
            candidates = [b for b in cache.contents() if job.job_id not in reuse_map[b]]
            if candidates:
                batch = max(candidates, key=weighted_cus)

        hit = cache.access(batch)
        if hit:
            job.cache_hits += 1
        else:
            job.cache_misses += 1
            cache.insert(batch)

        reuse_map[batch].add(job.job_id)
        job.cursor += 1
        job.next_time = time + job.speed
        heapq.heappush(event_q, (job.next_time, job))

    total_hits = sum(j.cache_hits for j in jobs)
    total_accesses = total_hits + sum(j.cache_misses for j in jobs)
    return {
        'cache_hit_percent': 100 * total_hits / total_accesses,
        'avg_reuse': np.mean([len(v) for v in reuse_map.values()]),
        'reuse_map': dict(reuse_map)
    }

# --- Run both simulations ---
fifo = run_sim("fifo")
cus = run_sim("cus")

print("\nComparison of FIFO vs CUS:")
print(f"Cache Hit Rate - FIFO: {fifo['cache_hit_percent']:.2f}% | CUS: {cus['cache_hit_percent']:.2f}%")
print(f"Avg Reuse Per Batch - FIFO: {fifo['avg_reuse']:.2f} | CUS: {cus['avg_reuse']:.2f}")
