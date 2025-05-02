import numpy as np
import pandas as pd
from collections import defaultdict
class MinibatchSimulator:
    def __init__(self,
                 num_jobs=3,
                 epoch_batches=90,
                 cache_capacity=20,
                 sim_duration=300,
                 job_speeds=None,
                 use_reuse_score=True,
                 use_prefetch=True,
                 use_safe_eviction=True,
                 prefetch_limit=1):

        self.N = num_jobs
        self.B = epoch_batches
        self.C = cache_capacity
        self.T = sim_duration
        self.speeds = job_speeds if job_speeds else [1.0] * self.N

        self.use_reuse_score = use_reuse_score
        self.use_prefetch = use_prefetch
        self.use_safe_eviction = use_safe_eviction
        self.prefetch_limit = prefetch_limit

        self.weights = [1.0 for _ in range(self.N)]
        self.batch_seq = list(range(self.B))
        self.job_pos = [0] * self.N
        self.next_ready = [0.0] * self.N
        self.consumed = [set() for _ in range(self.N)]

        self.cache = {}
        self.evicted = []
        self.access_log = defaultdict(list)
        self.batch_in_progress = set()

        self.reuse_hits = 0
        self.reuse_total = 0

    def reuse_score(self, b, t_now):
        pending = [j for j in range(self.N) if b not in self.consumed[j]]
        if not pending:
            return 0
        return sum(self.weights[j] / max((b - self.job_pos[j]) / self.speeds[j], 0.1) for j in pending)

    def evict_batch(self, t_now):
        scores = []
        for b in list(self.cache.keys()):
            if self.use_safe_eviction:
                still_needed = any(b not in self.consumed[j] for j in range(self.N))
                if not still_needed:
                    scores.append((0, b))
                    continue
            score = self.reuse_score(b, t_now) if self.use_reuse_score else 1e9
            scores.append((score, b))
        if scores:
            _, evict_b = min(scores)
            del self.cache[evict_b]
            self.evicted.append((t_now, evict_b))

    def prefetch(self, t_now):
        if not self.use_prefetch:
            return
        candidates = [
            b for b in self.batch_seq
            if b not in self.cache and b not in self.batch_in_progress and any(b not in self.consumed[j] for j in range(self.N))
        ]
        scored = [(self.reuse_score(b, t_now), b) for b in candidates]
        for _, b in sorted(scored, reverse=True)[:self.prefetch_limit]:
            if len(self.cache) >= self.C:
                self.evict_batch(t_now)
            self.cache[b] = t_now

    def assign_batch(self, job_id, t_now):
        # Try to reuse
        reuse_cands = [b for b in self.cache if b not in self.consumed[job_id]]
        if reuse_cands:
            if self.use_reuse_score:
                scores = [(self.reuse_score(b, t_now), b) for b in reuse_cands]
                return max(scores)[1]
            return sorted(reuse_cands)[0]

        # Otherwise walk forward
        while self.job_pos[job_id] < self.B:
            b = self.batch_seq[self.job_pos[job_id]]
            self.job_pos[job_id] += 1
            if b not in self.consumed[job_id] and b not in self.batch_in_progress:
                self.batch_in_progress.add(b)
                return b
        return None

    def run(self):
        for t in range(self.T):
            self.prefetch(t)
            for j in range(self.N):
                if t < self.next_ready[j]:
                    continue
                b = self.assign_batch(j, t)
                if b is None:
                    continue

                self.consumed[j].add(b)
                self.access_log[b].append((t, j))
                self.reuse_total += 1
                if b in self.cache:
                    self.reuse_hits += 1
                else:
                    if len(self.cache) >= self.C:
                        self.evict_batch(t)
                    self.cache[b] = t
                self.batch_in_progress.discard(b)
                self.next_ready[j] = t + (1.0 / self.speeds[j])

        return self.summary(), self.access_df()

    def summary(self):
        return {
            "Reuse Hits": self.reuse_hits,
            "Total Requests": self.reuse_total,
            "Reuse Ratio": round(self.reuse_hits / self.reuse_total, 3),
            "Evictions": len(self.evicted),
            "Final Cache Size": len(self.cache)
        }

    def access_df(self):
        return pd.DataFrame([
            {"batch": b, "time": t, "job": j}
            for b, records in self.access_log.items()
            for t, j in records
        ])


# Example: enable all components
sim = MinibatchSimulator(
    num_jobs=3,
    job_speeds=[1.0, 0.75, 0.5],
    use_reuse_score=True,
    use_prefetch=True,
    use_safe_eviction=True
)
summary, df = sim.run()
print(summary)
print(df.head())
