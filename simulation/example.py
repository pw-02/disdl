import random
import heapq
from collections import defaultdict

# Sim config
NUM_JOBS = 20
NUM_BATCHES = 100
CACHE_SIZE = 10  # Much smaller cache to increase eviction pressure
EPOCH_LENGTH = NUM_BATCHES

# Job speeds: varied between fast and slow
job_iteration_times = {
    f'J{i+1}': random.uniform(0.5, 6.0) for i in range(NUM_JOBS)
}

job_weights = {j: 1.0 / t for j, t in job_iteration_times.items()}
jobs = list(job_weights.keys())

# Each job gets its own shuffled batch list to simulate divergent access
job_batches = {j: random.sample(range(NUM_BATCHES), NUM_BATCHES) for j in jobs}

# Simulate job state
job_cursors = {j: 0 for j in jobs}
job_next_step = {j: 0.0 for j in jobs}

# Policies to test
cache_hits = defaultdict(int)
cache = {
    'FIFO': [],
    'Reuse': [],
    'CUS': []
}

# Bookkeeping
batch_seen_by = defaultdict(set)
cache_contents = {
    'FIFO': set(),
    'Reuse': set(),
    'CUS': set()
}

# Eviction functions
def evict_FIFO():
    if len(cache['FIFO']) >= CACHE_SIZE:
        evicted = cache['FIFO'].pop(0)
        cache_contents['FIFO'].remove(evicted)

def evict_Reuse():
    if len(cache['Reuse']) >= CACHE_SIZE:
        worst = max(cache['Reuse'], key=lambda b: len(batch_seen_by[b]))
        cache['Reuse'].remove(worst)
        cache_contents['Reuse'].remove(worst)

def evict_CUS():
    if len(cache['CUS']) >= CACHE_SIZE:
        def cus(b):
            unseen_jobs = [j for j in jobs if j not in batch_seen_by[b]]
            return sum(job_weights[j] for j in unseen_jobs)
        worst = min(cache['CUS'], key=cus)
        cache['CUS'].remove(worst)
        cache_contents['CUS'].remove(worst)

# Simulate
for t in range(3000):
    for j in jobs:
        if t < job_next_step[j] or job_cursors[j] >= EPOCH_LENGTH:
            continue

        b = job_batches[j][job_cursors[j]]

        for policy in ['FIFO', 'Reuse', 'CUS']:
            if b in cache_contents[policy]:
                cache_hits[policy] += 1
            else:
                if policy == 'FIFO':
                    evict_FIFO()
                    cache['FIFO'].append(b)
                elif policy == 'Reuse':
                    evict_Reuse()
                    cache['Reuse'].append(b)
                elif policy == 'CUS':
                    evict_CUS()
                    cache['CUS'].append(b)
                cache_contents[policy].add(b)

        batch_seen_by[b].add(j)
        job_cursors[j] += 1
        job_next_step[j] = t + job_iteration_times[j]

print("Cache hit comparison (high pressure, divergent access):")
for policy in ['FIFO', 'Reuse', 'CUS']:
    print(f"{policy}: {cache_hits[policy]} hits")
