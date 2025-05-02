import heapq
import random
import matplotlib.pyplot as plt
from collections import defaultdict

# Sim config
NUM_JOBS = 5
NUM_BATCHES = 100
CACHE_SIZE = 10
EPOCH_LENGTH = NUM_BATCHES
CACHE_MISS_PENALTY = 0.2
 
job_iteration_times = {
    f"J{i+1}": t for i, t in enumerate([0.104419951,  0.33947309, 0.514980298, 0.062516984, 0.062516984])
}

job_weights = {j: 1.0 / t for j, t in job_iteration_times.items()}
jobs = list(job_iteration_times.keys())
global_batches = list(range(NUM_BATCHES))
random.shuffle(global_batches)

# Define policies to compare
policies = ["CUS", "FIFO"]
results = {}

for policy in policies:
    job_cursors = {j: 0 for j in jobs}
    job_next_step = {j: 0.0 for j in jobs}
    job_wall_time = {j: 0.0 for j in jobs}
    cache_hits = defaultdict(int)
    cache_misses = defaultdict(int)
    cache = [] if policy == "FIFO" else set()
    seen_by = defaultdict(set)

    event_queue = [(job_iteration_times[j], j) for j in jobs]
    heapq.heapify(event_queue)

    def cus(batch_id):
        unseen_jobs = [j for j in jobs if j not in seen_by[batch_id]]
        return sum(job_weights[j] for j in unseen_jobs)

    def evict_one():
        if len(cache) < CACHE_SIZE:
            return
        if policy == "FIFO":
            cache.pop(0)
        else:  # CUS
            evict_batch = min(cache, key=cus)
            cache.remove(evict_batch)

    while event_queue:
        time, job = heapq.heappop(event_queue)
        if job_cursors[job] >= EPOCH_LENGTH:
            continue

        batch_id = global_batches[job_cursors[job]]

        if (batch_id in cache):
            cache_hits[job] += 1
            delay = job_iteration_times[job]
        else:
            cache_misses[job] += 1
            evict_one()
            if policy == "FIFO":
                cache.append(batch_id)
            else:
                cache.add(batch_id)
            delay = job_iteration_times[job] + CACHE_MISS_PENALTY

        seen_by[batch_id].add(job)
        job_cursors[job] += 1
        job_wall_time[job] = time + delay
        heapq.heappush(event_queue, (time + delay, job))

    policy_results = {}
    for j in jobs:
        hits = cache_hits[j]
        misses = cache_misses[j]
        total = hits + misses
        hit_rate = hits / total if total else 0.0
        baseline_time = total * (job_iteration_times[j] + CACHE_MISS_PENALTY)
        actual_time = job_wall_time[j]
        speedup = (baseline_time - actual_time) / baseline_time if baseline_time > 0 else 0.0
        policy_results[j] = {
            'hit_rate': hit_rate,
            'actual_time': actual_time,
            'baseline_time': baseline_time,
            'speedup': speedup,
        }

    total_batches = NUM_JOBS * NUM_BATCHES
    max_time = max(job_wall_time.values())
    throughput = total_batches / max_time if max_time > 0 else 0.0
    results[policy] = {
        'per_job': policy_results,
        'throughput': throughput
    }

# Print summary
for policy in policies:
    print(f"\n[{policy}] Per-job results:")
    for j in jobs:
        r = results[policy]['per_job'][j]
        print(f"{j}: hit rate = {r['hit_rate']:.2%}, speedup = {r['speedup']:.2%}, actual_time = {r['actual_time']:.1f}s")
    print(f"Aggregate throughput: {results[policy]['throughput']:.2f} batches/sec")

# Visualization
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
job_indices = range(len(jobs))
width = 0.35

# Plot per-job speedup
for i, policy in enumerate(policies):
    speedups = [results[policy]['per_job'][j]['speedup'] for j in jobs]
    ax[0].bar([x + i * width for x in job_indices], speedups, width=width, label=policy)

ax[0].set_title("Per-job Speedup")
ax[0].set_ylabel("Speedup")
ax[0].set_xticks([x + width/2 for x in job_indices])
ax[0].set_xticklabels(jobs)
ax[0].legend()

# Plot throughput
throughputs = [results[policy]['throughput'] for policy in policies]
ax[1].bar(policies, throughputs, color=['tab:blue', 'tab:orange'])
ax[1].set_title("Aggregate Throughput")
ax[1].set_ylabel("Batches/sec")

plt.tight_layout()
plt.show()
