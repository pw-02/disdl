import matplotlib.pyplot as plt
import numpy as np

# Parameters
average_batch_size_MB = 35
redis_cache_size_GB = 419.09
redis_cache_cost_per_hour = 6.981
cost_per_lambda_invocation = 0.00000830198

# Frequencies to compare
warm_up_freq_1 = 60   # every 1 minute
warm_up_freq_2 = 120  # every 2 minutes

# Monthly conversion
hours_per_month = 24 * 30

# Compute max batches Redis can hold
redis_cache_size_MB = redis_cache_size_GB * 1024
max_batches_in_redis = int(redis_cache_size_MB / average_batch_size_MB)

# Simulated batch counts
batch_counts = np.array([2500, 5000, 7500, 10000, 12500, 15000])

# Redis monthly cost (flat line)
redis_monthly_cost = redis_cache_cost_per_hour * hours_per_month
redis_costs = np.full_like(batch_counts, fill_value=redis_monthly_cost, dtype=float)

# Serverless costs
serverless_costs_60s = np.array([
    cost_per_lambda_invocation * (3600 / warm_up_freq_1) * count * hours_per_month
    for count in batch_counts
])
serverless_costs_120s = np.array([
    cost_per_lambda_invocation * (3600 / warm_up_freq_2) * count * hours_per_month
    for count in batch_counts
])

# Plot setup
fig, ax = plt.subplots(figsize=(6, 4))

# Serverless curves
ax.plot(batch_counts, serverless_costs_60s, marker='o', label='Serverless (1 min)', color='darkorange')
# ax.plot(batch_counts, serverless_costs_120s, marker='o', label='Serverless (2 min)', color='green')

# Redis curve
ax.plot(batch_counts, redis_costs, marker='o', label='Redis (fixed-size)', color='steelblue')

# Overflow marker
overflow_indices = np.where(batch_counts > max_batches_in_redis)[0]
if overflow_indices.size > 0:
    overflow_start_idx = overflow_indices[0]
    overflow_start = batch_counts[overflow_start_idx]

    ax.axvline(x=overflow_start, color='gray', linestyle='--', alpha=0.6)
    ax.plot(overflow_start, 0, marker='x', color='red', markersize=10)
    ax.text(overflow_start, 0.3 * redis_monthly_cost, 'Redis overflow', ha='center', va='bottom', color='red', fontsize=9)

# Labels and grid
ax.set_title('Monthly Cost Comparison: Redis vs. Serverless Cache')
ax.set_xlabel('Number of Mini-Batches (35MB each)')
ax.set_ylabel('Monthly Cost (USD)')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.6)

# Annotate serverless lines (just one for clarity)
for xval, yval in zip(batch_counts, serverless_costs_60s):
    ax.text(xval, yval + 0.02 * yval, f"${yval:.0f}", ha='center', fontsize=8, color='darkorange')

plt.tight_layout()
plt.show()
