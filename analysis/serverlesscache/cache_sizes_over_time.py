import matplotlib.pyplot as plt
import numpy as np

# Parameters
average_batch_size_MB = 35
redis_cache_size_GB = 419.09
redis_cache_cost_per_hour = 6.981
cost_per_lambda_invocation = 0.00000830198
warm_up_frequency_sec = 60  # One request per batch per minute

# Compute max batches Redis can hold
redis_cache_size_MB = redis_cache_size_GB * 1024
max_batches_in_redis = int(redis_cache_size_MB / average_batch_size_MB)

# Simulated batch counts (increasing load)
batch_counts = [2500, 5000, 7500, 10000, 12500, 15000]

redis_costs = []
serverless_costs = []

for count in batch_counts:
    # Redis cost is fixed if within capacity; undefined if it overflows
    if count <= max_batches_in_redis:
        redis_costs.append(redis_cache_cost_per_hour)
    else:
        redis_costs.append(None)

    # Serverless cost = number of invocations per hour * cost per invocation
    invocations_per_hour = (3600 / warm_up_frequency_sec) * count
    serverless_hourly_cost = cost_per_lambda_invocation * invocations_per_hour
    serverless_costs.append(serverless_hourly_cost)

# Plot setup
x = np.arange(len(batch_counts))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - width/2, [c if c is not None else 0 for c in redis_costs], width, label='Redis (fixed-size)', color='steelblue')
bars2 = ax.bar(x + width/2, serverless_costs, width, label='Serverless (pay-per-request)', color='darkorange')

# Mark Redis overflow cases
for idx, cost in enumerate(redis_costs):
    if cost is None:
        ax.text(x[idx] - width/2, 0.1, 'overflow', ha='center', va='bottom', rotation=90, color='red', fontsize=9)
        ax.bar(x[idx] - width/2, 0.1, width, color='red', hatch='//')

# Vertical line to show Redis capacity limit
if any(c is None for c in redis_costs):
    overflow_index = np.argmax([c is None for c in redis_costs])
    ax.axvline(x=overflow_index - 0.5, color='gray', linestyle='--', alpha=0.5)
    ax.text(overflow_index, redis_cache_cost_per_hour + 0.2,
            f'Max Redis capacity ≈ {max_batches_in_redis:,} batches',
            ha='left', va='bottom', fontsize=9, color='gray')

# Labels and styling
ax.set_ylabel('Hourly Cost (USD)')
ax.set_xlabel('Number of Mini-Batches (35MB each)')
ax.set_title('Hourly Cost Comparison: Redis vs. Serverless Cache')
ax.set_xticks(x)
ax.set_xticklabels([f"{n:,}" for n in batch_counts])
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.6)

# Annotate serverless bars with costs
for bar in bars2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"${yval:.2f}", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()
