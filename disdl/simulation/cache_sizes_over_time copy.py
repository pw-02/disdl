import matplotlib.pyplot as plt

# Given data
average_size_of_batch_mb = 35  # MB
redis_cache_size_gb = 419.09
redis_cache_size_mb = redis_cache_size_gb * 1024
total_batches_in_cache = redis_cache_size_mb / average_size_of_batch_mb
redis_cache_cost_hourly = 6.981

print(f"Total batches in cache: {total_batches_in_cache}")

cost_per_lambda_warmup = 0.00000830198
warm_up_frequency_sec = 60  # 1 minute

hourly_warmup_cost = cost_per_lambda_warmup * (3600 / warm_up_frequency_sec) * total_batches_in_cache

print(f"Hourly warmup cost: {hourly_warmup_cost}")
print(f"Hourly redis cache cost: {redis_cache_cost_hourly}")

# Scaling up
monthly_warmup_cost = hourly_warmup_cost * 24 * 30
monthly_redis_cache_cost = redis_cache_cost_hourly * 24 * 30

print(f"Monthly warmup cost: {monthly_warmup_cost}")
print(f"Monthly redis cache cost: {monthly_redis_cache_cost}")

# Number of batches and corresponding serverless cache costs
batch_counts = [2500, 5000, 7500, 10000, 12500, 15000,17500,20000]
serverless_cache_1_min_warmup = [1.245297, 2.490594, 3.735891, 4.981188, 6.226485, 7.471782, 8.717079, 9.962376]
serverless_cache_2_min_warmup = [0.6226485, 1.245297, 1.8679455, 2.490594, 3.1132425, 3.735891, 4.3585395, 4.981188]

# Plotting the figure
plt.figure(figsize=(8, 5))
plt.plot(batch_counts, serverless_cache_1_min_warmup, marker='o', linestyle='-', label='Warmup = 1min')
plt.plot(batch_counts, serverless_cache_2_min_warmup, marker='*', linestyle='-', label='Warmup = 2min')

plt.axhline(y=redis_cache_cost_hourly, color='r', linestyle='--', label='Fixed Redis Cache Cost')

# Labels and title
plt.xlabel("Number of Batches")
plt.ylabel("Cost (USD per Hour)")
plt.title("Cost Comparison: Serverless Cache vs Redis")
plt.legend()
plt.grid(True)
plt.show()
