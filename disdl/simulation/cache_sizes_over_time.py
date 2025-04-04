averga_Size_of_bacth_mb = 35 #mb
redis_cache_size_gb = 419.09
redis_Cache_size_mb = redis_cache_size_gb * 1024
total_batchs_in_cache = redis_Cache_size_mb / averga_Size_of_bacth_mb
redis_cache_cost_hourly = 6.981


print(f"Total batches in cache: {total_batchs_in_cache}")

cost_per_lambda_warmup = 0.00000830198
warm_up_frequency_sec = 60 *2 #minute

hourly_warmup_cost = cost_per_lambda_warmup * (3600 / warm_up_frequency_sec) *total_batchs_in_cache

print(f"Hourly warmup cost: {hourly_warmup_cost}")
print(f"Hourly redis cache cost: {redis_cache_cost_hourly}")



#scaling up
monthly_warmup_cost = hourly_warmup_cost * 24 * 30
monthly_redis_cache_cost = redis_cache_cost_hourly * 24 * 30

print(f"Monthly warmup cost: {monthly_warmup_cost}")
print(f"Monthly redis cache cost: {monthly_redis_cache_cost}")

costs = []

for batch_count in [2500, 5000, 7500,10000,12500,15000, 17500,20000]:
    # Calculate the cost of warmup for the given batch count
    hourly_warmup_cost = cost_per_lambda_warmup * (3600 / warm_up_frequency_sec) * batch_count
    monthly_warmup_cost = hourly_warmup_cost * 24 * 30
    costs.append(hourly_warmup_cost)
    print(f"Hourly warmup cost for {batch_count} batches: {hourly_warmup_cost}")
    # print(f"Monthly warmup cost for {batch_count} batches: {monthly_warmup_cost}")

print(costs)

