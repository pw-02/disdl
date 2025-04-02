averga_Size_of_bacth_mb = 17 #mb
redis_Cache_size_gb = 203
redis_Cache_size_mb = redis_Cache_size_gb * 1024
total_batchs_in_cache = redis_Cache_size_mb / averga_Size_of_bacth_mb
redis_cache_cost_hourly = 3.64 
print(f"Total batches in cache: {total_batchs_in_cache}")

cost_per_lambda_warmup = 0.00000830198
warm_up_frequency_sec = 180 #1 minute

hhorly_warmup_cost = cost_per_lambda_warmup * (3600 / warm_up_frequency_sec) *total_batchs_in_cache
print(f"Hourly warmup cost: {hhorly_warmup_cost}")
print(f"Hourly redis cache cost: {redis_cache_cost_hourly}")

#scaling up
monthly_warmup_cost = hhorly_warmup_cost * 24 * 30
monthly_redis_cache_cost = redis_cache_cost_hourly * 24 * 30

print(f"Monthly warmup cost: {monthly_warmup_cost}")
print(f"Monthly redis cache cost: {monthly_redis_cache_cost}")


