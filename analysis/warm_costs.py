import math
memory_per_lambda_function_gb = 5
memory_reuqired_gb = 2000
num_lambdas_needed = math.ceil(memory_reuqired_gb / memory_per_lambda_function_gb)
warm_up_frequency_sec = 60 # Frequency of warm-up in seconds
num_jobs = 20 # Number of jobs
num_epochs_per_job = 100 # Number of epochs per job
mini_batches_per_epoch = 10000 # Number of mini-batches per epoch
cache_invccation_cost = 0.0000095 # cost per invocation
prefetch_cost = 0.000113256


# #cost used aws lambda function, 0.20/million invocations
# number_warmup_invocations_per_hour = (3600 / warm_up_frequency_sec) * num_lambdas_needed
# #cost of warmup invocations
# warmup_cost = number_warmup_invocations_per_hour * cache_invccation_cost
# number_cache_requets_by_jobs = num_jobs * num_epochs_per_job * mini_batches_per_epoch 
# number_cache_requests_by_prefetcer = num_epochs_per_job * mini_batches_per_epoch 
# prefetch_cost = (number_cache_requests_by_prefetcer * prefetch_cost) #+ (number_cache_requests_by_prefetcer * cache_invccation_cost)
# cost_to_serve_jobs = number_cache_requets_by_jobs * cache_invccation_cost
# total_cost = cost_to_serve_jobs + prefetch_cost + warmup_cost

# #print all details
# print(f"Memory per Lambda Function: {memory_per_lambda_function_gb} GB")
# print(f"GB Required: {memory_reuqired_gb} GB")
# print(f"Number Lambdas Needed: {num_lambdas_needed}")
# print(f"Warm-up Frequency: {warm_up_frequency_sec} seconds")
# print(f"Horly Warmup cost: {warmup_cost}")
# print(f"Serving Job Cost: {cost_to_serve_jobs}")
# print(f"Serving Prefetcher: {prefetch_cost}")
# print(f"Total Cost: {total_cost}")

# print('*' * 40)
#how much data could I store if I want to cap warm costs per hpur a 300$?
target_warm_up_cost = 300
max_warmup_invocations_per_hour = target_warm_up_cost / cache_invccation_cost
max_warmup_invocations_per_hour = math.floor(max_warmup_invocations_per_hour)
warmup_cost = max_warmup_invocations_per_hour * cache_invccation_cost
#how much memory could I store if I want to cap warm costs per hpur a 300$?
max_memory_gb = max_warmup_invocations_per_hour * memory_per_lambda_function_gb
#how many lambda functions do I need to store this amount of memory?
max_num_lambdas_needed = math.ceil(max_memory_gb / memory_per_lambda_function_gb)
print(f"Target Warmup Cost: ${target_warm_up_cost}")
print(f"Max Number Lambdas Needed: {max_num_lambdas_needed}")
print(f"Max Memory that can be stored GB: {max_memory_gb}")

print('*' * 40)
num_jobs = 10
num_epochs_per_job = 10
mini_batches_per_epoch = 10000

