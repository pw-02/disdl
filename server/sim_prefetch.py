import math

def compute_lambda_concurrency(gpu_batch_time, lambda_processing_time, job_load_time, enable_lambda_prefetch=True):
    gpu_batch_rate = 1 / gpu_batch_time
    job_load_rate = 1 / job_load_time

    if enable_lambda_prefetch:
        required_prefetch_rate = gpu_batch_rate - job_load_rate
        if required_prefetch_rate <= 0:
            return 0, 0  # No additional concurrency needed, no delay

        concurrency = math.ceil(required_prefetch_rate * lambda_processing_time)
        actual_prefetch_rate = concurrency / lambda_processing_time  
    else:
        concurrency = 0
        actual_prefetch_rate = 0  

    total_load_rate = job_load_rate + actual_prefetch_rate  
    total_delay = max(0, (gpu_batch_rate - total_load_rate) * gpu_batch_time)  

    return concurrency, total_delay

# Define a list of different GPU speeds, Lambda processing times, and job load times
simulations = [
    {"gpu_batch_time": 0.2, "lambda_processing_time": 1.5, "job_load_time": 1.0},
    {"gpu_batch_time": 0.3, "lambda_processing_time": 2.0, "job_load_time": 1.2},
    {"gpu_batch_time": 0.15, "lambda_processing_time": 1.0, "job_load_time": 0.8},
    {"gpu_batch_time": 0.25, "lambda_processing_time": 2.5, "job_load_time": 1.5},
]

print("\nSimulation Results:")
print("-" * 60)
print(f"{'GPU Batch Time':<15}{'Lambda Proc Time':<18}{'Job Load Time':<15}{'Concurrency':<12}{'Delay (s)':<10}")
print("-" * 60)

for sim in simulations:
    concurrency, delay = compute_lambda_concurrency(
        sim["gpu_batch_time"], sim["lambda_processing_time"], sim["job_load_time"], enable_lambda_prefetch=True
    )
    print(f"{sim['gpu_batch_time']:<15}{sim['lambda_processing_time']:<18}{sim['job_load_time']:<15}{concurrency:<12}{delay:.4f}")

print("\nDone!")
