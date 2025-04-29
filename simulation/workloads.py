#based on batch size of 128 and imagenet dataset

#imagenet dataset
#batch size of 128
import os
import csv
from typing import Dict, List
import numpy as np
import time
def gen_report_data(dataloader_name, 
                    job_performances:List[Dict], 
                    cache_size_over_time:List[float], 
                    eviction_policy:str,
                    size_per_batch_gb:float,
                    cache_capacity_gb:float, 
                    cache_miss_penalty:float, 
                    hourly_ec2_cost:float, 
                    hourly_cache_cost:float, 
                    sim_id:str, 
                    workload_name:str, 
                    use_elasticache_severless_pricing:bool) -> Dict:
     # Calculate results
     #total batches processed by all jobs, sperate dict per job
    aggregated_batches_processed = sum(job['bacthes_processed'] for job in job_performances)
    aggregated_cache_hits = sum(job['cache_hit_count'] for job in job_performances)
    aggregated_cache_misses = sum(job['cache_miss_count'] for job in job_performances)
    aggregated_cache_hit_percent = (aggregated_cache_hits / (aggregated_cache_hits + aggregated_cache_misses)) * 100 if (aggregated_cache_hits + aggregated_cache_misses) > 0 else 0
    aggregated_compute_cost = sum(job['compute_cost'] for job in job_performances)
    aggregated_throuhgput = sum(job['throughput'] for job in job_performances)
    aggregated_time_sec = sum(job['elapsed_time'] for job in job_performances)
    elapsed_time_sec = max(job['elapsed_time'] for job in job_performances) if job_performances else 0

    max_cache_capacity_used = max(cache_size_over_time) if cache_size_over_time else 0
    average_cache_capacity_used = np.mean(cache_size_over_time) if cache_size_over_time else 0

    if use_elasticache_severless_pricing:
        cache_cost = calculate_elasticache_serverless_cost(average_gb_usage=average_cache_capacity_used)
    else:
        cache_cost = (hourly_cache_cost / 3600) * elapsed_time_sec

    # if dataloader_name == 'tensorsocket':
    #     cache_cost = 0

    total_cost = aggregated_compute_cost + cache_cost  # No additional costs in this simulation
    job_speeds = {job['job_id']: job['job_speed'] for job in job_performances}

    overall_results = {
        'sim_id': sim_id,
        'workload_name': workload_name,
        'job_speeds': job_speeds,
        'dataloader': dataloader_name,
        'cache_capacity': cache_capacity_gb,
        'cache_eviction_policy': eviction_policy,
        'size_per_batch': size_per_batch_gb,
        'num_jobs': len(job_performances),
        'cache_miss_penalty': cache_miss_penalty,
        'hourly_ec2_cost': hourly_ec2_cost,
        'hourly_cache_cost': hourly_cache_cost,
        'max_cache_capacity_used': max_cache_capacity_used,
        'average_cache_capacity_used': average_cache_capacity_used,
        'cache_hit_count': aggregated_cache_hits,
        'cache_miss_count': aggregated_cache_misses,
        'cache_hit_percent': aggregated_cache_hit_percent,
        'total_batches_processed': aggregated_batches_processed,
        'time_elapsed': elapsed_time_sec,
        'total_job_time': aggregated_time_sec,
        'throughput': aggregated_throuhgput,
        'compute_cost': aggregated_compute_cost,
        'cache_cost': cache_cost,
        'total_cost': total_cost,
    }
    print(f"{dataloader_name}")
    print(f"  Jobs: {job_speeds}"),
    print(f"  Time: {elapsed_time_sec:.2f} seconds")
    print(f"  Cache Size: {cache_capacity_gb} GB")
    print(f"  Cache Used: {max_cache_capacity_used:.4f} GB")
    print(f"  Cache Hit %: {aggregated_cache_hit_percent:.2f}%")
    print(f"  Total Batches Processed: {aggregated_batches_processed}")
    print(f"  Elapsed Time: {elapsed_time_sec:.2f}s, {elapsed_time_sec/60:.2f} min")
    print(f"  Overall Throughput: {aggregated_throuhgput:.2f} batches/sec")
    print(f"  Cache Cost: ${cache_cost:.2f}")
    print(f"  Compute Cost: ${aggregated_compute_cost:.2f}")
    print(f"  Total Cost : ${total_cost:.2f}")
    print("-" * 40)
    return overall_results

def calculate_elasticache_serverless_cost(
    average_gb_usage: float,
    duration_hours: float = 1,  # default to one hour
    price_per_gb_hour: float = 0.125,
    ecpu_cost: float = 0.0
) -> dict:
    gb_hours = average_gb_usage * duration_hours
    storage_cost = gb_hours * price_per_gb_hour
    total_cost = storage_cost + ecpu_cost
    return total_cost

def save_dict_list_to_csv(dict_list, output_file):
    if not dict_list:
        print("No data to save.")
        return
    headers = dict_list[0].keys()
    file_exists = os.path.isfile(output_file)
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers, delimiter=',')
        if not file_exists:
            writer.writeheader()
        for data in dict_list:
            writer.writerow(data)

workloads = {
    'imagenet_128': {
        'RESNET18': 0.104419951,
        'RESNET50': 0.33947309,
        'VGG16': 0.514980298,
        'SHUFFLENETV2': 0.062516984
    },
}


# imagenet_128_batch_size = {
#     'RESNET18': 0.104419951,
#     'RESNET50': 0.33947309,
#     'VGG16': 0.514980298,
#     'SHUFFLENETV2': 0.062516984
# }