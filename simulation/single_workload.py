import heapq
import numpy as np
import logging
from typing import List, Tuple, Dict, Set, Any
import sys
sys.path.append(".")
from simulation.workloads import workloads, save_dict_list_to_csv,gen_report_data 
from simulation.sim_coordl import run_coordl_simulation
from simulation.sim_tensorsocket import run_tensorsocket_simualtion
import os
import csv
import time
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
    






if __name__ == "__main__":
    #print name variable name 'imagenet_128_batch_size'
    
    #compute_sim_id based on date time
    sim_id= str(int(time.time()))

    workload_name = 'imagenet_128'
    workload =  workloads[workload_name]
    simulation_time_sec = None #3600 # None  #3600 * 1 # Simulate 1 hour
    max_batches_per_job = 8500 # 8500 #np.inf
    hourly_ec2_cost = 12.24 
    hourly_cache_cost = 3.25
    cache_capacity_gb =  200
    size_per_batch_gb = 20 / 1024
    cache_miss_penalty = 0
    use_elasticache_severless_pricing = False
    cache_buffer_size = 10
    #save overall results to a file
    report_folder = os.path.join(os.getcwd(), "simulation", "reports", workload_name)
    overall_report_file = os.path.join(report_folder, "overall_results.csv")
    job_performance_file = os.path.join(report_folder, "job_results.csv")

    coordl_job_performances, cache_size_over_time = run_coordl_simulation(
        sim_id=sim_id,
        workload_name = workload_name,
        workload_jobs = workload.items(),
        cache_capacity_gb=cache_capacity_gb,
        size_per_batch_gb = size_per_batch_gb,
        cache_miss_penalty = cache_miss_penalty,
        hourly_cache_cost = hourly_cache_cost,
        simulation_time_sec=simulation_time_sec,
        batches_per_job=max_batches_per_job,
        use_elasticache_severless_pricing = use_elasticache_severless_pricing
    )

    coordl_overall_results = gen_report_data(
        dataloader_name = 'coordl',
        job_performances = coordl_job_performances,
        cache_size_over_time = cache_size_over_time,
        eviction_policy = "coordl",
        size_per_batch_gb = size_per_batch_gb,
        cache_capacity_gb = cache_capacity_gb,
        cache_miss_penalty = 0,
        hourly_ec2_cost = hourly_ec2_cost,
        hourly_cache_cost = hourly_cache_cost,
        sim_id = sim_id,
        workload_name = workload_name,
        use_elasticache_severless_pricing = use_elasticache_severless_pricing
    )
    os.makedirs(report_folder, exist_ok=True)
    save_dict_list_to_csv([coordl_overall_results], overall_report_file)
    save_dict_list_to_csv(coordl_job_performances, job_performance_file)

    ts_job_performances, cache_size_over_time = run_tensorsocket_simualtion(
        sim_id = sim_id,
        workload_name = workload_name,
        workload_jobs = workload.items(),
        cache_buffer_size = cache_buffer_size,
        cache_capacity_gb = cache_buffer_size * size_per_batch_gb,
        size_per_batch_gb = size_per_batch_gb,
        hourly_ec2_cost = hourly_ec2_cost,
        hourly_cache_cost = 0,
        simulation_time_sec=simulation_time_sec,
        batches_per_job=max_batches_per_job)
    
    ts_overall_results = gen_report_data(
        dataloader_name = 'tensorsocket',
        job_performances = ts_job_performances,
        cache_size_over_time = cache_size_over_time,
        eviction_policy = "tensorsocket",
        size_per_batch_gb = size_per_batch_gb,
        cache_capacity_gb = cache_buffer_size * size_per_batch_gb,
        cache_miss_penalty = 0,
        hourly_ec2_cost = hourly_ec2_cost,
        hourly_cache_cost = 0,
        sim_id = sim_id,
        workload_name = workload_name,
        use_elasticache_severless_pricing = use_elasticache_severless_pricing
    )

    save_dict_list_to_csv([ts_overall_results], overall_report_file)
    save_dict_list_to_csv(ts_job_performances, job_performance_file)


