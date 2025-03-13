import numpy as np
import math
def compute_wasted_gpu_cost_syncronous_jobs(job_speeds, batches_per_job, gpu_cost_per_hour):
    #p3.2xlarge $3.06 per hour
    # Simulation parameters
    job_batches_per_sec = [1/speed for speed in job_speeds]  # Batches per hour per job
    num_jobs = len(job_speeds)
    total_batches = batches_per_job * num_jobs  # Total batches for all jobs
    gpu_cost_per_sec = gpu_cost_per_hour / 3600  # Convert to cost per second

    #compute best-case scenario
    best_case_throughput = math.floor(sum(job_batches_per_sec))  #Batches per hour
    worst_case_throughput = math.floor(min(job_batches_per_sec) * num_jobs)  #Batches per hour

    best_case_total_time_sec = total_batches / best_case_throughput
    worst_case_total_time_sec = total_batches / worst_case_throughput
    wasted_time_sec = worst_case_total_time_sec - best_case_total_time_sec

    # Compute total wasted cost
    print(f"Best-Case - Compute Time: {best_case_total_time_sec:.0f} sec, Cost: ${best_case_total_time_sec * gpu_cost_per_sec:,.2f}")
    print(f"Worst-Case Compute Time: {worst_case_total_time_sec:.0f} sec", f"Cost: ${worst_case_total_time_sec * gpu_cost_per_sec:,.2f}")
    print(f"Wasted - Time: {wasted_time_sec:.0f} sec, Cost: ${wasted_time_sec * gpu_cost_per_sec:,.2f}")

if __name__ == "__main__":
    job_speeds = [0.1,0.25,0.09,0.3]  # Fixed batch processing times per job (seconds per batch)
    batches_per_job = 10000  # Total batches per job
    buffer_size = 0  # Buffer size (batches)
    gpu_cost_per_hour = 12.24  # Example: $3 per hour for an EC2 instance
    
    compute_wasted_gpu_cost_syncronous_jobs(job_speeds, batches_per_job, gpu_cost_per_hour)
    print("-------------------")
