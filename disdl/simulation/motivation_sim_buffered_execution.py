import heapq
import math
import numpy as np
import matplotlib.pyplot as plt

def compute_optimal_buffer(worst_case_efficiency, total_batches_best, total_batches_worst, max_buffer_best, compute_cost, cache_cost_per_gb):
    """Computes the largest buffer size while maintaining better cost efficiency than the worst case."""
    
    # Estimate N(B) as a linear interpolation between worst-case and best-case
    def N_B(B):
        return total_batches_worst + (total_batches_best - total_batches_worst) * (B / max_buffer_best)
    
    # Solve for B where cost efficiency remains better than the worst-case scenario
    B_optimal = (N_B(max_buffer_best) - compute_cost * worst_case_efficiency) / (worst_case_efficiency * cache_cost_per_gb)
    
    return min(B_optimal, max_buffer_best)  # Ensure it does not exceed best case

def calc_ec2_compute_cost(duarion_seconds, instance_cost_per_hour = 12.24):
    """Compute the total EC2 compute cost for a given duration and instance cost."""
    instacne_cost_per_second = instance_cost_per_hour / 3600
    return duarion_seconds * instacne_cost_per_second

def compute_prefetch_lambda_requests_cost(total_requests, cost_per_request =0.00012546):
    return total_requests * cost_per_request

def compute_lambda_requests_cost(total_requests, simulation_time_seconds, cost_per_request =0.000010017):
    """
    Compute the total AWS Lambda requests and request cost.
    
    :param total_batches: Total number of batches processed
    :param batch_size: Size of each batch (number of samples per batch)
    :param cost_per_million_requests: Cost per million AWS Lambda requests (default: $0.20)
    :return: (total_requests, lambda_request_cost)
    """
    proxy_cost_per_hour = 0.4
    proxy_cost_per_second = proxy_cost_per_hour / 3600
    lambda_request_cost = total_requests * cost_per_request
    proxy_cost = simulation_time_seconds * proxy_cost_per_second
    return lambda_request_cost + proxy_cost

def fill_range(int_list):
    """Generate a list of all integers between the min and max values in int_list."""
    if not int_list:
        return []
    
    min_val = min(int_list)
    max_val = max(int_list)
    
    return list(range(min_val, max_val + 1))


# Simulate buffered execution of jobs
def simulate_jobs_with_buffer(job_speeds, buffer_size, bacthes_per_job):
    num_jobs = len(job_speeds)
    job_progress = [0] * num_jobs  # Tracks how many batches each job has completed
    event_queue = []  # Priority queue for next event times
    largest_distance_to_slowest = 0
    lambda_get_request_count = 0
    # job_speeds.append(5) # the last job is the prefetch lambda function that warms up cached functions
    # # warm_up_job = len(job_speeds) - 1
    # job_speeds.append(1) # the second to last job is the lambda function that fetches data from the cache
    # compute_efficiency_job = len(job_speeds) - 1
    last_invocation_time = {}
    last_job_completion_time = 0
 
   
    # Initialize event queue with first batch completion times
    for job_id, speed in enumerate(job_speeds):
        heapq.heappush(event_queue, (speed, job_id))  # (time to complete first batch, job_id)

    time_elapsed = 0  # Global simulation time

    #check if all jobs have completed processing bacthes per job
    while not all([progress >= bacthes_per_job for progress in job_progress]):
        time_elapsed, job_id = heapq.heappop(event_queue)  # Get next job event

        # if job_id == compute_efficiency_job:
        #     total_batches_processed = sum(job_progress)
        #     #total cost so far
        #     lambda_get_cost = compute_lambda_requests_cost(lambda_get_request_count, time_elapsed)
        #     lambda_prefetch_cost = compute_prefetch_lambda_requests_cost(max(job_progress))
        #     compute_cost = calc_ec2_compute_cost(time_elapsed)
        #     total_cost = compute_cost + lambda_get_cost + lambda_prefetch_cost
        #     cost_efficiency = total_batches_processed / total_cost
        #     cost_efficiency_over_time[time_elapsed] = cost_efficiency
        #     next_event_time = time_elapsed + (job_speeds[job_id])
        #     heapq.heappush(event_queue, (next_event_time, job_id))
        #     continue
        # #here is the warm up part to keep data in cache until it is used
        # if job_id == warm_up_job:
        #     #also compute current cost efficiency

        #     filled_list = fill_range(job_progress)
        #     for i in filled_list:
        #         if i not in last_invocation_time:
        #             last_invocation_time[i] = time_elapsed
        #         elif i in last_invocation_time and (time_elapsed - last_invocation_time[i]) > 60: #1 minute
        #             lambda_get_request_count += 1
        #             last_invocation_time[i] = time_elapsed
        #         else:
        #             continue
        #     next_event_time = time_elapsed + (job_speeds[job_id])
        #     heapq.heappush(event_queue, (next_event_time, job_id))
        #     continue

        # Ensure buffer constraint is met before progressing
        slowest_progress = min(job_progress)
        distance_to_slowest = job_progress[job_id] - slowest_progress
        largest_distance_to_slowest = max(largest_distance_to_slowest, distance_to_slowest)
        if distance_to_slowest >= buffer_size:
            # Job is ahead of the buffer, so it must wait
            heapq.heappush(event_queue, (time_elapsed + 0.01, job_id))  # Re-check soon
            continue  # Skip progressing this job for now

        # Process batch completion
        job_progress[job_id] += 1

        if job_progress[job_id] in last_invocation_time:
             last_invocation_time[job_progress[job_id]] = time_elapsed

        lambda_get_request_count += 1
        # Schedule the next batch completion if still within time limit
        next_event_time = time_elapsed + (job_speeds[job_id])

        if job_progress[job_id] != bacthes_per_job:
            heapq.heappush(event_queue, (next_event_time, job_id))
        else:
            #compute throuhgput for job
            job_throuhgputput = (job_progress[job_id] / (next_event_time)) * 128
            job_cost = calc_ec2_compute_cost(next_event_time) / 4
            print(f"Job {job_id} completed {job_progress[job_id]} batches in {next_event_time:.2f} seconds. Throughput: {job_throuhgputput:.2f} samples/sec. Cost: ${job_cost:.2f}")
            

            last_job_completion_time = time_elapsed

    total_batches_processed = sum(job_progress)
    throughput = total_batches_processed / last_job_completion_time  # Batches per second
    # #plot cost efficiency over time
    # plt.plot(cost_efficiency_over_time.values())
    # plt.xlabel('Time (s)')
    # plt.ylabel('Cost Efficiency (batches/s)')
    # plt.show()
    return time_elapsed, throughput, job_progress, largest_distance_to_slowest
# 

def run_simulation_case(case_name, 
                        job_speeds, 
                        buffer_size,
                        batches_per_job, 
                        hourly_ec2_cost=12.24, 
                        hourly_cache_cost_per_gb=0.10, 
                        batch_size_mb=10,
                        files_per_batch=128):
    """Runs a simulation and prints the results for a given scenario."""
    simulation_time, throughput, job_progress, max_buffer_size = simulate_jobs_with_buffer(job_speeds, buffer_size, batches_per_job)

    # Assume each batch in the buffer needs 10MB of storage
    cache_size_gb = (max_buffer_size * batch_size_mb) / 1024  # Convert MB to GB
    # Compute cost per second
    compute_cost = (hourly_ec2_cost / 3600) * simulation_time
    
    if buffer_size > 1:
        redis_cost = (hourly_cache_cost_per_gb * cache_size_gb) * (simulation_time / 3600)
    else:
        redis_cost = 0

    total_batches= batches_per_job * len(job_speeds)
    toal_samples = total_batches * files_per_batch
    samples_per_second = throughput * files_per_batch
    total_cost_with_redis = compute_cost + redis_cost
    cost_efficiency_with_redis = total_batches / total_cost_with_redis  # Batches per dollar

    print(f"{case_name}:")
    print(f"  Total Batches Processed: {total_batches}")
    print(f"  Overall Throughput: {samples_per_second:.2f} smaples/sec")

    print(f"  Overall Throughput: {throughput:.2f} batches/sec")
    print(f"  Max Buffer Size Used: {max_buffer_size} batches")
    print(f"  Cache Size Used: {cache_size_gb:.2f} GB")
    print(f"  Compute Cost: ${compute_cost:.2f}")
    print(f"  Redis Cache Cost: ${redis_cost:.2f}. Per Job: ${redis_cost / len(job_speeds):.2f}")
    print(f"  Total Cost (with Redis): ${total_cost_with_redis:.2f}")
    print(f"  Cost Efficiency (with Redis): {cost_efficiency_with_redis:.2f} batches per dollar")
    print("-" * 40)

    return samples_per_second, cost_efficiency_with_redis, total_cost_with_redis, cache_size_gb, max_buffer_size


if __name__ == "__main__":


    '''This script simulates the execution of multiple jobs with different speeds and buffer sizes.
    It serves as motivation for my work as it shows how caching bacthes can be used to imporve throughput, yet it comes with a cost. 
    When using Redis there is a sinfificant reduction in cost efficiency to achive higher throughput. With AWS Lambda the cost is 
    much lower but there is still a reduction in cost efficiency. If we find a way to limit the number of requests to the cache we can
    achive a higher cost efficiency. This is the motivation for my work to find a way to limit the number of requests to the cache
    while still achiving high throughput. Note some organization may only care about throughput and not cost efficiency. While others
    may care about cost efficiency and not throughput.'''

   # Define simulation parameters
    throughputs = []
    costs = []
    num_jobs = [4]
    job_speed = 0.523961767  # Speed of each job in batches per second
    inatcnes_cost_per_hour = 12.24
    for jobs in num_jobs:
        job_speeds = [0.137222914, 0.14272167, 0.351509787, 0.519805225]  
        print(len(job_speeds))
        batches_per_job = 8564  # Number of batches to process per job
        hourly_ec2_cost = inatcnes_cost_per_hour 
        hourly_cache_cost_per_gb = 0.125  # aws serverless redis
        files_per_batch = 37  # Size of each batch in MB
        lambda_cost_per_get_request = 0.000010017  # Cost per AWS Lambda request
        lambda_cost_per_prefetch_request = 0.00012546  # Cost per AWS Lambda request
        # Worst-case scenario: All jobs are constrained by the slowest speed
        worst_case_buffer = 5000
        batch_size = 128
        worst_case_throughput, worst_case_cost_efficiency, wost_case_cost, cache_size_gb, max_buffer_size = run_simulation_case(
            "Worst-Case Scenario (No Cache)", 
            job_speeds, 
            worst_case_buffer, 
            batches_per_job, 
            hourly_ec2_cost, 
            hourly_cache_cost_per_gb, 
            files_per_batch)
        throughputs.append(worst_case_throughput)
        costs.append(wost_case_cost)
    print(throughputs)
    print(costs)
        

    # # Best-case scenario: Jobs run at their full speed (infinite buffer)
    # best_case_buffer = np.inf
    # bestcase_throughput, bestcase_cost_efficiency, best_case_cost, cache_size_gb, max_buffer_best = run_simulation_case(
    #     "Best-Case Scenario (Throughput)", 
    #     job_speeds, 
    #     best_case_buffer, 
    #     batches_per_job, 
    #     hourly_ec2_cost, 
    #     hourly_cache_cost_per_gb, 
    #     batch_size_mb,
    #     lambda_cost_per_get_request,
    #     lambda_cost_per_prefetch_request)
    
    #so by running at best scenario we can achive a speed up of x in terms of throughput and cost efficiency
    # speedup_throughput = worst_case_throughput / bestcase_throughput
    # speedup_cost_efficiency = worst_case_cost_efficiency / bestcase_cost_efficiency
