import heapq
import math
import numpy as np


def compute_optimal_buffer(worst_case_efficiency, total_batches_best, total_batches_worst, max_buffer_best, compute_cost, cache_cost_per_gb):
    """Computes the largest buffer size while maintaining better cost efficiency than the worst case."""
    
    # Estimate N(B) as a linear interpolation between worst-case and best-case
    def N_B(B):
        return total_batches_worst + (total_batches_best - total_batches_worst) * (B / max_buffer_best)
    
    # Solve for B where cost efficiency remains better than the worst-case scenario
    B_optimal = (N_B(max_buffer_best) - compute_cost * worst_case_efficiency) / (worst_case_efficiency * cache_cost_per_gb)
    
    return min(B_optimal, max_buffer_best)  # Ensure it does not exceed best case

def compute_lambda_requests_cost(total_requests, cost_per_million_requests=0.20):
    """
    Compute the total AWS Lambda requests and request cost.
    
    :param total_batches: Total number of batches processed
    :param batch_size: Size of each batch (number of samples per batch)
    :param cost_per_million_requests: Cost per million AWS Lambda requests (default: $0.20)
    :return: (total_requests, lambda_request_cost)
    """
    # total_requests = total_batches / batch_size  # Number of Lambda invocations
    lambda_request_cost = (total_requests / 1_000_000) * cost_per_million_requests  # Cost in dollars
    lambda_request_cost = (total_requests * 0.00003125) * 0.4 # Cost in dollars
    proxy_cost = 0.108* (simulation_time / 3600)
    return lambda_request_cost + proxy_cost


# Simulate buffered execution of jobs
def simulate_jobs_with_buffer(job_speeds, buffer_size, simulation_time=3600):
    num_jobs = len(job_speeds)
    job_progress = [0] * num_jobs  # Tracks how many batches each job has completed
    event_queue = []  # Priority queue for next event times
    largest_distance_to_slowest = 0
    lambda_request_cost = 0
    job_speeds.append(5) # the last job is the prefetch lambda function that warms up cached functions
    warm_up_job = len(job_speeds) - 1
    # Initialize event queue with first batch completion times
    for job_id, speed in enumerate(job_speeds):
        heapq.heappush(event_queue, (speed, job_id))  # (time to complete first batch, job_id)

    time_elapsed = 0  # Global simulation time

    while time_elapsed < simulation_time:
        time_elapsed, job_id = heapq.heappop(event_queue)  # Get next job event

        if job_id == warm_up_job:
            # print(f"Warm Up: {(max(job_progress) - min(job_progress))}")
            lambda_request_cost +=  (max(job_progress) - min(job_progress))
            next_event_time = time_elapsed + (job_speeds[job_id])
            heapq.heappush(event_queue, (next_event_time, job_id))
            continue

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
        lambda_request_cost += 1
        # Schedule the next batch completion if still within time limit
        next_event_time = time_elapsed + (job_speeds[job_id])
        if next_event_time < simulation_time:
            heapq.heappush(event_queue, (next_event_time, job_id))
        else:
            break

    total_batches_processed = sum(job_progress)
    throughput = total_batches_processed / simulation_time  # Batches per second

    return total_batches_processed, throughput, job_progress, largest_distance_to_slowest, lambda_request_cost

def run_simulation_case(case_name, job_speeds, buffer_size, simulation_time, hourly_ec2_cost=12.24, hourly_cache_cost_per_gb=0.10, batch_size_mb=10):
    """Runs a simulation and prints the results for a given scenario."""
    total_batches, throughput, job_progress, max_buffer_size, lambda_request_cost = simulate_jobs_with_buffer(job_speeds, buffer_size, simulation_time)

    # Assume each batch in the buffer needs 10MB of storage
    cache_size_gb = (max_buffer_size * batch_size_mb) / 1024  # Convert MB to GB
    # Compute cost per second
    compute_cost = (hourly_ec2_cost / 3600) * simulation_time
    redis_cost = (hourly_cache_cost_per_gb * cache_size_gb) * (simulation_time / 3600)
    total_cost_with_redis = compute_cost + redis_cost
    cost_efficiency_with_redis = total_batches / total_cost_with_redis  # Batches per dollar
    
    lambda_cache_cost = compute_lambda_requests_cost(lambda_request_cost, simulation_time)
    total_cost_with_lambda = compute_cost + lambda_cache_cost
    cost_efficiency_with_lambda = total_batches / total_cost_with_lambda  # Batches per dollar


    
    print(f"{case_name}:")
    print(f"  Total Batches Processed: {total_batches}")
    print(f"  Overall Throughput: {throughput:.2f} batches/sec")
    print(f"  Max Buffer Size Used: {max_buffer_size} batches")
    print(f"  Cache Size Used: {cache_size_gb:.2f} GB")
    print(f"  Compute Cost: ${compute_cost:.2f}")
    print(f"  Redis Cache Cost: ${redis_cost:.2f}")
    print(f"  Total Cost (with Redis): ${total_cost_with_redis:.2f}")
    print(f"  Cost Efficiency (with Redis): {cost_efficiency_with_redis:.2f} batches per dollar")
    print(f"  Lambda Requests: {lambda_request_cost}")
    print(f"  Lambda Cache Cost: ${lambda_cache_cost:.2f}")
    print(f"  Total Cost (with Lambda): ${total_cost_with_lambda:.2f}")
    print(f"  Cost Efficiency (with Lambda): {cost_efficiency_with_lambda:.2f} batches per dollar")
    print("-" * 40)

    return throughput, cost_efficiency_with_redis, total_cost_with_redis, cache_size_gb, max_buffer_size


if __name__ == "__main__":
   # Define simulation parameters
    job_speeds = [0.1, 0.05, 0.2, 0.15]  # Speeds in batches per second
    simulation_time = 3600 * 1 # Simulate 1 hour
    hourly_ec2_cost = 12.24  # Example: $3 per hour for an EC2 instance
    hourly_cache_cost_per_gb = 0.125  # aws serverless redis
    batch_size_mb = 37  # Size of each batch in MB
    # Worst-case scenario: All jobs are constrained by the slowest speed
    worst_case_buffer = 1
    worst_case_throughput, worst_case_cost_efficiency, wost_case_cost, cache_size_gb, max_buffer_size = run_simulation_case(
        "Worst-Case Scenario (Throughput)", job_speeds, worst_case_buffer, simulation_time, hourly_ec2_cost, hourly_cache_cost_per_gb, batch_size_mb)

    # Best-case scenario: Jobs run at their full speed (infinite buffer)
    best_case_buffer = np.inf
    bestcase_throughput, bestcase_cost_efficiency, best_case_cost, cache_size_gb, max_buffer_best = run_simulation_case(
        "Best-Case Scenario (Throughput)", job_speeds, best_case_buffer, simulation_time, hourly_ec2_cost, hourly_cache_cost_per_gb, batch_size_mb)
    
    #so by running at best scenario we can achive a speed up of x in terms of throughput and cost efficiency
    speedup_throughput = worst_case_throughput / bestcase_throughput
    speedup_cost_efficiency = worst_case_cost_efficiency / bestcase_cost_efficiency
