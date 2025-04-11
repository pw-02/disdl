import heapq
import numpy as np

def run_tensorsocket_simualtion(
        job_speeds,
        buffer_size,
        hourly_ec2_cost=12.24,
        simulation_time=3600,
        max_batches_per_job=np.inf):
    
    num_jobs = len(job_speeds)
    job_progress = [0] * num_jobs  # Tracks how many batches each job has completed
    event_queue = []  # Priority queue for next event times
    max_distance_between_jobs = 0
    next_batch_is_cahe_miss = {}
    cache_miss_count = 0
    cache_hit_count = 0

     # Initialize event queue with first batch completion times
    for job_id, speed in enumerate(job_speeds):
        heapq.heappush(event_queue, (speed, job_id))  # (time to complete first batch, job_id)

    time_elapsed = 0  # Global simulation time

    while True:
        
        if simulation_time is not None and time_elapsed >= simulation_time:
            break
        if max_batches_per_job is not None and all(progress >= max_batches_per_job for progress in job_progress):
            break
        
        time_elapsed, job_id = heapq.heappop(event_queue)  # Get next job event

        # Check buffer constraint
        slowest_progress = min(job_progress)
        distance_to_slowest = job_progress[job_id] - slowest_progress
        max_distance_between_jobs = max(max_distance_between_jobs, distance_to_slowest)
        if distance_to_slowest >= buffer_size:
            # Job must wait, but estimate a smarter retry time
            next_batch_is_cahe_miss[job_id] = True
            target_distance = buffer_size - 1
            progress_gap = distance_to_slowest - target_distance
            catch_up_rate = min(job_speeds)  # slowest job's time per batch
            estimated_catch_up_time = progress_gap * catch_up_rate
            retry_time = time_elapsed + estimated_catch_up_time
            heapq.heappush(event_queue, (retry_time, job_id))
            continue  # Skip this job for now
        
        # Process batch completion
        job_progress[job_id] += 1
        # Schedule the next batch completion if still within time limit
        if job_id in next_batch_is_cahe_miss and next_batch_is_cahe_miss[job_id] == True:
            cache_miss_count += 1
            next_batch_is_cahe_miss[job_id] = False
        else:
            cache_hit_count += 1
        next_event_time = time_elapsed + (job_speeds[job_id])
        heapq.heappush(event_queue, (next_event_time, job_id))
       
    total_batches_processed = sum(job_progress)
    throughput = total_batches_processed / time_elapsed  # Batches per second
    compute_cost = (hourly_ec2_cost / 3600) * time_elapsed
    total_cost = compute_cost  # No additional costs in this simulation
    cache_hit_percent = (cache_hit_count / (cache_hit_count + cache_miss_count)) * 100 if (cache_hit_count + cache_miss_count) > 0 else 0
    #print some results
    print(f"TensorSocket:")
    print(f"  Buffer Size: {buffer_size} batches")
    print(f"  Max Distance Between Jobs: {max_distance_between_jobs} batches")
    print(f"  Cache Miss Count: {cache_miss_count}")
    print(f"  Cache Hit Count: {cache_hit_count}") 
    print(f"  Cache Hit Percentage: {cache_hit_percent:.2f}%")
    print(f"  Total Batches Processed: {total_batches_processed}")
    print(f"  Elapsed Time: {time_elapsed}s, {time_elapsed/60:.2f} min")
    print(f"  Overall Throughput: {throughput:.2f} batches/sec")
    print(f"  Compute Cost: ${compute_cost:.2f}")
    print(f"  Total Cost : ${total_cost:.2f}")
    print("-" * 40)


if __name__ == "__main__":

   # Define simulation parameters
    job_speeds = [0.137222914, 0.14272167, 0.351509787, 0.519805225]  # Speeds in batches per second
    simulation_time =  3600 * 1 # Simulate 1 hour
    hourly_ec2_cost = 12.24  # Example: $3 per hour for an EC2 instance
    
    run_tensorsocket_simualtion(
        job_speeds = job_speeds,
        buffer_size= 10,  # Example buffer size in batches
        hourly_ec2_cost=hourly_ec2_cost,
        simulation_time=3600,
        max_batches_per_job=None)
    
    
    