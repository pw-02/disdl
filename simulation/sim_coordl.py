import heapq
import numpy as np
import logging
from typing import List, Tuple, Dict, Set, Any
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_cv(job_speeds):
    mean_speed = np.mean(job_speeds)
    std_dev = np.std(job_speeds)
    return std_dev / mean_speed

def generate_job_speeds(target_cv, num_jobs=4, seed=None):
    if seed is not None:
        np.random.seed(seed)

    sigma = np.sqrt(np.log(1 + target_cv ** 2))
    mu = -0.5 * sigma**2  # ensures mean = 1

    speeds = np.random.lognormal(mean=mu, sigma=sigma, size=num_jobs)
    return np.round(speeds, 3)

# def generate_job_speeds(cv, num_jobs=4, mean=1.0, seed=None):
#     if seed is not None:
#         np.random.seed(seed)
#     std_dev = cv * mean
#     speeds = np.random.normal(loc=mean, scale=std_dev, size=num_jobs)
#     return np.round(speeds, 3)  # Rounded for readability
# # import numpy as np

# def generate_job_speeds(target_cv, num_jobs=4, seed=None):
#     if seed is not None:
#         np.random.seed(seed)

#     sigma = np.sqrt(np.log(1 + target_cv ** 2))
#     mu = -0.5 * sigma**2  # ensures mean = 1

#     speeds = np.random.lognormal(mean=mu, sigma=sigma, size=num_jobs)
#     return np.round(speeds, 3)


class RedisCache:
    def __init__(self, 
                 max_cache_size_gb,
                 size_per_batch_gb, 
                 num_jobs, 
                 eviction_policy="LRU"):
        
        self.data:Dict[Any,Set] = {}  # Stores cached batches
        self.size_per_batch_gb = size_per_batch_gb
        self.num_jobs = num_jobs
        self.max_cache_size_gb = max_cache_size_gb
        self.eviction_policy = eviction_policy
        self.num_jobs = num_jobs
        self.batch_ref_count = {} #counter for the number of times a batch has been accessed
        self.cache_size_over_time = []

    def handle_request(self, key, job_id):
        cache_hit = False
        if key in self.data:
            logger.debug(f"Cache hit: Job {job_id} accessed {key}")
            self.batch_ref_count[key] = self.batch_ref_count.get(key, 0) + 1
            if self.batch_ref_count[key] >= self.num_jobs:
                self.data.pop(key, None)                 #evict the batch from the cache
            cache_hit = True
        else:
            # Check if the cache is full and apply eviction policy
            if (len(self.data) + 1) * self.size_per_batch_gb >= self.max_cache_size_gb:
                    pass
                # if self.eviction_policy == "LRU":
            else:
                # Add the new batch to the cache
                self.data[key] = set()
                logger.debug(f"Cache miss: Job {job_id} added {key} to cache")
                self.data[key].add(job_id)

                #Check if its the first time the batch is added to the cache
                if self.batch_ref_count.get(key, 0) == 0:
                    cache_hit = True
                self.batch_ref_count[key] = self.batch_ref_count.get(key, 0) + 1 #cache get(key, 1) for now to assume prefetching, but it should be 0
        self.cache_size_over_time.append(len(self.data))
        return cache_hit

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
    # return {
    #     "gb_hours": round(gb_hours, 2),
    #     "storage_cost_usd": round(storage_cost, 2),
    #     "ecpu_cost_usd": round(ecpu_cost, 2),
    #     "total_cost_usd": round(total_cost, 2)
    # }


def run_coordl_simulation(
        job_speeds,
        max_cache_size_gb,
        size_per_batch_gb,
        cache_miss_penalty,
        hourly_cache_cost,
        hourly_ec2_cost=12.24,
        simulation_time=3600,
        batches_per_job=np.inf,
        use_elasticache_severless_pricing=False):
    
    num_jobs = len(job_speeds)
    job_progress = [0] * num_jobs  # Tracks how many batches each job has completed
    event_queue = []  # Priority queue for next event times
    cache_miss_count = 0
    cache_hit_count = 0
    
    cache = RedisCache(
        max_cache_size_gb = max_cache_size_gb,
        size_per_batch_gb=size_per_batch_gb,
        num_jobs = num_jobs
    )
   
    for job_id, speed in enumerate(job_speeds):
        heapq.heappush(event_queue, (speed, job_id))  # (time to complete first batch, job_id)

    time_elapsed = 0  # Global simulation time

    while True:
        if simulation_time is not None and time_elapsed >= simulation_time:
            break
        #check all jobs have completed their batches
        if batches_per_job is not None and all(job_progress[job_id] >= batches_per_job for job_id in range(num_jobs)):
            break
        
        time_elapsed, job_id = heapq.heappop(event_queue)  # Get next job event
        current_batch = job_progress[job_id]  # Current batch for the job
        cache_hit = cache.handle_request(current_batch, job_id)  # Handle cache request
        if cache_hit:
            cache_hit_count +=1
            next_event_time = time_elapsed + (job_speeds[job_id])
        else:
            cache_miss_count += 1
            next_event_time = time_elapsed + (job_speeds[job_id]) + cache_miss_penalty
        
        job_progress[job_id] += 1
        if batches_per_job is not None and job_progress[job_id] < batches_per_job:
            heapq.heappush(event_queue, (next_event_time, job_id))
        else:
            heapq.heappush(event_queue, (next_event_time, job_id))

    
    total_batches_processed = sum(job_progress)
    throughput = total_batches_processed / time_elapsed  # Batches per second
    compute_cost = (hourly_ec2_cost / 3600) * time_elapsed


    # cache_cost = hourly_cache_cost
    cache_hit_percent = (cache_hit_count / (cache_hit_count + cache_miss_count)) * 100 if (cache_hit_count + cache_miss_count) > 0 else 0
    max_cached_bacthes = max(cache.cache_size_over_time)
    max_cache_capacity_used = max_cached_bacthes * size_per_batch_gb
    if use_elasticache_severless_pricing:
        cache_cost = calculate_elasticache_serverless_cost(average_gb_usage=max_cache_capacity_used)
    else:
        cache_cost = (hourly_cache_cost / 3600) * time_elapsed
    total_cost = compute_cost + cache_cost  # No additional costs in this simulation

    #resote these results in a dict to be used in the report
    results = {
        'cache_miss_count': cache_miss_count,
        'cache_hit_count': cache_hit_count,
        'cache_hit_percent': cache_hit_percent,
        'total_batches_processed': total_batches_processed,
        'time_elapsed': time_elapsed,
        'throughput': throughput,
        'compute_cost': compute_cost,
        'cache_cost': cache_cost,
        'total_cost': total_cost,
        'max_cache_capacity_used_gb': max_cache_capacity_used,
        'max_cached_bacthes': max_cached_bacthes
    }
    
    #print some results
    print(f"CoorDL:")
    print(f"  Time: {time_elapsed}")
    print(f"  Cache Size: {max_cache_size_gb} GB")
    print(f"  Cache Used: {max_cache_capacity_used} GB")
    # print(f"  Max Cached Batches: {max_cached_bacthes} GB")
    print(f"  Cache Miss Count: {cache_miss_count}")
    print(f"  Cache Hit Count: {cache_hit_count}") 
    print(f"  Cache Hit Percentage: {cache_hit_percent:.2f}%")
    print(f"  Total Batches Processed: {total_batches_processed}")
    print(f"  Elapsed Time: {time_elapsed}s, {time_elapsed/60:.2f} min")
    print(f"  Overall Throughput: {throughput:.2f} batches/sec")
    print(f"  Cache Cost: ${cache_cost:.2f}")
    print(f"  Compute Cost: ${compute_cost:.2f}")
    print(f"  Total Cost : ${total_cost:.2f}")
    print("-" * 40)
    return results
   


def cache_requitmenest_for_job_speed_disparity():
   # Define simulation parameters
    job_speeds_list = [
        [1.01, 1.01, 1.01, 1.01],  # Very Low Variability
        [1.01, 0.99, 1.00, 1.02],  # Very Low Variability
        [1.05, 0.97, 1.02, 0.98],  # Low Variability
        [0.9, 1.0, 1.1, 1.0],      # Mild Variability
        [0.8, 0.9, 1.2, 1.1],      # Moderate Variability
        # [0.6, 0.8, 1.4, 1.2],      # Medium-High Variability
        [0.5, 0.7, 1.5, 1.2],      # High Variability
        [0.3, 0.6, 2.0, 1.5],      # Very High Variability
        [0.2, 0.5, 2.5, 2.0],      # Extreme Variability
        [0.1, 0.4, 2.8, 2.2],      # Ultra-Extreme Variability
        [0.05, 0.3, 3.5, 2.5],      # Maximum Variability
        # [0.05, 0.3, 3.5, 4.5]      # Maximum Variability
    ]

    #update te jobs lists sso that all entired are the same, and match the highest value in the list
    # for i in range(len(job_speeds_list)):
    #     max_value = max(job_speeds_list[i])
    #     job_speeds_list[i] = [max_value] * len(job_speeds_list[i])

    # Example usage:
    # target_cvs = [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.3, 1.7, 2.0]
    # job_speeds_list = [generate_job_speeds(cv, num_jobs=10, seed=i) for i, cv in enumerate(target_cvs)]

    simulation_time = 3600  # Simulate 1 hour
    num_epochs = 100
    batches_per_epoch = 10000
    max_batches_per_job = None #batches_per_epoch * num_epochs
    hourly_ec2_cost = 12.24  # Example: $3 per hour for an EC2 instance
    hourly_redis_cache_cost = 3.25
    redis_cache_size_gb = 100 #np.inf
    size_per_batch_gb = 20 / 1024
    cache_miss_penalty = 0
    use_elasticache_severless_pricing = True
    cvs = []
    cache_capcity_requires = []
    cache_costs = []
    cache_hit_percentages = []
    # job_speeds_list = job_speeds_list * 2
    for idx, job_speeds in enumerate(job_speeds_list):
        #sanity check that the cv fucntion is working
        # cv = calculate_cv(job_speeds)
        # if cv != target_cvs[idx]:
        #     logger.error(f"CV mismatch: Expected {target_cvs[idx]}, but got {cv} for job speeds {job_speeds}")
        #     continue
        # Run the simulation with the given parameters
        cv = calculate_cv(job_speeds)
        logger.info(f"Running simulation with CV: {cv:.2f}")
        logger.info(f"Job Speeds: {job_speeds}")
        results = {'job_speeds': job_speeds, 'cv': cv}
        sim_results = run_coordl_simulation(
            job_speeds = job_speeds,
            max_cache_size_gb=redis_cache_size_gb,
            size_per_batch_gb = size_per_batch_gb,
            cache_miss_penalty = cache_miss_penalty,
            hourly_cache_cost = hourly_redis_cache_cost,
            simulation_time=simulation_time,
            batches_per_job=max_batches_per_job,
            use_elasticache_severless_pricing = use_elasticache_severless_pricing
        )
        results.update(sim_results)
        cvs.append(cv)
        cache_capcity_requires.append(sim_results['max_cache_capacity_used_gb'])
        cache_costs.append(sim_results['cache_cost'])
        cache_hit_percentages.append(sim_results['cache_hit_percent'])

    # #compuet potential aggregated throughput
    # aggregated_throughput = 0
    # for joblist in job_speeds_list:
    #     for speed in joblist:
    #         #time to compute jobs per epoch
    #         time_to_fiinsh = speed * batches_per_epoch
    #         tp = batches_per_epoch / time_to_fiinsh
    #         aggregated_throughput.append(tp)

    #     aggregated_throughput += sum(job_speeds_list[i]) / len(job_speeds_list[i])

    # logger.info(f"Aggregated Throughput: {aggrehated_throughput:.2f} batches/sec")
    #plot the results
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cvs, cache_capcity_requires, marker='o', linestyle='-', color='b')

    # Set larger font sizes
    ax.set_xlabel('Coefficient of Variation (CV)', fontsize=14)
    ax.set_ylabel('Cache Capacity Usage (GB)', fontsize=14)

    # Increase tick label font size
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Configure axis tick formatting
    ax.xaxis.set_major_locator(mticker.AutoLocator())
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    

   # Convert CVs to strings for labels
    labels = [f"{cv:.2f}" for cv in cvs]
    x = np.arange(len(cvs))  # Evenly spaced x positions

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, cache_hit_percentages, color='b', width=0.6)

    # Set x-tick labels to actual CV values
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)

    # Set axis labels with larger font
    ax.set_xlabel('Coefficient of Variation (CV)', fontsize=14)
    ax.set_ylabel('Cache Hit Percentage (%)', fontsize=14)

    # Adjust tick font sizes
    ax.tick_params(axis='y', labelsize=12)

    plt.tight_layout()
    plt.show()

    # #also plot the cache cost vs cv
    # fig, ax = plt.subplots(figsize=(6, 4))
    # ax.plot(cvs, cache_costs, marker='o', linestyle='-', color='b')

    # ax.set_xlabel('Coefficient of Variation (CV)')
    # ax.set_ylabel('Hourly Cache Cost ($)')
    # # ax.set_title('Required Cache Capacity vs. CV')

    # ax.xaxis.set_major_locator(mticker.AutoLocator())
    # ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    # ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    # #Cate a legens to shw its AWS ElastiCache Severless Pricing
    # ax.legend(['AWS ElastiCache']) 
    # ax.grid(True)
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    cache_requitmenest_for_job_speed_disparity()
    # # Define simulation parameters
    # job_speeds = [1.01, 0.99, 1.00, 1.02]  # Example job speeds
    # simulation_time =  None #3600 * 5 # Simulate 1 hour
    # max_batches_per_job = 10000
    # hourly_ec2_cost = 12.24  # Example: $3 per hour for an EC2 instance
    # hourly_redis_cache_cost = 3.25
    # redis_cache_size_gb = 150 #np.inf
    # size_per_batch_gb = 20 / 1024
    # cache_miss_penalty = 0
    # use_elasticache_severless_pricing = False
    # run_coordl_simulation(
    #     job_speeds = job_speeds,
    #     max_cache_size_gb=redis_cache_size_gb,
    #     size_per_batch_gb = size_per_batch_gb,
    #     cache_miss_penalty = cache_miss_penalty,
    #     hourly_cache_cost = hourly_redis_cache_cost,
    #     simulation_time=simulation_time,
    #     batches_per_job=max_batches_per_job,
    #     use_elasticache_severless_pricing = use_elasticache_severless_pricing
    # )

