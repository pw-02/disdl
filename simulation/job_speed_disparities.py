from sim_coordl import run_coordl_simulation
import logging
import sys
import numpy as np
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


if __name__ == "__main__":
    # Define simulation parameters
    job_speeds_list = [
        [1.01, 1.01, 1.01, 1.01],  # Very Low Variability
        [1.01, 0.99, 1.00, 1.02],  # Very Low Variability
        [1.05, 0.97, 1.02, 0.98],  # Low Variability
        [0.9, 1.0, 1.1, 1.0],      # Mild Variability
        [0.8, 0.9, 1.2, 1.1],      # Moderate Variability
        [0.5, 0.7, 1.5, 1.2],      # High Variability
        [0.3, 0.6, 2.0, 1.5],      # Very High Variability
        [0.2, 0.5, 2.5, 2.0],      # Extreme Variability
        [0.1, 0.4, 2.8, 2.2],      # Ultra-Extreme Variability
        [0.05, 0.3, 3.5, 2.5],      # Maximum Variability
    ]

    
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
    create_plot = False
    # job_speeds_list = job_speeds_list * 2
    for idx, job_speeds in enumerate(job_speeds_list):
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
    if create_plot:
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





