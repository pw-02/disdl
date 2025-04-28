import math
from itertools import chain

# Helper function to compute factors of a number
def factors(n):
    return set(i for i in range(1, n + 1) if n % i == 0)

# Function to compute the number of blocks and fragmentation penalty
def num_blocks_and_penalty(job_batch_size, B_j, penalty_factor):
    num_blocks = math.ceil(job_batch_size / B_j)
    penalty = penalty_factor if job_batch_size % B_j != 0 else 0
    return num_blocks, penalty

# Function to calculate total overhead for a given minibatch size B_j
def calculate_overhead(job_batch_sizes, B_j, penalty_factor):
    total_overhead = 0
    for b_i in job_batch_sizes:
        num_blocks, penalty = num_blocks_and_penalty(b_i, B_j, penalty_factor)
        total_overhead += num_blocks + penalty
    return total_overhead

# Function to find the common factors among all job batch sizes
def common_factors(job_batch_sizes):
    # Find the factors of the first batch size
    common_factors_set = factors(job_batch_sizes[0])
    # Compute intersection of factors across all batch sizes
    for batch_size in job_batch_sizes[1:]:
        common_factors_set &= factors(batch_size)
    return common_factors_set

# Main function to find the optimal minibatch size
def find_optimal_minibatch_size(job_batch_sizes, penalty_factor=1000):
    # Step 1: Find common factors among all batch sizes
    common_factors_set = common_factors(job_batch_sizes)
    
    # Step 2: Calculate total overhead for each factor size
    best_B = None
    min_overhead = float('inf')
    for B_j in common_factors_set:
        total_overhead = calculate_overhead(job_batch_sizes, B_j, penalty_factor)
        print(f"Minibatch size {B_j}: Total Overhead = {total_overhead}")
        if total_overhead < min_overhead:
            min_overhead = total_overhead
            best_B = B_j
    
    return best_B

# Example usage
job_batch_sizes = [32, 64, 128]  # Example job batch sizes
penalty_factor = 1000  # Penalty for non-divisible batches (adjust as needed)

optimal_B = find_optimal_minibatch_size(job_batch_sizes, penalty_factor)
print(f"Optimal minibatch size for caching: {optimal_B}")
