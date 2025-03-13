def compute_optimal_buffer(worst_case_efficiency, total_batches_best, total_batches_worst, max_buffer_best, compute_cost, cache_cost_per_batch):
    """Computes the largest buffer size while maintaining better cost efficiency than the worst case."""

    # Estimate N(B) using linear scaling
    def N_B(B):
        return total_batches_worst + (total_batches_best - total_batches_worst) * (B / max_buffer_best)

    # Solve for B where cost efficiency remains better than the worst-case scenario
    B_optimal = (N_B(max_buffer_best) - compute_cost * worst_case_efficiency) / (worst_case_efficiency * cache_cost_per_batch)
    
    return min(B_optimal, max_buffer_best)  # Ensure it does not exceed best case

# Given values from previous runs
worst_case_efficiency = 7842.81  # Worst-case batches per dollar
total_batches_best = 251987  # Best-case total batches
total_batches_worst = 95996  # Worst-case total batches
max_buffer_best = 95995  # Best-case buffer size
compute_cost = 12.24  # Compute cost per hour
cache_cost_per_batch = 0.001  # 10MB per batch, $0.10 per GB per hour

# Compute the optimal buffer size
optimal_buffer = compute_optimal_buffer(worst_case_efficiency, total_batches_best, total_batches_worst, max_buffer_best, compute_cost, cache_cost_per_batch)
print(f"Optimal Buffer Size: {optimal_buffer:.2f} batches")
print(f"Optimal Cache Capacity: {optimal_buffer * 10 / 1000:.2f} GB")  # Convert to GB
