import numpy as np
import matplotlib.pyplot as plt

def pareto_frontier(costs, throughputs, dataset_sizes):
    """Finds the Pareto frontier points (highest throughput for lowest cost) and keeps dataset sizes."""
    sorted_indices = np.argsort(costs)  # Sort by cost
    pareto_costs, pareto_throughputs, pareto_sizes = [], [], []

    max_throughput = -np.inf
    for idx in sorted_indices:
        if throughputs[idx] > max_throughput:  # Pareto condition
            pareto_costs.append(costs[idx])
            pareto_throughputs.append(throughputs[idx])
            pareto_sizes.append(dataset_sizes[idx])
            max_throughput = throughputs[idx]

    return np.array(pareto_costs), np.array(pareto_throughputs), np.array(pareto_sizes)

# Figure settings
tensorsocket_label = "TensorSocket"
disdl_label = "DisDL"
line_width = 5.5
font_size = 30

visual_map_plot = {
    tensorsocket_label: {'color': '#F39D00', 'linestyle': '-', 'marker':'o', 'linewidth': line_width, 'alpha': 1.0}, 
    disdl_label: {'color': 'black', 'linestyle': '-', 'marker':'s', 'linewidth': line_width, 'alpha': 1.0}, 
}

# Workload data
workload = {
    "IncreasingDataSize": {
        "dataset_sizes": np.array([8564, 16000, 32000, 64000, 128000]),
        disdl_label: {"costs": np.array([10.405, 20.81,31.215, 38.14, 52.025]), "throughputs": np.array([2436, 2436, 2436, 2436, 2436])},
        tensorsocket_label: {"costs": np.array([15.26, 30.51, 45.77, 61.03, 76.28]), "throughputs": np.array([973.9, 973.9,973.9,973.9, 973.9])}
    },
    "IncreasingNumJobs": {
        "num_jobs": np.array([4,8,12,16]),
        disdl_label: {"costs": np.array([4, 8, 12, 16]), "throughputs": np.array([488.59, 973.9, 2436, 2436])},
        tensorsocket_label: {"costs": np.array([15.26, 30.51, 45.77, 61.03]), "throughputs": np.array([973.9,2245.34,2931.51,3908.68])}
    },
    "IncreasingVariability": {
        "variabilities": np.array([0.1, 0.2, 0.4, 0.8, 1.6]),
        disdl_label: {"costs": np.array([8, 5, 9, 20, 35]), "throughputs": np.array([2436, 2436, 2436, 2436, 2436])},
        tensorsocket_label: {"costs": np.array([18, 36, 72, 144, 288]), "throughputs": np.array([973.9, 973.9, 973.9, 973.9, 973.9])}
    },
      "IncreasingBatchSize": {
        "batch_sizes": np.array([32, 64, 128, 256, 512]),
        disdl_label: {"costs": np.array([8, 5, 9, 20, 35]), "throughputs": np.array([2436, 2436, 2436, 2436, 2436])},
        tensorsocket_label: {"costs": np.array([18, 36, 72, 144, 288]), "throughputs": np.array([973.9, 973.9, 973.9, 973.9, 973.9])}
    }
}

for workload_name, workload_data in workload.items():
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))  # Two subplots

    xaxis_values = workload_data[list(workload_data.keys())[0]]
    tensorsocket_costs = workload_data[tensorsocket_label]["costs"]
    tensorsocket_throughputs = workload_data[tensorsocket_label]["throughputs"]
    disdl_costs = workload_data[disdl_label]["costs"]
    disdl_throughputs = workload_data[disdl_label]["throughputs"]

    # First subplot: Cost vs Dataset Size
    axs[0].plot(xaxis_values, tensorsocket_costs, label=tensorsocket_label, **visual_map_plot[tensorsocket_label])
    axs[0].plot(xaxis_values, disdl_costs, label=disdl_label, **visual_map_plot[disdl_label])
    
    axs[0].set_xlabel("Batches/Epoch", fontsize=font_size)
    axs[0].set_ylabel("Cost ($)", fontsize=font_size)
    axs[0].set_xscale("log")
    axs[0].set_xticks(xaxis_values)
    if min(xaxis_values) > 1000:
        axs[0].set_xticklabels([f"{size//1000}K" for size in xaxis_values])
    else:
        axs[0].set_xticklabels([f"{size}" for size in xaxis_values])
    # axs[0].set_xticklabels([f"{size//1000}K" for size in xaxis_values])
    axs[0].legend(fontsize=font_size, loc="upper left")
    axs[0].grid(True, linestyle="--", linewidth=0.5)
    
    # Second subplot: Throughput vs Cost
    axs[1].plot(xaxis_values, tensorsocket_throughputs, label=tensorsocket_label, **visual_map_plot[tensorsocket_label])
    axs[1].plot(xaxis_values, disdl_throughputs, label=disdl_label, **visual_map_plot[disdl_label])
    axs[1].set_xlabel("Batches/Epoch", fontsize=font_size)
    axs[1].set_xscale("log")
    axs[1].set_xticks(xaxis_values)
    if min(xaxis_values) > 1000:
        axs[1].set_xticklabels([f"{size//1000}K" for size in xaxis_values])
    else:
        axs[1].set_xticklabels([f"{size}" for size in xaxis_values])
    axs[1].set_ylabel("Throughput (samples/s)", fontsize=font_size)
    axs[1].legend(fontsize=font_size, loc="center left")
    axs[1].grid(True, linestyle="--", linewidth=0.5)

    axs[0].tick_params(axis='both', which='major', labelsize=font_size)
    axs[1].tick_params(axis='both', which='major', labelsize=font_size)

    plt.tight_layout()
    plt.show()
