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

#figure data
tensorsocket_label = "TensorSocket"
disdl_label = "DisDL"
line_width = 2.5
font_size = 18
bar_width = 0.35

visual_map_plot = {
    tensorsocket_label: {'color': 'red', 'linestyle': '-', 'marker':'', 'linewidth': line_width, 'alpha': 1.0}, #indianred
    disdl_label: {'color': 'black', 'linestyle': '-', 'marker':'','linewidth': line_width, 'alpha': 1.0}, #steelblue
}

workload = {
    "IncreasingDataSize":
     {
         "dataset_sizes": np.array([8000, 16000, 32000, 64000, 128000]),
          tensorsocket_label: {"costs": np.array([8, 5, 9, 20, 35]), "throughputs": np.array([2436, 95, 85, 70, 55])},
          disdl_label: {"costs": np.array([18, 36, 72, 144, 288]), "throughputs": np.array([973.9196926, 973.9196926, 973.9196926, 973.9196926, 973.9196926])}
    },
    # "IncreasingNumJobs":
    # {
    #     "num_jobs": np.array([1, 2, 4, 8, 16]),
    #     "tensorsocket": {"costs": np.array([1, 5, 9, 20, 35]), "throughputs": np.array([100, 95, 85, 70, 55])},
    #     "disdl": {"costs": np.array([2, 6, 10, 22, 38]), "throughputs": np.array([95, 90, 80, 65, 50])}
    # },
    # "IncreasingVariability":
    # {
    #     "variabilities": np.array([0.1, 0.2, 0.4, 0.8, 1.6]),
    #     "tensorsocket": {"costs": np.array([1, 5, 9, 20, 35]), "throughputs": np.array([100, 95, 85, 70, 55])},
    #     "disdl": {"costs": np.array([2, 6, 10, 22, 38]), "throughputs": np.array([95, 90, 80, 65, 50])}
    # },
    # "IncraesingBatchSize":
    # {
    #     "batch_sizes": np.array([32, 64, 128, 256, 512]),
    #     "tensorsocket": {"costs": np.array([1, 5, 9, 20, 35]), "throughputs": np.array([100, 95, 85, 70, 55])},
    #     "disdl": {"costs": np.array([2, 6, 10, 22, 38]), "throughputs": np.array([95, 90, 80, 65, 50])}
    # }
}

for workload_name, workload_data in workload.items():
    fig, ax = plt.subplots(figsize=(8, 3.5))

    #get valeus from first item in dictionary
    xaxis_values = workload_data[list(workload_data.keys())[0]]
    tput_disdl = workload_data[disdl_label]["throughputs"]
    tput_tensorsocket =workload_data[tensorsocket_label]["throughputs"]
    cost_disdl =workload_data[disdl_label]["costs"]
    cost_tensorsocket = workload_data[tensorsocket_label]["costs"]
    
    plt.plot(cost_disdl, tput_disdl, label=disdl_label, 
             marker= visual_map_plot[disdl_label]['marker'],
             color = visual_map_plot[disdl_label]['color'],
             linestyle=visual_map_plot[disdl_label]['linestyle'],
             alpha=visual_map_plot[disdl_label]['alpha'],
             linewidth=visual_map_plot[disdl_label]['linewidth'])
    
    plt.plot(cost_tensorsocket, tput_tensorsocket, label=tensorsocket_label,
                marker= visual_map_plot[tensorsocket_label]['marker'],
                color = visual_map_plot[tensorsocket_label]['color'],
                linestyle=visual_map_plot[tensorsocket_label]['linestyle'],
                alpha=visual_map_plot[tensorsocket_label]['alpha'],
                linewidth=visual_map_plot[tensorsocket_label]['linewidth'])
              
    # Annotate points with dataset sizes
    for i in range(len(xaxis_values)):
        size_label = f"{xaxis_values[i]//1000}K" if xaxis_values[i] >= 1000 else str(xaxis_values[i])

    for i in range(len(xaxis_values)):
        size_label = f"{xaxis_values[i]//1000}K" if xaxis_values[i] >= 1000 else str(xaxis_values[i])

        plt.annotate(size_label, 
                    (cost_disdl[i], tput_disdl[i]), 
                    textcoords="offset points", xytext=(0, 0), ha='center', 
                    fontsize=14, color=visual_map_plot[disdl_label]['color'],
                    bbox=dict(facecolor='white', edgecolor='none', alpha=1.0, boxstyle="round,pad=0.1"))  # Label background

        plt.annotate(size_label, 
                    (cost_tensorsocket[i], tput_tensorsocket[i]), 
                    textcoords="offset points", xytext=(0, 0), ha='center', 
                    fontsize=14, color=visual_map_plot[tensorsocket_label]['color'],
                    bbox=dict(facecolor='white', edgecolor='none', alpha=1.0, boxstyle="round,pad=0.1"))  

   

    # Labels and formatting

    #set font size for all lables and ticks
    plt.tick_params(axis='both', which='major', labelsize=font_size)
    # Labels and legend
    plt.xlabel("Cost ($)", fontsize=font_size)
    plt.ylabel("Throughput (Images/sec)", fontsize=font_size)
    # plt.title(f"Pareto Frontier: Cost vs Throughput for {workload_name}")
    plt.legend(fontsize=font_size)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

    