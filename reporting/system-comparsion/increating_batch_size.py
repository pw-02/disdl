import matplotlib.pyplot as plt
import numpy as np

# Figure settings
tensorsocket_label = "TensorSocket"
disdl_label = "DisDL"
line_width = 5
font_size = 28

# Line styles for clarity
cost_linestyle = "--"
throughput_linestyle = "-"

visual_map_plot = {
    tensorsocket_label: {'color': '#F39D00', 'marker': 'o', 'linewidth': line_width}, 
    disdl_label: {'color': 'black', 'marker': 's', 'linewidth': line_width}, 
}

# Workload data
workload_data = {
    "IncreasingBatchSizes": {
        "batch_sizes": [16, 32, 64, 128, 256],
        disdl_label: {"costs": [0.02, 0.03, 0.05, 0.08, 0.12], "throughputs": [500, 900, 1600, 2500, 3500]},
        tensorsocket_label: {"costs": [0.015, 0.025, 0.04, 0.06, 0.09], "throughputs": [550, 950, 1700, 2700, 3800]}
    }
}

# Compute Pareto Frontier
def pareto_frontier(cost, throughput):
    sorted_indices = np.argsort(cost)  
    cost, throughput = cost[sorted_indices], throughput[sorted_indices]  
    pareto_x, pareto_y = [cost[0]], [throughput[0]]  

    for i in range(1, len(cost)):
        if throughput[i] > pareto_y[-1]:  
            pareto_x.append(cost[i])
            pareto_y.append(throughput[i])

    return np.array(pareto_x), np.array(pareto_y)

# Plot settings
plt.figure(figsize=(12, 7))

# Plotting both systems
for label, data in workload_data.items():

    didsl_costs = data[disdl_label]["costs"]
    didsl_throughputs = data[disdl_label]["throughputs"]

    batch_sizes = data[list(data.keys())[0]]

    tensorsocket_costs = data[tensorsocket_label]["costs"]
    tensorsocket_throughputs = data[tensorsocket_label]["throughputs"]

    # Plot cost vs throughput for each system
    plt.plot(didsl_costs, didsl_throughputs, visual_map_plot[disdl_label]['marker'] + throughput_linestyle, 
             color=visual_map_plot[disdl_label]['color'],
            linewidth=visual_map_plot[disdl_label]['linewidth'], 
               label=disdl_label, markersize=8)
             
    
    #add batch size annotations
    for i, batch_size in enumerate(batch_sizes):
        plt.annotate(batch_size, (didsl_costs[i], didsl_throughputs[i]), fontsize=font_size - 6, 
                     xytext=(5, 5), textcoords='offset points')
        
    plt.plot(tensorsocket_costs, tensorsocket_throughputs, visual_map_plot[tensorsocket_label]['marker'] + throughput_linestyle,
                color=visual_map_plot[tensorsocket_label]['color'],
                linewidth=visual_map_plot[tensorsocket_label]['linewidth'],
                  label=tensorsocket_label, markersize=8)
    
    #add batch size annotations
    for i, batch_size in enumerate(batch_sizes):
        plt.annotate(batch_size, (tensorsocket_costs[i], tensorsocket_throughputs[i]), fontsize=font_size - 6, 
                     xytext=(5, 5), textcoords='offset points')


# Labels, legend, and grid
plt.xlabel("Cost per Batch ($)", fontsize=font_size)
plt.ylabel("Throuhgput (samples/s)", fontsize=font_size)
plt.legend(loc="lower right", fontsize=font_size)
# plt.xscale("log")  # Logarithmic scale for cost
plt.grid(True, linestyle="--", alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=font_size)


# Adjust layout for better spacing
plt.tight_layout()

# Show plot
plt.show()
