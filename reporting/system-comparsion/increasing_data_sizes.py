import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Figure settings
tensorsocket_label = "TensorSocket"
disdl_label = "DisDL"
line_width = 2.5
font_size = 22

# Line styles for clarity
cost_linestyle = "--"
throughput_linestyle = "-"

visual_map_plot = {
    tensorsocket_label: {'color': '#F39D00', 'marker': 'o', 'linewidth': line_width}, 
    disdl_label: {'color': 'black', 'marker': 's', 'linewidth': line_width}, 
}

# Workload data
workload = {
    "IncreasingDataSize": {
        "dataset_sizes": np.array([8000, 16000, 32000, 64000, 128000]),
        disdl_label: {"costs": np.array([8, 5, 9, 20, 35]), "throughputs": np.array([2436, 2436, 2436, 2436, 2436])},
        tensorsocket_label: {"costs": np.array([18, 36, 72, 144, 288]), "throughputs": np.array([973.9, 973.9, 973.9, 973.9, 973.9])}
    }
}

for workload_name, workload_data in workload.items():
    fig, ax = plt.subplots(figsize=(8, 6))

    xaxis_values = workload_data["dataset_sizes"]
    tensorsocket_costs = workload_data[tensorsocket_label]["costs"]
    tensorsocket_throughputs = workload_data[tensorsocket_label]["throughputs"]
    disdl_costs = workload_data[disdl_label]["costs"]
    disdl_throughputs = workload_data[disdl_label]["throughputs"]

    # Plot lines and points for TensorSocket
    ax.plot(tensorsocket_costs, tensorsocket_throughputs, linestyle='-', color=visual_map_plot[tensorsocket_label]["color"],
            marker=visual_map_plot[tensorsocket_label]["marker"], markersize=8, label=f"{tensorsocket_label}", zorder=5)
    
    # Plot lines and points for DisDL
    ax.plot(disdl_costs, disdl_throughputs, linestyle='-', color=visual_map_plot[disdl_label]["color"],
            marker=visual_map_plot[disdl_label]["marker"], markersize=8, label=f"{disdl_label}", zorder=5)

    # Setting labels
    ax.set_xlabel("Cost ($)", fontsize=font_size)
    ax.set_ylabel("Throughput (samples/s)", fontsize=font_size)
    ax.set_xscale("log")
    ax.set_xticks(tensorsocket_costs)
    ax.set_xticklabels([f"{size//1000}K" for size in xaxis_values])
    
    # Custom Legend
    legend_elements = [
        Line2D([0], [0], color=visual_map_plot[tensorsocket_label]["color"], marker="o", lw=line_width, label=f"{tensorsocket_label}"),
        Line2D([0], [0], color=visual_map_plot[disdl_label]["color"], marker="s", lw=line_width, label=f"{disdl_label}")
    ]
    
    ax.legend(handles=legend_elements, loc="upper left", fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    plt.tight_layout()
    plt.show()
