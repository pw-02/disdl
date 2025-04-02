import numpy as np
import matplotlib.pyplot as plt

# Figure settings
tensorsocket_label = "TensorSocket"
disdl_label = "DisDL"
line_width = 2.5
font_size = 12

visual_map_plot = {
    tensorsocket_label: {'color': '#F39D00', 'linestyle': '-', 'marker':'o', 'linewidth': line_width, 'alpha': 1.0}, 
    disdl_label: {'color': 'black', 'linestyle': '-', 'marker':'s', 'linewidth': line_width, 'alpha': 1.0}, 
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
    fig, ax1 = plt.subplots(figsize=(8, 5))

    xaxis_values = workload_data["dataset_sizes"]
    tensorsocket_costs = workload_data[tensorsocket_label]["costs"]
    tensorsocket_throughputs = workload_data[tensorsocket_label]["throughputs"]
    disdl_costs = workload_data[disdl_label]["costs"]
    disdl_throughputs = workload_data[disdl_label]["throughputs"]

    # Primary y-axis: Cost
    cost_tensorsocket, = ax1.plot(xaxis_values, tensorsocket_costs, label=f"{tensorsocket_label} (Cost)", **visual_map_plot[tensorsocket_label])
    cost_disdl, = ax1.plot(xaxis_values, disdl_costs, label=f"{disdl_label} (Cost)", **visual_map_plot[disdl_label])
    ax1.set_xlabel("Batches/Epoch", fontsize=font_size)
    ax1.set_ylabel("Cost ($)", fontsize=font_size, color='black')
    ax1.set_xscale("log")
    ax1.set_xticks(xaxis_values)
    ax1.set_xticklabels([f"{size//1000}K" for size in xaxis_values])
    ax1.grid(True, linestyle="--", linewidth=0.5)

    # Secondary y-axis: Throughput
    ax2 = ax1.twinx()
    throughput_tensorsocket, = ax2.plot(xaxis_values, tensorsocket_throughputs, linestyle="--", marker="o", color="#F39D00", linewidth=line_width, label=f"{tensorsocket_label} (Throughput)")
    throughput_disdl, = ax2.plot(xaxis_values, disdl_throughputs, linestyle="--", marker="s", color="black", linewidth=line_width, label=f"{disdl_label} (Throughput)")
    ax2.set_ylabel("Throughput (samples/s)", fontsize=font_size, color='black')

    # Combine legends from both axes
    combined_handles = [cost_tensorsocket, throughput_tensorsocket, cost_disdl, throughput_disdl]
    combined_labels = [f"{tensorsocket_label} (Cost)", f"{tensorsocket_label} (Throughput)", 
                       f"{disdl_label} (Cost)", f"{disdl_label} (Throughput)"]
    
    # ax1.legend(combined_handles, combined_labels, fontsize=font_size,  bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax1.legend(combined_handles, combined_labels, fontsize=font_size-2, loc="center left", frameon=True)

    # Tick customization
    ax1.tick_params(axis='both', which='major', labelsize=font_size)
    ax2.tick_params(axis='y', which='major', labelsize=font_size)

    plt.title(workload_name, fontsize=font_size)
    plt.tight_layout()
    plt.show()
