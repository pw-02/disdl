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
    fig, ax1 = plt.subplots(figsize=(8, 6))

    xaxis_values = workload_data["dataset_sizes"]
    tensorsocket_costs = workload_data[tensorsocket_label]["costs"]
    tensorsocket_throughputs = workload_data[tensorsocket_label]["throughputs"]
    disdl_costs = workload_data[disdl_label]["costs"]
    disdl_throughputs = workload_data[disdl_label]["throughputs"]

    # First y-axis (Cost)
    ax1.set_xlabel("Batches/Epoch", fontsize=font_size)
    ax1.set_ylabel("Cost ($)", fontsize=font_size, color="black")
    ax1.set_xscale("log")
    ax1.set_xticks(xaxis_values)
    ax1.set_xticklabels([f"{size//1000}K" for size in xaxis_values])

    ax1.plot(xaxis_values, tensorsocket_costs, linestyle=cost_linestyle, label=f"{tensorsocket_label} Cost", **visual_map_plot[tensorsocket_label])
    ax1.plot(xaxis_values, disdl_costs, linestyle=cost_linestyle, label=f"{disdl_label} Cost", **visual_map_plot[disdl_label])

    ax1.tick_params(axis="y", labelcolor="black")
    ax1.grid(True, linestyle="--", linewidth=0.5)

    # Second y-axis (Throughput)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Throughput (samples/s)", fontsize=font_size, color="black")

    ax2.plot(xaxis_values, tensorsocket_throughputs, linestyle=throughput_linestyle, label=f"{tensorsocket_label} Throughput", **visual_map_plot[tensorsocket_label])
    ax2.plot(xaxis_values, disdl_throughputs, linestyle=throughput_linestyle, label=f"{disdl_label} Throughput", **visual_map_plot[disdl_label])

    ax2.tick_params(axis="y", labelcolor="black")

    # Custom Legend: Show both metrics clearly
    legend_elements = [
        Line2D([0], [0], color=visual_map_plot[tensorsocket_label]["color"], linestyle=cost_linestyle, marker="o", lw=line_width, label=f"{tensorsocket_label} Cost"),
        Line2D([0], [0], color=visual_map_plot[tensorsocket_label]["color"], linestyle=throughput_linestyle, marker="o", lw=line_width, label=f"{tensorsocket_label} Throughput"),
        Line2D([0], [0], color=visual_map_plot[disdl_label]["color"], linestyle=cost_linestyle, marker="s", lw=line_width, label=f"{disdl_label} Cost"),
        Line2D([0], [0], color=visual_map_plot[disdl_label]["color"], linestyle=throughput_linestyle, marker="s", lw=line_width, label=f"{disdl_label} Throughput"),
    ]

    ax1.legend(handles=legend_elements, loc="center left", fontsize=15)
    ax1.tick_params(axis='both', which='major', labelsize=font_size)
    ax2.tick_params(axis='y', which='major', labelsize=font_size)

    plt.tight_layout()
    plt.show()
