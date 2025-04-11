import matplotlib.pyplot as plt
import numpy as np

# Figure settings
tensorsocket_label = "TensorSocket"
disdl_label = "DisDL"
line_width = 2.5
font_size = 14

# Line styles
cost_linestyle = "--"
throughput_linestyle = "-"

visual_map_plot = {
    tensorsocket_label: {'color': '#F39D00', 'marker': 'o', 'linewidth': line_width}, 
    disdl_label: {'color': 'black', 'marker': 's', 'linewidth': line_width}, 
}

# Workload data
workload_data = {
    "IncreasingBatchSizes": {
        "batch_sizes": [16, 32, 64, 128],
        disdl_label: {"costs": [0.02, 0.03, 0.05, 0.07], "throughputs": [1002, 1024, 1054, 1078]},
        tensorsocket_label: {"costs": [0.015, 0.025, 0.04, 0.06], "throughputs": [902, 936, 965, 987]}
    }
}

# Plot settings
plt.figure(figsize=(6, 3))

for label, data in workload_data.items():
    batch_sizes = data["batch_sizes"]

    # DISDL
    didsl_costs = data[disdl_label]["costs"]
    didsl_throughputs = data[disdl_label]["throughputs"]
    plt.plot(didsl_costs, didsl_throughputs, visual_map_plot[disdl_label]['marker'] + throughput_linestyle, 
             color=visual_map_plot[disdl_label]['color'],
             linewidth=visual_map_plot[disdl_label]['linewidth'], 
             label=disdl_label, markersize=8)

    for i, batch_size in enumerate(batch_sizes):
        plt.annotate(str(batch_size), 
                     (didsl_costs[i], didsl_throughputs[i]),
                     fontsize=font_size - 2,
                     xytext=(8, -12),  # better positioning
                     textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6),
                     arrowprops=dict(arrowstyle='-', lw=0.5, color='gray'))

    # TensorSocket
    tensorsocket_costs = data[tensorsocket_label]["costs"]
    tensorsocket_throughputs = data[tensorsocket_label]["throughputs"]
    plt.plot(tensorsocket_costs, tensorsocket_throughputs, visual_map_plot[tensorsocket_label]['marker'] + throughput_linestyle,
             color=visual_map_plot[tensorsocket_label]['color'],
             linewidth=visual_map_plot[tensorsocket_label]['linewidth'],
             label=tensorsocket_label, markersize=8)

    for i, batch_size in enumerate(batch_sizes):
        plt.annotate(str(batch_size), 
                     (tensorsocket_costs[i], tensorsocket_throughputs[i]),
                     fontsize=font_size - 2,
                     xytext=(8, -12),
                     textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6),
                     arrowprops=dict(arrowstyle='-', lw=0.5, color='gray'))

# Axis Labels
plt.xlabel("Cost per Batch ($)", fontsize=font_size)
plt.ylabel("Throughput (samples/s)", fontsize=font_size)  # fixed spelling
plt.legend(loc="lower right", fontsize=font_size)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tick_params(axis='both', which='major', labelsize=font_size)

# Set limits for breathing room
plt.xlim(0.01, 0.08)
plt.ylim(880, 1100)

# Tight layout
plt.tight_layout()

# Show plot
plt.show()
