import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as mticker
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator


def fill_range(int_list):
    """Generate a list of all integers between the min and max values in int_list."""
    if not int_list:
        return []
    
    min_val = min(int_list)
    max_val = max(int_list)
    
    return list(range(min_val, max_val + 1))

#figure data
tensorsocket_label = "TensorSocket"
disdl_label = "DisDL"
line_width = 1
visual_map_plot = {
    disdl_label: {'color': 'black', 'linestyle': '-', 'linewidth': line_width},
    tensorsocket_label: {'color': 'black', 'linestyle': ':', 'linewidth': line_width}}

workload = {
    "imagenet": {
        "over_time_data": {
            disdl_label: r"C:\Users\pw\Desktop\disdl(600)\imagenet_nas\disdl\2025-03-13_20-34-34\2025-03-13_20-34-34_imagenet_nas_disdl_over_time.csv",
            tensorsocket_label: r"C:\Users\pw\Desktop\disdl(600)\imagenet_nas\tensorsocket\2025-03-13_19-41-51\2025-03-13_19-41-51_imagenet_nas_tensorsocket_over_time.csv"}
    },"openimages": {
        "over_time_data": {
            disdl_label: r"C:\Users\pw\Desktop\disdl(600)\openimages_nas\disdl\2025-03-13_20-48-17\2025-03-13_20-48-17_openimages_nas_disdl_over_time.csv",
            tensorsocket_label: r"C:\Users\pw\Desktop\disdl(600)\openimages_nas\tensorsocket\2025-03-13_19-55-06\2025-03-13_19-55-06_openimages_nas_tensorsocket_over_time.csv"}
    },"coco": {
        "over_time_data": {
            disdl_label: r"C:\Users\pw\Desktop\disdl(600)\coco_nas\disdl\2025-03-13_21-10-51\2025-03-13_21-10-51_coco_nas_disdl_over_time.csv",
            tensorsocket_label: r"C:\Users\pw\Desktop\disdl(600)\coco_nas\tensorsocket\2025-03-13_20-08-04\2025-03-13_20-08-04_coco_nas_tensorsocket_over_time.csv"}
            }
    }


for workload_name, workload_data in workload.items():
    fig = plt.figure(figsize=(16.5, 3.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])  # First two plots are twice as wide
    ax1 = fig.add_subplot(gs[0, 0])  # First plot
    ax2 = fig.add_subplot(gs[0, 1])  # Second plot
    ax3 = fig.add_subplot(gs[0, 2])  # Third plot

    #batches over time plot first get data from disdl
    df = pd.read_csv(workload_data["over_time_data"][disdl_label])
    elapsed_times = df["elapsed_time"].values
    time_steps = list(set(int(x) for x in elapsed_times))
    time_steps = list(set(time_steps))

    #trim off the first 100 time steps
    # time_steps = time_steps[100:]

    #now get the throughput for each time step
    disdl_throughput = []
    tensorsocket_throughput = []
    disdl_cost_efficiency = []
    tensorsocket_cost_efficiency = []

    for time in time_steps:
        #find the first elapsed time that is greater than or equal to time
        idx = np.argmax(elapsed_times >= time)
        disdl_throughput.append(df["throughput"].values[idx])
        disdl_cost_efficiency.append(df["cost_efficiency"].values[idx])

    df = pd.read_csv(workload_data["over_time_data"][tensorsocket_label])
    for time in time_steps:
        idx = np.argmax(df["elapsed_time"].values >= time)
        tensorsocket_throughput.append(df["throughput"].values[idx])
        tensorsocket_cost_efficiency.append(df["cost_efficiency"].values[idx])

    #plot throughput over time
    ax1.plot(time_steps, disdl_throughput, label=disdl_label, **visual_map_plot[disdl_label])
    ax1.plot(time_steps, tensorsocket_throughput, label=tensorsocket_label, **visual_map_plot[tensorsocket_label])
    # ax1.set_ylim(0, 25)
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Throughput (batches/sec)")
    ax1.legend()


    #plot cost efficiency over time
    ax2.plot(time_steps, disdl_cost_efficiency, label=disdl_label, **visual_map_plot[disdl_label])
    ax2.plot(time_steps, tensorsocket_cost_efficiency, label=tensorsocket_label, **visual_map_plot[tensorsocket_label])
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Cost Efficiency (batches/$)")
    ax2.legend()

    plt.tight_layout()
    plt.show()