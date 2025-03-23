import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as mticker
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import json
# Function to load and process NDJSON file
def load_usage_data(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")  # Convert timestamp
    return df

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
line_width = 2
font_size = 14


visual_map_plot = {
    tensorsocket_label: {'color': '#FDA300', 'linestyle': '-', 'linewidth': line_width, 'edgecolor': 'black', 'hatch': '...','alpha': 1.0}, #indianred
    disdl_label: {'color': 'black', 'linestyle': '-', 'linewidth': line_width,'edgecolor': 'black', 'hatch':'', 'alpha': 1.0}, #steelblue
}


workload = {
    "imagenet": {"C:\\Users\\pw\Desktop\\disdl(600)\\imagenet_nas\\overall_summary_imagenet_nas.csv"},
    "openimages": {"C:\\Users\\pw\Desktop\\disdl(600)\\openimages_nas\\overall_summary_openimages_nas.csv"},
    "coco": {"C:\\Users\\pw\Desktop\\disdl(600)\\coco_nas\\overall_summary_coco_nas.csv"}
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
    time_steps = time_steps[:]

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

    #set font size for all lables and ticks
    ax1.tick_params(axis='both', which='major', labelsize=font_size)
    ax1.tick_params(axis='both', which='minor', labelsize=font_size)
    ax1.xaxis.label.set_size(font_size)
    ax1.yaxis.label.set_size(font_size)
    #lengened font size
    ax1.legend(fontsize=font_size)


    #plot cost efficiency over time
    ax2.plot(time_steps, disdl_cost_efficiency, label=disdl_label, **visual_map_plot[disdl_label])
    ax2.plot(time_steps, tensorsocket_cost_efficiency, label=tensorsocket_label, **visual_map_plot[tensorsocket_label])
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Cost Efficiency (batches/$)")
    ax2.legend()

    #set font size for all lables and ticks
    ax2.tick_params(axis='both', which='major', labelsize=font_size)
    ax2.tick_params(axis='both', which='minor', labelsize=font_size)
    ax2.xaxis.label.set_size(font_size)
    ax2.yaxis.label.set_size(font_size)
    ax2.legend(fontsize=font_size)


    # # Resource usage plot
    disdl_usage = load_usage_data(workload_data["resource_usage"][disdl_label])
    tenorsocket_usage = load_usage_data(workload_data["resource_usage"][tensorsocket_label])
    categories = ["CPU", "GPU"]
    disdl_values = [disdl_usage["Average CPU Usage (%)"].mean(), disdl_usage["Average GPU Usage (%)"].max()]
    tenorsocket_values = [tenorsocket_usage["Average CPU Usage (%)"].mean(), tenorsocket_usage["Average GPU Usage (%)"].mean()]
    x = np.arange(len(categories))  # X positions for categories
    width = 0.3  # Width of bars 
    ax3.bar(x - width / 2, tenorsocket_values, width, label=tensorsocket_label, color=visual_map_plot[tensorsocket_label]['color'],  edgecolor = 'black', alpha=0.7)
    ax3.bar(x + width / 2, disdl_values, width, label=disdl_label, color=visual_map_plot[disdl_label]['color'], edgecolor = 'black')

    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)  # Set x-axis labels as CPU and GPU
    ax3.set_ylabel("Average Utilization (%)")  
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax3.set_ylim(0, 100)  # Assuming percentage values
    ax3.legend(loc="upper center", ncol=2,fontsize=font_size)  # Moves legend above plot

    #set font size for all lables and ticks
    ax3.tick_params(axis='both', which='major', labelsize=font_size)
    ax3.tick_params(axis='both', which='minor', labelsize=font_size)
    ax3.xaxis.label.set_size(font_size)
    ax3.yaxis.label.set_size(font_size)


    plt.tight_layout()
    plt.show()