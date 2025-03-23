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

# Normalize the data to percentages
def compute_percentage_breakdown(compute, io, transform):
    total = np.array(compute) + np.array(io) + np.array(transform)
    return (np.array(compute) / total * 100, 
            np.array(io) / total * 100, 
            np.array(transform) / total * 100)

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
bar_width = 0.35

visual_map_plot = {
    tensorsocket_label: {'color': '#FDA300', 'linestyle': '-', 'linewidth': line_width, 'edgecolor': 'black', 'hatch': '...','alpha': 1.0}, #indianred
    disdl_label: {'color': 'black', 'linestyle': '-', 'linewidth': line_width,'edgecolor': 'black', 'hatch':'', 'alpha': 1.0}, #steelblue
}

# visual_map_plot = {
#     tensorsocket_label: {'color': '#FEA400', 'linestyle': '-', 'linewidth': line_width, 'edgecolor': 'black', 'hatch': '', 'alpha': 1.0}, #indianred
#     disdl_label: {'color': '#005250', 'linestyle': '-', 'linewidth': line_width,'edgecolor': 'black', 'hatch':'', 'alpha': 1.0}, #steelblue
# }

workload = {
    "imagenet":
     {
         "samples_per_epoch": 1281167,
         "samples_per_batch": 128,
         "ec2_instance_cost_per_hour": 12.24,
         "summary": r"C:\Users\pw\Desktop\disdl(600)\imagenet_nas\jobsummary.csv"
    },
    # "openimages": {"C:\\Users\\pw\Desktop\\disdl(600)\\openimages_nas\\overall_summary_openimages_nas.csv"},
    # "coco": {"C:\\Users\\pw\Desktop\\disdl(600)\\coco_nas\\overall_summary_coco_nas.csv"}
    }

for workload_name, workload_data in workload.items():
    fig = plt.figure(figsize=(14, 2.4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])  # First two plots are twice as wide
    ax1 = fig.add_subplot(gs[0, 0])  # First plot
    ax2 = fig.add_subplot(gs[0, 1])  # Second plot
    ax4 = fig.add_subplot(gs[0, 2])  # Third plot
    # ax4 = fig.add_subplot(gs[0, 3])  # Fourth plot
    ec2_instance_cost_per_hour = workload_data["ec2_instance_cost_per_hour"]
    #batches over time plot first get data from disdl
    df = pd.read_csv(workload_data["summary"])
    model_names = list(df["model"])
    # Extract throughput values for each system
    disdl_throughputs = list(df["disdl_throughput(samples/s)"])
    tensorsocket_throughputs = list(df["tensorsocket_throughput(samples/s)"])
    #based on the through, lets compute the time to complete one epoch
    disdl_times_epoch_time_min = [(workload_data["samples_per_epoch"]/x)/60 for x in disdl_throughputs] #convert to minutes
    tsocket_times_epoch_time_min = [(workload_data["samples_per_epoch"]/x)/60 for x in tensorsocket_throughputs] #convert to minutes

    # Bars for DISDL and TensorSocket
    x = np.arange(len(model_names))  # X-axis positions for models
    ax1.bar(x + bar_width/2, disdl_times_epoch_time_min, bar_width, label=disdl_label,
             color=visual_map_plot[disdl_label]['color'],
             edgecolor=visual_map_plot[disdl_label]['edgecolor'],
             hatch=visual_map_plot[disdl_label]['hatch'])
    ax1.bar(x - bar_width/2, tsocket_times_epoch_time_min, bar_width, label=tensorsocket_label, 
            color=visual_map_plot[tensorsocket_label]['color'], 
            edgecolor=visual_map_plot[tensorsocket_label]['edgecolor'], 
            hatch=visual_map_plot[tensorsocket_label]['hatch'])
    
    # Labels and formatting
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=0, ha="center")
    ax1.set_ylabel("Epoch Time (min)", fontsize=font_size)
    ax1.legend()
    #set font size for all lables and ticks
    ax1.tick_params(axis='both', which='major', labelsize=font_size)


    #compute hour cost for each model
    hourly_cost = workload_data["ec2_instance_cost_per_hour"]/len(model_names)
    #get epoch time in hours for each model, for both disdl and tensorsocket
    disdl_times_to_complete_one_epoch = [x/60 for x in disdl_times_epoch_time_min]
    tensorsocket_times_to_complete_one_epoch = [x/60 for x in tsocket_times_epoch_time_min]

    #get hourly cost for each model for both disdl and tensorsocket
    disdl_epoch_cost = [hourly_cost * _ for _ in disdl_times_to_complete_one_epoch]
    tensorsocket_epoch_cost = [hourly_cost * _ for _ in tensorsocket_times_to_complete_one_epoch]
    # Bars for DISDL and TensorSocket
  
    # Bars for DISDL and TensorSocket
    x = np.arange(len(model_names))  # X-axis positions for models
    ax2.bar(x + bar_width/2, disdl_epoch_cost, bar_width, label=disdl_label,
             color=visual_map_plot[disdl_label]['color'],
             edgecolor=visual_map_plot[disdl_label]['edgecolor'],
             hatch=visual_map_plot[disdl_label]['hatch'],
             alpha=visual_map_plot[disdl_label]['alpha'])
    ax2.bar(x - bar_width/2, tensorsocket_epoch_cost, bar_width, label=tensorsocket_label,
            color=visual_map_plot[tensorsocket_label]['color'],
            edgecolor=visual_map_plot[tensorsocket_label]['edgecolor'],
            hatch=visual_map_plot[tensorsocket_label]['hatch'],
            alpha=visual_map_plot[disdl_label]['alpha']
                         )

    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=0, ha="center")
    ax2.set_ylabel("Epoch Cost ($)", fontsize=font_size)
    ax2.legend()
    #set font size for all lables and ticks


    # #nots plot the cache hit % for each model
    # disdl_cache_hit = list(df["disdl_cache_hit"])
    # tensorsocket_cache_hit = list(df["tensorsocket_cache_hit"])

    # #convery cache hit to percentage
    # disdl_cache_hit = [x*100 for x in disdl_cache_hit]
    # tensorsocket_cache_hit = [x*100 for x in tensorsocket_cache_hit]

    # # Bars for DISDL and TensorSocket
    # x = np.arange(len(model_names))  # X-axis positions for models
    # ax3.bar(x + bar_width/2, disdl_cache_hit, bar_width, label=disdl_label,
    #          color=visual_map_plot[disdl_label]['color'],
    #          edgecolor=visual_map_plot[disdl_label]['edgecolor'],
    #          hatch=visual_map_plot[disdl_label]['hatch'],
    #            alpha=visual_map_plot[disdl_label]['alpha']
    #                      )
    # ax3.bar(x - bar_width/2, tensorsocket_cache_hit, bar_width, label=tensorsocket_label,
    #         color=visual_map_plot[tensorsocket_label]['color'],
    #         edgecolor=visual_map_plot[tensorsocket_label]['edgecolor'],
    #         hatch=visual_map_plot[tensorsocket_label]['hatch'],
    #           alpha=visual_map_plot[disdl_label]['alpha']
    #                      )
    # ax3.set_xticks(x)
    # ax3.set_xticklabels(model_names, rotation=0, ha="center")
    # ax3.set_ylabel("Cache Hit (%)", fontsize=font_size)
    # ax3.legend()
    # #set font size for all lables and ticks
    # ax3.tick_params(axis='both', which='major', labelsize=font_size)
    # ax3.set_yticks(ticks=np.arange(0, 101, 20), labels=[f'{i}%' for i in np.arange(0, 101, 20)])

    #finally craete a stacked bar to show a breakdown of where time is spent between compute, I/O and Tranform
    disdl_compute = list(df["disdl_compute"])
    disdl_io = list(df["disdl_io"])
    disdl_transform = list(df["disdl_transform"])
    tensorsocket_compute = list(df["tensorsocket_compute"])
    tensorsocket_io = list(df["tensorsocket_io"])
    tensorsocket_transform = list(df["tensorsocket_transform"])
    # Compute percentage breakdown for each system
    disdl_compute_pct, disdl_io_pct, disdl_transform_pct = compute_percentage_breakdown(disdl_compute, disdl_io, disdl_transform)
    tensorsocket_compute_pct, tensorsocket_io_pct, tensorsocket_transform_pct = compute_percentage_breakdown(tensorsocket_compute, tensorsocket_io, tensorsocket_transform)
    # Bars for DISDL and TensorSocket
    x = np.arange(len(model_names))  # X-axis positions for models
    ax4.bar(x, disdl_compute_pct, bar_width, label="Compute", color=visual_map_plot[disdl_label]["color"], edgecolor="black", hatch='///', alpha=visual_map_plot[disdl_label]['alpha'])
    ax4.bar(x, disdl_io_pct, bar_width, label="I/O", color=visual_map_plot[disdl_label]["color"], edgecolor="black", hatch="...", bottom=disdl_compute_pct, alpha=0.8)
    ax4.bar(x, disdl_transform_pct, bar_width, label="Transform", color=visual_map_plot[disdl_label]["color"], edgecolor="black", hatch="---", alpha=0.6,
            bottom=disdl_compute_pct + disdl_io_pct)
    ax4.bar(x + bar_width, tensorsocket_compute_pct, bar_width, label="Compute", color=visual_map_plot[tensorsocket_label]["color"], edgecolor="black", hatch='///', alpha=visual_map_plot[tensorsocket_label]['alpha'])
    ax4.bar(x + bar_width, tensorsocket_io_pct, bar_width, label="I/O", color=visual_map_plot[tensorsocket_label]["color"], edgecolor="black", hatch="...", bottom=tensorsocket_compute_pct,   alpha=0.8)
    ax4.bar(x + bar_width, tensorsocket_transform_pct, bar_width, label="Transform", color=visual_map_plot[tensorsocket_label]["color"], edgecolor="black", hatch="---", alpha=0.6,
             bottom=tensorsocket_compute_pct + tensorsocket_io_pct)
    ax4.set_xticks(x + bar_width / 2)
    ax4.set_xticklabels(model_names, rotation=0, ha="center")
    ax4.set_ylabel("Time Breakdown (%)", fontsize=font_size)
    # ax4.legend()
    #set font size for all lables and ticks
    ax4.tick_params(axis='both', which='major', labelsize=font_size)
    ax4.set_yticks(ticks=np.arange(0, 101, 20), labels=[f'{i}%' for i in np.arange(0, 101, 20)])
    # ax4.legend(loc="upper center", ncol=3, fontsize=9)  # Moves legend above plot


    plt.tight_layout()
    plt.show()