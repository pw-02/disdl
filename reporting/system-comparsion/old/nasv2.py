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
font_size = 40
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
       "openimages":
     {
         "samples_per_epoch": 1281167,
         "samples_per_batch": 128,
         "ec2_instance_cost_per_hour": 12.24,
         "summary": r"C:\Users\pw\Desktop\disdl(600)\imagenet_nas\jobsummary.csv"
    },
       "coco":
     {
         "samples_per_epoch": 1281167,
         "samples_per_batch": 128,
         "ec2_instance_cost_per_hour": 12.24,
         "summary": r"C:\Users\pw\Desktop\disdl(600)\imagenet_nas\jobsummary.csv"
    },
  "audio":
     {
         "samples_per_epoch": 1281167,
         "samples_per_batch": 128,
         "ec2_instance_cost_per_hour": 12.24,
         "summary": r"C:\Users\pw\Desktop\disdl(600)\imagenet_nas\jobsummary.csv"
    },

    # "coco": {"C:\\Users\\pw\Desktop\\disdl(600)\\openimages_nas\\overall_summary_openimages_nas.csv"},
    # "coco": {"C:\\Users\\pw\Desktop\\disdl(600)\\coco_nas\\overall_summary_coco_nas.csv"}
    }

#aggrated throughput plot
fig, ax = plt.subplots(figsize=(16.5, 7.2))
categories = list(workload.keys())
disdl_throughputs = [4,1,8,3]
tensorsocket_throughputs = [5,2,7,4]
x = np.arange(len(categories))  # X-axis positions
ax.bar(x - bar_width/2, disdl_throughputs, bar_width, label=disdl_label,
        color=visual_map_plot[disdl_label]['color'],
        edgecolor=visual_map_plot[disdl_label]['edgecolor'],
        hatch=visual_map_plot[disdl_label]['hatch'])
ax.bar(x + bar_width/2, tensorsocket_throughputs, bar_width, label=tensorsocket_label,
        color=visual_map_plot[tensorsocket_label]['color'],
        edgecolor=visual_map_plot[tensorsocket_label]['edgecolor'],
        hatch=visual_map_plot[tensorsocket_label]['hatch'])
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=0, ha="center")
ax.set_ylabel("Throughput (samples/s)", fontsize=font_size)
ax.legend(fontsize=font_size)
#set font size for all lables and ticks
ax.tick_params(axis='both', which='major', labelsize=font_size)
plt.tight_layout()
plt.show()





