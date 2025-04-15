import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as mticker
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import json
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

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
coordl_label = 'ReCoorDL'
tensorsocket_label = "TensorSocket"
disdl_label = "DisDL"
line_width = 2
font_size = 16
bar_width = 0.285

coordl_cahe_cost_per_hour = 1.82

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
        "openimages":{
                "samples_per_epoch": 9000000,
                "samples_per_batch": 128,
                "ec2_instance_cost_per_hour": 12.24,
                "summary": r"C:\Users\pw\Desktop\disdl(600)\openimages_nas\jobsummary.csv"
        },
        "coco":
        {
                "samples_per_epoch": 118287,
                "samples_per_batch": 128,
                "ec2_instance_cost_per_hour": 12.24,
                "summary": r"C:\Users\pw\Desktop\disdl(600)\coco_nas\jobsummary.csv"
        },
        #     "openimages": {"C:\\Users\\pw\Desktop\\disdl(600)\\openimages_nas\\overall_summary_openimages_nas.csv"},
        #     "coco": {"C:\\Users\\pw\Desktop\\disdl(600)\\coco_nas\\overall_summary_coco_nas.csv"}
        }

for workload_name, workload_data in workload.items():
        visual_map_plot = {
        coordl_label: {'color': '#A3B6CE', 'linestyle': '-', 'linewidth': line_width, 'edgecolor': 'black', 'hatch': '','alpha': 1.0}, #indianred
        disdl_label: {'color': 'black', 'linestyle': '-', 'linewidth': line_width,'edgecolor': 'black', 'hatch':'', 'alpha': 1.0}, #steelblue
        tensorsocket_label: {'color': '#FDA300', 'linestyle': '-', 'linewidth': line_width, 'edgecolor': 'black', 'hatch': '','alpha': 1.0}, #indianred
        }
        fig = plt.figure(figsize=(17, 3.2))
        gs = gridspec.GridSpec(1, 3, width_ratios=[0.95, 0.95, 1.1])  # First two plots are twice as wide
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
        coordl_throughputs = list(df["coordl_throughput(samples/s)"])
        #based on the through, lets compute the time to complete one epoch
        disdl_times_epoch_time_min = [(workload_data["samples_per_epoch"]/x)/60 for x in disdl_throughputs] #convert to minutes
        tsocket_times_epoch_time_min = [(workload_data["samples_per_epoch"]/x)/60 for x in tensorsocket_throughputs] #convert to minutes
        coodl_times_epoch_time_min = [(workload_data["samples_per_epoch"]/x)/60 for x in coordl_throughputs] #convert to minutes
        
        # X-axis positions for models
        x = np.arange(len(model_names))

        # Bars for DISDL, TensorSocket, and CoordL, properly spaced
        

        # Bars for TensorSocket, CoordL, and DISDL in the correct order
        ax1.bar(x - bar_width, tsocket_times_epoch_time_min, bar_width, label=tensorsocket_label, 
                color=visual_map_plot[tensorsocket_label]['color'], 
                edgecolor=visual_map_plot[tensorsocket_label]['edgecolor'], 
                hatch=visual_map_plot[tensorsocket_label]['hatch'],
                alpha=visual_map_plot[disdl_label]['alpha'])  # Adjusted alpha for visibility

        ax1.bar(x, coodl_times_epoch_time_min, bar_width, label=coordl_label,
                color=visual_map_plot[coordl_label]['color'], 
                edgecolor=visual_map_plot[coordl_label]['edgecolor'], 
                hatch=visual_map_plot[coordl_label]['hatch'],
                alpha=visual_map_plot[disdl_label]['alpha'])  # Adjusted alpha for visibility

        ax1.bar(x + bar_width, disdl_times_epoch_time_min, bar_width, label=disdl_label,
                color=visual_map_plot[disdl_label]['color'],
                edgecolor=visual_map_plot[disdl_label]['edgecolor'],
                hatch=visual_map_plot[disdl_label]['hatch'],
                alpha=visual_map_plot[disdl_label]['alpha'])  # Adjusted alpha for visibility
        # Labels and formatting
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, rotation=0, ha="center")
        ax1.set_ylabel("Avg epoch time (min)", fontsize=font_size)
        ax1.legend()
        #set font size for all lables and ticks
        ax1.tick_params(axis='both', which='major', labelsize=font_size)
        ax1.tick_params(axis='both', which='major', labelsize=font_size)
        # ax1.legend(loc="upper center", ncol=3, fontsize=9)  # Moves legend above plot
        # plt.tight_layout()
        # plt.show()

        #compute hour cost for each model
        hourly_cost = workload_data["ec2_instance_cost_per_hour"]/len(model_names)
        #get epoch time in hours for each model, for both disdl and tensorsocket
        disdl_times_to_complete_one_epoch = [x/60 for x in disdl_times_epoch_time_min]
        tensorsocket_times_to_complete_one_epoch = [x/60 for x in tsocket_times_epoch_time_min]
        coordl_times_to_complete_one_epoch = [x/60 for x in coodl_times_epoch_time_min]

        #get cache cost for each model
        disdl_cache_costs = list(df["disdl_cache_cost($)"])
        tensortsocket_cache_costs = list(df["tensorsocket_cache_cost($)"])
        coordl_cache_costs = list(df["coordl_cache_cost($)"])

        #get hourly compute cost for each model for both disdl and tensorsocket
        disdl_epoch_cost = [hourly_cost * _ for _ in disdl_times_to_complete_one_epoch]
        tensorsocket_epoch_cost = [hourly_cost * _ for _ in tensorsocket_times_to_complete_one_epoch]
        
        coordl_horly_cost = (workload_data["ec2_instance_cost_per_hour"] + coordl_cahe_cost_per_hour)/len(model_names)
        coordl_epoch_cost = [coordl_horly_cost * _ for _ in coordl_times_to_complete_one_epoch]
        # Bars for DISDL and TensorSocket

        #compute total cost for each model
        disdl_epoch_cost = [x + y + z for x, y, z in zip(disdl_epoch_cost, disdl_cache_costs, disdl_epoch_cost)]
        tensorsocket_epoch_cost = [x + y + z for x, y, z in zip(tensorsocket_epoch_cost, tensortsocket_cache_costs, tensorsocket_epoch_cost)]
        coordl_epoch_cost = [x + y + z for x, y, z in zip(coordl_epoch_cost, coordl_cache_costs, coordl_epoch_cost)]

        #     #print the aggregated cost for each system
        #     print(f"Aggregated cost for {workload_name} workload")
        #     print(f"DisDL: {sum(disdl_epoch_cost)}")
        #     print(f"TensorSocket: {sum(tensorsocket_epoch_cost)}")
        #     print(f"Cost difference:{abs(sum(disdl_epoch_cost) - sum(tensorsocket_epoch_cost))}")

        #print aggregated throughput for each system
        print(f"Aggregated throughput for {workload_name} workload")
        print(f"DisDL: {sum(disdl_throughputs)}")
        print(f"TensorSocket: {sum(tensorsocket_throughputs)}")
        print(f"CoordL: {sum(coordl_throughputs)}")

        # Bars for DISDL and TensorSocket
        x = np.arange(len(model_names))  # X-axis positions for models
        # Bars for TensorSocket, CoordL, and DISDL in the correct order
        ax2.bar(x - bar_width, tensorsocket_epoch_cost, bar_width, label=tensorsocket_label,
                color=visual_map_plot[tensorsocket_label]['color'],
                edgecolor=visual_map_plot[tensorsocket_label]['edgecolor'],
                hatch=visual_map_plot[tensorsocket_label]['hatch'],
                alpha=visual_map_plot[disdl_label]['alpha'])
        ax2.bar(x, coordl_epoch_cost, bar_width, label=coordl_label,
                color=visual_map_plot[coordl_label]['color'],
                edgecolor=visual_map_plot[coordl_label]['edgecolor'],
                hatch=visual_map_plot[coordl_label]['hatch'],
                alpha=visual_map_plot[disdl_label]['alpha'])
        
        ax2.bar(x + bar_width, disdl_epoch_cost, bar_width, label=disdl_label,
                color=visual_map_plot[disdl_label]['color'],
                edgecolor=visual_map_plot[disdl_label]['edgecolor'],
                hatch=visual_map_plot[disdl_label]['hatch'],
                alpha=visual_map_plot[disdl_label]['alpha'])

        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names, rotation=0, ha="center")
        ax2.set_ylabel("Avg epoch Cost ($)", fontsize=font_size)
        ax2.tick_params(axis='both', which='major', labelsize=font_size)
        ax2.tick_params(axis='both', which='major', labelsize=font_size)
        ax2.legend()


        
        #set font size for all lables and ticks

        #finally craete a stacked bar to show a breakdown of where time is spent between compute, I/O and Tranform
        disdl_compute = list(df["disdl_compute"])
        disdl_io = list(df["disdl_io"])
        disdl_transform = list(df["disdl_transform"])
        tensorsocket_compute = list(df["tensorsocket_compute"])
        tensorsocket_io = list(df["tensorsocket_io"])
        tensorsocket_transform = list(df["tensorsocket_transform"])

        coordl_compute = list(df["coordl_compute"])
        coordl_io = list(df["coordl_io"])
        coordl_transform = list(df["coordl_transform"])

        # Compute percentage breakdown for each system
        disdl_compute_pct, disdl_io_pct, disdl_transform_pct = compute_percentage_breakdown(disdl_compute, disdl_io, disdl_transform)
        tensorsocket_compute_pct, tensorsocket_io_pct, tensorsocket_transform_pct = compute_percentage_breakdown(tensorsocket_compute, tensorsocket_io, tensorsocket_transform)
        coodl_compute_pct, coordl_io_pct, coordl_transform_pct = compute_percentage_breakdown(coordl_compute, coordl_io, coordl_transform)
        
        
        visual_map_plot = {
                "GPU": {"color": "#A3B6CE", "hatch": "", "alpha": 0.8},        # Blue for GPU computation
                "I/O": {"color": "black", "hatch": "", "alpha": 0.8},    # Orange for I/O operations
                "Transform": {"color": "#902424", "hatch": "", "alpha": 0.8}, # Green for Transformations
        }

        

        visual_map_plot = {
        coordl_label: {'color': '#A3B6CE', 'linestyle': '-', 'linewidth': line_width, 'edgecolor': 'black', 'hatch': '//','alpha': 1.0}, #indianred
        disdl_label: {'color': 'black', 'linestyle': '-', 'linewidth': line_width,'edgecolor': 'black', 'hatch':'', 'alpha': 1.0}, #steelblue
        tensorsocket_label: {'color': '#FDA300', 'linestyle': '-', 'linewidth': line_width, 'edgecolor': 'black', 'hatch': '...','alpha': 1.0}, #indianred
        }

        gpu_hatch = ''
        io_hatch = '..'
        transform_hatch = '///'

        gpu_aplha = 1.0
        io_alpha = 1.0
        transform_alpha = 1.0


        x = np.arange(len(model_names))  # X-axis positions for models

        # Bars for TensorSocket
        ax4.bar(x - bar_width, tensorsocket_compute_pct, bar_width, label="GPU", 
                color=visual_map_plot[tensorsocket_label]["color"], 
                edgecolor="black", 
                hatch=gpu_hatch, 
                alpha=gpu_aplha)
        
        ax4.bar(x - bar_width, tensorsocket_transform_pct, bar_width, label="Transform", 
                color=visual_map_plot[tensorsocket_label]["color"], 
                edgecolor="black", 
                hatch=transform_hatch,
                bottom=tensorsocket_compute_pct,
                alpha=0.7)
        

        ax4.bar(x - bar_width, tensorsocket_io_pct, bar_width, label="I/O", 
                color=visual_map_plot[tensorsocket_label]["color"], 
                edgecolor="black", 
                hatch=io_hatch,
                alpha=io_alpha,
                bottom=tensorsocket_compute_pct + tensorsocket_transform_pct)
  



 
        # Bars for CoordL
        
        ax4.bar(x, coodl_compute_pct, 
                bar_width, label="GPU", 
                color=visual_map_plot[coordl_label]["color"], 
                edgecolor="black", 
                hatch=gpu_hatch, 
                alpha=gpu_aplha,
                )
        ax4.bar(x, coordl_transform_pct, bar_width, label="Transform", 
                color=visual_map_plot[coordl_label]["color"], 
                edgecolor="black", 
                hatch=transform_hatch,
                alpha=0.7,
                bottom=coodl_compute_pct)
        
        ax4.bar(x, coordl_io_pct, bar_width, label="I/O", 
                color=visual_map_plot[coordl_label]["color"], 
                edgecolor="black", 
                hatch=io_hatch,
                alpha=io_alpha,
                bottom=coodl_compute_pct + coordl_transform_pct)
        


        # Bars for DISDL (placed last)
                
        ax4.bar(x + bar_width, disdl_compute_pct, bar_width, label="GPU", 
                color=visual_map_plot[disdl_label]["color"], 
                edgecolor="black", 
                hatch=gpu_hatch, 
                alpha=gpu_aplha)
        
        ax4.bar(x + bar_width, 
                disdl_transform_pct, 
                bar_width, label="Transform", 
                color=visual_map_plot[disdl_label]["color"], 
                edgecolor="black", 
                hatch=transform_hatch,
                bottom=disdl_compute_pct, 
                alpha=0.4)


        ax4.bar(x + bar_width, disdl_io_pct, bar_width, label="I/O", 
                color='white', 
                edgecolor="black", 
                hatch=io_hatch,
                alpha=io_alpha,
                bottom=disdl_compute_pct + disdl_transform_pct)
        


        # ax4.legend(ncol=6, fontsize=9, loc="upper center")  # Moves legend above plot
        ax4.set_xticks(x + bar_width / 2)
        ax4.set_xticklabels(model_names, rotation=0, ha="center")
        ax4.set_ylabel("Time Breakdown (%)", fontsize=font_size)
        #set font size for all lables and ticks
        ax4.tick_params(axis='both', which='major', labelsize=font_size)
        # ax4.set_yticks(ticks=np.arange(0, 101, 20), labels=[f'{i}%' for i in np.arange(0, 101, 20)])
        ax4.set_ylim(0, 120)

        # Get existing legend handles and labels
        handles, labels = ax4.get_legend_handles_labels()

        # Remove duplicates while preserving hatches from the bars
        unique = {}
        for h, l in zip(handles, labels):
                if l not in unique:
                        # If it's a BarContainer, take the first actual bar in *that* group
                        if hasattr(h, "patches") and h.patches:
                                patch = h.patches[3]
                        else:
                                patch = h
                        unique[l] = patch
        legend_handles = [
        Patch(
                facecolor="white",
                edgecolor=patch.get_edgecolor(),
                hatch=patch.get_hatch(),
                label=label
        )
        for label, h in unique.items()
        ]
    # Remove duplicates from the legend (preserving order)
#     unique_labels = dict(zip(labels, handles))
        handles = [Patch(facecolor="white", edgecolor="black", label=label) for label in labels]
        unique_labels = dict(zip(labels, handles))


        # Update the legend with proper formatting
        ax4.legend(
                legend_handles, 
                list(unique.keys()),   # Legend handles (with empty handles for system names)
                loc="upper center",  # Preferred location
                ncol=3,  # Number of columns
                fontsize=11,
                handlelength=1,  # Set handle length to 0 to avoid color patches
                handleheight=1,  # Set handle height to 0 to remove unnecessary space
        )








        plt.tight_layout()
        plt.show()


