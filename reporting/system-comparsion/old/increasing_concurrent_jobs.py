import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator


# Set global font properties
plt.rc('font', family='serif')  # Set font family, weight, and size
plt.rc('axes', titlesize=16)  # Set the font size for axes titles
# plt.rc('axes', labelsize=14)  # Set the font size for axes labels


# Labels and data
coordl_label = 'CoorDL'
disdl_label = r'$\bf{DisDP}$'
baseline_label = 'Pytorch'
dataset_label = 'ImageNet'
line_width = 1.5

visual_map_plot_1 = {
    coordl_label: {'color': 'black', 'linestyle': ':', 'marker': '', 'linewidth': line_width},
    disdl_label: {'color': 'black', 'linestyle': '-', 'marker': '', 'linewidth': line_width},
    baseline_label: {'color': 'black', 'linestyle': '-', 'marker': '', 'linewidth': line_width},
}






# Data
concurrent_jobs = np.array([1, 2, 4, 6, 8, 10, 12, 14, 16])  # Number of concurrent jobs
coordl = np.array([136.5865588, 267.86081, 401.7902429, 535.7215442, 669.6590489, 803.5965539,
                   937.5354311, 1071.462984, 1205.394251, 1339.324019, 1473.263421])
super_vals = np.array([292.8484753, 426.7897239, 595.4783372, 795.8563516, 1328.265121, 1860.721967,
                       2701.305051, 2846.832603, 3046.519497, 3220.134065, 4300.065485])
ideal = super_vals  # Ideal follows "Super" values

# Create subplots
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8,2.7), sharex=False)

# Plot the throughput graph
ax1.plot(concurrent_jobs, coordl, 
         label=coordl_label,
         color=visual_map_plot_1[coordl_label]['color'], linestyle=visual_map_plot_1[coordl_label]['linestyle'], linewidth=visual_map_plot_1[coordl_label]['linewidth'])
ax1.plot(concurrent_jobs, super_vals, 
         label=disdl_label,
         color=visual_map_plot_1[disdl_label]['color'], linestyle=visual_map_plot_1[disdl_label]['linestyle'], linewidth=visual_map_plot_1[disdl_label]['linewidth'])
# ax1.plot(concurrent_jobs, ideal,
#          label=disdl_label,
#          color=visual_map_plot_1[disdl_label]['color'], linestyle=visual_map_plot_1[disdl_label]['linestyle'], linewidth=visual_map_plot_1[disdl_label]['linewidth'])

# Labels and Title for ax1
ax1.set_xticks(concurrent_jobs)
ax1.set_ylabel("Throughput (Samples/sec)", fontsize=12)
ax1.set_xlabel("Number of Concurrent Jobs", fontsize=12)
ax1.legend()

# Plot the latency graph
batch_retrieval_latency = np.array([10, 12, 14, 16, 18, 20, 22, 24, 27, 30, 33])  # in milliseconds
preprocessing_latency = np.array([5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18])  # in milliseconds

ax2.plot(concurrent_jobs, batch_retrieval_latency, 
         label='Batch Retrieval Latency',
         color=visual_map_plot_1[coordl_label]['color'], linestyle=visual_map_plot_1[coordl_label]['linestyle'], linewidth=visual_map_plot_1[coordl_label]['linewidth'])
ax2.plot(concurrent_jobs, preprocessing_latency, 
         label='Preprocessing Latency',
         color=visual_map_plot_1[disdl_label]['color'], linestyle=visual_map_plot_1[disdl_label]['linestyle'], linewidth=visual_map_plot_1[disdl_label]['linewidth'])

# Labels and Title for ax2
ax2.set_xticks(concurrent_jobs)
ax2.set_ylabel("Average Latency/Batch (ms)", fontsize=12)
ax2.set_xlabel("Number of Concurrent Jobs", fontsize=12)
ax2.legend()

# Improve x-tick formatting
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # Ensures ticks are integers
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))  # Ensures ticks are integers

# Make sure the layout is tight and no overlaps
plt.tight_layout()

# Show the plot
plt.show()
