import matplotlib.pyplot as plt
import pandas as pd

# Set global font size
plt.rcParams.update({'font.size': 18})

# Cache hit percentage data
cache_data = {
    'Workload': ['ImageNet NAS', 'ImageNet NAS', 'ImageNet NAS',
                 'OpenImages NAS', 'OpenImages NAS', 'OpenImages NAS',
                 'COCO NAS', 'COCO NAS', 'COCO NAS'],
    'System': ['DisDL', 'TensorSocket', 'ReCoorDL',
               'DisDL', 'TensorSocket', 'ReCoorDL',
               'DisDL', 'TensorSocket', 'ReCoorDL'],
    'Cache Hit (%)': [98.1, 75.2, 74.5, 98.2, 74.6, 69.0, 96.2, 75.6, 85.7]
}

df_cache = pd.DataFrame(cache_data)

# Workload order and system colors
workloads = ['ImageNet NAS', 'OpenImages NAS', 'COCO NAS']
system_colors = {
    'DisDL': '#FDA300',
    'TensorSocket': '#A3B6CE',
    'ReCoorDL': 'black'
}

# Set up the figure
fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True)

# Plot each workload
for ax, workload in zip(axes, workloads):
    df_workload = df_cache[df_cache['Workload'] == workload]
    
    systems = df_workload['System'].values
    values = df_workload['Cache Hit (%)'].values
    colors = [system_colors[sys] for sys in systems]

    # Bar positions and width
    x = range(len(systems))
    bar_width = 0.5

    # Plot bars
    bars = ax.bar(x, values, color=colors, edgecolor='black', width=bar_width, linewidth=2)
    
    # Customize axes
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=18)
    ax.set_ylabel('Cache Hit (%)', fontsize=18)
    ax.set_title(workload, fontsize=18)

# Layout adjustment
plt.tight_layout()
plt.show()
