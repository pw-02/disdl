import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set global font size
plt.rcParams.update({'font.size': 18})

# Data
data = {
    'Workload': ['ImageNet NAS', 'ImageNet NAS', 'ImageNet NAS',
                 'OpenImages NAS', 'OpenImages NAS', 'OpenImages NAS',
                 'COCO NAS', 'COCO NAS', 'COCO NAS'],
    'System': ['DisDL', 'TensorSocket', 'ReCoorDL',
               'DisDL', 'TensorSocket', 'ReCoorDL',
               'DisDL', 'TensorSocket', 'ReCoorDL'],
    'CPU (%)': [37.7, 25.7, 45.7, 49.5, 27.1, 62.4, 60.8, 26.8, 64.1],
    'GPU (%)': [75.8, 49.6, 36.4, 67.4, 35.7, 27.2, 42.0, 35.5, 29.4]
}

df = pd.DataFrame(data)

# Set up subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True)
workloads = ['ImageNet NAS', 'OpenImages NAS', 'COCO NAS']
systems = ['DisDL', 'TensorSocket', 'ReCoorDL']
bar_width = 0.35

# Colors
color_cpu = '#FDA300'
color_gpu = '#A3B6CE'

# Plot
for ax, workload in zip(axes, workloads):
    df_w = df[df['Workload'] == workload]

    x = np.arange(len(systems))
    cpu_vals = df_w['CPU (%)'].values
    gpu_vals = df_w['GPU (%)'].values

    bars1 = ax.bar(x - bar_width/2, cpu_vals, width=bar_width, label='CPU (%)',
                   color=color_cpu, edgecolor='black', linewidth=2)
    bars2 = ax.bar(x + bar_width/2, gpu_vals, width=bar_width, label='GPU (%)',
                   color=color_gpu, edgecolor='black', linewidth=2)

    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=18)
    ax.set_ylabel('Usage (%)', fontsize=18)
    ax.set_title(workload, fontsize=18)

    if ax == axes[0]:
        ax.legend(fontsize=18)

# Layout
plt.tight_layout()
plt.show()
