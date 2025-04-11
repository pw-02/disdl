import matplotlib.pyplot as plt

# Data from the ablation table
variants = ['Full System', 'No Coord. Cache', 'No Preloading', 'No Central Sampler']
throughput = [92.5, 88.7, 85.4, 84.0]
cost = [1.050, 0.031, 0.890, 0.027]
latency = [24.1, 27.3, 30.5, 33.2]

# Set up subplots with a slightly larger figure size for clarity
fig, axs = plt.subplots(3, 1, figsize=(6, 4), sharex=True)

# Set color palette with more professional tones
bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Throughput plot
axs[0].bar(variants, throughput, color=bar_colors[0], width=0.6)
axs[0].set_ylabel('Throughput (samples/s)', fontsize=12)
axs[0].set_title('Ablation Study: Impact on System Metrics', fontsize=14)
axs[0].grid(True, linestyle='--', alpha=0.6)

# Cost plot
axs[1].bar(variants, cost, color=bar_colors[1], width=0.6)
axs[1].set_ylabel('Cost ($)', fontsize=12)
axs[1].grid(True, linestyle='--', alpha=0.6)

# Latency plot
axs[2].bar(variants, latency, color=bar_colors[2], width=0.6)
axs[2].set_ylabel('Latency (ms)', fontsize=12)
axs[2].set_xticklabels(variants, rotation=15, fontsize=10)
axs[2].grid(True, linestyle='--', alpha=0.6)

# Improve layout and spacing
plt.tight_layout()

# Show the plot
plt.show()
