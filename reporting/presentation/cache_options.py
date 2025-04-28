import matplotlib.pyplot as plt

# Strategies and their characteristics
strategies = [
    "Raw Images (No Transform)",
    "Raw Images + Transform",
    "Tensor (No Transform)",
    "Tensor + Transform"
]

# X-axis: Relative storage size (larger = uses more memory)
storage_size = [1, 6.5, 4, 8]

# Y-axis: Preprocessing needed at runtime (higher = slower training)
runtime_preprocessing = [10, 5, 2, 0.5]

# Plot setup
plt.figure(figsize=(6, 4))
colors = ['C0', 'C1', 'C2', 'C3']

for i, strategy in enumerate(strategies):
    plt.scatter(storage_size[i], runtime_preprocessing[i],
                s=200, color=colors[i], label=strategy, edgecolor='black')
    # Annotate each point with strategy name
    plt.text(storage_size[i] + 0.2, runtime_preprocessing[i] + 0.2,
             strategy, fontsize=10, color='black')

# Axes labels and title
plt.xlabel("Relative Storage Size", fontsize=14)
plt.ylabel("Relative Preprocessing Time", fontsize=14)
plt.title("Trade-offs Between Storage and Preprocessing Efficiency", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# Grid, legend, and styling
plt.grid(True, linestyle='--', alpha=0.2)
plt.xlim(0, 12)
plt.ylim(0, 11)
plt.legend().set_visible(False)  # legend is now embedded in point labels
plt.tight_layout()
plt.show()
#set x adn y ticks font size

