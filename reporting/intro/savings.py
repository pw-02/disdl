import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Configuration
epochs = 90
batch_size = 128
dataset_size = 1280000  # ImageNet-1K
batches_per_epoch = dataset_size // batch_size
shared_batches = epochs * batches_per_epoch  # 900,000

# X-axis: Number of concurrent jobs
job_counts = list(range(1, 31))

# Y-axis: Total minibatches generated
without_sharing = [jobs * shared_batches for jobs in job_counts]
with_sharing = [shared_batches for _ in job_counts]

# Formatter: display y-axis in millions (e.g., 18M)
def millions(x, pos):
    return f'{x * 1e-6:.0f}M'

# Set font size globally
plt.rcParams.update({'font.size': 17})

# Plot
plt.figure(figsize=(8, 5))

# Use colorblind-safe, print-friendly colors
plt.plot(job_counts, without_sharing, label="Without sharing", linewidth=2, marker='o', color='#1f77b4')  # Dark blue
plt.plot(job_counts, with_sharing, label="With Ideal Sharing", linewidth=4, linestyle='--', color='black')  # Black line

# Light gray fill for redundancy region
plt.fill_between(job_counts, with_sharing, without_sharing, color='orange', alpha=0.1, label='Redundant Minibatches')

# Label the ideal line (900K)
# plt.text(job_counts[-1], shared_batches, "900K Minibatches", ha='right', va='bottom', fontsize=17, color='black')

# Axes formatting
plt.xlabel("Number of Concurrent Jobs", fontsize=17)
plt.ylabel("Total Minibatches Generated", fontsize=17)
plt.gca().yaxis.set_major_formatter(FuncFormatter(millions))
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.legend(fontsize=17)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()

# Optional: Save the figure
# plt.savefig("minibatch_sharing_plot.pdf", bbox_inches='tight')

plt.show()
