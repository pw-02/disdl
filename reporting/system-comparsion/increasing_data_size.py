import numpy as np
import matplotlib.pyplot as plt

# Simulated dataset sizes (number of batches per epoch)
dataset_sizes = np.array([100, 500, 1000, 5000, 10000, 50000])

# Simulated cost values for two data loading systems
costs_system1 = np.array([5, 20, 40, 100, 200, 500])
costs_system2 = np.array([4, 18, 35, 90, 180, 450])

# Simulated throughput values for two data loading systems
throughput_system1 = np.array([1200, 1000, 900, 700, 600, 400])
throughput_system2 = np.array([1300, 1100, 950, 750, 650, 450])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot cost comparison
ax1.set_xlabel("Dataset Size (Batches per Epoch)")
ax1.set_ylabel("Cost (USD)")
ax1.plot(dataset_sizes, costs_system1, 'o-', color='red', label='System 1')
ax1.plot(dataset_sizes, costs_system2, 's-', color='blue', label='System 2')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_title("Cost vs Dataset Size")
ax1.legend()

# Plot throughput comparison
ax2.set_xlabel("Dataset Size (Batches per Epoch)")
ax2.set_ylabel("Throughput (images/sec)")
ax2.plot(dataset_sizes, throughput_system1, 'o-', color='red', label='System 1')
ax2.plot(dataset_sizes, throughput_system2, 's-', color='blue', label='System 2')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_title("Throughput vs Dataset Size")
ax2.legend()

# Layout adjustments
fig.tight_layout()
plt.show()