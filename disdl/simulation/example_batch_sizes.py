import matplotlib.pyplot as plt
import numpy as np

# Example data (adjust to your actual experiment results)
batch_sizes = [16, 32, 64, 128, 256, 1024]
serverless_times = [1.2, 2.3, 3.5, 4.8, 5.9, 7.2]  # Time for serverless
traditional_times = [1.1, 2.0, 3.2, 4.4, 5.6, 6.9]  # Time for traditional
serverless_cost = [0.05, 0.10, 0.15, 0.22, 0.30, 0.35]  # Cost for serverless
traditional_cost = [0.04, 0.08, 0.12, 0.18, 0.25, 0.30]  # Cost for traditional
throughput_serverless = [100, 95, 90, 85, 80, 75]  # Throughput for serverless (images/sec)
throughput_traditional = [110, 105, 100, 95, 90, 85]  # Throughput for traditional

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Data Retrieval and Preprocessing Time (Stacked Bars)
axs[0, 0].bar(batch_sizes, serverless_times, color='blue', label='Serverless')
axs[0, 0].bar(batch_sizes, traditional_times, color='orange', label='Traditional', alpha=0.7)
axs[0, 0].set_title('Data Retrieval & Preprocessing Time per Batch')
axs[0, 0].set_xlabel('Batch Size')
axs[0, 0].set_ylabel('Time (seconds)')
axs[0, 0].legend()

# Plot 2: Throughput vs. Batch Size
axs[0, 1].plot(batch_sizes, throughput_serverless, label='Serverless', marker='o', color='blue')
axs[0, 1].plot(batch_sizes, throughput_traditional, label='Traditional', marker='s', color='orange')
axs[0, 1].set_title('Throughput vs. Batch Size')
axs[0, 1].set_xlabel('Batch Size')
axs[0, 1].set_ylabel('Throughput (images/sec)')
axs[0, 1].legend()

# Plot 3: Cost vs. Batch Size
axs[1, 0].plot(batch_sizes, serverless_cost, label='Serverless', marker='o', color='blue')
axs[1, 0].plot(batch_sizes, traditional_cost, label='Traditional', marker='s', color='orange')
axs[1, 0].set_title('Cost vs. Batch Size')
axs[1, 0].set_xlabel('Batch Size')
axs[1, 0].set_ylabel('Cost (USD)')
axs[1, 0].legend()

# Plot 4: Total Time vs. Batch Size
total_time_serverless = np.array(serverless_times) + np.array([0.5] * len(batch_sizes))  # Adding arbitrary preprocessing time
total_time_traditional = np.array(traditional_times) + np.array([0.5] * len(batch_sizes))
axs[1, 1].plot(batch_sizes, total_time_serverless, label='Serverless', marker='o', color='blue')
axs[1, 1].plot(batch_sizes, total_time_traditional, label='Traditional', marker='s', color='orange')
axs[1, 1].set_title('Total Time vs. Batch Size')
axs[1, 1].set_xlabel('Batch Size')
axs[1, 1].set_ylabel('Total Time (seconds)')
axs[1, 1].legend()

# Layout adjustments
plt.tight_layout()
plt.show()
