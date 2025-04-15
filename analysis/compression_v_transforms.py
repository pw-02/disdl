import matplotlib.pyplot as plt
import numpy as np

# Dummy data for memory sizes (in MB)
batch_sizes = [64, 128, 256]
transformations_memory = [75, 150, 300]  # Memory sizes with transformations
compression_memory_50 = [30, 60, 120]  # Memory sizes with 50% compression
compression_memory_75 = [20, 40, 80]  # Memory sizes with 75% compression

# Plot
plt.figure(figsize=(8, 6))
plt.plot(batch_sizes, transformations_memory, label="With Transformations", marker='o')
plt.plot(batch_sizes, compression_memory_50, label="50% Compression", marker='o')
plt.plot(batch_sizes, compression_memory_75, label="75% Compression", marker='o')

plt.xlabel("Mini-batch Size", fontsize=12)
plt.ylabel("Memory Size (MB)", fontsize=12)
plt.title("Memory Size Comparison: Transformations vs. Compression", fontsize=14)
plt.legend()
plt.grid(True)
plt.xticks(batch_sizes)
plt.tight_layout()
plt.show()
# Dummy data for time (in seconds)
transformation_time = [0.8, 1.5, 2.5]  # Time to apply transformations
decompression_time_50 = [0.3, 0.5, 0.7]  # Time to decompress with 50% compression

# Plot
plt.figure(figsize=(8, 6))
plt.plot(batch_sizes, transformation_time, label="Transformation Time", marker='o')
plt.plot(batch_sizes, decompression_time_50, label="Decompression Time (50% Compression)", marker='o')

plt.xlabel("Mini-batch Size", fontsize=12)
plt.ylabel("Time (seconds)", fontsize=12)
plt.title("Transformation Time vs. Decompression Time", fontsize=14)
plt.legend()
plt.grid(True)
plt.xticks(batch_sizes)
plt.tight_layout()
plt.show()

# Dummy data for retrieval time (in seconds)
retrieval_time_transformations = [0.4, 0.8, 1.2]  # Time for transformed data
retrieval_time_compression_50 = [0.3, 0.5, 0.6]  # Time for compressed data (50%)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(batch_sizes, retrieval_time_transformations, label="Retrieval Time (Transformations)", marker='o')
plt.plot(batch_sizes, retrieval_time_compression_50, label="Retrieval Time (50% Compression)", marker='o')

plt.xlabel("Mini-batch Size", fontsize=12)
plt.ylabel("Retrieval Time (seconds)", fontsize=12)
plt.title("Data Retrieval Time: Transformations vs. Compression", fontsize=14)
plt.legend()
plt.grid(True)
plt.xticks(batch_sizes)
plt.tight_layout()
plt.show()


# Dummy data for costs (in dollars)
cost_transformations = [0.10, 0.20, 0.40]  # Cost with transformations
cost_compression_50 = [0.05, 0.10, 0.15]  # Cost with 50% compression
cost_compression_75 = [0.03, 0.06, 0.10]  # Cost with 75% compression

# Plot
plt.figure(figsize=(8, 6))
plt.plot(batch_sizes, cost_transformations, label="With Transformations", marker='o')
plt.plot(batch_sizes, cost_compression_50, label="50% Compression", marker='o')
plt.plot(batch_sizes, cost_compression_75, label="75% Compression", marker='o')

plt.xlabel("Mini-batch Size", fontsize=12)
plt.ylabel("Cost (USD)", fontsize=12)
plt.title("Cost Comparison: Transformations vs. Compression", fontsize=14)
plt.legend()
plt.grid(True)
plt.xticks(batch_sizes)
plt.tight_layout()
plt.show()
