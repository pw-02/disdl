import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to load and process NDJSON file
def load_usage_data(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")  # Convert timestamp
    return df

# Load both datasets
file1 = r"C:\Users\pw\Desktop\disdl(today)\nas\imagenet_nas\disdl\2025-03-11_15-54-09\resource_usage_metrics.json"  
file2 = r"C:\Users\pw\Desktop\disdl(today)\nas\imagenet_nas\tensorsocket\2025-03-11_17-16-11\resource_usage_metrics.json"

df1 = load_usage_data(file1)
df2 = load_usage_data(file2)

# Extract maximum values for CPU and GPU usage
max_cpu_1 = df1["Average CPU Usage (%)"].mean()
max_gpu_1 = df1["Average GPU Usage (%)"].mean()
max_cpu_2 = df2["Average CPU Usage (%)"].mean()
max_gpu_2 = df2["Average GPU Usage (%)"].mean()

# Data for plotting
categories = ["CPU", "GPU"]
system_1_values = [max_cpu_1, max_gpu_1]
system_2_values = [max_cpu_2, max_gpu_2]

x = np.arange(len(categories))  # X positions for categories
width = 0.3  # Width of bars

# Plot bar chart
plt.figure(figsize=(6, 5))
plt.bar(x - width / 2, system_1_values, width, label="System 1", color="blue", alpha=0.7)
plt.bar(x + width / 2, system_2_values, width, label="System 2", color="red", alpha=0.7)

# Formatting
plt.xticks(x, categories)  # Set x-axis labels as CPU and GPU
plt.ylabel("Max Utilization (%)")
plt.title("Maximum CPU and GPU Utilization Comparison")
plt.ylim(0, 100)  # Assuming percentage values
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.show()
