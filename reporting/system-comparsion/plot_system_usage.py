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
file1 = r"C:\Users\pw\Desktop\disdl(600)\imagenet_nas\disdl\2025-03-13_20-34-34\resource_usage_metrics.json"  
file2 = r"C:\Users\pw\Desktop\disdl(600)\imagenet_nas\tensorsocket\2025-03-13_19-41-51\resource_usage_metrics.json"
fil3 = r"C:\Users\pw\Desktop\disdl(600)\imagenet_nas\coordl\2025-04-04_12-16-09\resource_usage_metrics.json"
df1 = load_usage_data(file1)
df2 = load_usage_data(file2)
df3 = load_usage_data(fil3)

# Extract maximum values for CPU and GPU usage
max_cpu_1 = df1["Average CPU Usage (%)"].mean()
max_gpu_1 = df1["Average GPU Usage (%)"].mean()
max_cpu_2 = df2["Average CPU Usage (%)"].mean()
max_gpu_2 = df2["Average GPU Usage (%)"].mean()
max_cpu_3 = df3["Average CPU Usage (%)"].mean()
max_gpu_3 = df3["Average GPU Usage (%)"].mean()

# Data for plotting
categories = ["CPU", "GPU"]
system_1_values = [max_cpu_1, max_gpu_1]
system_2_values = [max_cpu_2, max_gpu_2]
system_3_values = [max_cpu_3, max_gpu_3]

x = np.arange(len(categories))  # X positions for categories
width = 0.3  # Width of bars

# Plot bar chart
plt.figure(figsize=(6, 5))
plt.bar(x - width / 2, system_1_values, width, label="System 1", color="blue", alpha=0.7)
plt.bar(x + width / 2, system_2_values, width, label="System 2", color="red", alpha=0.7)
plt.bar(x + width, system_3_values, width, label="System 3", color="green", alpha=0.7)

# Formatting
plt.xticks(x, categories)  # Set x-axis labels as CPU and GPU
plt.ylabel("Max Utilization (%)")
plt.title("Maximum CPU and GPU Utilization Comparison")
plt.ylim(0, 100)  # Assuming percentage values
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.show()
