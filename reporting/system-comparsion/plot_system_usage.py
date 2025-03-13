import json
import pandas as pd
import matplotlib.pyplot as plt

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
file1 = r"C:\Users\pw\Desktop\disdl(today)\nas\imagenet_nas\disdl\2025-03-11_15-54-09\resource_usage_metrics.json"  # Change to your actual file path
file2 = r"C:\Users\pw\Desktop\disdl(today)\nas\imagenet_nas\tensorsocket\2025-03-11_17-16-11\resource_usage_metrics.json"

df1 = load_usage_data(file1)
df2 = load_usage_data(file2)

# Plot comparison
plt.figure(figsize=(12, 6))

# CPU Usage Comparison
# plt.plot(df1["timestamp"], df1["Average CPU Usage (%)"], label="System 1 - CPU", marker="o")
plt.plot(df2["timestamp"], df2["Average CPU Usage (%)"], label="System 2 - CPU", marker="o", linestyle="dashed")

# GPU Usage Comparison
# plt.plot(df1["timestamp"], df1["Average GPU Usage (%)"], label="System 1 - GPU", marker="s")
plt.plot(df2["timestamp"], df2["Average GPU Usage (%)"], label="System 2 - GPU", marker="s", linestyle="dashed")

plt.xlabel("Time")
plt.ylabel("Usage (%)")
plt.title("Comparison of Average CPU & GPU Usage Over Time")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)

plt.show()
