import json
import matplotlib.pyplot as plt

# Load JSON data from file
with open(r"C:\Users\pw\Desktop\disdl(600)\imagenet_nas\disdl\2025-03-13_20-34-34\resource_usage_metrics.json", "r") as file:
    data = json.load(file)

# Extract elapsed time and average CPU usage
elapsed_time = [entry["elapsed_time"] for entry in data]
avg_cpu_usage = [entry["Average CPU Usage (%)"] for entry in data]

# Plot the data
plt.figure(figsize=(8, 5))
plt.plot(elapsed_time, avg_cpu_usage, marker='o', linestyle='-', color='b', label="Average CPU Usage (%)")

# Labels and title
plt.xlabel("Elapsed Time (s)")
plt.ylabel("Average CPU Usage (%)")
plt.title("Average CPU Usage Over Time")
plt.legend()
plt.grid(True)

# Show plot
plt.show()
