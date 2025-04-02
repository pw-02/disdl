import matplotlib.pyplot as plt
import numpy as np

# Sample batch sizes
batch_sizes = [16, 32, 64, 128, 256, 512]

# System A (e.g., serverless)
cost_a = np.array([0.02, 0.03, 0.05, 0.08, 0.12, 0.20])  
throughput_a = np.array([500, 900, 1600, 2500, 3500, 4500])  

# System B (e.g., ElastiCache)
cost_b = np.array([0.015, 0.025, 0.04, 0.06, 0.09, 0.15])  
throughput_b = np.array([550, 950, 1700, 2700, 3800, 4800])  

plt.figure(figsize=(6, 3.5))

# Plot system A
plt.plot(cost_a, throughput_a, 'o-', color='blue', label="System A")
for i, txt in enumerate(batch_sizes):
    plt.annotate(txt, (cost_a[i], throughput_a[i]), fontsize=10, xytext=(5,5), textcoords='offset points')

# Plot system B
plt.plot(cost_b, throughput_b, 's-', color='red', label="System B")
for i, txt in enumerate(batch_sizes):
    plt.annotate(txt, (cost_b[i], throughput_b[i]), fontsize=10, xytext=(5,-10), textcoords='offset points')

# Compute Pareto Frontier
def pareto_frontier(cost, throughput):
    sorted_indices = np.argsort(cost)  
    cost, throughput = cost[sorted_indices], throughput[sorted_indices]  
    pareto_x, pareto_y = [cost[0]], [throughput[0]]  

    for i in range(1, len(cost)):
        if throughput[i] > pareto_y[-1]:  
            pareto_x.append(cost[i])
            pareto_y.append(throughput[i])

    return np.array(pareto_x), np.array(pareto_y)

pareto_x_a, pareto_y_a = pareto_frontier(cost_a, throughput_a)
pareto_x_b, pareto_y_b = pareto_frontier(cost_b, throughput_b)

# Plot Pareto Frontier
plt.plot(pareto_x_a, pareto_y_a, '--', color='blue', alpha=0.5, label="Pareto Frontier A")
plt.plot(pareto_x_b, pareto_y_b, '--', color='red', alpha=0.5, label="Pareto Frontier B")

# Labels and legend
plt.xlabel("Cost per Batch ($)")
plt.ylabel("Throughput (Images per Second)")
plt.title("Impact of Batch Size on Throughput and Cost with Pareto Frontier")
plt.legend()
plt.grid(True)

plt.show()
