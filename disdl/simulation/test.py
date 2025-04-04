import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_batches = np.linspace(1, 1_000_000, 1000)  # Increasing number of batches
lambda_cost_per_request = 0.20 / 1_000_000  # $0.20 per million requests
redis_fixed_cost = 5096.13   # 419.09 GiB

#25480650000 requests per month




# Cost functions
serverless_cost = num_batches * lambda_cost_per_request

# Plot
plt.figure(figsize=(8, 5))
plt.plot(num_batches, serverless_cost, label='Serverless Cache Cost', color='blue')
plt.axhline(y=redis_fixed_cost, color='red', linestyle='--', label='Redis Fixed Cost')

# Labels and Legend
plt.xlabel('Number of Batches')
plt.ylabel('Cost ($)')
plt.title('Serverless vs Redis Caching Cost Over Time')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
