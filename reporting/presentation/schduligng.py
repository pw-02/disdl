import matplotlib.pyplot as plt
import numpy as np

# Example data
subgraphs = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']
costs = [90, 85, 30, 25, 10, 10, 5]  # time units
num_workers = 3

# Round-Robin assignment
rr_assignments = {w: [] for w in range(num_workers)}
for i in range(len(costs)):
    rr_assignments[i % num_workers].append(i)

# Greedy (LPT) assignment
order = sorted(range(len(costs)), key=lambda i: costs[i], reverse=True)
greedy_assignments = {w: [] for w in range(num_workers)}
loads = [0] * num_workers
for idx in order:
    w = loads.index(min(loads))
    greedy_assignments[w].append(idx)
    loads[w] += costs[idx]

# Function to compute makespan
def makespan(assignments):
    return max(sum(costs[i] for i in tasks) for tasks in assignments.values())

makespan_rr = makespan(rr_assignments)
makespan_greedy = makespan(greedy_assignments)

# Create side-by-side subplots
fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
y_pos = np.arange(num_workers)
colors = plt.cm.tab10(np.linspace(0, 1, len(costs)))  # Unique colors for each subgraph

for ax, assignments, title, ms in zip(
    axes,
    [rr_assignments, greedy_assignments],
    ['Round-Robin Scheduling', 'Greedy (LPT) Scheduling'],
    [makespan_rr, makespan_greedy]
):
    # Plot bars with labels
    for w in range(num_workers):
        start = 0
        for idx in assignments[w]:
            ax.barh(y_pos[w], costs[idx], left=start, color=colors[idx], alpha=0.6, edgecolor='black', height=0.5)
            ax.text(start + costs[idx] / 2, y_pos[w], f'{costs[idx]}', va='center', ha='center', color='black')
            start += costs[idx]
            #plot y axis label for first subplot only
            if ax is axes[0]:
                # ax.text(-5, y_pos[w], f'Worker {w+1}', va='center', ha='right', fontsize=10, color='black')
                ax.set_yticklabels([f'Worker {i+1}' for i in range(num_workers)] if ax is axes[0] else [])
    
    # Makespan line and legend
    # ax.axvline(ms, linestyle='--', color='gray', label=f'Makespan = {ms}')
    # ax.set_title(title)
    # ax.set_xlabel('Cost (time units)')
    # ax.set_yticks(y_pos)
    # ax.set_yticklabels([f'Worker {i+1}' for i in range(num_workers)] if ax is axes[0] else [])
    # ax.legend(loc='upper right')
    ax.grid(axis='x', linestyle=':', linewidth=0.5)

# Super title and layout adjustments
# fig.suptitle('Scheduling Comparison for Subgraph Proving', fontsize=14)
plt.tight_layout()

plt.show()
