import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Data
sub_circuits = ['Global Model', 'Conv', 'Clip', 'Conv', 'Clip', 'Conv', 'Add']
zk_constraints = [43_076_174, 15_353_870, 18_063_379, 15_353_867, 7_125_014, 1_655_810, 0]

# Divide all by 1 million
zk_constraints = [x / 1_000_000 for x in zk_constraints]

# Define colors
global_color = '#F26D6D'
split_color = '#A3B6CE'
colors = [global_color] + [split_color] * (len(sub_circuits) - 1)

# Plot
plt.figure(figsize=(8, 4.5))
bars = plt.bar(range(len(sub_circuits)), zk_constraints, 
               color=colors, edgecolor='black', linewidth=1.5, width=0.55)

# Set alpha manually per bar: only Global visible
bars[0].set_alpha(1.0)
for bar in bars[1:]:
    bar.set_alpha(0.0)

# X ticks: only show label for Global Model
tick_labels = ['Global Model'] + [''] * (len(sub_circuits) - 1)
plt.xticks(range(len(sub_circuits)), tick_labels, rotation=15, fontsize=13)

# Y axis
plt.yticks(fontsize=13)
plt.ylabel('ZK Constraints (Millions)', fontsize=14)

# Annotate only Global Model
# plt.text(0, zk_constraints[0] + 0.5, f'{zk_constraints[0]:.1f}', ha='center', va='bottom', fontsize=11)

# Custom Legend
legend_elements = [
    Patch(facecolor=global_color, edgecolor='black', label='Global Model'),
    # Patch(facecolor=split_color, edgecolor='black', label='Subcircuits')
]
# plt.legend(handles=legend_elements, fontsize=12, loc='upper right')

plt.tight_layout()
plt.show()
