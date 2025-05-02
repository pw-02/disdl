import matplotlib.pyplot as plt
import numpy as np

# Parameters
speed_ratios = np.linspace(1.0, 5.0, 100)  # r_fast / r_slow

# Compute normalized cache requirement: C_required / B
normalized_cache = 1 - 1 / speed_ratios

# Plot
plt.figure(figsize=(6, 4))
plt.plot(speed_ratios, normalized_cache, linewidth=2, label=r'Normalized Cache Requirement ($C_{\mathrm{required}} / B$)')
plt.axhline(1.0, linestyle='--', color='gray', label='Epoch Size (B)')
plt.xlabel(r'Job Speed Ratio ($r_{\mathrm{fast}} / r_{\mathrm{slow}}$)')
plt.ylabel(r'Fraction of Epoch in Cache')
plt.title('Normalized Cache Requirement vs. Job Speed Divergence')
plt.ylim(0, 1.05)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
