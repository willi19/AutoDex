import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# DATA INPUT - MODIFY THIS SECTION
# =============================================================================

# X-axis for real methods (1-100)
x_real = np.arange(1, 101, 1)

# X-axis for sim methods (1-10000)
x_sim = np.concatenate([
    np.arange(1, 101, 1),  # 1-100: dense
    np.logspace(np.log10(100), np.log10(10000), 50)  # 100-10000: sparse
])

# TODO: Load your data here
# Example: data = np.load('coverage_data.npy')
# real_ours = data['real_ours']
# teleop = data['teleop']
# ...

# Real methods coverage (length = 100)
real_ours = 100 * (1 - np.exp(-x_real / 30))  # Replace with your data
teleop = 100 * (1 - np.exp(-x_real / 50))     # Replace with your data

# Sim methods coverage (length = len(x_sim))
tabletop = 100 * (1 - np.exp(-x_sim / 200))   # Replace with your data
collision = 100 * (1 - np.exp(-x_sim / 350))  # Replace with your data
sampling = 100 * (1 - np.exp(-x_sim / 400))   # Replace with your data
floating = 100 * (1 - np.exp(-x_sim / 500))   # Replace with your data

# =============================================================================
# PLOTTING - DO NOT MODIFY BELOW
# =============================================================================

# Figure setup - RSS single column width
fig, ax = plt.subplots(figsize=(3.5, 2.5))

# Plot lines
ax.plot(x_real, real_ours, 'o-', linewidth=2, markersize=3, markevery=10, 
        label='Real (Ours)', color='#d62728')
ax.plot(x_real, teleop, 's-', linewidth=2, markersize=3, markevery=10, 
        label='Teleop', color='#ff7f0e')
ax.plot(x_sim, tabletop, '^-', linewidth=2, markersize=3, markevery=10, 
        label='Tabletop', color='#2ca02c')
ax.plot(x_sim, collision, 'v-', linewidth=2, markersize=3, markevery=10, 
        label='Collision', color='#9467bd')
ax.plot(x_sim, sampling, 'd-', linewidth=2, markersize=3, markevery=10, 
        label='Sampling', color='#8c564b')
ax.plot(x_sim, floating, 'x-', linewidth=2, markersize=3, markevery=10, 
        label='Floating', color='#1f77b4')

# Vertical line at x=100 to mark real methods limit
# ax.axvline(x=100, color='gray', linestyle=':', linewidth=1, alpha=0.5)
# ax.text(100, 50, 'Real limit', rotation=90, va='center', fontsize=8, color='gray')

# Axes settings
ax.set_xscale('log')
ax.set_xlim(1, 10000)
ax.set_ylim(0, 105)

# Labels
ax.set_xlabel('Number of Grasps', fontsize=10)
ax.set_ylabel('Scene Coverage (%)', fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.legend(fontsize=8, loc='lower right', framealpha=0.9)

# X-axis ticks
ax.set_xticks([1, 10, 100, 1000, 10000])
ax.set_xticklabels(['1', '10', '100', '1K', '10K'])

# Save
plt.tight_layout()
plt.savefig('coverage_vs_grasps.pdf', bbox_inches='tight', pad_inches=0.01)
print("Saved: coverage_vs_grasps.pdf")