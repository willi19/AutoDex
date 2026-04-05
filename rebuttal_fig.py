import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ['Wood', 'Plastic', 'Metal', 'Paper', 'Table', 'Wall', 'Clutt.', 'Light', 'Middle', 'Heavy']
group_labels = ['Material', 'Scene', 'Weight']
group_sizes = [4, 3, 3]

sim_vals = [19.5, 51.7, 30.9, 42.6, 36.8, 13.5, 6.7, 59.1, 18.5, 22.5]
sim_errs = [8.3, 12.6, 7.7, 12.4, 5.7, 11.0, 12.6, 9.0, 8.5, 6.7]
real_vals = [67.5, 73.3, 75.7, 90.3, 81.7, 67.6, 86.7, 73.7, 73.8, 83.9]
real_errs = [14.5, 11.2, 13.8, 10.4, 7.9, 15.1, 17.2, 9.9, 11.0, 12.9]

# Layout
fig, ax = plt.subplots(figsize=(7, 1.5))

x = np.arange(len(categories))
bar_w = 0.35

# Bars
ax.bar(x - bar_w/2, sim_vals, bar_w, yerr=sim_errs, label='Sim-validated',
       color='#6C9BD2', edgecolor='#3A6EA5', linewidth=0.5,
       capsize=2, error_kw={'linewidth': 0.7})
ax.bar(x + bar_w/2, real_vals, bar_w, yerr=real_errs, label='Real-filtered',
       color='#E8836B', edgecolor='#C0392B', linewidth=0.5,
       capsize=2, error_kw={'linewidth': 0.7})

# Group separators
sep_positions = [3.5, 6.5]  # between Paper/Table, Clutt./Light
for sp in sep_positions:
    ax.axvline(sp, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

# Group labels at top
group_centers = [1.5, 5.0, 8.0]
for label, cx in zip(group_labels, group_centers):
    ax.text(cx, 103, label, ha='center', va='bottom', fontsize=7, fontstyle='italic', color='gray')

# Axes
ax.set_ylabel('Success (%)', fontsize=8)
ax.set_ylim(0, 110)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=7, rotation=0, ha='center')
ax.tick_params(axis='y', labelsize=7)
ax.yaxis.grid(True, linewidth=0.3, alpha=0.5)
ax.set_axisbelow(True)
ax.legend(fontsize=7, loc='upper left', framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout(pad=0.3)
plt.savefig('rebuttal_fig1.pdf', bbox_inches='tight', dpi=300)
plt.savefig('rebuttal_fig1.png', bbox_inches='tight', dpi=300)
print("Saved rebuttal_fig1.pdf and rebuttal_fig1.png")
