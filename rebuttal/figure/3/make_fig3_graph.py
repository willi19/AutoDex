"""Figure 3 bar chart: text-free version + legend separate."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

categories = ['Wood', 'Plastic', 'Metal', 'Paper', 'Table', 'Wall', 'Clutt.', 'Light', 'Middle', 'Heavy']
sim_vals = [19.5, 51.7, 30.9, 42.6, 36.8, 13.5, 6.7, 59.1, 18.5, 22.5]
sim_errs = [8.3, 12.6, 7.7, 12.4, 5.7, 11.0, 12.6, 9.0, 8.5, 6.7]
real_vals = [67.5, 73.3, 75.7, 90.3, 81.7, 67.6, 86.7, 73.7, 73.8, 83.9]
real_errs = [14.5, 11.2, 13.8, 10.4, 7.9, 15.1, 17.2, 9.9, 11.0, 12.9]

x = np.arange(len(categories))
bar_w = 0.35

TICK_SIZE = 7

# --- Bar chart, text-free ---
fig, ax = plt.subplots(figsize=(7, 1.5), dpi=300)

ax.bar(x - bar_w/2, sim_vals, bar_w, yerr=sim_errs,
       color='#6C9BD2', edgecolor='#3A6EA5', linewidth=0.5,
       capsize=2, error_kw={'linewidth': 0.7})
ax.bar(x + bar_w/2, real_vals, bar_w, yerr=real_errs,
       color='#E8836B', edgecolor='#C0392B', linewidth=0.5,
       capsize=2, error_kw={'linewidth': 0.7})

# Group separators
for sp in [3.5, 6.5]:
    ax.axvline(sp, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

ax.set_ylim(0, 110)
ax.set_xticks([])
ax.tick_params(axis='y', labelsize=TICK_SIZE)
ax.yaxis.grid(True, linewidth=0.3, alpha=0.5)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.savefig(os.path.join(OUT_DIR, "fig3_graph.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
fig.savefig(os.path.join(OUT_DIR, "fig3_graph.pdf"), bbox_inches='tight', pad_inches=0.02, dpi=300)
plt.close(fig)
print("Saved fig3_graph")

# --- Legend only ---
fig_leg, ax_leg = plt.subplots(figsize=(2.5, 0.3), dpi=300)
ax_leg.bar([], [], bar_w, color='#6C9BD2', edgecolor='#3A6EA5', linewidth=0.5, label='w/o Real Valid.')
ax_leg.bar([], [], bar_w, color='#E8836B', edgecolor='#C0392B', linewidth=0.5, label='w/ Real Valid.')
ax_leg.axis('off')
ax_leg.legend(fontsize=TICK_SIZE, loc='center', ncol=2, frameon=False)
fig_leg.savefig(os.path.join(OUT_DIR, "fig3_legend.png"), bbox_inches='tight', pad_inches=0.01, dpi=300)
fig_leg.savefig(os.path.join(OUT_DIR, "fig3_legend.pdf"), bbox_inches='tight', pad_inches=0.01, dpi=300)
plt.close(fig_leg)
print("Saved fig3_legend")
