"""Rebuttal figure 3: Physics validation bar plot with confidence intervals.

Sim-validated vs Real-filtered success rates by Material / Scene / Weight.

Usage:
    python rebuttal/plot_phys_validation.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure", "3")
OBJ_LIST_IMG = os.path.join(OUT_DIR, "obj_list.jpg")

# Data
categories = ['Wood', 'Plastic', 'Metal', 'Paper', 'Table', 'Wall', 'Clutt.', 'Light', 'Middle', 'Heavy']
group_labels = ['Material', 'Scene', 'Weight']
group_sizes = [4, 3, 3]

sim_vals = [19.5, 51.7, 30.9, 42.6, 36.8, 13.5, 6.7, 59.1, 18.5, 22.5]
sim_errs = [8.3, 12.6, 7.7, 12.4, 5.7, 11.0, 12.6, 9.0, 8.5, 6.7]
real_vals = [67.5, 73.3, 75.7, 90.3, 81.7, 67.6, 86.7, 73.7, 73.8, 83.9]
real_errs = [14.5, 11.2, 13.8, 10.4, 7.9, 15.1, 17.2, 9.9, 11.0, 12.9]

# Layout
fig, ax = plt.subplots(figsize=(5, 1.5))

x = np.arange(len(categories))
bar_w = 0.35

# Bars
ax.bar(x - bar_w/2, sim_vals, bar_w, yerr=sim_errs, label='w/o Real Valid.',
       color='#6C9BD2', edgecolor='#3A6EA5', linewidth=0.5,
       capsize=2, error_kw={'linewidth': 0.7})
ax.bar(x + bar_w/2, real_vals, bar_w, yerr=real_errs, label='w/ Real Valid.',
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
ax.legend(fontsize=6, loc='upper left', framealpha=0.5, edgecolor='none')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

os.makedirs(OUT_DIR, exist_ok=True)
plt.tight_layout(pad=0.3)

# Render bar plot to image
fig.canvas.draw()
buf = fig.canvas.buffer_rgba()
plot_img = np.asarray(buf)[:, :, :3].copy()  # RGB
plt.close()

# Load obj_list.jpg and stitch: obj_list (left) + bar plot (right)
obj_img = cv2.imread(OBJ_LIST_IMG)
obj_img = cv2.cvtColor(obj_img, cv2.COLOR_BGR2RGB)

# Match heights
target_h = obj_img.shape[0]
plot_h, plot_w = plot_img.shape[:2]
scale = target_h / plot_h
plot_img = cv2.resize(plot_img, (int(plot_w * scale), target_h))

combined = np.hstack([obj_img, plot_img])
combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

cv2.imwrite(os.path.join(OUT_DIR, 'rebuttal_fig3.png'), combined_bgr)

# Also save as PDF via matplotlib
fig2, ax2 = plt.subplots(figsize=(combined.shape[1] / 300, combined.shape[0] / 300), dpi=300)
ax2.imshow(combined)
ax2.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(os.path.join(OUT_DIR, 'rebuttal_fig3.pdf'), bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()

print(f"Saved to {OUT_DIR}/rebuttal_fig3.pdf and rebuttal_fig3.png")
print(f"  Combined size: {combined.shape[1]}x{combined.shape[0]}")
