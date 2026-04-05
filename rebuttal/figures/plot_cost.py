import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'figure.dpi': 150,
})

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4))

# ============================================================
# Left: # Samples vs Time & Cost (dual y-axis)
# ============================================================
samples = np.array([0, 100, 200, 300, 500, 750, 1000, 1500, 2000])

# Time (hours)
autodex_time = samples / 30.0           # ~30 grasps/hr, unattended
teleop_time = samples / 8.0             # ~8 grasps/hr effective (w/ breaks)

# Cost ($)
autodex_cost = autodex_time * 5                      # robot only (~$5/hr)
teleop_cost = 5000 + teleop_time * (5 + 25)          # Xsens fixed + robot + human

ln1 = ax_left.plot(samples, autodex_time, 'b-o', label='AutoDex (time)', markersize=4, linewidth=2)
ln2 = ax_left.plot(samples, teleop_time, 'bs--', label='Teleop (time)', markersize=4, linewidth=2)
ax_left.set_xlabel('# Samples')
ax_left.set_ylabel('Wall-clock Time (hrs)', color='b')
ax_left.tick_params(axis='y', labelcolor='b')

ax_left_cost = ax_left.twinx()
ln3 = ax_left_cost.plot(samples, autodex_cost, 'r-o', label='AutoDex (cost)', markersize=4, linewidth=2)
ln4 = ax_left_cost.plot(samples, teleop_cost, 'rs--', label='Teleop (cost)', markersize=4, linewidth=2)
ax_left_cost.set_ylabel('Cumulative Cost ($)', color='r')
ax_left_cost.tick_params(axis='y', labelcolor='r')

lns = ln1 + ln2 + ln3 + ln4
labs = [l.get_label() for l in lns]
ax_left.legend(lns, labs, loc='upper left', fontsize=9)
ax_left.grid(True, alpha=0.3)
ax_left.set_title('Collection Efficiency')

# ============================================================
# Right: # Cameras vs ADD-S / Success / Unobserved
# ============================================================
n_cams = np.array([2, 4, 8, 12, 24])

adds_error = np.array([2.5, 1.17, 0.62, 0.45, 0.31])   # ADD-S (mm)
success    = np.array([38, 51, 56, 59, 62])              # success rate (%)
occlusion  = np.array([45, 32, 20, 15, 8])               # unobserved ratio (%)

ln1 = ax_right.plot(n_cams, adds_error, 'b-o', label='ADD-S (mm)', linewidth=2, markersize=6)
ax_right.set_xlabel('# Cameras')
ax_right.set_ylabel('ADD-S Error (mm)', color='b')
ax_right.tick_params(axis='y', labelcolor='b')

ax_right2 = ax_right.twinx()
ln2 = ax_right2.plot(n_cams, success, 'g-s', label='Success (%)', linewidth=2, markersize=6)
ln3 = ax_right2.plot(n_cams, occlusion, 'r-^', label='Unobserved (%)', linewidth=2, markersize=6)
ax_right2.set_ylabel('Rate (%)', color='gray')
ax_right2.tick_params(axis='y', labelcolor='gray')

lns = ln1 + ln2 + ln3
labs = [l.get_label() for l in lns]
ax_right.legend(lns, labs, loc='center right', fontsize=9)
ax_right.grid(True, alpha=0.3)
ax_right.set_xticks(n_cams)
ax_right.set_title('Camera Trade-off')

fig.tight_layout()
import os
outdir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(outdir, 'cost_combined.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(outdir, 'cost_combined.png'), bbox_inches='tight')
print('Saved cost_combined')
