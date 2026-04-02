"""Figure 4: text-free graphs (left/right separate) + legends."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
TICK_SIZE = 7

# ============================================================
# Parameters (same as cost_combined.py)
# ============================================================
CAL_HOURS = 0.5
TELEOP_RAW_RATE = 60
TELEOP_WORK_MIN = 30
TELEOP_REST_MIN = 12
TELEOP_HRS_PER_DAY = 15
TELEOP_EFF_RATE = (TELEOP_WORK_MIN / (TELEOP_WORK_MIN + TELEOP_REST_MIN)) * TELEOP_RAW_RATE
AUTODEX_RAW_RATE = 3600 / 40
AUTODEX_WORK_MIN_PER_HR = 55
AUTODEX_EFF_RATE = (AUTODEX_WORK_MIN_PER_HR / 60) * AUTODEX_RAW_RATE

SHARED_HW = 8_399 + 16_000 + 20_000
TELEOP_HW = 6_598 + 3_790
LABOR_KRW_PER_HR = 10_000
KRW_PER_USD = 1_350
LABOR_USD_PER_HR = LABOR_KRW_PER_HR / KRW_PER_USD

# ============================================================
# Compute curves
# ============================================================
samples = np.arange(0, 2001)
autodex_time = samples / AUTODEX_EFF_RATE
teleop_samples_per_day = TELEOP_EFF_RATE * TELEOP_HRS_PER_DAY
teleop_full_days = np.floor(samples / teleop_samples_per_day)
teleop_remaining = samples - teleop_full_days * teleop_samples_per_day
teleop_remaining_hrs = teleop_remaining / TELEOP_EFF_RATE
teleop_time = teleop_full_days * 24 + teleop_remaining_hrs

teleop_work_hours = samples / TELEOP_EFF_RATE
# Variable cost only (ignore shared HW / Xsens), both start at $0
teleop_cost = LABOR_USD_PER_HR * teleop_work_hours
autodex_cost = np.zeros_like(samples, dtype=float)

# ============================================================
# Left graph: Time & Cost (dual y-axis), text-free
# ============================================================
fig, ax = plt.subplots(figsize=(1000/300, 300/300), dpi=300)
# Time: solid lines, Cost: dotted lines
# AutoDex: red, Teleop: blue
ax.plot(samples, autodex_time, '-', color='#d62728', linewidth=1.5)
ax.plot(samples, teleop_time, '-', color='#1f77b4', linewidth=1.5)
ax.set_ylim(-5, None)
ax.tick_params(axis='both', labelsize=TICK_SIZE)
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_position(('data', -5))

ax_cost = ax.twinx()
ax_cost.set_ylim(-35, 500)

# Style variants for cost lines
styles = {
    'a_dashed':   dict(linestyle='--', linewidth=0.6, alpha=0.6),
    'b_dashdot':  dict(linestyle='-.', linewidth=0.7, alpha=0.7),
    'c_thin':     dict(linestyle='-',  linewidth=0.5, alpha=0.4),
}

for name, kw in styles.items():
    fig_v, ax_v = plt.subplots(figsize=(1000/300, 300/300), dpi=300)
    ax_v.plot(samples, autodex_time, '-', color='#d62728', linewidth=0.8)
    ax_v.plot(samples, teleop_time, '-', color='#1f77b4', linewidth=0.8)
    ax_v.set_ylim(-5, None)
    ax_v.spines['bottom'].set_position(('data', -5))
    ax_v.set_xticks([0, 500, 1000, 1500, 2000])
    ax_v.set_yticks([0, 20, 40, 60])
    ax_v.tick_params(axis='both', labelsize=4, colors='gray')
    ax_v.grid(True, alpha=0.3)
    ax_v.spines['top'].set_visible(False)
    ax_v.spines['right'].set_visible(False)

    ax_c = ax_v.twinx()
    ax_c.plot(samples, autodex_cost, color='#d62728', **kw)
    ax_c.plot(samples, teleop_cost, color='#1f77b4', **kw)
    ax_c.set_ylim(-35, 500)
    ax_c.set_yticks([0, 125, 250, 375, 500])
    ax_c.tick_params(axis='y', labelsize=4, colors='gray')
    ax_c.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax_c.spines['top'].set_visible(False)

    fig_v.savefig(os.path.join(OUT_DIR, f"fig4_left_{name}.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
    plt.close(fig_v)
    print(f"Saved fig4_left_{name}")

# Keep default as dashed
ax_cost.plot(samples, autodex_cost, '--', color='#d62728', linewidth=1.0, alpha=0.6)
ax_cost.plot(samples, teleop_cost, '--', color='#1f77b4', linewidth=1.0, alpha=0.6)
ax_cost.tick_params(axis='y', labelsize=TICK_SIZE)
ax_cost.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax_cost.spines['top'].set_visible(False)

fig.savefig(os.path.join(OUT_DIR, "fig4_left.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
fig.savefig(os.path.join(OUT_DIR, "fig4_left.pdf"), bbox_inches='tight', pad_inches=0.02, dpi=300)
plt.close(fig)
print("Saved fig4_left")

# ============================================================
# Right graph: # Cameras vs Success Rate, text-free
# ============================================================
n_cams = np.arange(1, 25)
wr_mean = np.array([
    28.0, 40.9, 48.6, 53.2, 56.9, 59.6, 61.5, 62.9,
    64.3, 65.3, 66.2, 66.9, 67.6, 68.1, 68.6, 69.0,
    69.4, 69.7, 70.0, 70.3, 70.6, 70.8, 71.0, 71.3,
])
wr_max = np.array([
    36.6, 53.8, 59.5, 62.3, 64.3, 65.7, 66.8, 67.5,
    68.2, 68.8, 69.1, 69.6, 69.9, 70.2, 70.4, 70.6,
    70.7, 70.9, 71.0, 71.1, 71.2, 71.2, 71.3, 71.3,
])
nr_mean = np.array([
    34.4, 47.6, 55.1, 59.3, 62.6, 64.9, 66.3, 67.8,
    68.7, 69.6, 70.2, 70.8, 71.4, 71.7, 72.1, 72.4,
    72.7, 73.0, 73.2, 73.5, 73.6, 73.8, 74.0, 74.1,
])
nr_max = np.array([
    40.0, 60.6, 65.4, 67.5, 69.0, 70.1, 70.7, 71.4,
    71.8, 72.2, 72.5, 72.8, 73.1, 73.2, 73.5, 73.6,
    73.7, 73.8, 73.9, 74.0, 74.1, 74.1, 74.1, 74.1,
])

fig, ax = plt.subplots(figsize=(1000/300, 300/300), dpi=300)
ax.plot(n_cams, wr_mean, '-o', color='#d62728', linewidth=0.8, markersize=1)
ax.set_ylim(20, 80)
ax.set_yticks([20, 40, 60, 80])
ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
ax.tick_params(axis='both', labelsize=4, colors='gray')
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.savefig(os.path.join(OUT_DIR, "fig4_right.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
fig.savefig(os.path.join(OUT_DIR, "fig4_right.pdf"), bbox_inches='tight', pad_inches=0.02, dpi=300)
plt.close(fig)
print("Saved fig4_right")

# ============================================================
# Legends (separate)
# ============================================================
# Left legend
fig_leg, ax_leg = plt.subplots(figsize=(3.5, 0.3), dpi=300)
ax_leg.plot([], [], '-', color='#d62728', linewidth=1.5, label='AutoDex (time)')
ax_leg.plot([], [], '--', color='#1f77b4', linewidth=1.5, label='Teleop (time)')
ax_leg.plot([], [], ':', color='#d62728', linewidth=1.5, alpha=0.5, label='AutoDex (cost)')
ax_leg.plot([], [], ':', color='#1f77b4', linewidth=1.5, alpha=0.5, label='Teleop (cost)')
ax_leg.axis('off')
ax_leg.legend(fontsize=TICK_SIZE, loc='center', ncol=4, frameon=False)
fig_leg.savefig(os.path.join(OUT_DIR, "fig4_left_legend.png"), bbox_inches='tight', pad_inches=0.01, dpi=300)
fig_leg.savefig(os.path.join(OUT_DIR, "fig4_left_legend.pdf"), bbox_inches='tight', pad_inches=0.01, dpi=300)
plt.close(fig_leg)
print("Saved fig4_left_legend")

# Right legend
fig_leg, ax_leg = plt.subplots(figsize=(2, 0.3), dpi=300)
ax_leg.plot([], [], '-o', color='#d62728', linewidth=1.5, markersize=3, label='With Robot')
ax_leg.plot([], [], '--s', color='#1f77b4', linewidth=1.5, markersize=3, label='No Robot')
ax_leg.axis('off')
ax_leg.legend(fontsize=TICK_SIZE, loc='center', ncol=2, frameon=False)
fig_leg.savefig(os.path.join(OUT_DIR, "fig4_right_legend.png"), bbox_inches='tight', pad_inches=0.01, dpi=300)
fig_leg.savefig(os.path.join(OUT_DIR, "fig4_right_legend.pdf"), bbox_inches='tight', pad_inches=0.01, dpi=300)
plt.close(fig_leg)
print("Saved fig4_right_legend")
