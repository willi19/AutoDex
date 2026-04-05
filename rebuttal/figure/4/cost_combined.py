import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'figure.dpi': 150,
})

# ============================================================
# Parameters
# ============================================================
# Calibration bias (both methods)
CAL_HOURS = 0.5  # 30 min

# Teleop
TELEOP_RAW_RATE = 60          # samples/hr (1 sample/min)
TELEOP_WORK_MIN = 30          # min work per cycle
TELEOP_REST_MIN = 12          # min rest per cycle
TELEOP_HRS_PER_DAY = 15       # work hours per day
TELEOP_EFF_RATE = (TELEOP_WORK_MIN / (TELEOP_WORK_MIN + TELEOP_REST_MIN)) * TELEOP_RAW_RATE  # ~42.86/hr

# AutoDex
AUTODEX_RAW_RATE = 3600 / 40  # 90 samples/hr (1 sample/40s)
AUTODEX_WORK_MIN_PER_HR = 55  # 5 min thermal rest per hour
AUTODEX_EFF_RATE = (AUTODEX_WORK_MIN_PER_HR / 60) * AUTODEX_RAW_RATE  # 82.5/hr

# Cost (USD)
SHARED_HW = 8_399 + 16_000 + 20_000   # arm + hand + cameras/rig = $44,399
TELEOP_HW = 6_598 + 3_790             # Xsens MetaGloves + software = $10,388
LABOR_KRW_PER_HR = 10_000
KRW_PER_USD = 1_350
LABOR_USD_PER_HR = LABOR_KRW_PER_HR / KRW_PER_USD  # ~$7.41/hr

# ============================================================
# Compute curves
# ============================================================
samples = np.arange(0, 2001)

# --- Wall-clock time ---
# AutoDex: 24/7 continuous
autodex_time = CAL_HOURS + samples / AUTODEX_EFF_RATE

# Teleop: 15hr work/day, 9hr overnight gap
teleop_samples_per_day = TELEOP_EFF_RATE * TELEOP_HRS_PER_DAY  # ~642.9
teleop_full_days = np.floor(samples / teleop_samples_per_day)
teleop_remaining = samples - teleop_full_days * teleop_samples_per_day
teleop_remaining_hrs = teleop_remaining / TELEOP_EFF_RATE
teleop_time = CAL_HOURS + teleop_full_days * 24 + teleop_remaining_hrs

# --- Cumulative cost ---
# Shared HW is bias for both
teleop_work_hours = CAL_HOURS + samples / TELEOP_EFF_RATE  # calibration + collection labor
teleop_cost = SHARED_HW + TELEOP_HW + LABOR_USD_PER_HR * teleop_work_hours
autodex_cost = SHARED_HW * np.ones_like(samples, dtype=float)  # no variable cost

# ============================================================
# Plot
# ============================================================
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 3.2))

# --- Left: Time & Cost (dual y-axis) ---
ln1, = ax_left.plot(samples, autodex_time, '-', color='#d62728', linewidth=2, label='AutoDex (time)')
ln2, = ax_left.plot(samples, teleop_time, '--', color='#1f77b4', linewidth=2, label='Teleop (time)')
ax_left.set_xlabel('# Grasp Samples')
ax_left.set_ylabel('Wall-clock Time (hrs)', color='b')
ax_left.tick_params(axis='y', labelcolor='b')

ax_left_cost = ax_left.twinx()
ln3, = ax_left_cost.plot(samples, autodex_cost, color='#d62728', linewidth=2, alpha=0.5, linestyle='dotted', label='AutoDex (cost)')
ln4, = ax_left_cost.plot(samples, teleop_cost, color='#1f77b4', linewidth=2, alpha=0.5, linestyle='dotted', label='Teleop (cost)')
ax_left_cost.set_ylabel('Cumulative Cost (USD)', color='r')
ax_left_cost.tick_params(axis='y', labelcolor='r')
ax_left_cost.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

lns = [ln1, ln2, ln3, ln4]
ax_left.legend(lns, [l.get_label() for l in lns], loc='upper left', fontsize=8)
ax_left.grid(True, alpha=0.3)
ax_left.set_title('Collection Efficiency')

# --- Right: # Cameras vs Success Rate (Max) ---
n_cams = np.arange(1, 25)
max_with_robot = np.array([
    37.0, 54.3, 60.0, 63.3, 65.4, 66.7, 67.8, 68.6,
    69.2, 69.7, 70.3, 70.5, 70.9, 71.2, 71.4, 71.6,
    71.7, 71.9, 72.0, 72.1, 72.1, 72.2, 72.2, 72.2,
])
max_no_robot = np.array([
    40.0, 60.6, 65.4, 67.7, 69.1, 70.2, 70.9, 71.6,
    72.0, 72.4, 72.7, 73.0, 73.2, 73.4, 73.6, 73.8,
    73.9, 74.0, 74.1, 74.2, 74.3, 74.3, 74.3, 74.3,
])
ax_right.plot(n_cams, max_with_robot, '-o', color='#d62728', linewidth=2, markersize=4, label='With Robot')
ax_right.plot(n_cams, max_no_robot, '--s', color='#1f77b4', linewidth=2, markersize=4, label='No Robot')
ax_right.set_xlabel('# Cameras')
ax_right.set_ylabel('Success Rate (%)')
ax_right.legend(fontsize=9)
ax_right.grid(True, alpha=0.3)
ax_right.set_title('Camera Coverage (Best Subset)')
ax_right.set_xticks([1, 4, 8, 12, 16, 20, 24])

fig.tight_layout()
outdir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(outdir, 'cost_combined.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(outdir, 'cost_combined.png'), bbox_inches='tight')

# Print summary
print(f'AutoDex effective rate: {AUTODEX_EFF_RATE:.1f} samples/hr')
print(f'Teleop effective rate:  {TELEOP_EFF_RATE:.1f} samples/hr')
print(f'Teleop samples/day:     {teleop_samples_per_day:.0f}')
print(f'\n{"Samples":>8} {"AutoDex hrs":>12} {"Teleop hrs":>12} {"AutoDex $":>12} {"Teleop $":>12}')
for n in [100, 500, 1000, 2000]:
    i = n
    print(f'{n:>8} {autodex_time[i]:>12.1f} {teleop_time[i]:>12.1f} {autodex_cost[i]:>12,.0f} {teleop_cost[i]:>12,.0f}')
print('\nSaved cost_combined.pdf/.png')
