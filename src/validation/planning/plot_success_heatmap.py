"""
Planning Success Heatmap

Visualize planning results as heatmaps per object/pose.
Categories:
  - Green:  IK ok + plan always ok
  - Yellow: IK ok + plan stochastic
  - Red:    IK ok + plan always fail
  - Gray:   IK always fail (no candidate passes IK)

Usage:
    python src/validation/planning/plot_success_heatmap.py
    python src/validation/planning/plot_success_heatmap.py --obj apple
    python src/validation/planning/plot_success_heatmap.py --data_dir outputs/planning_success_rate
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


def load_results(data_dir, obj_name):
    obj_dir = os.path.join(data_dir, obj_name)
    for f in sorted(os.listdir(obj_dir)):
        if f.startswith("plan_vs_ik_") and f.endswith(".json") and "partial" not in f and "prev" not in f:
            with open(os.path.join(obj_dir, f)) as fh:
                return json.load(fh)
    return None


def categorize(result):
    """Return category for a grid point."""
    ik_mean = result.get("ik_mean", 0)
    rate = result["success_rate"]

    if ik_mean == 0:
        return 0  # IK fail
    elif rate == 1.0:
        return 1  # IK ok + plan always ok
    elif rate == 0.0:
        return 3  # IK ok + plan always fail
    else:
        return 2  # IK ok + plan stochastic


COLORS = ["#cccccc", "#4caf50", "#ffeb3b", "#f44336"]  # gray, green, yellow, red
LABELS = ["IK fail", "IK+Plan ok", "Plan stochastic", "IK ok, Plan fail"]
CMAP = ListedColormap(COLORS)


def plot_object(data, obj_name, save_dir=None):
    results = data["results"]

    # Get unique axes
    poses = sorted(set(r["pose_idx"] for r in results))
    x_offsets = sorted(set(r["x_offset"] for r in results))
    z_rotations = sorted(set(r["z_rotation_deg"] for r in results))

    n_poses = len(poses)
    fig, axes = plt.subplots(1, n_poses, figsize=(5 * n_poses, 4), squeeze=False)
    fig.suptitle(f"{obj_name}  (rate={data['overall_mean_rate']*100:.1f}%)", fontsize=14)

    for pi, pose_idx in enumerate(poses):
        ax = axes[0, pi]
        grid = np.full((len(x_offsets), len(z_rotations)), np.nan)

        for r in results:
            if r["pose_idx"] != pose_idx:
                continue
            xi = x_offsets.index(r["x_offset"])
            zi = z_rotations.index(r["z_rotation_deg"])
            grid[xi, zi] = categorize(r)

        im = ax.imshow(grid, cmap=CMAP, vmin=-0.5, vmax=3.5, aspect="auto",
                        origin="lower", interpolation="nearest")

        ax.set_xticks(range(len(z_rotations)))
        ax.set_xticklabels([f"{int(z)}" for z in z_rotations], rotation=45, fontsize=7)
        ax.set_yticks(range(len(x_offsets)))
        ax.set_yticklabels([f"{x:.2f}" for x in x_offsets], fontsize=8)
        ax.set_xlabel("z_rotation (deg)")
        ax.set_ylabel("x_offset")
        ax.set_title(f"pose {pose_idx}")

    # Legend
    patches = [mpatches.Patch(color=COLORS[i], label=LABELS[i]) for i in range(4)]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{obj_name}_heatmap.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Planning success heatmap")
    parser.add_argument("--obj", type=str, default=None, help="Object name (omit for all)")
    parser.add_argument("--data_dir", type=str, default="outputs/planning_success_rate")
    parser.add_argument("--save_dir", type=str, default="outputs/planning_success_rate_plots")
    args = parser.parse_args()

    if args.obj:
        objects = [args.obj]
    else:
        objects = sorted([d for d in os.listdir(args.data_dir)
                          if os.path.isdir(os.path.join(args.data_dir, d)) and d != "plots"])

    for obj_name in objects:
        data = load_results(args.data_dir, obj_name)
        if data is None:
            print(f"{obj_name}: no results found")
            continue
        plot_object(data, obj_name, save_dir=args.save_dir)
