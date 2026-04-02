"""Plot expected coverage vs number of grasps for selected_100 vs baseline_100.

X-axis: N (number of grasps sampled, 1..100)
Y-axis: expected coverage = mean P(>=1 valid in N from 100) across all scenes

Usage:
    python rebuttal/plot_coverage.py
    python rebuttal/plot_coverage.py --obj blue_alarm
    python rebuttal/plot_coverage.py --top 10
"""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from math import comb

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def p_at_least_one(X, N, total=100):
    if X <= 0: return 0.0
    if X >= total or N > total - X: return 1.0
    return 1.0 - comb(total - X, N) / comb(total, N)


def load_survive_counts(obj_name):
    """Returns {version: [n_surviving per scene]}."""
    data = {}
    for ver in ["selected_100", "baseline_100"]:
        ver_dir = os.path.join(CACHE_DIR, obj_name, ver)
        if not os.path.isdir(ver_dir):
            continue
        counts = []
        for f in sorted(os.listdir(ver_dir)):
            if not f.endswith(".json"):
                continue
            with open(os.path.join(ver_dir, f)) as fh:
                d = json.load(fh)
            counts.append(d["n_surviving"])
        data[ver] = counts
    return data


def compute_curve(survive_counts, n_range):
    """For each N, compute mean P(>=1) across scenes."""
    curve = []
    for N in n_range:
        probs = [p_at_least_one(x, N) for x in survive_counts]
        curve.append(np.mean(probs) * 100)
    return np.array(curve)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, nargs="+", default=None)
    parser.add_argument("--top", type=int, default=None, help="Plot top N objects by diff")
    parser.add_argument("--aggregate", action="store_true", help="Plot aggregate across all objects")
    args = parser.parse_args()

    n_range = np.arange(1, 101)

    # Determine which objects
    all_objs = sorted(os.listdir(CACHE_DIR))
    if args.obj:
        obj_list = args.obj
    elif args.top or args.aggregate:
        # Rank by diff at N=50
        ranked = []
        for obj in all_objs:
            data = load_survive_counts(obj)
            if not all(v in data for v in ["selected_100", "baseline_100"]):
                continue
            sel = np.mean([p_at_least_one(x, 50) for x in data["selected_100"]])
            base = np.mean([p_at_least_one(x, 50) for x in data["baseline_100"]])
            ranked.append((obj, sel - base, data))
        ranked.sort(key=lambda x: x[1], reverse=True)

        if args.aggregate:
            obj_list = [r[0] for r in ranked]
        else:
            obj_list = [r[0] for r in ranked[:args.top]]
    else:
        obj_list = all_objs[:5]

    if args.aggregate:
        # Single plot: aggregate all objects
        all_sel, all_base = [], []
        for obj in obj_list:
            data = load_survive_counts(obj)
            if "selected_100" in data:
                all_sel.extend(data["selected_100"])
            if "baseline_100" in data:
                all_base.extend(data["baseline_100"])

        sel_curve = compute_curve(all_sel, n_range)
        base_curve = compute_curve(all_base, n_range)

        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(n_range, sel_curve, "-", linewidth=2, color="#d62728", label="Ours (selected_100)")
        ax.plot(n_range, base_curve, "-", linewidth=2, color="#1f77b4", label="Baseline")
        ax.fill_between(n_range, base_curve, sel_curve, alpha=0.15, color="#d62728")
        ax.set_xlabel("Number of Grasps (N)", fontsize=11)
        ax.set_ylabel("Expected Coverage (%)", fontsize=11)
        ax.set_xlim(1, 100)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Aggregate ({len(obj_list)} objects, {len(all_sel)} scenes)", fontsize=11)
        plt.tight_layout()
        out = os.path.join(OUT_DIR, "coverage_aggregate.pdf")
        plt.savefig(out, bbox_inches="tight", dpi=300)
        print(f"Saved {out}")

    else:
        # Per-object plots
        n_objs = len(obj_list)
        cols = min(n_objs, 5)
        rows = (n_objs + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3 * rows), squeeze=False)

        for idx, obj in enumerate(obj_list):
            ax = axes[idx // cols][idx % cols]
            data = load_survive_counts(obj)
            if "selected_100" not in data or "baseline_100" not in data:
                ax.set_title(obj, fontsize=9)
                continue

            sel_curve = compute_curve(data["selected_100"], n_range)
            base_curve = compute_curve(data["baseline_100"], n_range)

            ax.plot(n_range, sel_curve, "-", linewidth=1.5, color="#d62728", label="Ours")
            ax.plot(n_range, base_curve, "-", linewidth=1.5, color="#1f77b4", label="Base")
            ax.fill_between(n_range, base_curve, sel_curve, alpha=0.15, color="#d62728")
            ax.set_xlim(1, 100)
            ax.set_ylim(0, 105)
            ax.set_title(obj, fontsize=9)
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend(fontsize=7)

        # Hide empty axes
        for idx in range(n_objs, rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)

        fig.supxlabel("Number of Grasps (N)", fontsize=11)
        fig.supylabel("Expected Coverage (%)", fontsize=11)
        plt.tight_layout()
        if args.top:
            out = os.path.join(OUT_DIR, f"coverage_top{args.top}.pdf")
        elif args.obj and len(args.obj) == 1:
            out = os.path.join(OUT_DIR, f"coverage_{args.obj[0]}.pdf")
        else:
            out = os.path.join(OUT_DIR, "coverage_objects.pdf")
        plt.savefig(out, bbox_inches="tight", dpi=300)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
