"""Rebuttal figure 1: coverage curves + experiment photos.

Two-pass approach:
  1. Compose graph (text-free) + photos
  2. Add all text with consistent font size

Usage:
    python rebuttal/figure/make_figure.py
"""
import os
import json
import numpy as np
from math import comb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

CAND_ROOT = "/home/robot/shared_data/AutoDex/candidates/allegro"
PHOTO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "1")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

OBJECTS = ["blue_alarm", "brown_ramen", "soaptray"]

# --- Unified font size ---
FONT_SIZE = 8
TICK_SIZE = 7
LEGEND_SIZE = 7


def p_at_least_one(X, N, total=100):
    if X <= 0: return 0.0
    if X >= total or N > total - X: return 1.0
    return 1.0 - comb(total - X, N) / comb(total, N)


def get_indices(cand_dir):
    succ, fail, untest = [], [], []
    i = 0
    for st in sorted(os.listdir(cand_dir)):
        st_dir = os.path.join(cand_dir, st)
        if not os.path.isdir(st_dir): continue
        for sid in sorted(os.listdir(st_dir)):
            sid_dir = os.path.join(st_dir, sid)
            if not os.path.isdir(sid_dir): continue
            for gn in sorted(os.listdir(sid_dir)):
                gdir = os.path.join(sid_dir, gn)
                if not os.path.isdir(gdir): continue
                if not os.path.exists(os.path.join(gdir, "wrist_se3.npy")): continue
                rpath = os.path.join(gdir, "result.json")
                if os.path.exists(rpath):
                    with open(rpath) as f:
                        r = json.load(f)
                    if r.get("success"): succ.append(i)
                    else: fail.append(i)
                else:
                    untest.append(i)
                i += 1
    return succ, fail, untest


def compute_real_curve(vm, success_idx, n_range):
    curve = []
    for N in n_range:
        probs = []
        for si in range(vm.shape[0]):
            x = int(vm[si, success_idx].sum())
            probs.append(p_at_least_one(x, N))
        curve.append(np.mean(probs) * 100)
    return np.array(curve)


def load_photo(path):
    """Load and center-crop to square."""
    img = Image.open(path)
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def main():
    n_range = np.arange(1, 101)

    # --- Compute curves ---
    all_curves = []
    for obj in OBJECTS:
        ours_vm = np.load(f"/tmp/{obj}_valid_matrix.npy")
        base_vm = np.load(f"/tmp/{obj}_baseline_valid_matrix.npy")
        os_, _, _ = get_indices(os.path.join(CAND_ROOT, "selected_100", obj))
        bs_, _, _ = get_indices(os.path.join(CAND_ROOT, "baseline_100", obj))
        ours_curve = compute_real_curve(ours_vm, os_, n_range)
        base_curve = compute_real_curve(base_vm, bs_, n_range) if bs_ else np.zeros_like(n_range, dtype=float)
        all_curves.append((obj, ours_curve, base_curve))
        print(f"{obj}: ours_succ={len(os_)}, base_succ={len(bs_)}")

    ours_agg = np.mean([c[1] for c in all_curves], axis=0)
    base_agg = np.mean([c[2] for c in all_curves], axis=0)

    # --- Load photos ---
    photos = [load_photo(os.path.join(PHOTO_DIR, f"{i}.jpg")) for i in [1, 2, 3, 4]]

    # --- Compose figure ---
    # Layout: [graph | photo1 photo2]
    #         [      | photo3 photo4]
    def make_graph(with_labels=False):
        fig, ax = plt.subplots(figsize=(4, 2), dpi=300)
        ax.plot(n_range, ours_agg, "-", linewidth=1.5, color="#d62728")
        ax.plot(n_range, base_agg, "--", linewidth=1.5, color="#1f77b4")
        ax.fill_between(n_range, base_agg, ours_agg, alpha=0.12, color="#d62728")
        ax.set_xlim(1, 100)
        ax.set_ylim(0, 80)
        ax.tick_params(axis='both', labelsize=TICK_SIZE)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if with_labels:
            ax.set_xlabel("Number of Grasps", fontsize=FONT_SIZE)
            ax.set_ylabel("Expected Scene Coverage (%)", fontsize=FONT_SIZE)
        return fig

    # V1: no axis titles
    fig = make_graph(with_labels=False)
    fig.savefig(os.path.join(OUT_DIR, "1", "rebuttal_fig1_graph.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
    fig.savefig(os.path.join(OUT_DIR, "1", "rebuttal_fig1_graph.pdf"), bbox_inches='tight', pad_inches=0.02, dpi=300)
    plt.close(fig)
    print("Saved graph (no labels)")

    # V2: with axis titles
    fig = make_graph(with_labels=True)
    fig.savefig(os.path.join(OUT_DIR, "1", "rebuttal_fig1_graph_labeled.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
    fig.savefig(os.path.join(OUT_DIR, "1", "rebuttal_fig1_graph_labeled.pdf"), bbox_inches='tight', pad_inches=0.02, dpi=300)
    plt.close(fig)
    print("Saved graph (with labels)")

    # V3: legend only
    fig_leg, ax_leg = plt.subplots(figsize=(2, 0.3), dpi=300)
    ax_leg.plot([], [], "-", linewidth=1.5, color="#d62728", label="Ours (real)")
    ax_leg.plot([], [], "--", linewidth=1.5, color="#1f77b4", label="Baseline (real)")
    ax_leg.axis('off')
    legend = ax_leg.legend(fontsize=LEGEND_SIZE, loc='center', ncol=2, frameon=False)
    fig_leg.savefig(os.path.join(OUT_DIR, "1", "rebuttal_fig1_legend.png"), bbox_inches='tight', pad_inches=0.01, dpi=300)
    fig_leg.savefig(os.path.join(OUT_DIR, "1", "rebuttal_fig1_legend.pdf"), bbox_inches='tight', pad_inches=0.01, dpi=300)
    plt.close(fig_leg)
    print("Saved legend")


if __name__ == "__main__":
    main()
