"""Rebuttal figure: PR curve (left) + 2x4 grasp examples (right).

Top row: FN example (sim fail, real success) - plant_mister sim_rate=0.07
Bottom row: FP example (sim pass, real fail) - pepsi_light sim_rate=0.93

Usage:
    python rebuttal/figure/make_simdr_figure.py
"""

import os
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_PATH = os.path.join(REPO_ROOT, "outputs", "sim_dr", "results_25grid.json")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

FN_STRIP = os.path.join(OUT_DIR, "2", "false_negative", "plant_mister_20260326_161415", "25322652.png")
FP_STRIP = os.path.join(OUT_DIR, "2", "false_positive", "pepsi_light_20260326_214633", "25322652.png")


def filter_trials(results, max_mass=0.3):
    filtered = []
    for r in results:
        trials = [t for t in r["trials"] if t["params"]["obj_mass"] <= max_mass + 1e-9]
        if trials:
            filtered.append({**r, "trials": trials})
    return filtered


def compute_pr(results):
    thresholds = np.arange(0, 1.001, 0.02)
    prec_list, rec_list = [], []
    for t in thresholds:
        tp = fp = fn = tn = 0
        for r in results:
            n_succ = sum(1 for tr in r["trials"] if tr["success"])
            sim_pass = (n_succ / len(r["trials"])) >= t if r["trials"] else False
            real = r["real_success"]
            if real and sim_pass: tp += 1
            elif not real and sim_pass: fp += 1
            elif real and not sim_pass: fn += 1
            else: tn += 1
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        prec_list.append(prec)
        rec_list.append(rec)
    return rec_list, prec_list


def main():
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    results = filter_trials(data["results"], max_mass=0.3)
    rec, prec = compute_pr(results)

    # Filter noisy low-recall points
    rec_f = [r for r, p in zip(rec, prec) if r >= 0.15]
    prec_f = [p for r, p in zip(rec, prec) if r >= 0.15]

    # Load strips
    fn_img = np.array(Image.open(FN_STRIP))
    fp_img = np.array(Image.open(FP_STRIP))

    # Split each strip into 4 frames
    fn_frames = np.array_split(fn_img, 4, axis=1)
    fp_frames = np.array_split(fp_img, 4, axis=1)

    # Figure layout: 2 rows, 5 cols (PR curve spans left column)
    fig = plt.figure(figsize=(16, 5), dpi=200)
    gs = GridSpec(2, 5, figure=fig, width_ratios=[1.2, 1, 1, 1, 1],
                  wspace=0.05, hspace=0.08)

    # PR curve (left, spans both rows)
    ax_pr = fig.add_subplot(gs[:, 0])
    ax_pr.plot(rec_f, prec_f, "o-", linewidth=2.5, markersize=4, color="#2ca02c")
    ax_pr.set_xlabel("Recall", fontsize=13)
    ax_pr.set_ylabel("Precision", fontsize=13)
    ax_pr.set_xlim(0, 1.05)
    ax_pr.set_ylim(0, 1.05)
    ax_pr.grid(True, alpha=0.3)
    ax_pr.set_title("Precision-Recall", fontsize=13, fontweight="bold")
    ax_pr.tick_params(labelsize=10)

    # Time labels
    time_labels = ["-1.5s", "-1.0s", "-0.5s", "end"]

    # Top row: FN (real success, sim fail)
    for i in range(4):
        ax = fig.add_subplot(gs[0, i + 1])
        ax.imshow(fn_frames[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(time_labels[i], fontsize=10)
        for spine in ax.spines.values():
            spine.set_color("#2ca02c")
            spine.set_linewidth(2.5)
        if i == 0:
            ax.set_ylabel("Real: Pass\nSim: Fail (0.07)", fontsize=9, color="#2ca02c",
                          fontweight="bold")

    # Bottom row: FP (real fail, sim pass)
    for i in range(4):
        ax = fig.add_subplot(gs[1, i + 1])
        ax.imshow(fp_frames[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(time_labels[i], fontsize=10)
        for spine in ax.spines.values():
            spine.set_color("#d62728")
            spine.set_linewidth(2.5)
        if i == 0:
            ax.set_ylabel("Real: Fail\nSim: Pass (0.93)", fontsize=9, color="#d62728",
                          fontweight="bold")

    plt.savefig(os.path.join(OUT_DIR, "simdr_figure.png"), bbox_inches="tight",
                pad_inches=0.1, facecolor="white")
    plt.savefig(os.path.join(OUT_DIR, "simdr_figure.pdf"), bbox_inches="tight",
                pad_inches=0.1, facecolor="white")
    plt.close()
    print(f"Saved to {OUT_DIR}/simdr_figure.png and .pdf")


if __name__ == "__main__":
    main()
