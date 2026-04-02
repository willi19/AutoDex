"""Plot threshold vs metrics for sim DR confusion matrix.

Usage:
    python rebuttal/plot_sim_dr.py
    python rebuttal/plot_sim_dr.py --max_mass 0.3
    python rebuttal/plot_sim_dr.py --results outputs/sim_dr/results_25grid.json
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_RESULTS = os.path.join(REPO_ROOT, "outputs", "sim_dr", "results_25grid.json")


def filter_trials(results, max_mass):
    """Keep only trials with obj_mass <= max_mass."""
    filtered = []
    for r in results:
        trials = [t for t in r["trials"] if t["params"]["obj_mass"] <= max_mass + 1e-9]
        if trials:
            filtered.append({**r, "trials": trials})
    return filtered


def sweep(results, thresholds):
    rows = []
    for t in thresholds:
        tp = fp = fn = tn = 0
        for r in results:
            n_succ = sum(1 for tr in r["trials"] if tr["success"])
            sim_pass = (n_succ / len(r["trials"])) >= t if len(r["trials"]) > 0 else False
            real = r["real_success"]
            if real and sim_pass: tp += 1
            elif not real and sim_pass: fp += 1
            elif real and not sim_pass: fn += 1
            else: tn += 1
        total = tp + fp + fn + tn
        rows.append({
            "threshold": t,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "fpr": fp / (fp + tn) if (fp + tn) else 0,
            "fnr": fn / (fn + tp) if (fn + tp) else 0,
            "accuracy": (tp + tn) / total if total else 0,
            "precision": tp / (tp + fp) if (tp + fp) else 0,
            "recall": tp / (tp + fn) if (tp + fn) else 0,
            "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--max_mass", type=float, default=0.3)
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "rebuttal", "figure", "2"))
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    results = filter_trials(data["results"], args.max_mass)
    n_trials = len(results[0]["trials"]) if results else 0
    n_real_pos = sum(1 for r in results if r["real_success"])
    n_real_neg = sum(1 for r in results if not r["real_success"])
    print(f"Mass <= {args.max_mass}: {len(results)} experiments, {n_trials} trials/grasp, Real+={n_real_pos}, Real-={n_real_neg}")

    # Use sim_rate breakpoints (n_succ/n_trials) for clean curves
    n_trials_per = len(results[0]["trials"]) if results else 25
    thresholds = np.unique(np.concatenate([
        np.arange(0, n_trials_per + 1) / n_trials_per,
        [1.0]
    ]))
    rows = sweep(results, thresholds)

    th = [r["threshold"] for r in rows]
    fpr = [r["fpr"] for r in rows]
    fnr = [r["fnr"] for r in rows]
    prec = [r["precision"] for r in rows]
    rec = [r["recall"] for r in rows]
    f1 = [r["f1"] for r in rows]
    acc = [r["accuracy"] for r in rows]

    os.makedirs(args.out, exist_ok=True)

    # Plot 1: FPR and FNR vs threshold
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.plot(th, fpr, "-", linewidth=2, color="#d62728", label="FPR")
    ax.plot(th, fnr, "-", linewidth=2, color="#1f77b4", label="FNR")
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Rate", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Sim DR: FPR/FNR vs Threshold (mass≤{args.max_mass}kg)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "sim_dr_fpr_fnr.pdf"))
    plt.savefig(os.path.join(args.out, "sim_dr_fpr_fnr.png"))
    plt.close()

    # Plot 2: Precision-Recall curve (interpolated precision)
    # Sort by recall ascending, apply interpolated precision (monotone decreasing envelope)
    rec_arr = np.array(rec)
    prec_arr = np.array(prec)
    order = np.argsort(rec_arr)
    rec_sorted = rec_arr[order]
    prec_sorted = prec_arr[order]
    # Interpolated: prec[i] = max(prec[i:])
    prec_interp = np.maximum.accumulate(prec_sorted[::-1])[::-1]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.plot(rec_sorted, prec_interp, "-", linewidth=2, color="#2ca02c")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Sim DR: Precision-Recall (mass≤{args.max_mass}kg)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "sim_dr_pr_curve.pdf"))
    plt.savefig(os.path.join(args.out, "sim_dr_pr_curve.png"))
    plt.close()

    # Plot 2b: PR curve text-free, 749x534 px
    fig, ax = plt.subplots(figsize=(749/300, 534/300), dpi=300)
    ax.plot(rec_sorted, prec_interp, "-", linewidth=1.5, color="#2ca02c")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis='both', labelsize=7)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(os.path.join(args.out, "sim_dr_pr_graph.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
    fig.savefig(os.path.join(args.out, "sim_dr_pr_graph.pdf"), bbox_inches='tight', pad_inches=0.02, dpi=300)
    plt.close()

    # Plot 3: All metrics
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.plot(th, acc, "-", linewidth=2, label="Accuracy", color="#1f77b4")
    ax.plot(th, prec, "-", linewidth=2, label="Precision", color="#2ca02c")
    ax.plot(th, rec, "-", linewidth=2, label="Recall", color="#d62728")
    ax.plot(th, f1, "-", linewidth=2, label="F1", color="#9467bd")
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Sim DR Metrics vs Threshold (mass≤{args.max_mass}kg)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "sim_dr_metrics.pdf"))
    plt.savefig(os.path.join(args.out, "sim_dr_metrics.png"))
    plt.close()

    # Print table
    print(f"\n{'Thresh':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}  {'FPR':>5} {'FNR':>5} {'Prec':>5} {'Rec':>5} {'F1':>5}")
    for r in rows:
        if r["threshold"] in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            print(f"{r['threshold']:>6.1f} {r['tp']:>4} {r['fp']:>4} {r['fn']:>4} {r['tn']:>4}  {r['fpr']:>5.3f} {r['fnr']:>5.3f} {r['precision']:>5.3f} {r['recall']:>5.3f} {r['f1']:>5.3f}")

    print(f"\nSaved to {args.out}/sim_dr_*.png/pdf")


if __name__ == "__main__":
    main()
