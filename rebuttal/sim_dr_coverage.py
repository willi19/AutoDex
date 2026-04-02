"""Compute and plot real scene coverage filtered by sim DR threshold.

For each threshold (0.0~1.0, step 0.1), filter candidates by sim_pass_rate >= thr,
then compute coverage curve using only real-success grasps among the filtered set.
X-axis = N (up to #filtered candidates), Y-axis = expected coverage.

Usage:
    conda run -n planner python rebuttal/sim_dr_coverage.py
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import comb

SIM_DR_PATH = "/home/robot/AutoDex/outputs/sim_dr/candidates_sim_dr.json"
CAND_ROOT = "/home/robot/shared_data/AutoDex/candidates/allegro"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
OBJECTS = ["blue_alarm", "brown_ramen", "soaptray"]


def p_at_least_one(X, N, total=100):
    if X <= 0: return 0.0
    if X >= total or N > total - X: return 1.0
    return 1.0 - comb(total - X, N) / comb(total, N)


def get_success_indices(cand_dir):
    succ = []
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
                    if r.get("success"):
                        succ.append(i)
                i += 1
    return succ


def compute_coverage_at_n(vm, grasp_indices, N, total=100):
    """P(>=1 grasp from grasp_indices in N picks from total)."""
    if not grasp_indices or N <= 0:
        return 0.0
    probs = []
    for si in range(vm.shape[0]):
        x = int(vm[si, grasp_indices].sum())
        probs.append(p_at_least_one(x, N, total))
    return np.mean(probs) * 100


def main():
    with open(SIM_DR_PATH) as f:
        sim_data = json.load(f)

    thresholds = np.arange(0, 1.1, 0.1)
    n_range = np.arange(1, 101)

    # Per-object results
    all_results = {}

    for obj in OBJECTS:
        vm = np.load(f"/tmp/{obj}_valid_matrix.npy")
        succ_set = set(get_success_indices(os.path.join(CAND_ROOT, "selected_100", obj)))
        candidates = sim_data["results"][obj]

        obj_results = {}
        for thr in thresholds:
            thr_key = f"{thr:.1f}"
            pass_idx = [i for i, c in enumerate(candidates) if c["sim_pass_rate"] >= thr]
            real_succ = [i for i in pass_idx if i in succ_set]
            n_cand = len(pass_idx)

            # Compute curve up to n_cand, total=n_cand (picking from filtered set)
            curve = []
            for N in n_range:
                if N > n_cand:
                    break
                curve.append(compute_coverage_at_n(vm, real_succ, N, total=n_cand))

            obj_results[thr_key] = {
                "n_candidates": n_cand,
                "n_real_success": len(real_succ),
                "curve": curve,
            }

        all_results[obj] = obj_results
        print(f"{obj}: done")

    # Save results
    save_path = os.path.join(OUT_DIR, "sim_dr_coverage.json")
    # Convert for JSON
    save_data = {}
    for obj, res in all_results.items():
        save_data[obj] = {}
        for thr_key, v in res.items():
            save_data[obj][thr_key] = {
                "n_candidates": v["n_candidates"],
                "n_real_success": v["n_real_success"],
                "curve": v["curve"],
            }
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved: {save_path}")

    # === Compute Ours real and Baseline real curves (full N=1..100) ===
    ours_curves, base_curves = [], []
    for obj in OBJECTS:
        ours_vm = np.load(f"/tmp/{obj}_valid_matrix.npy")
        base_vm = np.load(f"/tmp/{obj}_baseline_valid_matrix.npy")
        os_succ = get_success_indices(os.path.join(CAND_ROOT, "selected_100", obj))
        bs_succ = get_success_indices(os.path.join(CAND_ROOT, "baseline_100", obj))

        ours_c = [compute_coverage_at_n(ours_vm, os_succ, N) for N in n_range]
        base_c = [compute_coverage_at_n(base_vm, bs_succ, N) for N in n_range]
        ours_curves.append(ours_c)
        base_curves.append(base_c)

    ours_avg = np.mean(ours_curves, axis=0)
    base_avg = np.mean(base_curves, axis=0)

    # === Plot: aggregate across objects ===
    fig, ax = plt.subplots(figsize=(6, 4))

    # Ours real and Baseline real
    ax.plot(n_range, ours_avg, "-", linewidth=2.5, color="#d62728", label="Ours (real)")
    ax.plot(n_range, base_avg, "--", linewidth=2.5, color="#1f77b4", label="Baseline (real)")
    ax.fill_between(n_range, base_avg, ours_avg, alpha=0.1, color="#d62728")

    # Sim DR threshold curves
    cmap = plt.cm.viridis
    for ti, thr in enumerate(thresholds):
        thr_key = f"{thr:.1f}"
        curves = [all_results[obj][thr_key]["curve"] for obj in OBJECTS]
        if all(len(c) == 0 for c in curves):
            continue

        min_len = min(len(c) for c in curves if len(c) > 0)
        if min_len == 0:
            continue

        valid_curves = [c[:min_len] for c in curves if len(c) > 0]
        avg_curve = np.mean(valid_curves, axis=0)

        n_cands = [all_results[obj][thr_key]["n_candidates"] for obj in OBJECTS]
        avg_cand = np.mean(n_cands)

        color = cmap(ti / len(thresholds))
        label = f"sim_dr≥{thr:.1f} ({int(avg_cand)})"
        ax.plot(np.arange(1, min_len + 1), avg_curve, "-", linewidth=1.2,
                color=color, label=label, alpha=0.8)

    ax.set_xlabel("Number of Grasps (N)", fontsize=11)
    ax.set_ylabel("Expected Scene Coverage (%)", fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 80)
    ax.legend(fontsize=6, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Scene Coverage ({len(OBJECTS)} objects)", fontsize=11)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "sim_dr_coverage.pdf")
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()