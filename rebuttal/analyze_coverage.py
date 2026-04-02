"""Analyze scene coverage results with N-sampling probability.

For each (object, version, scene), given X surviving grasps out of 100:
  P(at least 1 valid in N samples) = 1 - C(100-X, N) / C(100, N)

Reports results for N = 10, 50, 100.

Usage:
    python rebuttal/analyze_coverage.py
    python rebuttal/analyze_coverage.py --obj attached_container
"""
import os
import sys
import json
import argparse
import numpy as np
from math import comb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REBUTTAL_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(REBUTTAL_DIR, "cache")
SCENE_DIR = os.path.join(REBUTTAL_DIR, "scenes")
VERSIONS = ["selected_100", "baseline_100"]
N_SAMPLES = [10, 50, 100]
N_TOTAL = 100  # total grasps per version


def prob_at_least_one(X, N, total=N_TOTAL):
    """P(at least 1 valid when sampling N from total, where X are valid).

    = 1 - C(total-X, N) / C(total, N)
    """
    if X <= 0:
        return 0.0
    if X >= total or N >= total:
        return 1.0 if X > 0 else 0.0
    if N > total - X:
        return 1.0  # must pick at least one valid
    return 1.0 - comb(total - X, N) / comb(total, N)


def load_all_caches():
    """Load all cached results. Returns {obj: {version: {scene_type: [n_surviving, ...]}}}."""
    results = {}
    if not os.path.isdir(CACHE_DIR):
        return results

    for obj_name in sorted(os.listdir(CACHE_DIR)):
        obj_dir = os.path.join(CACHE_DIR, obj_name)
        if not os.path.isdir(obj_dir):
            continue
        results[obj_name] = {}
        for version in VERSIONS:
            ver_dir = os.path.join(obj_dir, version)
            if not os.path.isdir(ver_dir):
                continue
            by_type = {}
            for fname in sorted(os.listdir(ver_dir)):
                if not fname.endswith(".json"):
                    continue
                with open(os.path.join(ver_dir, fname)) as f:
                    data = json.load(f)
                stype = data["scene_type"]
                by_type.setdefault(stype, []).append(data["n_surviving"])
            results[obj_name][version] = by_type
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, default=None)
    args = parser.parse_args()

    all_data = load_all_caches()
    if args.obj:
        all_data = {k: v for k, v in all_data.items() if k == args.obj}

    if not all_data:
        print("No cached results found. Run scene_coverage.py first.")
        return

    # Per-object analysis
    obj_summary = {}

    for obj_name in sorted(all_data.keys()):
        obj_data = all_data[obj_name]
        if not all(v in obj_data for v in VERSIONS):
            continue

        summary = {"per_type": {}, "overall": {}}

        for stype in ["wall", "shelf", "cluttered"]:
            for version in VERSIONS:
                surv_list = obj_data.get(version, {}).get(stype, [])
                if not surv_list:
                    continue
                key = f"{version}"

                # Coverage (any survive)
                coverage = sum(1 for x in surv_list if x > 0) / len(surv_list) * 100

                # Mean surviving
                avg_surv = np.mean(surv_list)

                # N-sampling probabilities: average P(>=1) across scenes
                n_probs = {}
                for N in N_SAMPLES:
                    probs = [prob_at_least_one(x, N) for x in surv_list]
                    n_probs[N] = np.mean(probs) * 100  # as percentage

                summary["per_type"].setdefault(stype, {})[key] = {
                    "coverage": coverage,
                    "avg_surviving": avg_surv,
                    "n_probs": n_probs,
                }

        # Overall (across all scene types)
        for version in VERSIONS:
            all_surv = []
            for stype in ["wall", "shelf", "cluttered"]:
                all_surv.extend(obj_data.get(version, {}).get(stype, []))
            if not all_surv:
                continue
            coverage = sum(1 for x in all_surv if x > 0) / len(all_surv) * 100
            avg_surv = np.mean(all_surv)
            n_probs = {}
            for N in N_SAMPLES:
                probs = [prob_at_least_one(x, N) for x in all_surv]
                n_probs[N] = np.mean(probs) * 100
            summary["overall"][version] = {
                "coverage": coverage,
                "avg_surviving": avg_surv,
                "n_probs": n_probs,
                "n_scenes": len(all_surv),
            }

        obj_summary[obj_name] = summary

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"SCENE COVERAGE ANALYSIS  (N-sampling: probability of at least 1 valid grasp)")
    print(f"{'='*100}")

    # ── Top-K diff ranking ──────────────────────────────────────────────────
    # For each (object, scene_type), compute per-scene P_diff = P_sel(N) - P_base(N).
    # Pick top-10 per scene_type (wall/shelf/cluttered) = 30 scenes total.
    # Score = sum of those 30 diffs.
    TOP_K_PER_TYPE = 10

    obj_scores = {}  # obj -> {N -> score}

    for obj_name in sorted(obj_summary.keys()):
        obj_data = all_data[obj_name]
        if not all(v in obj_data for v in VERSIONS):
            continue

        scores_by_n = {}
        detail_by_n = {}

        for N in N_SAMPLES:
            all_top_k = []  # collected top-K from each scene type

            for stype in ["wall", "shelf", "cluttered"]:
                sel_surv = obj_data.get("selected_100", {}).get(stype, [])
                base_surv = obj_data.get("baseline_100", {}).get(stype, [])
                n_scenes = min(len(sel_surv), len(base_surv))

                type_diffs = []
                for si in range(n_scenes):
                    p_sel = prob_at_least_one(sel_surv[si], N)
                    p_base = prob_at_least_one(base_surv[si], N)
                    diff = p_sel - p_base
                    type_diffs.append((stype, si, p_sel, p_base, diff))

                # Top-K per scene type
                type_diffs.sort(key=lambda x: x[4], reverse=True)
                all_top_k.extend(type_diffs[:TOP_K_PER_TYPE])

            score = sum(d[4] for d in all_top_k)
            scores_by_n[N] = score
            detail_by_n[N] = all_top_k

        obj_scores[obj_name] = {"scores": scores_by_n, "details": detail_by_n}

    # Rank by N=10 score (most discriminative)
    ranked_objs = sorted(obj_scores.items(), key=lambda x: x[1]["scores"][10], reverse=True)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"SCENE COVERAGE ANALYSIS — Top-{TOP_K_PER_TYPE} scene diff ranking")
    print(f"P(>=1 valid) = 1 - C(100-X,N)/C(100,N),  score = sum of top-{TOP_K_PER_TYPE} per-scene diffs")
    print(f"{'='*100}")

    print(f"\n{'Rank':<5} {'Object':<25} {'Score N=10':>10} {'Score N=50':>10} {'Score N=100':>11}"
          f"  {'Cov Ours':>8} {'Cov Base':>9} {'Gap':>6}")
    print("-" * 90)

    for rank, (obj, info) in enumerate(ranked_objs, 1):
        s = info["scores"]
        overall_sel = obj_summary[obj]["overall"].get("selected_100", {})
        overall_base = obj_summary[obj]["overall"].get("baseline_100", {})
        cov_sel = overall_sel.get("coverage", 0)
        cov_base = overall_base.get("coverage", 0)
        gap = cov_sel - cov_base
        sign = "+" if gap > 0 else ""
        print(f"{rank:<5} {obj:<25} {s[10]:>9.3f} {s[50]:>9.3f} {s[100]:>10.3f}"
              f"  {cov_sel:>7.1f}% {cov_base:>8.1f}% {sign}{gap:>5.1f}%")

    # Detailed top 10
    print(f"\n\n{'='*100}")
    print(f"TOP 10 OBJECTS — Per-scene breakdown (N=10, top-{TOP_K_PER_TYPE} per type = {TOP_K_PER_TYPE*3} scenes)")
    print(f"{'='*100}")

    for rank, (obj, info) in enumerate(ranked_objs[:10], 1):
        score = info["scores"][10]
        print(f"\n  #{rank}  {obj}  (N=10 score: {score:.3f})")
        details = info["details"][10]
        for stype in ["wall", "shelf", "cluttered"]:
            type_items = [(si, p_sel, p_base, diff) for st, si, p_sel, p_base, diff in details if st == stype]
            if type_items:
                type_score = sum(d for _, _, _, d in type_items)
                print(f"     {stype} (sub-score: {type_score:.3f}):")
                for si, p_sel, p_base, diff in type_items:
                    print(f"       scene {si:3d}:  P_sel={p_sel*100:6.2f}%  P_base={p_base*100:6.2f}%  diff={diff*100:+6.2f}%")

    # Save full results
    out_path = os.path.join(REBUTTAL_DIR, "analysis_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "ranking": [(obj, info["scores"]) for obj, info in ranked_objs],
            "per_object": obj_summary,
            "n_samples": N_SAMPLES,
        }, f, indent=2, default=float)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
