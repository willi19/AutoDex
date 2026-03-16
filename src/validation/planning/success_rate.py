"""
Planning Success Rate — Only on IK-Reachable Points

Loads reachability results, then runs full planning only on points
where IK succeeded. Directly compares IK reachability vs planning success.

Usage:
    # Single object (uses reachability data from outputs/reachability/)
    python src/validation/planning/success_rate.py --obj attached_container --version selected_100

    # All objects with reachability data
    python src/validation/planning/success_rate.py --version selected_100

    # Custom trials and reachability dir
    python src/validation/planning/success_rate.py --version selected_100 --n_trials 3 \
        --reach_dir outputs/reachability
"""

import os
import sys
import time
import argparse
import json
import logging
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "paradex"))

from autodex.planner import GraspPlanner
from autodex.utils.path import obj_path
from autodex.utils.conversion import se32cart


def load_tabletop_scene(obj_name, pose_idx="000", x_offset=0.4, z_rotation=0.0):
    """Load object pose with configurable x offset and z rotation."""
    pose_dir = os.path.join(obj_path, obj_name, "processed_data", "info", "tabletop")
    pose_file = os.path.join(pose_dir, f"{pose_idx}.npy")
    if not os.path.exists(pose_file):
        available = sorted(os.listdir(pose_dir))
        raise FileNotFoundError(f"Pose {pose_idx} not found. Available: {available}")

    obj_pose = np.load(pose_file)
    obj_pose[0, 3] += x_offset

    if z_rotation != 0.0:
        c, s = np.cos(z_rotation), np.sin(z_rotation)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        obj_pose[:3, :3] = Rz @ obj_pose[:3, :3]

    mesh_path = os.path.join(obj_path, obj_name, "processed_data", "mesh", "simplified.obj")

    scene_cfg = {
        "mesh": {
            "target": {
                "pose": se32cart(obj_pose).tolist(),
                "file_path": mesh_path,
            },
        },
        "cuboid": {
            "table": {
                "dims": [2, 3, 0.2],
                "pose": [1.1, 0, -0.1, 1, 0, 0, 0],
            },
        },
    }
    return scene_cfg


def load_reachable_points(reach_dir, obj_name):
    """Load IK-reachable grid points from reachability data."""
    obj_dir = os.path.join(reach_dir, obj_name)
    if not os.path.isdir(obj_dir):
        return None

    grid_files = [f for f in os.listdir(obj_dir)
                  if f.endswith(".json") and "_viz" not in f]
    if not grid_files:
        return None

    with open(os.path.join(obj_dir, grid_files[0])) as f:
        data = json.load(f)

    # Only keep points where IK succeeded in all trials
    reachable = []
    for r in data["grid"]:
        if r["trials_with_ik"] == r["n_trials"]:
            reachable.append({
                "pose_idx": r["pose_idx"],
                "x_offset": r["x_offset"],
                "z_rotation_deg": r["z_rotation_deg"],
                "ik_mean": r["ik_mean"],
            })

    return {
        "reachable": reachable,
        "grasp_version": data["grasp_version"],
        "n_reachable": data["n_reachable"],
        "n_partial": data["n_partial"],
        "n_unreachable": data["n_unreachable"],
        "total": data["n_reachable"] + data["n_partial"] + data["n_unreachable"],
    }


def run_planning_on_reachable(obj_name, grasp_version, reachable_points,
                               n_trials, save_dir=None):
    """Run full planning on IK-reachable points."""
    n_points = len(reachable_points)
    print(f"Object: {obj_name}")
    print(f"Grasp version: {grasp_version}")
    print(f"IK-reachable points: {n_points}")
    print(f"Trials per point: {n_trials}")
    print(f"Total plans: {n_points * n_trials}")
    print("=" * 60)

    logging.getLogger("curobo").setLevel(logging.ERROR)
    planner = GraspPlanner()

    all_results = []
    stage_keys = ["load_candidates_s", "world_setup_s", "collision_check_s",
                  "arm_plan_s", "finger_refine_s"]

    n_plan_success = 0
    n_plan_fail = 0

    pbar = tqdm(reachable_points, desc=f"{obj_name}", unit="pt")
    for pt in pbar:
        pose_idx = pt["pose_idx"]
        x_off = pt["x_offset"]
        z_deg = pt["z_rotation_deg"]
        z_rad = np.radians(z_deg)

        scene_cfg = load_tabletop_scene(obj_name, pose_idx,
                                        x_offset=x_off, z_rotation=z_rad)

        successes = 0
        trial_timings = []

        for trial in range(n_trials):
            result = planner.plan(
                scene_cfg, obj_name=obj_name,
                grasp_version=grasp_version, mode="batch",
                seed=trial,
            )
            trial_timings.append(result.timing)
            if result.success:
                successes += 1
                break  # no need to keep trying once we know planning can succeed

        # Aggregate timing
        avg_timing = {}
        for key in stage_keys:
            vals = [t.get(key, 0) for t in trial_timings if t]
            if vals:
                avg_timing[key] = round(float(np.mean(vals)), 3)

        rate = successes / max(1, len(trial_timings))

        entry = {
            "pose_idx": pose_idx,
            "x_offset": x_off,
            "z_rotation_deg": z_deg,
            "ik_mean": pt["ik_mean"],
            "success_count": successes,
            "n_trials": n_trials,
            "success_rate": rate,
            "avg_timing": avg_timing,
        }
        all_results.append(entry)

        if rate == 1.0:
            n_plan_success += 1
        else:
            n_plan_fail += 1

        pbar.set_postfix(ok=n_plan_success, fail=n_plan_fail,
                         rate=f"{rate*100:.0f}%")

    # Summary
    print(f"\n{'=' * 60}")
    print("PLANNING ON IK-REACHABLE POINTS")
    print(f"{'=' * 60}")

    rates = [r["success_rate"] for r in all_results]
    always_ok = sum(1 for r in rates if r == 1.0)
    never_ok = sum(1 for r in rates if r == 0.0)
    partial = len(rates) - always_ok - never_ok

    print(f"IK-reachable points tested: {len(all_results)}")
    print(f"  Planning always succeeds: {always_ok}")
    print(f"  Planning sometimes fails: {partial}")
    print(f"  Planning always fails:    {never_ok}")
    print(f"Overall mean rate: {np.mean(rates)*100:.1f}%")

    if never_ok + partial > 0:
        print(f"\nIK-reachable but planning fails ({never_ok + partial} points):")
        failures = [r for r in all_results if r["success_rate"] < 1.0]
        for r in sorted(failures, key=lambda x: x["success_rate"]):
            print(f"  pose={r['pose_idx']} x={r['x_offset']:.2f} z={r['z_rotation_deg']:3.0f}°  "
                  f"plan={r['success_count']}/{r['n_trials']} ({r['success_rate']*100:.0f}%)  "
                  f"ik_mean={r['ik_mean']:.1f}")

    # Per-stage timing breakdown
    success_results = [r for r in all_results if r["success_rate"] > 0]
    fail_results = [r for r in all_results if r["success_rate"] == 0]

    def compute_timing_stats(results, stage_keys):
        stats = {}
        for key in stage_keys:
            vals = [r["avg_timing"].get(key, 0) for r in results if r["avg_timing"]]
            if vals:
                stats[key] = {
                    "mean": round(float(np.mean(vals)), 3),
                    "total": round(float(np.sum(vals)), 3),
                    "min": round(float(np.min(vals)), 3),
                    "max": round(float(np.max(vals)), 3),
                }
        # Total wall time per point
        totals = []
        for r in results:
            if r["avg_timing"]:
                totals.append(sum(r["avg_timing"].get(k, 0) for k in stage_keys))
        if totals:
            stats["total_per_point"] = {
                "mean": round(float(np.mean(totals)), 3),
                "min": round(float(np.min(totals)), 3),
                "max": round(float(np.max(totals)), 3),
                "total": round(float(np.sum(totals)), 3),
            }
        return stats

    timing_all = compute_timing_stats(all_results, stage_keys)
    timing_success = compute_timing_stats(success_results, stage_keys) if success_results else {}
    timing_fail = compute_timing_stats(fail_results, stage_keys) if fail_results else {}

    # Save
    if save_dir is None:
        save_dir = os.path.join("outputs", "planning_success_rate", obj_name)
    os.makedirs(save_dir, exist_ok=True)

    summary = {
        "obj_name": obj_name,
        "grasp_version": grasp_version,
        "n_trials": n_trials,
        "n_ik_reachable": len(all_results),
        "n_plan_always_ok": always_ok,
        "n_plan_partial": partial,
        "n_plan_always_fail": never_ok,
        "overall_mean_rate": round(float(np.mean(rates)), 3),
        "timing": {
            "all": timing_all,
            "success": timing_success,
            "fail": timing_fail,
        },
        "results": all_results,
    }

    out_path = os.path.join(save_dir, f"plan_vs_ik_{grasp_version}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Planning success on IK-reachable points")
    parser.add_argument("--obj", type=str, default=None, help="Object name (omit for all)")
    parser.add_argument("--version", type=str, required=True, help="Grasp candidate version")
    parser.add_argument("--n_trials", type=int, default=3, help="Trials per point")
    parser.add_argument("--reach_dir", type=str, default="outputs/reachability",
                        help="Reachability results directory")
    parser.add_argument("--save_dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    # Find objects with reachability data
    if args.obj is not None:
        objects = [args.obj]
    else:
        objects = sorted([d for d in os.listdir(args.reach_dir)
                          if os.path.isdir(os.path.join(args.reach_dir, d))])
        print(f"Found {len(objects)} objects with reachability data\n")

    all_summaries = {}
    for obj_name in objects:
        print(f"\n{'#' * 60}")
        print(f"# {obj_name}")
        print(f"{'#' * 60}")

        reach = load_reachable_points(args.reach_dir, obj_name)
        if reach is None:
            print(f"  No reachability data found, skipping")
            continue

        if not reach["reachable"]:
            print(f"  No IK-reachable points (0/{reach['total']}), skipping")
            continue

        print(f"  IK: {reach['n_reachable']}/{reach['total']} reachable, "
              f"{reach['n_partial']} partial, {reach['n_unreachable']} unreachable")

        try:
            summary = run_planning_on_reachable(
                obj_name=obj_name,
                grasp_version=args.version,
                reachable_points=reach["reachable"],
                n_trials=args.n_trials,
                save_dir=args.save_dir,
            )
            all_summaries[obj_name] = summary
        except Exception as e:
            print(f"  SKIPPED: {e}")
            import traceback
            traceback.print_exc()
            all_summaries[obj_name] = {"error": str(e)}

    if len(all_summaries) > 1:
        print(f"\n{'=' * 60}")
        print("ALL OBJECTS: IK REACHABLE vs PLANNING SUCCESS")
        print(f"{'=' * 60}")
        for obj_name, s in all_summaries.items():
            if "error" in s:
                print(f"  {obj_name}: ERROR - {s['error']}")
            else:
                print(f"  {obj_name}: ik_reachable={s['n_ik_reachable']}  "
                      f"plan_ok={s['n_plan_always_ok']}  "
                      f"plan_fail={s['n_plan_always_fail']}  "
                      f"partial={s['n_plan_partial']}  "
                      f"rate={s['overall_mean_rate']*100:.1f}%")
