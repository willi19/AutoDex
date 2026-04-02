"""Sim Domain Randomization vs Real experiment confusion matrix.

Step 1: Run MuJoCo sim eval over a grid of (friction_tangent × obj_mass) for
        each grasp with a real experiment result. Save per-combo raw data.
Step 2: Load raw data, sweep thresholds, compute confusion matrices.

Grid: friction_tangent (5) × obj_mass (5) = 25 combos per grasp.

Usage:
    # Run sim DR (saves raw results)
    python src/validation/simulation/sim_dr_confusion.py --obj plant_mister
    python src/validation/simulation/sim_dr_confusion.py

    # Analyze saved results with different thresholds
    python src/validation/simulation/sim_dr_confusion.py --analyze outputs/sim_dr/results_25grid.json --sim_threshold 0.5
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm

import mujoco
from autodex.simulator.hand_object import MjHO
from autodex.utils.conversion import se32cart
from autodex.simulator.rot_util import np_get_delta_qpos

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

FORCE_DIRECTIONS = np.array([
    [-1,0,0, 0,0,0], [1,0,0, 0,0,0],
    [0,-1,0, 0,0,0], [0,1,0, 0,0,0],
    [0,0,-1, 0,0,0], [0,0,1, 0,0,0],
], dtype=float)

TRANS_THRE = 0.05
ANGLE_THRE = 15.0
FORCE_SCALE = 9.8

HAND_PATH = os.path.join(
    REPO_ROOT, "src", "grasp_generation", "sim_filter",
    "assets", "hand", "allegro", "right_hand.xml",
)

import transforms3d.quaternions as tq
_q_delta = np.array([0, 1, 0, 1], dtype=np.float64)
_q_delta /= np.linalg.norm(_q_delta)
R_DELTA = tq.quat2mat(_q_delta)

DR_GRID = {
    "friction_tangent": np.linspace(0.3, 1.0, 5).tolist(),
    "obj_mass": [0.1, 0.2, 0.3, 0.4, 0.5],
}
FRICTION_TORSION = 0.02  # fixed

EXPERIMENT_ROOT = os.path.expanduser(
    "~/shared_data/AutoDex/experiment/selected_100/allegro"
)
CANDIDATE_ROOT = os.path.expanduser(
    "~/shared_data/AutoDex/candidates/allegro/selected_100"
)


def build_param_grid():
    """5x5 = 25 combinations."""
    params = []
    for ft in DR_GRID["friction_tangent"]:
        for om in DR_GRID["obj_mass"]:
            params.append({"friction_tangent": ft, "obj_mass": om})
    return params


def eval_grasp_with_params(mj, wrist_se3, pregrasp, grasp, obj_mass, record_traj=False):
    """Run single sim eval. Returns (success, traj) if record_traj else success."""
    w = wrist_se3.copy()
    w[:3, :3] = w[:3, :3] @ np.linalg.inv(R_DELTA)
    wrist_cart = se32cart(w)

    squeeze = grasp * 2 - pregrasp
    pregrasp_qpos = np.concatenate([wrist_cart, pregrasp])
    grasp_qpos = np.concatenate([wrist_cart, grasp])
    squeeze_qpos = np.concatenate([wrist_cart, squeeze])
    obj_pose = np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)

    traj = {"robot_qpos": [], "object_pose": [], "phase": []} if record_traj else None

    def _rec(phase):
        if traj is not None:
            traj["robot_qpos"].append(mj.data.qpos[:-7].tolist())
            traj["object_pose"].append(mj.get_obj_pose().tolist())
            traj["phase"].append(phase)

    mj.reset_pose_qpos(pregrasp_qpos, obj_pose)
    _rec("pregrasp")
    mj.control_hand_with_interp(pregrasp_qpos, grasp_qpos)
    _rec("grasp")
    mj.control_hand_with_interp(grasp_qpos, squeeze_qpos)
    _rec("squeeze")

    for fi, fdir in enumerate(FORCE_DIRECTIONS):
        mj.reset_pose_qpos(pregrasp_qpos, obj_pose)
        mj.control_hand_with_interp(pregrasp_qpos, grasp_qpos)
        mj.control_hand_with_interp(grasp_qpos, squeeze_qpos)

        mj.set_ext_force_on_obj(fdir * FORCE_SCALE * obj_mass)
        pre_obj = mj.get_obj_pose().copy()

        for _ in range(50):
            mj.control_hand_step(step_inner=10)
            _rec(f"force_{fi}")
            cur_obj = mj.get_obj_pose()
            dp, da = np_get_delta_qpos(pre_obj, cur_obj)
            if dp > TRANS_THRE or da > ANGLE_THRE:
                mj.set_ext_force_on_obj(np.zeros(6))
                if record_traj:
                    return False, traj
                return False

        mj.set_ext_force_on_obj(np.zeros(6))

    if record_traj:
        return True, traj
    return True


def collect_experiments(obj_name):
    """Collect experiments that were actually executed (not perception/planning fail)."""
    obj_dir = os.path.join(EXPERIMENT_ROOT, obj_name)
    if not os.path.isdir(obj_dir):
        return []

    experiments = []
    for dirname in sorted(os.listdir(obj_dir)):
        result_path = os.path.join(obj_dir, dirname, "result.json")
        if not os.path.exists(result_path):
            continue
        with open(result_path) as f:
            result = json.load(f)

        scene_info = result.get("scene_info")
        if scene_info is None or len(scene_info) != 3:
            continue

        reason = result.get("reason", "")
        if reason in ("perception_failed", "planning_failed"):
            continue

        scene_type, scene_id, grasp_idx = scene_info
        cand_dir = os.path.join(CANDIDATE_ROOT, obj_name, scene_type, scene_id, grasp_idx)
        if not os.path.isdir(cand_dir):
            continue

        required = ["wrist_se3.npy", "pregrasp_pose.npy", "grasp_pose.npy"]
        if not all(os.path.exists(os.path.join(cand_dir, f)) for f in required):
            continue

        experiments.append({
            "real_success": result["success"],
            "cand_dir": cand_dir,
            "scene_info": scene_info,
            "dir_idx": dirname,
        })

    return experiments


def _cache_key(params):
    return f"ft{params['friction_tangent']:.4f}_m{params['obj_mass']:.4f}"


def _load_cache(cand_dir):
    cache_path = os.path.join(cand_dir, "sim_dr_cache.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    return {}


def _save_cache(cand_dir, cache):
    cache_path = os.path.join(cand_dir, "sim_dr_cache.json")
    with open(cache_path, "w") as f:
        json.dump(cache, f)


def run_sim_dr(obj_name):
    """Run DR sim eval on grid. Caches per-combo results in candidate dir.

    Loop order: params (outer) × experiments (inner) so MjHO is created once per param combo.
    """
    experiments = collect_experiments(obj_name)
    if not experiments:
        print(f"  {obj_name}: no valid experiments found")
        return []

    param_grid = build_param_grid()

    # Load all caches, figure out what needs running per param combo
    caches = {}  # cand_dir -> cache dict
    grasp_data = {}  # cand_dir -> (wrist_se3, pregrasp, grasp)
    work = {}  # (ft, om) -> [exp indices that need this combo]

    for i, exp in enumerate(experiments):
        cd = exp["cand_dir"]
        if cd not in caches:
            caches[cd] = _load_cache(cd)
        cache = caches[cd]
        for params in param_grid:
            key = _cache_key(params)
            if key not in cache:
                pk = (params["friction_tangent"], params["obj_mass"])
                work.setdefault(pk, set()).add(i)

    total = sum(len(param_grid) for _ in experiments)
    total_new = sum(len(exps) for exps in work.values())
    total_cached = total - total_new
    print(f"  {len(experiments)} experiments × {len(param_grid)} combos: {total_cached} cached, {total_new} new")

    if total_new > 0:
        # Pre-load grasp data for experiments that need computation
        need_load = set()
        for exp_idxs in work.values():
            need_load.update(exp_idxs)
        for i in need_load:
            cd = experiments[i]["cand_dir"]
            if cd not in grasp_data:
                grasp_data[cd] = (
                    np.load(os.path.join(cd, "wrist_se3.npy")),
                    np.load(os.path.join(cd, "pregrasp_pose.npy")),
                    np.load(os.path.join(cd, "grasp_pose.npy")),
                )

        # Outer loop: param combos (one MjHO per combo)
        pbar = tqdm(total=total_new, desc=f"  {obj_name}", leave=False)
        for (ft, om), exp_idxs in work.items():
            mj = MjHO(
                obj_name, HAND_PATH, weld_body_name="world",
                obj_mass=om, friction_coef=(ft, FRICTION_TORSION),
            )
            params = {"friction_tangent": ft, "obj_mass": om}
            key = _cache_key(params)

            for i in exp_idxs:
                cd = experiments[i]["cand_dir"]
                wrist_se3, pregrasp, grasp = grasp_data[cd]
                try:
                    succ = eval_grasp_with_params(mj, wrist_se3, pregrasp, grasp, om)
                except Exception:
                    succ = False
                caches[cd][key] = succ
                pbar.update(1)

            mj.close()
        pbar.close()

        # Save updated caches
        for cd, cache in caches.items():
            _save_cache(cd, cache)

    # Build results from caches
    results = []
    for exp in experiments:
        cache = caches[exp["cand_dir"]]
        trials = [{"params": p, "success": cache[_cache_key(p)]} for p in param_grid]
        results.append({
            "obj": obj_name,
            "dir_idx": exp["dir_idx"],
            "scene_info": exp["scene_info"],
            "real_success": exp["real_success"],
            "trials": trials,
        })

    return results


def _compute_confusion(results, sim_threshold):
    """Returns (tp, fp, fn, tn)."""
    tp = fp = fn = tn = 0
    for r in results:
        n_succ = sum(1 for t in r["trials"] if t["success"])
        sim_pass = (n_succ / len(r["trials"])) >= sim_threshold
        real = r["real_success"]
        if real and sim_pass: tp += 1
        elif not real and sim_pass: fp += 1
        elif real and not sim_pass: fn += 1
        else: tn += 1
    return tp, fp, fn, tn


def _bootstrap_ci(results, sim_threshold, n_boot=1000, ci=0.95, seed=42):
    """Bootstrap confidence intervals for FPR, FNR, accuracy, precision, recall, F1."""
    rng = np.random.default_rng(seed)
    n = len(results)
    metrics = {"fpr": [], "fnr": [], "accuracy": [], "precision": [], "recall": [], "f1": []}

    for _ in range(n_boot):
        idxs = rng.integers(0, n, size=n)
        sample = [results[i] for i in idxs]
        tp, fp, fn, tn = _compute_confusion(sample, sim_threshold)
        total = tp + fp + fn + tn

        metrics["fpr"].append(fp / (fp + tn) if (fp + tn) else 0)
        metrics["fnr"].append(fn / (fn + tp) if (fn + tp) else 0)
        metrics["accuracy"].append((tp + tn) / total if total else 0)
        metrics["precision"].append(tp / (tp + fp) if (tp + fp) else 0)
        metrics["recall"].append(tp / (tp + fn) if (tp + fn) else 0)
        p, r = metrics["precision"][-1], metrics["recall"][-1]
        metrics["f1"].append(2 * p * r / (p + r) if (p + r) else 0)

    alpha = (1 - ci) / 2
    ci_results = {}
    for key, vals in metrics.items():
        vals = np.array(vals)
        ci_results[key] = (float(np.percentile(vals, 100 * alpha)),
                           float(np.percentile(vals, 100 * (1 - alpha))))
    return ci_results


def print_confusion_matrix(results, sim_threshold, label=""):
    tp, fp, fn, tn = _compute_confusion(results, sim_threshold)

    total = len(results)
    if total == 0:
        print("No results.")
        return

    ci = _bootstrap_ci(results, sim_threshold)

    fpr = fp / (fp + tn) if (fp + tn) else 0
    fnr = fn / (fn + tp) if (fn + tp) else 0
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print(f"\n{'=' * 60}")
    if label:
        print(f"  {label} (threshold={sim_threshold})")
    print(f"  Total: {total}  (Real+: {tp+fn}, Real-: {fp+tn})")
    print(f"{'=' * 60}")
    print(f"                    Real Success   Real Fail")
    print(f"  Sim  Pass    {tp:>8d} (TP)    {fp:>8d} (FP)")
    print(f"  Sim  Fail    {fn:>8d} (FN)    {tn:>8d} (TN)")
    print(f"{'=' * 60}")
    print(f"  FPR:       {fpr:.3f}  [{ci['fpr'][0]:.3f}, {ci['fpr'][1]:.3f}]")
    print(f"  FNR:       {fnr:.3f}  [{ci['fnr'][0]:.3f}, {ci['fnr'][1]:.3f}]")
    print(f"  Accuracy:  {accuracy:.3f}  [{ci['accuracy'][0]:.3f}, {ci['accuracy'][1]:.3f}]")
    print(f"  Precision: {precision:.3f}  [{ci['precision'][0]:.3f}, {ci['precision'][1]:.3f}]")
    print(f"  Recall:    {recall:.3f}  [{ci['recall'][0]:.3f}, {ci['recall'][1]:.3f}]")
    print(f"  F1:        {f1:.3f}  [{ci['f1'][0]:.3f}, {ci['f1'][1]:.3f}]")
    print()


def analyze(results_path, sim_threshold):
    """Load saved results and compute confusion matrix."""
    with open(results_path) as f:
        data = json.load(f)

    results = data["results"]
    objs = sorted(set(r["obj"] for r in results))

    for obj in objs:
        obj_results = [r for r in results if r["obj"] == obj]
        print_confusion_matrix(obj_results, sim_threshold, label=obj)

    if len(objs) > 1:
        print_confusion_matrix(results, sim_threshold, label="ALL OBJECTS")


def save_fp_fn_traj(results_path, sim_threshold=0.5, output_dir=None):
    """For FP/FN cases, re-run sim with median param and save trajectory."""
    with open(results_path) as f:
        data = json.load(f)

    if output_dir is None:
        output_dir = os.path.join(REPO_ROOT, "outputs", "sim_dr", "traj")
    os.makedirs(output_dir, exist_ok=True)

    fp_fn = []
    for r in data["results"]:
        n_succ = sum(1 for t in r["trials"] if t["success"])
        sim_pass = (n_succ / len(r["trials"])) >= sim_threshold
        real = r["real_success"]
        if (not real and sim_pass) or (real and not sim_pass):
            label = "fp" if (not real and sim_pass) else "fn"
            fp_fn.append((label, r))

    print(f"Saving trajectories for {len(fp_fn)} FP/FN cases...")

    # Use median grid params for trajectory
    median_ft = float(np.median(DR_GRID["friction_tangent"]))
    median_m = float(np.median(DR_GRID["obj_mass"]))
    print(f"  Using median params: friction={median_ft}, mass={median_m}")

    for i, (label, r) in enumerate(fp_fn):
        obj_name = r["obj"]
        scene_info = r["scene_info"]
        dir_idx = r["dir_idx"]
        cand_dir = os.path.join(CANDIDATE_ROOT, obj_name, *scene_info)

        if not os.path.isdir(cand_dir):
            print(f"  [{i}] {label} {obj_name}/{dir_idx}: candidate dir missing, skip")
            continue

        wrist_se3 = np.load(os.path.join(cand_dir, "wrist_se3.npy"))
        pregrasp = np.load(os.path.join(cand_dir, "pregrasp_pose.npy"))
        grasp = np.load(os.path.join(cand_dir, "grasp_pose.npy"))

        mj = MjHO(
            obj_name, HAND_PATH, weld_body_name="world",
            obj_mass=median_m, friction_coef=(median_ft, FRICTION_TORSION),
        )
        try:
            succ, traj = eval_grasp_with_params(
                mj, wrist_se3, pregrasp, grasp, median_m, record_traj=True,
            )
        except Exception as e:
            print(f"  [{i}] {label} {obj_name}/{dir_idx}: sim error {e}")
            mj.close()
            continue
        mj.close()

        # Save
        traj_dir = os.path.join(output_dir, label, f"{obj_name}_{dir_idx}")
        os.makedirs(traj_dir, exist_ok=True)
        with open(os.path.join(traj_dir, "sim_traj.json"), "w") as f:
            json.dump(traj, f)
        with open(os.path.join(traj_dir, "info.json"), "w") as f:
            json.dump({
                "obj": obj_name, "dir_idx": dir_idx, "scene_info": scene_info,
                "label": label, "real_success": r["real_success"],
                "sim_success": succ,
                "params": {"friction_tangent": median_ft, "obj_mass": median_m},
            }, f, indent=2)

        sim_str = "pass" if succ else "fail"
        real_str = "succ" if r["real_success"] else "fail"
        print(f"  [{i}] {label} {obj_name}/{dir_idx} sim={sim_str} real={real_str} -> {traj_dir}")

    print(f"Done. Trajectories in {output_dir}")


def run_sim_dr_all_candidates(obj_name):
    """Run DR sim eval on ALL candidates in selected_100 (not just experiments).
    Saves sim_dr_cache.json per candidate and returns summary."""
    cand_root = os.path.join(CANDIDATE_ROOT, obj_name)
    if not os.path.isdir(cand_root):
        print(f"  {obj_name}: no candidates found")
        return []

    param_grid = build_param_grid()

    # Collect all candidates
    candidates = []
    for scene_type in sorted(os.listdir(cand_root)):
        st_path = os.path.join(cand_root, scene_type)
        if not os.path.isdir(st_path):
            continue
        for scene_id in sorted(os.listdir(st_path)):
            si_path = os.path.join(st_path, scene_id)
            if not os.path.isdir(si_path):
                continue
            for grasp_idx in sorted(os.listdir(si_path)):
                gp = os.path.join(si_path, grasp_idx)
                if os.path.isdir(gp) and os.path.exists(os.path.join(gp, "wrist_se3.npy")):
                    candidates.append({
                        "cand_dir": gp,
                        "scene_info": [scene_type, scene_id, grasp_idx],
                    })

    if not candidates:
        print(f"  {obj_name}: no candidates found")
        return []

    # Load caches, find work
    caches = {}
    grasp_data = {}
    work = {}

    for i, cand in enumerate(candidates):
        cd = cand["cand_dir"]
        if cd not in caches:
            caches[cd] = _load_cache(cd)
        cache = caches[cd]
        for params in param_grid:
            key = _cache_key(params)
            if key not in cache:
                pk = (params["friction_tangent"], params["obj_mass"])
                work.setdefault(pk, set()).add(i)

    total = len(candidates) * len(param_grid)
    total_new = sum(len(exps) for exps in work.values())
    total_cached = total - total_new
    print(f"  {len(candidates)} candidates × {len(param_grid)} combos: {total_cached} cached, {total_new} new")

    if total_new > 0:
        need_load = set()
        for exp_idxs in work.values():
            need_load.update(exp_idxs)
        for i in need_load:
            cd = candidates[i]["cand_dir"]
            if cd not in grasp_data:
                grasp_data[cd] = (
                    np.load(os.path.join(cd, "wrist_se3.npy")),
                    np.load(os.path.join(cd, "pregrasp_pose.npy")),
                    np.load(os.path.join(cd, "grasp_pose.npy")),
                )

        pbar = tqdm(total=total_new, desc=f"  {obj_name}", leave=False)
        for (ft, om), cand_idxs in work.items():
            mj = MjHO(
                obj_name, HAND_PATH, weld_body_name="world",
                obj_mass=om, friction_coef=(ft, FRICTION_TORSION),
            )
            params = {"friction_tangent": ft, "obj_mass": om}
            key = _cache_key(params)

            for i in cand_idxs:
                cd = candidates[i]["cand_dir"]
                wrist_se3, pregrasp, grasp = grasp_data[cd]
                try:
                    succ = eval_grasp_with_params(mj, wrist_se3, pregrasp, grasp, om)
                except Exception:
                    succ = False
                caches[cd][key] = succ
                pbar.update(1)

            mj.close()
        pbar.close()

        for cd, cache in caches.items():
            _save_cache(cd, cache)

    # Build results
    results = []
    for cand in candidates:
        cache = caches[cand["cand_dir"]]
        n_pass = sum(1 for p in param_grid if cache.get(_cache_key(p), False))
        results.append({
            "scene_info": cand["scene_info"],
            "sim_pass_rate": n_pass / len(param_grid),
            "n_pass": n_pass,
            "n_total": len(param_grid),
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, default=None, help="Single object (default: all)")
    parser.add_argument("--output", type=str, default=None, help="Save results JSON path")
    parser.add_argument("--analyze", type=str, default=None,
                        help="Path to saved results JSON (skip sim, just compute confusion)")
    parser.add_argument("--sim_threshold", type=float, default=0.5,
                        help="Threshold for --analyze mode")
    parser.add_argument("--save_traj", type=str, default=None,
                        help="Path to results JSON — re-run FP/FN cases and save trajectories")
    parser.add_argument("--all_candidates", nargs="+", default=None,
                        help="Run sim DR on ALL candidates for these objects (not just experiments)")
    args = parser.parse_args()

    if args.all_candidates:
        all_cand_results = {}
        for obj_name in args.all_candidates:
            print(f"\n{obj_name}:")
            results = run_sim_dr_all_candidates(obj_name)
            all_cand_results[obj_name] = results
        output_path = args.output or os.path.join(
            REPO_ROOT, "outputs", "sim_dr", "candidates_sim_dr.json"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "grid": {k: list(v) for k, v in DR_GRID.items()},
                "friction_torsion": FRICTION_TORSION,
                "force_scale": FORCE_SCALE,
                "results": all_cand_results,
            }, f, indent=2)
        print(f"\nSaved to {output_path}")
        return

    if args.save_traj:
        save_fp_fn_traj(args.save_traj, args.sim_threshold)
        return

    if args.analyze:
        analyze(args.analyze, args.sim_threshold)
        return

    if args.obj:
        obj_list = [args.obj]
    else:
        obj_list = sorted(os.listdir(EXPERIMENT_ROOT))
        obj_list = [o for o in obj_list if os.path.isdir(os.path.join(EXPERIMENT_ROOT, o))]

    all_results = []
    for obj_name in obj_list:
        print(f"\n{obj_name}:")
        results = run_sim_dr(obj_name)
        all_results.extend(results)

    # Save raw results (params + success per trial)
    n_combos = len(build_param_grid())
    output_path = args.output or os.path.join(
        REPO_ROOT, "outputs", "sim_dr", f"results_{n_combos}grid.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "args": vars(args),
            "grid": {k: list(v) for k, v in DR_GRID.items()},
            "friction_torsion": FRICTION_TORSION,
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Print quick summary with default threshold
    print_confusion_matrix(all_results, 0.5, label="ALL (threshold=0.5)")


if __name__ == "__main__":
    main()
