#!/usr/bin/env python3
"""Step 4: Evaluate pose quality and depth comparison. (conda: any — numpy only)

Computes:
- Multi-view consensus pose (quaternion averaging of inliers)
- Per-view pose error vs consensus (translation, rotation, ADD)
- Depth comparison (DA3 vs stereo) in mask region
- View ranking by pose quality
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

AUTODEX_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(AUTODEX_ROOT))

MESH_ROOT = Path("/home/mingi/shared_data/object_6d/data/mesh")


def load_poses(pose_dir):
    """Load per-view world-frame poses. Returns {serial: 4x4}."""
    poses = {}
    world_dir = pose_dir / "ob_in_world"
    if not world_dir.exists():
        return poses
    for f in sorted(world_dir.glob("*.txt")):
        poses[f.stem] = np.loadtxt(str(f))
    return poses


def rotation_error_deg(R1, R2):
    """Geodesic rotation error in degrees."""
    R_diff = R1.T @ R2
    cos_angle = (np.trace(R_diff) - 1) / 2
    cos_angle = np.clip(cos_angle, -1, 1)
    return np.degrees(np.arccos(cos_angle))


def translation_error_mm(t1, t2):
    """Translation error in mm."""
    return float(np.linalg.norm(t1 - t2) * 1000)


def add_error_mm(pose_est, pose_gt, vertices):
    """Average Distance of model points (ADD) in mm."""
    pts_est = (pose_est[:3, :3] @ vertices.T + pose_est[:3, 3:4]).T
    pts_gt = (pose_gt[:3, :3] @ vertices.T + pose_gt[:3, 3:4]).T
    return float(np.mean(np.linalg.norm(pts_est - pts_gt, axis=1)) * 1000)


def add_s_error_mm(pose_est, pose_gt, vertices):
    """Symmetric ADD (ADD-S) in mm — closest point matching."""
    from scipy.spatial import cKDTree
    pts_est = (pose_est[:3, :3] @ vertices.T + pose_est[:3, 3:4]).T
    pts_gt = (pose_gt[:3, :3] @ vertices.T + pose_gt[:3, 3:4]).T
    tree = cKDTree(pts_gt)
    dists, _ = tree.query(pts_est)
    return float(np.mean(dists) * 1000)


def consensus_pose(poses_dict):
    """Compute consensus pose from multiple views using RANSAC-like inlier selection.

    1. For each pair, compute translation distance
    2. Pick the pose with most inliers (translation within threshold)
    3. Average inlier poses (quaternion averaging for rotation)
    """
    if len(poses_dict) < 1:
        return None, []

    serials = list(poses_dict.keys())
    poses = [poses_dict[s] for s in serials]

    if len(poses) == 1:
        return poses[0], serials

    # Compute pairwise translation distances
    translations = np.array([p[:3, 3] for p in poses])
    n = len(poses)

    # Find inliers: pose with most neighbors within 30mm
    threshold = 0.03  # 30mm
    best_inliers = []
    for i in range(n):
        dists = np.linalg.norm(translations - translations[i], axis=1)
        inliers = [j for j in range(n) if dists[j] < threshold]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    if len(best_inliers) < 2:
        # Relax threshold
        threshold = 0.05  # 50mm
        for i in range(n):
            dists = np.linalg.norm(translations - translations[i], axis=1)
            inliers = [j for j in range(n) if dists[j] < threshold]
            if len(inliers) > len(best_inliers):
                best_inliers = inliers

    if len(best_inliers) == 0:
        best_inliers = [0]

    inlier_serials = [serials[i] for i in best_inliers]
    inlier_poses = [poses[i] for i in best_inliers]

    # Average translation
    avg_t = np.mean([p[:3, 3] for p in inlier_poses], axis=0)

    # Average rotation via quaternion averaging
    quats = np.array([Rotation.from_matrix(p[:3, :3]).as_quat() for p in inlier_poses])
    # Ensure consistent quaternion sign
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[0]) < 0:
            quats[i] *= -1
    avg_quat = np.mean(quats, axis=0)
    avg_quat /= np.linalg.norm(avg_quat)
    avg_R = Rotation.from_quat(avg_quat).as_matrix()

    result = np.eye(4)
    result[:3, :3] = avg_R
    result[:3, 3] = avg_t

    return result, inlier_serials


def evaluate_depth_comparison(out_dir, serials):
    """Compare DA3 vs stereo depth in mask regions."""
    da3_dir = out_dir / "depth_da3"
    stereo_dir = out_dir / "depth_stereo"
    masks_dir = out_dir / "masks"

    if not da3_dir.exists() or not stereo_dir.exists():
        return {}

    results = {}
    for s in serials:
        mask_path = masks_dir / f"{s}.png"
        da3_path = da3_dir / f"{s}.png"
        stereo_path = stereo_dir / f"{s}.png"

        if not all(p.exists() for p in [mask_path, da3_path, stereo_path]):
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127
        da3 = cv2.imread(str(da3_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        stereo = cv2.imread(str(stereo_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

        # Compare in mask region where both have valid depth
        valid = mask & (da3 > 0.001) & (stereo > 0.001)
        if valid.sum() < 50:
            continue

        da3_v = da3[valid]
        stereo_v = stereo[valid]
        diff = np.abs(da3_v - stereo_v)

        results[s] = {
            "n_pixels": int(valid.sum()),
            "da3_mean_m": float(da3_v.mean()),
            "stereo_mean_m": float(stereo_v.mean()),
            "mean_abs_diff_mm": float(diff.mean() * 1000),
            "median_abs_diff_mm": float(np.median(diff) * 1000),
            "agree_5mm_pct": float((diff < 0.005).mean() * 100),
            "agree_10mm_pct": float((diff < 0.010).mean() * 100),
            "agree_20mm_pct": float((diff < 0.020).mean() * 100),
            "correlation": float(np.corrcoef(da3_v, stereo_v)[0, 1]),
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--episode", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_root) / args.obj / args.episode

    # Load camera data from cam_param/
    with open(out_dir / "cam_param" / "intrinsics.json") as f:
        intr_raw = json.load(f)

    img_dir = out_dir / "images"
    if not img_dir.exists():
        img_dir = out_dir / "raw" / "images"
    serials = sorted(p.stem for p in img_dir.glob("*.png"))

    # Load mesh vertices for ADD
    import trimesh
    mesh_dir = MESH_ROOT / args.obj
    mesh_candidates = [
        mesh_dir / "simplified.obj", mesh_dir / f"{args.obj}.obj",
        mesh_dir / "coacd.obj", mesh_dir / "raw.obj",
    ]
    mesh_path = next((c for c in mesh_candidates if c.exists()), None)
    if mesh_path is None:
        objs = list(mesh_dir.glob("*.obj"))
        mesh_path = objs[0] if objs else None

    vertices = None
    if mesh_path:
        mesh = trimesh.load(str(mesh_path), force="mesh")
        # Subsample for speed
        vertices = mesh.vertices.astype(np.float32)
        if len(vertices) > 2000:
            idx = np.random.RandomState(42).choice(len(vertices), 2000, replace=False)
            vertices = vertices[idx]

    results = {"obj": args.obj, "episode": args.episode}

    # Evaluate each depth method
    for depth_name in ["da3", "stereo"]:
        pose_dir = out_dir / f"pose_{depth_name}"
        if not pose_dir.exists():
            continue

        poses = load_poses(pose_dir)
        if not poses:
            continue

        print(f"\n=== {depth_name}: {len(poses)} poses ===")

        # Consensus
        gt_pose, inlier_serials = consensus_pose(poses)
        if gt_pose is None:
            continue

        print(f"Consensus from {len(inlier_serials)} inliers: {inlier_serials}")
        print(f"  t = {gt_pose[:3,3]}")

        # Per-view errors
        per_view = {}
        for s, pose in poses.items():
            err = {
                "trans_err_mm": translation_error_mm(pose[:3, 3], gt_pose[:3, 3]),
                "rot_err_deg": rotation_error_deg(pose[:3, :3], gt_pose[:3, :3]),
                "is_inlier": s in inlier_serials,
            }
            if vertices is not None:
                err["add_mm"] = add_error_mm(pose, gt_pose, vertices)
                err["add_s_mm"] = add_s_error_mm(pose, gt_pose, vertices)
            per_view[s] = err

        # Sort by ADD
        sort_key = "add_mm" if vertices is not None else "trans_err_mm"
        ranked = sorted(per_view.items(), key=lambda x: x[1][sort_key])

        print(f"\nView ranking ({depth_name}):")
        print(f"  {'Serial':<12} {'Trans(mm)':>10} {'Rot(deg)':>10} {'ADD(mm)':>10} {'Inlier':>8}")
        for s, err in ranked:
            add_str = f"{err.get('add_mm', -1):>10.1f}" if 'add_mm' in err else f"{'N/A':>10}"
            print(f"  {s:<12} {err['trans_err_mm']:>10.1f} {err['rot_err_deg']:>10.2f} "
                  f"{add_str} {'*' if err['is_inlier'] else '':>8}")

        results[depth_name] = {
            "consensus_pose": gt_pose.tolist(),
            "n_inliers": len(inlier_serials),
            "inlier_serials": inlier_serials,
            "per_view": per_view,
            "summary": {
                "median_trans_err_mm": float(np.median([e["trans_err_mm"] for e in per_view.values()])),
                "median_rot_err_deg": float(np.median([e["rot_err_deg"] for e in per_view.values()])),
            },
        }
        if vertices is not None:
            results[depth_name]["summary"]["median_add_mm"] = float(
                np.median([e["add_mm"] for e in per_view.values()]))

    # Depth comparison
    print(f"\n=== Depth comparison (DA3 vs Stereo) ===")
    depth_cmp = evaluate_depth_comparison(out_dir, serials)
    if depth_cmp:
        results["depth_comparison"] = depth_cmp
        all_diffs = [v["mean_abs_diff_mm"] for v in depth_cmp.values()]
        all_corrs = [v["correlation"] for v in depth_cmp.values()]
        print(f"  Views with both: {len(depth_cmp)}")
        print(f"  Mean abs diff: {np.mean(all_diffs):.1f}mm (median {np.median(all_diffs):.1f}mm)")
        print(f"  Correlation: {np.mean(all_corrs):.4f}")
        print(f"  Agree <5mm: {np.mean([v['agree_5mm_pct'] for v in depth_cmp.values()]):.1f}%")
        print(f"  Agree <20mm: {np.mean([v['agree_20mm_pct'] for v in depth_cmp.values()]):.1f}%")

    # Save results
    # Convert numpy types for JSON
    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj

    results_path = out_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(to_serializable(results), f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()