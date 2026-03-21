#!/usr/bin/env python3
"""Step 5: Rank views using GT pose, compare DA3 vs stereo, select best 5. (conda: foundationpose)

Uses the GT pose from step4 to evaluate per-view pose quality.
Also runs FPose with stereo depth for comparison.
Outputs best 5 views for each depth method.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
AUTODEX_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(AUTODEX_ROOT))

MESH_ROOT = Path("/home/mingi/shared_data/object_6d/data/mesh")


def find_mesh(obj_name):
    candidates = [
        MESH_ROOT / obj_name / "simplified.obj",
        MESH_ROOT / obj_name / f"{obj_name}.obj",
        MESH_ROOT / obj_name / "coacd.obj",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    objs = list((MESH_ROOT / obj_name).glob("*.obj"))
    return str(objs[0]) if objs else None


def rotation_error_deg(R1, R2):
    R_diff = R1.T @ R2
    cos_angle = np.clip((np.trace(R_diff) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cos_angle)))


def add_error_mm(pose_est, pose_gt, vertices):
    pts_est = (pose_est[:3, :3] @ vertices.T + pose_est[:3, 3:4]).T
    pts_gt = (pose_gt[:3, :3] @ vertices.T + pose_gt[:3, 3:4]).T
    return float(np.mean(np.linalg.norm(pts_est - pts_gt, axis=1)) * 1000)


def run_fpose_stereo(cameras, masks_dir, stereo_dir, mesh_path, out_dir, downscale=0.5):
    """Run FPose with stereo depth on all views. Returns {serial: pose_world}."""
    from autodex.perception import PoseTracker

    cam_dir = out_dir / "ob_in_cam"
    world_dir = out_dir / "ob_in_world"
    for d in [out_dir, cam_dir, world_dir]:
        d.mkdir(parents=True, exist_ok=True)

    tracker = PoseTracker(mesh_path, device_id=0)
    poses = {}

    for cam in cameras:
        s = cam["serial"]
        mask_path = masks_dir / f"{s}.png"
        depth_path = stereo_dir / f"{s}.png"

        if not mask_path.exists() or not depth_path.exists():
            continue

        rgb = cam["rgb"].copy()
        mask = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        K = cam["K"].copy()

        if downscale != 1.0:
            h, w = rgb.shape[:2]
            nw, nh = int(w * downscale), int(h * downscale)
            rgb = cv2.resize(rgb, (nw, nh))
            depth = cv2.resize(depth, (nw, nh), interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
            K[0] *= downscale
            K[1] *= downscale

        if mask.sum() < 100 or (depth[mask > 0] > 0.001).sum() < 50:
            continue

        tracker.reset()
        try:
            pose_cam = tracker.init(rgb, depth, mask, K, iteration=5)
        except Exception:
            continue

        np.savetxt(str(cam_dir / f"{s}.txt"), pose_cam)
        T_world_cam = np.linalg.inv(cam["T"])
        pose_world = T_world_cam @ pose_cam
        np.savetxt(str(world_dir / f"{s}.txt"), pose_world)
        poses[s] = pose_world
        print(f"  {s}: ok (stereo)")

    return poses


def evaluate_views(poses_dict, gt_pose, vertices, label):
    """Compute per-view errors vs GT, return ranked list."""
    results = {}
    for s, pose in poses_dict.items():
        trans_err = float(np.linalg.norm(pose[:3, 3] - gt_pose[:3, 3]) * 1000)
        rot_err = rotation_error_deg(pose[:3, :3], gt_pose[:3, :3])
        add_err = add_error_mm(pose, gt_pose, vertices)
        results[s] = {"trans_mm": trans_err, "rot_deg": rot_err, "add_mm": add_err}

    ranked = sorted(results.items(), key=lambda x: x[1]["add_mm"])

    print(f"\n{'='*60}")
    print(f"View ranking ({label}):")
    print(f"  {'Rank':<5} {'Serial':<12} {'Trans(mm)':>10} {'Rot(deg)':>10} {'ADD(mm)':>10}")
    for rank, (s, err) in enumerate(ranked, 1):
        print(f"  {rank:<5} {s:<12} {err['trans_mm']:>10.1f} {err['rot_deg']:>10.2f} {err['add_mm']:>10.1f}")

    return ranked, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--episode", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--downscale", type=float, default=0.5)
    args = parser.parse_args()

    capture_dir = Path(args.data_root) / args.obj / args.episode
    out_dir = capture_dir
    pose_dir = out_dir / "pose"

    # Load GT pose
    gt_path = pose_dir / "gt.npy"
    if not gt_path.exists():
        print(f"ERROR: no GT pose at {gt_path}, run step4 first")
        return
    gt_pose = np.load(str(gt_path)).reshape(4, 4)
    print(f"GT pose t = {gt_pose[:3, 3]}")

    # Skip if view ranking already exists
    if (out_dir / "view_ranking.json").exists():
        print(f"  View ranking already exists, skipping")
        return

    # Load mesh vertices
    import trimesh
    mesh_path = find_mesh(args.obj)
    mesh = trimesh.load(str(mesh_path), force="mesh")
    vertices = mesh.vertices.astype(np.float32)
    if len(vertices) > 2000:
        vertices = vertices[np.random.RandomState(42).choice(len(vertices), 2000, replace=False)]

    # Load camera data from cam_param/
    with open(capture_dir / "cam_param" / "intrinsics.json") as f:
        intr_raw = json.load(f)
    with open(capture_dir / "cam_param" / "extrinsics.json") as f:
        extr_raw = json.load(f)

    img_dir = capture_dir / "images"
    if not img_dir.exists():
        img_dir = capture_dir / "raw" / "images"
    serials = sorted(p.stem for p in img_dir.glob("*.png"))

    intrinsics = np.array([intr_raw[s]["intrinsics_undistort"] for s in serials], dtype=np.float32)
    extrinsics = []
    for s in serials:
        T = np.array(extr_raw[s], dtype=np.float64)
        if T.shape == (3, 4):
            T = np.vstack([T, [0, 0, 0, 1]])
        extrinsics.append(T)
    extrinsics = np.array(extrinsics)

    # Evaluate DA3 poses (already computed in step3)
    da3_poses = {}
    da3_world_dir = pose_dir
    for f in sorted(da3_world_dir.glob("*.npy")):
        if f.stem in ("gt", ) or "_cam" in f.stem:
            continue
        da3_poses[f.stem] = np.load(str(f)).reshape(4, 4)

    da3_ranked, da3_results = evaluate_views(da3_poses, gt_pose, vertices, "DA3 depth")
    da3_best5 = [s for s, _ in da3_ranked[:args.top_k]]
    print(f"\nBest {args.top_k} DA3 views: {da3_best5}")

    # Run FPose with stereo depth (if available)
    stereo_dir = out_dir / "depth_stereo"
    stereo_ranked, stereo_results = None, None
    stereo_best5 = []
    if stereo_dir.exists():
        cameras = []
        for i, s in enumerate(serials):
            rgb = cv2.cvtColor(cv2.imread(str(img_dir / f"{s}.png")), cv2.COLOR_BGR2RGB)
            cameras.append({"serial": s, "rgb": rgb, "K": intrinsics[i], "T": extrinsics[i]})

        print(f"\nRunning FPose with stereo depth...")
        stereo_poses = run_fpose_stereo(
            cameras, out_dir / "masks", stereo_dir, mesh_path,
            out_dir / "pose_stereo", args.downscale,
        )
        if stereo_poses:
            stereo_ranked, stereo_results = evaluate_views(stereo_poses, gt_pose, vertices, "Stereo depth")
            stereo_best5 = [s for s, _ in stereo_ranked[:args.top_k]]
            print(f"\nBest {args.top_k} stereo views: {stereo_best5}")

    # Save evaluation results
    eval_results = {
        "gt_pose": gt_pose.tolist(),
        "top_k": args.top_k,
        "da3": {
            "per_view": da3_results,
            "best_views": da3_best5,
            "ranking": [s for s, _ in da3_ranked],
        },
    }
    if stereo_results:
        eval_results["stereo"] = {
            "per_view": stereo_results,
            "best_views": stereo_best5,
            "ranking": [s for s, _ in stereo_ranked],
        }

    with open(out_dir / "view_ranking.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nView ranking saved to {out_dir / 'view_ranking.json'}")


if __name__ == "__main__":
    main()