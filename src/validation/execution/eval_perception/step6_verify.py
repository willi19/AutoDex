#!/usr/bin/env python3
"""Step 6: Verify — rerun with best 5 views only, NMS+sil, compare with GT. (conda: foundationpose)

Takes the best 5 views from step5, runs FPose + NMS + silhouette matching,
and compares the resulting pose against the full-24-view GT pose.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

AUTODEX_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(AUTODEX_ROOT))

_FP_ROOT = AUTODEX_ROOT / "autodex/perception/thirdparty/_object_6d_tracking/FoundationPose"
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


def run_fpose_subset(view_serials, capture_dir, out_dir, masks_dir, depth_dir,
                     intrinsics_dict, extrinsics_dict, mesh_path, downscale=0.5):
    """Run FPose on a subset of views. Returns {serial: pose_world}."""
    from autodex.perception import PoseTracker
    tracker = PoseTracker(mesh_path, device_id=0)
    img_dir = capture_dir / "images"
    if not img_dir.exists():
        img_dir = capture_dir / "raw" / "images"
    poses = {}

    for s in view_serials:
        mask_path = masks_dir / f"{s}.png"
        depth_path = depth_dir / f"{s}.png"
        if not mask_path.exists() or not depth_path.exists():
            continue

        rgb = cv2.cvtColor(cv2.imread(str(img_dir / f"{s}.png")), cv2.COLOR_BGR2RGB)
        mask = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        K = intrinsics_dict[s].copy()

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

        T_world_cam = np.linalg.inv(extrinsics_dict[s])
        poses[s] = T_world_cam @ pose_cam

    return poses


def nms_and_silhouette(poses_dict, mesh_path, serials_all, intrinsics, extrinsics,
                       masks_dir, H, W, iou_threshold=0.5):
    """NMS + silhouette matching → refined pose. Same as step4 but on subset poses."""
    import trimesh
    from scipy.spatial.transform import Rotation

    # NMS
    mesh = trimesh.load(str(mesh_path), force="mesh")
    corners = _bbox_corners(mesh.bounds)

    pose_serials = list(poses_dict.keys())
    pose_list = [poses_dict[s] for s in pose_serials]
    aabbs = [_aabb(_transform(corners, p)) for p in pose_list]

    n = len(aabbs)
    if n == 0:
        return None, {}
    if n == 1:
        return pose_list[0], {pose_serials[0]: 1.0}

    iou_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            v = _iou(aabbs[i], aabbs[j])
            iou_matrix[i, j] = iou_matrix[j, i] = v

    overlap = np.where(iou_matrix >= 0.5, iou_matrix, 0).sum(axis=1)
    best_idx = int(np.argmax(overlap))
    nms_pose = pose_list[best_idx]

    # Silhouette rendering
    fp_path = str(_FP_ROOT)
    if fp_path not in sys.path:
        sys.path.insert(0, fp_path)
    import nvdiffrast.torch as dr
    from Utils import make_mesh_tensors, nvdiffrast_render

    mesh_tensors = make_mesh_tensors(mesh)
    glctx = dr.RasterizeCudaContext()

    iou_scores = {}
    for i, s in enumerate(serials_all):
        mask_path = masks_dir / f"{s}.png"
        if not mask_path.exists():
            iou_scores[s] = 0.0
            continue

        sam_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127
        K = intrinsics[i].astype(np.float32)
        pose_cam = extrinsics[i] @ nms_pose
        pose_t = torch.as_tensor(pose_cam, device="cuda", dtype=torch.float32).reshape(1, 4, 4)

        render_color, _, _ = nvdiffrast_render(
            K=K, H=H, W=W, ob_in_cams=pose_t, glctx=glctx,
            mesh_tensors=mesh_tensors, use_light=False,
        )
        sil = render_color[0].detach().cpu().numpy().sum(axis=2) > 0
        inter = (sil & sam_mask).sum()
        union = (sil | sam_mask).sum()
        iou_scores[s] = float(inter / union) if union > 0 else 0.0

    del mesh_tensors, glctx

    # Refine with high-IoU views
    inlier_serials = [s for s, iou in iou_scores.items()
                      if iou >= iou_threshold and s in poses_dict]
    if not inlier_serials:
        inlier_serials = sorted(iou_scores, key=iou_scores.get, reverse=True)[:3]
        inlier_serials = [s for s in inlier_serials if s in poses_dict]

    if not inlier_serials:
        return nms_pose, iou_scores

    inlier_poses = [poses_dict[s] for s in inlier_serials]
    avg_t = np.mean([p[:3, 3] for p in inlier_poses], axis=0)
    quats = np.array([Rotation.from_matrix(p[:3, :3]).as_quat() for p in inlier_poses])
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[0]) < 0:
            quats[i] *= -1
    avg_quat = np.mean(quats, axis=0)
    avg_quat /= np.linalg.norm(avg_quat)

    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(avg_quat).as_matrix()
    result[:3, 3] = avg_t
    return result, iou_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--episode", type=str, required=True)
    parser.add_argument("--downscale", type=float, default=0.5)
    args = parser.parse_args()

    capture_dir = Path(args.data_root) / args.obj / args.episode
    out_dir = capture_dir

    # Load view ranking
    ranking_path = out_dir / "view_ranking.json"
    if not ranking_path.exists():
        print(f"ERROR: no view ranking at {ranking_path}, run step5 first")
        return
    with open(ranking_path) as f:
        ranking = json.load(f)

    gt_pose = np.array(ranking["gt_pose"]).reshape(4, 4)

    # Skip if verify results already exist
    if (out_dir / "verify_results.json").exists():
        print(f"  Verify results already exist, skipping")
        return

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
    intrinsics_dict = {s: intrinsics[i] for i, s in enumerate(serials)}
    extrinsics_dict = {s: extrinsics[i] for i, s in enumerate(serials)}

    img = cv2.imread(str(img_dir / f"{serials[0]}.png"))
    H, W = img.shape[:2]

    mesh_path = find_mesh(args.obj)
    masks_dir = out_dir / "masks"

    # Load mesh for ADD metric
    import trimesh
    mesh = trimesh.load(str(mesh_path), force="mesh")
    vertices = mesh.vertices.astype(np.float32)
    if len(vertices) > 2000:
        vertices = vertices[np.random.RandomState(42).choice(len(vertices), 2000, replace=False)]

    results = {}

    for depth_name in ["da3", "stereo"]:
        if depth_name not in ranking:
            continue
        best_views = ranking[depth_name]["best_views"]
        depth_dir = out_dir / f"depth_{depth_name}"
        if not depth_dir.exists():
            continue

        print(f"\n{'='*60}")
        print(f"Verify: {depth_name} depth, best {len(best_views)} views: {best_views}")

        # Run FPose on best views only
        subset_poses = run_fpose_subset(
            best_views, capture_dir, out_dir, masks_dir, depth_dir,
            intrinsics_dict, extrinsics_dict, mesh_path, args.downscale,
        )
        print(f"  Got {len(subset_poses)} poses from {len(best_views)} views")

        # NMS + silhouette
        refined_pose, iou_scores = nms_and_silhouette(
            subset_poses, mesh_path, serials, intrinsics, extrinsics,
            masks_dir, H, W,
        )

        if refined_pose is None:
            print(f"  FAILED: no valid pose")
            continue

        # Compare with GT
        trans_err = float(np.linalg.norm(refined_pose[:3, 3] - gt_pose[:3, 3]) * 1000)
        R_diff = refined_pose[:3, :3].T @ gt_pose[:3, :3]
        rot_err = float(np.degrees(np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))))

        pts_est = (refined_pose[:3, :3] @ vertices.T + refined_pose[:3, 3:4]).T
        pts_gt = (gt_pose[:3, :3] @ vertices.T + gt_pose[:3, 3:4]).T
        add_err = float(np.mean(np.linalg.norm(pts_est - pts_gt, axis=1)) * 1000)

        mean_iou = np.mean([v for v in iou_scores.values() if v > 0])

        print(f"\n  Results ({depth_name}, {len(best_views)} views):")
        print(f"    Translation error: {trans_err:.1f}mm")
        print(f"    Rotation error:    {rot_err:.2f}deg")
        print(f"    ADD error:         {add_err:.1f}mm")
        print(f"    Mean sil IoU:      {mean_iou:.3f}")

        results[depth_name] = {
            "views_used": best_views,
            "n_poses": len(subset_poses),
            "trans_err_mm": trans_err,
            "rot_err_deg": rot_err,
            "add_err_mm": add_err,
            "mean_sil_iou": mean_iou,
            "refined_pose": refined_pose.tolist(),
        }

    # Save verification results
    with open(out_dir / "verify_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nVerification saved to {out_dir / 'verify_results.json'}")


# ── Geometry helpers ──

def _bbox_corners(bounds):
    mins, maxs = bounds
    return np.array([
        [mins[0], mins[1], mins[2]], [mins[0], mins[1], maxs[2]],
        [mins[0], maxs[1], mins[2]], [mins[0], maxs[1], maxs[2]],
        [maxs[0], mins[1], mins[2]], [maxs[0], mins[1], maxs[2]],
        [maxs[0], maxs[1], mins[2]], [maxs[0], maxs[1], maxs[2]],
    ], dtype=np.float32)

def _transform(points, pose):
    homo = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return (pose @ homo.T).T[:, :3]

def _aabb(points):
    return points.min(axis=0), points.max(axis=0)

def _iou(a, b):
    a_min, a_max = a
    b_min, b_max = b
    inter = np.maximum(np.minimum(a_max, b_max) - np.maximum(a_min, b_min), 0)
    inter_vol = inter[0] * inter[1] * inter[2]
    union = np.prod(a_max - a_min) + np.prod(b_max - b_min) - inter_vol
    return float(inter_vol / union) if union > 0 else 0.0


if __name__ == "__main__":
    main()