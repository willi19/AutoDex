#!/usr/bin/env python3
"""Step 4: NMS + silhouette optimization → GT pose. (conda: foundationpose)

1. NMS on all per-view world poses → initial pose
2. Differentiable silhouette rendering optimization (200 iters)
3. Save GT pose + grid visualization
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
AUTODEX_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(AUTODEX_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

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


def load_world_poses(pose_dir):
    poses = {}
    for f in sorted(pose_dir.glob("*.npy")):
        if f.stem == "gt":
            continue
        poses[f.stem] = np.load(str(f))
    return poses


def nms_select(poses_dict, mesh_path):
    import trimesh
    mesh = trimesh.load(str(mesh_path), force="mesh")
    corners = _bbox_corners(mesh.bounds)

    serials = list(poses_dict.keys())
    poses = [poses_dict[s] for s in serials]
    aabbs = [_aabb(_transform(corners, p)) for p in poses]

    n = len(aabbs)
    iou_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            v = _iou_3d(aabbs[i], aabbs[j])
            iou_matrix[i, j] = iou_matrix[j, i] = v

    overlap = np.where(iou_matrix >= 0.5, iou_matrix, 0).sum(axis=1)
    best_idx = int(np.argmax(overlap))

    print(f"  NMS: selected {serials[best_idx]} (overlap score {overlap[best_idx]:.2f})")
    return poses[best_idx], serials[best_idx]


def compute_iou(mask1, mask2):
    inter = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()
    return float(inter / union) if union > 0 else 0.0


from autodex.perception.silhouette import SilhouetteOptimizer  # noqa: E402


# ── Rendering for visualization ──

def render_silhouettes(pose_world, mesh_path, serials, intrinsics, extrinsics, H, W):
    import trimesh
    import nvdiffrast.torch as dr
    from render_utils import make_mesh_tensors, nvdiffrast_render

    mesh = trimesh.load(str(mesh_path), force="mesh")
    mesh_tensors = make_mesh_tensors(mesh)
    glctx = dr.RasterizeCudaContext()

    silhouettes = {}
    for i, s in enumerate(serials):
        K = intrinsics[i].astype(np.float32)
        pose_cam = extrinsics[i] @ pose_world
        pose_t = torch.as_tensor(pose_cam, device="cuda", dtype=torch.float32).reshape(1, 4, 4)
        color, _, _ = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=pose_t, glctx=glctx, mesh_tensors=mesh_tensors)
        sil = color[0].detach().cpu().numpy().sum(axis=2) > 0
        silhouettes[s] = sil

    del mesh_tensors, glctx
    return silhouettes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--episode", type=str, required=True)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    capture_dir = Path(args.data_root) / args.obj / args.episode
    pose_dir = capture_dir / "pose"
    masks_dir = capture_dir / "masks"

    # Skip if GT already exists
    if (pose_dir / "gt.npy").exists():
        print(f"  GT pose already exists, skipping")
        return

    # Load camera data
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

    img = cv2.imread(str(img_dir / f"{serials[0]}.png"))
    H, W = img.shape[:2]

    mesh_path = find_mesh(args.obj)
    print(f"Mesh: {mesh_path}")

    # Load per-view poses
    poses = load_world_poses(pose_dir)
    print(f"Loaded {len(poses)} per-view poses")

    if len(poses) == 0:
        print("ERROR: no poses found")
        return

    # NMS → initial pose
    nms_pose, nms_serial = nms_select(poses, mesh_path)
    print(f"NMS initial pose t = {nms_pose[:3, 3]}")

    # Prepare views for silhouette optimization
    views = []
    for i, s in enumerate(serials):
        mask_path = masks_dir / f"{s}.png"
        if not mask_path.exists():
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.sum() < 100:
            continue
        views.append({
            "mask": mask,
            "K": intrinsics[i].astype(np.float32),
            "extrinsic": extrinsics[i].astype(np.float32),
        })

    print(f"Optimizing with {len(views)} views, {args.iters} iters...")
    optimizer = SilhouetteOptimizer(mesh_path)
    gt_pose = optimizer.optimize(nms_pose, views, iters=args.iters, lr=args.lr)

    # Save GT
    np.save(str(pose_dir / "gt.npy"), gt_pose)
    print(f"GT pose saved, t = {gt_pose[:3, 3]}")

    # Render silhouettes + compute IoU
    print("Rendering GT silhouettes...")
    silhouettes = render_silhouettes(gt_pose, mesh_path, serials, intrinsics, extrinsics, H, W)

    iou_scores = {}
    for s in serials:
        mask_path = masks_dir / f"{s}.png"
        if not mask_path.exists():
            iou_scores[s] = 0.0
            continue
        sam_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127
        iou_scores[s] = compute_iou(silhouettes.get(s, np.zeros_like(sam_mask)), sam_mask)

    ranked = sorted(iou_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\nIoU scores (top 5):")
    for s, iou in ranked[:5]:
        print(f"  {s}: {iou:.3f}")
    print(f"Mean IoU: {np.mean(list(iou_scores.values())):.3f}")

    # Save per-view overlays + grid
    vis_dir = pose_dir / "silhouette_vis"
    vis_dir.mkdir(exist_ok=True)
    overlay_images = []
    for s in serials:
        img_bgr = cv2.imread(str(img_dir / f"{s}.png"))
        if img_bgr is None or s not in silhouettes:
            continue
        sil = silhouettes[s]
        overlay = img_bgr.copy()
        overlay[sil] = (overlay[sil].astype(np.float32) * 0.5 +
                        np.array([0, 200, 0], dtype=np.float32) * 0.5).astype(np.uint8)

        mask_path = masks_dir / f"{s}.png"
        if mask_path.exists():
            sam_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127
            contours, _ = cv2.findContours(sam_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 100, 0), 2)

        iou = iou_scores.get(s, 0)
        color = (0, 255, 0) if iou >= 0.5 else (0, 0, 255)
        cv2.putText(overlay, f"IoU={iou:.3f}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        cv2.putText(overlay, s, (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(str(vis_dir / f"{s}.png"), overlay)
        overlay_images.append(overlay)

    if overlay_images:
        cols = 4
        rows = (len(overlay_images) + cols - 1) // cols
        scale = 0.25
        oh, ow = overlay_images[0].shape[:2]
        th, tw = int(oh * scale), int(ow * scale)
        grid = np.ones((rows * th, cols * tw, 3), dtype=np.uint8) * 40
        for idx, img in enumerate(overlay_images):
            r, c = divmod(idx, cols)
            small = cv2.resize(img, (tw, th))
            grid[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = small
        cv2.imwrite(str(vis_dir / "grid.png"), grid)
        print(f"Grid saved to {vis_dir / 'grid.png'}")

    print(f"Overlays saved to {vis_dir}")


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

def _iou_3d(a, b):
    a_min, a_max = a
    b_min, b_max = b
    inter = np.maximum(np.minimum(a_max, b_max) - np.maximum(a_min, b_min), 0)
    inter_vol = inter[0] * inter[1] * inter[2]
    union = np.prod(a_max - a_min) + np.prod(b_max - b_min) - inter_vol
    return float(inter_vol / union) if union > 0 else 0.0


if __name__ == "__main__":
    main()
