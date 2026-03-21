#!/usr/bin/env python3
"""Step 6: Simulate real pipeline — best 5 views → FPose → best IoU → sil matching → grid overlay.

For each episode:
1. From pre-selected best 5 serials, pick the one with highest mask IoU (using existing FPose poses)
2. Run silhouette optimization from that pose
3. Render overlay grid for all 24 views
4. Save to tmp/simulate/

conda: foundationpose
"""
import argparse
import json
import sys
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
AUTODEX_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(AUTODEX_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

_FP_ROOT = AUTODEX_ROOT / "autodex/perception/thirdparty/_object_6d_tracking/FoundationPose"
sys.path.insert(0, str(_FP_ROOT))

from Utils import make_mesh_tensors, nvdiffrast_render, glcam_in_cvcam, to_homo_torch, projection_matrix_from_intrinsics
import trimesh
import nvdiffrast.torch as dr

MESH_ROOT = Path("/home/mingi/shared_data/object_6d/data/mesh")

# DA3 best 5
BEST5_DA3 = ['25322638', '25322645', '24080331', '25322639', '25322643']
# Stereo best 5
BEST5_STEREO = ['25305461', '25305463', '25322651', '25322639', '24122734']


def find_mesh(obj_name):
    for name in ["simplified.obj", f"{obj_name}.obj", "coacd.obj", "raw.obj"]:
        p = MESH_ROOT / obj_name / name
        if p.exists():
            return str(p)
    objs = list((MESH_ROOT / obj_name).glob("*.obj"))
    if objs:
        return str(objs[0])
    raise FileNotFoundError(f"No mesh for {obj_name}")


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


def compute_iou(mask1, mask2):
    inter = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()
    return float(inter / union) if union > 0 else 0.0


def silhouette_optimize(initial_pose_world, views, mesh_tensors, glctx, iters=200, lr=1e-3):
    """Differentiable silhouette optimization."""
    device = "cuda"
    glcam_t = torch.tensor(glcam_in_cvcam, device=device, dtype=torch.float32)
    pos_homo = to_homo_torch(mesh_tensors["pos"])
    faces = mesh_tensors["faces"]

    pose_init_t = torch.tensor(initial_pose_world, device=device, dtype=torch.float32)

    opt_views = []
    for view in views:
        mask_f = view["mask"].astype(np.float32)
        if mask_f.max() > 1.0:
            mask_f /= 255.0
        mask_t = torch.tensor(mask_f, device=device, dtype=torch.float32)
        ext_t = torch.tensor(view["extrinsic"], device=device, dtype=torch.float32)
        H, W = mask_t.shape
        proj = projection_matrix_from_intrinsics(view["K"], height=H, width=W, znear=0.001, zfar=100)
        proj_t = torch.as_tensor(proj.reshape(4, 4), device=device, dtype=torch.float32)
        opt_views.append({"mask_t": mask_t, "ext_t": ext_t, "proj_t": proj_t, "H": H, "W": W})

    # 6d rotation parameterization (first 2 rows of rotation matrix, flattened)
    r6d = pose_init_t[:3, :3][:2, :].reshape(6).clone().requires_grad_(True)
    t = pose_init_t[:3, 3].clone().requires_grad_(True)
    optimizer = torch.optim.Adam([r6d, t], lr=lr)

    for it in range(iters):
        optimizer.zero_grad()
        # Build rotation from 6d (rows, matching pytorch3d convention)
        a1, a2 = r6d[:3], r6d[3:]
        b1 = F.normalize(a1, dim=0)
        b2 = a2 - (b1 * a2).sum() * b1
        b2 = F.normalize(b2, dim=0)
        b3 = torch.cross(b1, b2)
        R = torch.stack([b1, b2, b3], dim=0)  # rows, not columns

        pose_world = torch.eye(4, device=device, dtype=torch.float32)
        pose_world[:3, :3] = R
        pose_world[:3, 3] = t

        total_loss = 0.0
        for view in opt_views:
            pose_cam = (view["ext_t"] @ pose_world).reshape(1, 4, 4)
            ob_in_glcams = glcam_t[None] @ pose_cam
            pos_clip = (view["proj_t"] @ ob_in_glcams)[:, None] @ pos_homo[None, ..., None]
            pos_clip = pos_clip[..., 0]
            rast_out, _ = dr.rasterize(glctx, pos_clip, faces, resolution=np.asarray([view["H"], view["W"]]))
            alpha = torch.clamp(rast_out[..., -1:], 0, 1)
            alpha = torch.flip(alpha, dims=[1])
            render_mask = alpha[0, :, :, 0]

            sil_mse = F.mse_loss(render_mask, view["mask_t"], reduction="mean")
            intersection = (render_mask * view["mask_t"]).sum()
            union = (render_mask + view["mask_t"]).clamp(0, 1).sum()
            sil_iou = 1 - (intersection / (union + 1e-9))
            total_loss = total_loss + sil_mse + sil_iou

        loss = total_loss / float(len(opt_views))
        loss.backward()
        optimizer.step()

        if it == 0 or (it + 1) % 50 == 0 or it + 1 == iters:
            print(f"      Iter {it+1}/{iters} loss={loss.item():.6f}")

    # Final pose
    with torch.no_grad():
        a1, a2 = r6d[:3], r6d[3:]
        b1 = F.normalize(a1, dim=0)
        b2 = a2 - (b1 * a2).sum() * b1
        b2 = F.normalize(b2, dim=0)
        b3 = torch.cross(b1, b2)
        R = torch.stack([b1, b2, b3], dim=0)  # rows
        pose_final = torch.eye(4, device=device, dtype=torch.float32)
        pose_final[:3, :3] = R
        pose_final[:3, 3] = t
    return pose_final.cpu().numpy()


def render_overlay_grid(pose_world, mesh_tensors, glctx, serials, intrinsics, extrinsics, img_dir, H, W, title=""):
    """Render overlay for all 24 views and return grid image."""
    purple = np.array([128, 0, 128], dtype=np.uint8)
    overlays = []

    for i, s in enumerate(serials):
        K = intrinsics[i].astype(np.float32)
        pose_cam = extrinsics[i] @ pose_world
        pose_t = torch.as_tensor(pose_cam, device="cuda", dtype=torch.float32).reshape(1, 4, 4)

        rc, _, _ = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=pose_t, glctx=glctx, mesh_tensors=mesh_tensors, use_light=False)
        render_mask = rc[0].detach().cpu().numpy().sum(axis=2) > 0

        img_rgb = cv2.cvtColor(cv2.imread(str(img_dir / f"{s}.png")), cv2.COLOR_BGR2RGB)
        overlay = img_rgb.copy()
        overlay[render_mask] = (overlay[render_mask].astype(np.float32) * 0.4 + purple.astype(np.float32) * 0.6).astype(np.uint8)
        cv2.putText(overlay, s, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        overlays.append(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    cols = 4
    rows = (len(overlays) + cols - 1) // cols
    scale = 0.25
    oh, ow = overlays[0].shape[:2]
    th, tw = int(oh * scale), int(ow * scale)
    grid = np.ones((rows * th, cols * tw, 3), dtype=np.uint8) * 40
    for idx, img in enumerate(overlays):
        r, c = divmod(idx, cols)
        small = cv2.resize(img, (tw, th))
        grid[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = small

    if title:
        cv2.putText(grid, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--episode", type=str, default=None)
    parser.add_argument("--depth_type", type=str, default="da3", choices=["da3", "stereo"])
    parser.add_argument("--init_method", type=str, default="best_iou", choices=["best_iou", "nms"])
    parser.add_argument("--sil_iters", type=int, default=200)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    obj_dir = data_root / args.obj
    best5 = BEST5_DA3 if args.depth_type == "da3" else BEST5_STEREO

    if args.episode:
        episodes = [args.episode]
    else:
        episodes = sorted([d.name for d in obj_dir.iterdir() if d.is_dir() and d.name != "cam_param"])

    mesh_path = find_mesh(args.obj)
    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])

    # Visualization mesh (purple)
    purple_color = np.array([128, 0, 128], dtype=np.uint8)
    mesh_vis = mesh.copy()
    vertex_colors = np.tile(np.append(purple_color, 255).reshape(1, 4), (len(mesh_vis.vertices), 1))
    mesh_vis.visual = trimesh.visual.ColorVisuals(vertex_colors=vertex_colors)
    mesh_tensors = make_mesh_tensors(mesh_vis)

    # Plain mesh for silhouette optimization
    mesh_tensors_plain = make_mesh_tensors(mesh)
    glctx = dr.RasterizeCudaContext()

    out_dir = data_root / "simulate"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep in episodes:
        capture_dir = obj_dir / ep
        print(f"\n=== {args.obj}/{ep} ===")

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

        # 1. Select initial pose
        pose_dir = capture_dir / "pose" if args.depth_type == "da3" else capture_dir / "pose_stereo"
        masks_dir = capture_dir / "masks"

        best_serial = None
        best_iou = -1
        best_pose_world = None

        if args.init_method == "best_iou":
            # From best 5 serials, pick highest mean IoU across all 24 views
            for s in best5:
                pose_path = pose_dir / f"{s}.npy"
                if not pose_path.exists():
                    continue

                pose_world_candidate = np.load(str(pose_path))
                view_ious = []
                for i, tgt_s in enumerate(serials):
                    mp = masks_dir / f"{tgt_s}.png"
                    if not mp.exists():
                        continue
                    sam_mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 127
                    K = intrinsics[i].astype(np.float32)
                    pose_cam = extrinsics[i] @ pose_world_candidate
                    pose_t = torch.as_tensor(pose_cam, device="cuda", dtype=torch.float32).reshape(1, 4, 4)
                    rc, _, _ = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=pose_t, glctx=glctx, mesh_tensors=mesh_tensors, use_light=False)
                    sil = rc[0].detach().cpu().numpy().sum(axis=2) > 0
                    view_ious.append(compute_iou(sil, sam_mask))

                mean_iou = np.mean(view_ious) if view_ious else 0.0
                print(f"  {s}: mean IoU={mean_iou:.3f} (24 views)")
                if mean_iou > best_iou:
                    best_iou = mean_iou
                    best_serial = s
                    best_pose_world = pose_world_candidate

        elif args.init_method == "nms":
            # NMS: AABB IoU across all views, pick most overlapping
            all_poses = {}
            for s in serials:
                p = pose_dir / f"{s}.npy"
                if p.exists():
                    all_poses[s] = np.load(str(p))

            if all_poses:
                mesh_for_nms = trimesh.load(mesh_path, process=False)
                if isinstance(mesh_for_nms, trimesh.Scene):
                    mesh_for_nms = trimesh.util.concatenate([g for g in mesh_for_nms.geometry.values() if isinstance(g, trimesh.Trimesh)])
                corners = _bbox_corners(mesh_for_nms.bounds)
                pose_serials = list(all_poses.keys())
                pose_list = [all_poses[s] for s in pose_serials]
                aabbs = [_aabb(_transform(corners, p)) for p in pose_list]
                n = len(aabbs)
                iou_matrix = np.zeros((n, n))
                for ii in range(n):
                    for jj in range(ii, n):
                        v = _iou_3d(aabbs[ii], aabbs[jj])
                        iou_matrix[ii, jj] = iou_matrix[jj, ii] = v
                overlap = np.where(iou_matrix >= 0.5, iou_matrix, 0).sum(axis=1)
                best_idx = int(np.argmax(overlap))
                best_serial = pose_serials[best_idx]
                best_pose_world = pose_list[best_idx]
                best_iou = overlap[best_idx]
                print(f"  NMS selected: {best_serial} (overlap={best_iou:.2f})")

        if best_serial is None or best_pose_world is None:
            print("  No valid pose found, skipping")
            continue

        # Compute pre-sil IoU for all 24 views
        pre_ious = []
        for i, s in enumerate(serials):
            mask_path = masks_dir / f"{s}.png"
            if not mask_path.exists():
                pre_ious.append(0.0)
                continue
            sam_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127
            K = intrinsics[i].astype(np.float32)
            pose_cam = extrinsics[i] @ best_pose_world
            pose_t = torch.as_tensor(pose_cam, device="cuda", dtype=torch.float32).reshape(1, 4, 4)
            rc, _, _ = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=pose_t, glctx=glctx, mesh_tensors=mesh_tensors, use_light=False)
            sil = rc[0].detach().cpu().numpy().sum(axis=2) > 0
            pre_ious.append(compute_iou(sil, sam_mask))
        pre_mean_iou = np.mean(pre_ious)

        print(f"  Selected ({args.init_method}): {best_serial}")
        print(f"  Pre-sil: mean IoU={pre_mean_iou:.3f} (min={min(pre_ious):.3f}, max={max(pre_ious):.3f})")

        # 2. Silhouette optimization from best pose
        print(f"  Running silhouette optimization ({args.sil_iters} iters)...")
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

        from autodex.perception.silhouette import SilhouetteOptimizer
        sil_optimizer = SilhouetteOptimizer(mesh_path)
        sil_views = [{"mask": v["mask"], "K": v["K"], "extrinsic": v["extrinsic"].astype(np.float64)} for v in views]
        optimized_pose = sil_optimizer.optimize(best_pose_world, sil_views, iters=100, lr=0.002, antialias=True)

        # Compute post-optimization IoU for all views
        post_ious = []
        for i, s in enumerate(serials):
            mask_path = masks_dir / f"{s}.png"
            if not mask_path.exists():
                continue
            sam_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127
            K = intrinsics[i].astype(np.float32)
            pose_cam = extrinsics[i] @ optimized_pose
            pose_t = torch.as_tensor(pose_cam, device="cuda", dtype=torch.float32).reshape(1, 4, 4)
            rc, _, _ = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=pose_t, glctx=glctx, mesh_tensors=mesh_tensors, use_light=False)
            sil = rc[0].detach().cpu().numpy().sum(axis=2) > 0
            post_ious.append(compute_iou(sil, sam_mask))

        mean_post_iou = np.mean(post_ious)
        print(f"  Before sil: best view IoU={best_iou:.3f}")
        print(f"  After sil:  mean IoU={mean_post_iou:.3f} (min={min(post_ious):.3f}, max={max(post_ious):.3f})")

        # 3. Render overlay grid
        grid = render_overlay_grid(
            optimized_pose, mesh_tensors, glctx,
            serials, intrinsics, extrinsics, img_dir, H, W,
            title=f"{args.obj}/{ep} src={best_serial} pre={best_iou:.3f} post={mean_post_iou:.3f} ({args.depth_type})"
        )

        out_path = out_dir / f"{args.obj}_{ep}_{args.depth_type}_{args.init_method}.png"
        cv2.imwrite(str(out_path), grid)

        # Save result json
        # Save optimized pose
        pose_path = out_dir / f"{args.obj}_{ep}_{args.depth_type}_{args.init_method}_pose.npy"
        np.save(str(pose_path), optimized_pose)

        result = {
            "obj": args.obj, "episode": ep,
            "depth_type": args.depth_type, "init_method": args.init_method,
            "selected_serial": best_serial,
            "pre_mean_iou": float(pre_mean_iou),
            "pre_min_iou": float(min(pre_ious)),
            "pre_max_iou": float(max(pre_ious)),
            "post_mean_iou": float(mean_post_iou),
            "post_min_iou": float(min(post_ious)),
            "post_max_iou": float(max(post_ious)),
            "post_per_view_iou": {s: float(v) for s, v in zip(serials, post_ious)},
        }
        json_path = out_dir / f"{args.obj}_{ep}_{args.depth_type}_{args.init_method}.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {out_path}")

    del mesh_tensors, mesh_tensors_plain, glctx


if __name__ == "__main__":
    main()