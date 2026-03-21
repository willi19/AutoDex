#!/usr/bin/env python3
"""Step 3: FPose register all 24 views with DA3 depth. (conda: foundationpose)

Saves per-view world poses as {serial}.npy (4x4).
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

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
        MESH_ROOT / obj_name / "raw.obj",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    objs = list((MESH_ROOT / obj_name).glob("*.obj"))
    if objs:
        return str(objs[0])
    raise FileNotFoundError(f"No mesh for {obj_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--episode", type=str, required=True)
    parser.add_argument("--downscale", type=float, default=0.5)
    args = parser.parse_args()

    capture_dir = Path(args.data_root) / args.obj / args.episode
    masks_dir = capture_dir / "masks"
    depth_dir = capture_dir / "depth_da3"
    pose_dir = capture_dir / "pose"
    pose_dir.mkdir(parents=True, exist_ok=True)

    # Load camera data
    import json
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

    mesh_path = find_mesh(args.obj)
    print(f"Mesh: {mesh_path}")
    print(f"Cameras: {len(serials)}, downscale: {args.downscale}")

    # Skip if all poses already exist
    existing = [s for s in serials if (pose_dir / f"{s}.npy").exists()]
    if len(existing) == len(serials):
        print(f"  All {len(serials)} poses already exist, skipping")
        return

    from autodex.perception import PoseTracker
    tracker = PoseTracker(mesh_path, device_id=0)

    n_success = 0
    for i, s in enumerate(serials):
        if (pose_dir / f"{s}.npy").exists():
            n_success += 1
            print(f"  {s}: exists, skip")
            continue

        mask_path = masks_dir / f"{s}.png"
        depth_path = depth_dir / f"{s}.png"

        if not mask_path.exists() or not depth_path.exists():
            print(f"  {s}: skip (missing mask or depth)")
            continue

        rgb = cv2.cvtColor(cv2.imread(str(img_dir / f"{s}.png")), cv2.COLOR_BGR2RGB)
        mask = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        depth[(depth < 0.001) | (depth >= 100)] = 0  # zero out invalid depth
        K = intrinsics[i].copy()

        ds = args.downscale
        if ds != 1.0:
            h, w = rgb.shape[:2]
            nw, nh = int(w * ds), int(h * ds)
            rgb = cv2.resize(rgb, (nw, nh))
            depth = cv2.resize(depth, (nw, nh), interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
            K[0, :] *= ds
            K[1, :] *= ds

        if mask.sum() < 100:
            print(f"  {s}: skip (mask too small)")
            continue
        if (depth[mask > 0] > 0.001).sum() < 50:
            print(f"  {s}: skip (no depth in mask)")
            continue

        tracker.reset()
        t0 = time.perf_counter()
        try:
            pose_cam = tracker.init(rgb, depth, mask, K, iteration=5)
        except Exception as e:
            print(f"  {s}: FAILED ({e})")
            continue
        infer_time = time.perf_counter() - t0

        T_world_cam = np.linalg.inv(extrinsics[i])
        pose_world = T_world_cam @ pose_cam
        np.save(str(pose_dir / f"{s}.npy"), pose_world)
        # Also save pose_cam directly for debug overlay
        np.save(str(pose_dir / f"{s}_cam.npy"), pose_cam)

        n_success += 1
        print(f"  {s}: ok [{infer_time:.3f}s]")

    print(f"\nPose: {n_success}/{len(serials)} views")

    # Render per-view overlays: for each source view's pose_world,
    # overlay onto ALL 24 views (cross-view verification)
    print("Rendering FPose cross-view overlays...")
    import torch
    import trimesh
    import nvdiffrast.torch as dr
    from render_utils import make_mesh_tensors, nvdiffrast_render

    mesh_vis = trimesh.load(mesh_path, process=False)
    if isinstance(mesh_vis, trimesh.Scene):
        mesh_vis = trimesh.util.concatenate([g for g in mesh_vis.geometry.values() if isinstance(g, trimesh.Trimesh)])
    purple = np.array([128, 0, 128], dtype=np.uint8)
    vertex_colors = np.tile(np.append(purple, 255).reshape(1, 4), (len(mesh_vis.vertices), 1))
    mesh_vis.visual = trimesh.visual.ColorVisuals(vertex_colors=vertex_colors)
    mesh_tensors = make_mesh_tensors(mesh_vis)
    glctx = dr.RasterizeCudaContext()

    # Load all images once
    all_images = {}
    for s in serials:
        img_bgr = cv2.imread(str(img_dir / f"{s}.png"))
        all_images[s] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = all_images[serials[0]].shape[:2]

    vis_dir = pose_dir / "fpose_vis"
    vis_dir.mkdir(exist_ok=True)

    # For each source view that has a pose
    source_serials = [s for s in serials if (pose_dir / f"{s}.npy").exists()]
    for src_s in source_serials:
        pose_world = np.load(str(pose_dir / f"{src_s}.npy"))

        # Overlay this pose onto all 24 views
        overlay_images = []
        for i, tgt_s in enumerate(serials):
            pose_cam = extrinsics[i] @ pose_world
            K = intrinsics[i].astype(np.float32)
            img_rgb = all_images[tgt_s].copy()

            pose_t = torch.as_tensor(pose_cam, device="cuda", dtype=torch.float32).reshape(1, 4, 4)
            render_color, _, _ = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=pose_t, glctx=glctx, mesh_tensors=mesh_tensors, use_light=False)
            render_mask = render_color[0].detach().cpu().numpy().sum(axis=2) > 0

            overlay = img_rgb.copy()
            overlay[render_mask] = (overlay[render_mask].astype(np.float32) * 0.4 + purple.astype(np.float32) * 0.6).astype(np.uint8)
            label = f"{tgt_s}" + (" (src)" if tgt_s == src_s else "")
            cv2.putText(overlay, label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            overlay_images.append(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        # Save grid for this source view
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
        cv2.imwrite(str(vis_dir / f"crossview_{src_s}.png"), grid)

    print(f"Cross-view grids saved to {vis_dir / 'crossview_*.png'} ({len(source_serials)} source views)")

    del mesh_tensors, glctx


if __name__ == "__main__":
    main()