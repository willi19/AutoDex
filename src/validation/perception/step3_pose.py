#!/usr/bin/env python3
"""
Step 3: FoundationPose + NMS + mesh overlay validation

Reads masks from step1, depth from step2.
For each object in object_info.json, runs FoundationPose on all cameras,
applies NMS to select the best pose, and renders mesh overlays.

Output structure:
    output_dir/
    └── objects/{name}/
        ├── ob_in_cam/           # 4x4 pose per camera (txt)
        ├── ob_in_world/         # 4x4 pose in world frame (txt)
        ├── selected_pose_world.txt
        └── visualizations/
            ├── {serial}_overlay.png
            └── grid.png

Usage:
    conda activate foundationpose
    python src/validation/perception/step3_pose.py \
        --output_dir ~/shared_data/.../20260214_231802/6d_output \
        --mesh_dir ~/robothome/mesh
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch

AUTODEX_ROOT = Path(__file__).resolve().parents[3]
FOUNDATION_POSE_PATH = AUTODEX_ROOT / "autodex/perception/thirdparty/_object_6d_tracking/FoundationPose"
sys.path.insert(0, str(FOUNDATION_POSE_PATH))
sys.path.insert(0, str(AUTODEX_ROOT))

from autodex.perception import PoseTracker

logging.basicConfig(level=logging.INFO, format="[pose] [%(levelname)s] %(message)s")


def load_camera_data(output_dir: Path):
    cam = np.load(str(output_dir / "camera_data.npz"), allow_pickle=True)
    serials = list(cam["serials"])
    intrinsics = {s: cam["intrinsics"][i] for i, s in enumerate(serials)}
    extrinsics = {s: cam["extrinsics"][i] for i, s in enumerate(serials)}
    return serials, intrinsics, extrinsics


def run_foundationpose(output_dir, seg_dir, depth_dir_root, obj_name, mesh_path,
                       serials, intrinsics, extrinsics,
                       device_id, downscale, est_refine_iter):
    tracker = PoseTracker(str(mesh_path), device_id=device_id)

    obj_dir = output_dir / "objects" / obj_name
    cam_dir = obj_dir / "ob_in_cam"
    world_dir = obj_dir / "ob_in_world"
    cam_dir.mkdir(exist_ok=True)
    world_dir.mkdir(exist_ok=True)

    images_dir = seg_dir / "images"
    depth_dir = depth_dir_root / "depth"
    masks_dir = seg_dir / "objects" / obj_name / "masks"

    start = time.perf_counter()
    for serial in serials:
        t0 = time.perf_counter()

        image_bgr = cv2.imread(str(images_dir / f"{serial}.png"))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(masks_dir / f"{serial}.png"), cv2.IMREAD_UNCHANGED)
        if mask is None or mask.sum() == 0:
            logging.warning(f"  {serial}: no mask, skipping")
            continue
        if mask.ndim == 3:
            mask = mask[..., 0]

        depth_raw = cv2.imread(str(depth_dir / f"{serial}.png"), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            logging.warning(f"  {serial}: no depth, skipping")
            continue
        depth = depth_raw.astype(np.float32) / 1000.0
        depth[(depth < 0.001) | (depth > 100.0)] = 0

        K = intrinsics[serial].copy()

        if downscale != 1.0:
            H, W = image_rgb.shape[:2]
            nH, nW = int(H * downscale), int(W * downscale)
            image_rgb = cv2.resize(image_rgb, (nW, nH))
            depth = cv2.resize(depth, (nW, nH), interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, (nW, nH), interpolation=cv2.INTER_NEAREST)
            K[0] *= downscale
            K[1] *= downscale

        tracker.reset()
        pose_cam = tracker.init(image_rgb, depth, mask, K, iteration=est_refine_iter)
        pose_world = np.linalg.inv(extrinsics[serial]) @ pose_cam

        np.savetxt(str(cam_dir / f"{serial}.txt"), pose_cam.reshape(4, 4))
        np.savetxt(str(world_dir / f"{serial}.txt"), pose_world.reshape(4, 4))
        logging.info(f"  {serial}: {time.perf_counter() - t0:.3f}s")

    logging.info(f"  FoundationPose done: {time.perf_counter() - start:.2f}s")


def run_nms(output_dir, obj_name, mesh_path, iou_threshold):
    import trimesh

    obj_dir = output_dir / "objects" / obj_name
    world_dir = obj_dir / "ob_in_world"
    pose_files = sorted(world_dir.glob("*.txt"))
    if not pose_files:
        logging.warning(f"  No pose files for {obj_name}")
        return None

    mesh = _load_mesh(mesh_path)
    corners = _bbox_corners(mesh.bounds)

    cam_ids = [p.stem for p in pose_files]
    poses = [np.loadtxt(str(p)).reshape(4, 4) for p in pose_files]
    aabbs = [_aabb(_transform(corners, p)) for p in poses]

    n = len(aabbs)
    iou_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            v = _iou(aabbs[i], aabbs[j])
            iou_matrix[i, j] = iou_matrix[j, i] = v

    overlap = np.where(iou_matrix >= iou_threshold, iou_matrix, 0).sum(axis=1)
    best_idx = int(np.argmax(overlap))
    best_pose = poses[best_idx]

    save_path = obj_dir / "selected_pose_world.txt"
    np.savetxt(str(save_path), best_pose.reshape(4, 4))
    logging.info(f"  NMS: selected {cam_ids[best_idx]} → {save_path}")
    return best_pose


def save_mesh_overlay(output_dir, seg_dir, obj_name, mesh_path, pose_world,
                      serials, intrinsics, extrinsics, device):
    import trimesh
    import nvdiffrast.torch as dr
    from Utils import make_mesh_tensors, nvdiffrast_render

    obj_dir = output_dir / "objects" / obj_name
    vis_dir = obj_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    images_dir = seg_dir / "images"

    mesh = _load_mesh(mesh_path)
    purple = np.array([128, 0, 128], dtype=np.uint8)
    vertex_colors = np.tile(np.append(purple, 255).reshape(1, 4), (len(mesh.vertices), 1))
    mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=vertex_colors)
    mesh_tensors = make_mesh_tensors(mesh, device=device)
    glctx = dr.RasterizeCudaContext()

    overlay_images = []
    for serial in serials:
        image_bgr = cv2.imread(str(images_dir / f"{serial}.png"))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        H, W = image_rgb.shape[:2]
        K = intrinsics[serial]

        pose_cam = extrinsics[serial] @ pose_world
        pose_t = torch.as_tensor(pose_cam, device=device, dtype=torch.float32).reshape(1, 4, 4)

        render_color, _, _ = nvdiffrast_render(
            K=K, H=H, W=W, ob_in_cams=pose_t, glctx=glctx,
            mesh_tensors=mesh_tensors, use_light=False,
        )
        render_mask = render_color[0].detach().cpu().numpy().sum(axis=2) > 0

        overlay = image_rgb.copy()
        overlay[render_mask] = (
            overlay[render_mask].astype(np.float32) * 0.4
            + purple.astype(np.float32) * 0.6
        ).astype(np.uint8)
        overlay_images.append(overlay)

        cv2.imwrite(str(vis_dir / f"{serial}_overlay.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    if overlay_images:
        cols = min(4, len(overlay_images))
        rows = (len(overlay_images) + cols - 1) // cols
        h, w = overlay_images[0].shape[:2]
        grid = np.ones((rows * h, cols * w, 3), dtype=np.uint8) * 255
        for idx, img in enumerate(overlay_images):
            r, c = divmod(idx, cols)
            grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
        imageio.imwrite(str(vis_dir / "grid.png"), grid)

    logging.info(f"  Overlays saved to {vis_dir}")


# ------------------------------------------------------------------ #
# Geometry helpers
# ------------------------------------------------------------------ #

def _load_mesh(mesh_path):
    import trimesh
    loaded = trimesh.load(str(mesh_path), process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        return trimesh.util.concatenate(meshes)
    return loaded


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


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = Path(args.mesh_dir)

    # seg_dir: where masks, images, camera_data, object_info live
    seg_dir = Path(args.seg_dir) if args.seg_dir else output_dir
    # depth_dir: where depth/ subdir lives
    depth_dir = Path(args.depth_dir) if args.depth_dir else output_dir

    with open(seg_dir / "object_info.json") as f:
        object_info = json.load(f)

    serials, intrinsics, extrinsics = load_camera_data(seg_dir)
    logging.info(f"Loaded {len(serials)} cameras, {len(object_info)} objects")

    t0_total = time.perf_counter()
    per_object_time = {}

    for obj_name in object_info:
        mesh_path = mesh_dir / obj_name / f"{obj_name}.obj"
        if not mesh_path.exists():
            mesh_path = mesh_dir / obj_name / "processed_data/mesh/simplified.obj"
        if not mesh_path.exists():
            mesh_path = next(mesh_dir.glob(f"{obj_name}/**/*.obj"), None) or \
                        next(mesh_dir.glob(f"{obj_name}/**/*.ply"), None)
        if not mesh_path or not mesh_path.exists():
            logging.warning(f"Mesh not found for {obj_name}, skipping")
            continue

        logging.info(f"--- {obj_name} ---")
        t0_obj = time.perf_counter()

        (output_dir / "objects" / obj_name).mkdir(parents=True, exist_ok=True)

        run_foundationpose(
            output_dir, seg_dir, depth_dir, obj_name, mesh_path,
            serials, intrinsics, extrinsics,
            device_id=args.device_id, downscale=args.downscale,
            est_refine_iter=args.est_refine_iter,
        )

        pose_world = run_nms(output_dir, obj_name, mesh_path, args.nms_iou_threshold)
        if pose_world is None:
            continue

        save_mesh_overlay(
            output_dir, seg_dir, obj_name, mesh_path, pose_world,
            serials, intrinsics, extrinsics, device=f"cuda:{args.device_id}",
        )
        per_object_time[obj_name] = round(time.perf_counter() - t0_obj, 2)

    elapsed = time.perf_counter() - t0_total
    timing_path = output_dir / "timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    timing["step3_pose"] = {"total_s": round(elapsed, 2),
                             "per_object_s": per_object_time,
                             "n_cameras": len(serials), "n_objects": len(object_info)}
    timing_path.write_text(json.dumps(timing, indent=2))

    # Save source dirs so step4_compare can find masks/depth
    sources = {"seg_dir": str(seg_dir), "depth_dir": str(depth_dir)}
    (output_dir / "sources.json").write_text(json.dumps(sources, indent=2))

    logging.info(f"Step 3 done. ({elapsed:.1f}s) → {timing_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to write pose results (e.g. validation_output/pose/sam3_da3)")
    parser.add_argument("--seg_dir", type=str, default=None,
                        help="Segmentation dir with masks, images, object_info (default: output_dir)")
    parser.add_argument("--depth_dir", type=str, default=None,
                        help="Depth dir with depth/ subdir (default: output_dir)")
    parser.add_argument("--mesh_dir", type=str, required=True,
                        help="Root dir containing {obj_name}/processed_data/mesh/simplified.obj")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--downscale", type=float, default=0.5)
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--nms_iou_threshold", type=float, default=0.5)
    main(parser.parse_args())