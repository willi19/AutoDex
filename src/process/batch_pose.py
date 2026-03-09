#!/usr/bin/env python3
"""Batch 6D pose estimation using FoundationPose (tracking).

Reads RGB video, depth video, and mask from local cache.
Mask can be either:
  - Full video:    obj_mask/{serial}.avi   (from batch_mask.py)
  - First frame:   obj_mask_first/{serial}.png  (from batch_mask_first.py)
Only the first frame's mask is used (for init). Tracking needs no mask.

Pre-requisites:
    - Videos downloaded: python src/perception/download_videos.py ...
    - First-frame mask:  python src/perception/batch_mask_first.py ...
      (OR full mask:     python src/perception/batch_mask.py ...)
    - Depth generated:   python src/perception/batch_depth.py ...

Run:
    conda activate foundationpose
    python -u src/perception/batch_pose.py \
        --base /home/mingi/paradex1/capture/eccv2026/inspire_f1 \
        --serials 22684755 23263780 \
        --mesh_dir /home/mingi/mesh

Upload results:
    python src/perception/upload_results.py --base ...
"""

import os
import sys
import gc
import time
import json
import logging
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

CACHE_ROOT = os.path.expanduser("~/video_cache")
NETWORK_PREFIX = "/home/mingi/paradex1/capture"

_FP_ROOT = Path(__file__).resolve().parents[2] / "autodex/perception/thirdparty/_object_6d_tracking/FoundationPose"
MESH_COLOR = np.array([128, 0, 128], dtype=np.uint8)
OVERLAY_ALPHA = 0.6


def _get_cache_base(base_dir):
    base = str(Path(base_dir).resolve())
    if base.startswith(NETWORK_PREFIX):
        rel = base[len(NETWORK_PREFIX):].lstrip("/")
    else:
        rel = Path(base_dir).name
    return os.path.join(CACHE_ROOT, rel)


# ── Camera params ────────────────────────────────────────────────────────────

def load_cam_param(capture_dir: Path):
    """Load intrinsics (undistorted) and extrinsics keyed by serial string."""
    param_dir = capture_dir / "cam_param"
    with open(param_dir / "intrinsics.json") as f:
        intr_raw = json.load(f)
    with open(param_dir / "extrinsics.json") as f:
        extr_raw = json.load(f)

    intrinsics = {s: np.array(v["intrinsics_undistort"], dtype=np.float64) for s, v in intr_raw.items()}
    extrinsics = {s: np.array(extr_raw[s], dtype=np.float64) for s in intr_raw}
    return intrinsics, extrinsics


# ── Depth decode ─────────────────────────────────────────────────────────────

def decode_depth_uint16(bgr: np.ndarray) -> np.ndarray:
    """Decode BGR frame (FFV1) back to depth in meters."""
    depth_mm = bgr[:, :, 1].astype(np.uint16) * 256 + bgr[:, :, 0].astype(np.uint16)
    return depth_mm.astype(np.float32) / 1000.0


# ── Mesh lookup ──────────────────────────────────────────────────────────────

def find_mesh(mesh_dir: Path, obj_name: str):
    """Find mesh file for an object. Tries common patterns."""
    candidates = [
        mesh_dir / obj_name / f"{obj_name}.obj",
        mesh_dir / obj_name / f"{obj_name}_remeshed.obj",
        mesh_dir / obj_name / "processed_data/mesh/simplified.obj",
        mesh_dir / f"{obj_name}.obj",
        mesh_dir / f"{obj_name}.ply",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Glob fallback
    found = list(mesh_dir.glob(f"{obj_name}/**/*.obj")) + list(mesh_dir.glob(f"{obj_name}/**/*.ply"))
    return found[0] if found else None


# ── Task collection ──────────────────────────────────────────────────────────

def _find_mask(idx_dir, serial):
    """Find mask: single image (obj_mask_first) or full video (obj_mask).

    Returns (mask_path, is_video) or (None, None).
    """
    # Prefer single-image mask (lighter, faster)
    first_mask = idx_dir / "obj_mask_first" / f"{serial}.png"
    if first_mask.exists():
        return str(first_mask), False
    # Fall back to full mask video
    mask_video = idx_dir / "obj_mask" / f"{serial}.avi"
    if mask_video.exists():
        return str(mask_video), True
    return None, None


def collect_tasks(base_dir, serials, mesh_dir):
    """Collect tasks — one per episode, picks first serial with mask+depth."""
    cache_base = Path(_get_cache_base(base_dir))
    if not cache_base.is_dir():
        return []
    tasks = []
    for obj_dir in sorted(cache_base.iterdir()):
        if not obj_dir.is_dir():
            continue
        obj_name = obj_dir.name
        mesh_path = find_mesh(Path(mesh_dir), obj_name)
        if mesh_path is None:
            continue
        for idx_dir in sorted(obj_dir.iterdir()):
            if not idx_dir.is_dir():
                continue
            # Find first serial with video + depth + mask
            found = False
            for serial in serials:
                video_path = idx_dir / "videos" / f"{serial}.avi"
                depth_path = idx_dir / "depth" / f"{serial}.avi"
                if not (video_path.exists() and depth_path.exists()):
                    continue
                mask_path, mask_is_video = _find_mask(idx_dir, serial)
                if mask_path is None:
                    continue
                # Need cam_param from network FS
                rel = str(idx_dir.relative_to(cache_base))
                net_dir = Path(base_dir) / rel
                if not (net_dir / "cam_param").is_dir():
                    continue
                tasks.append((str(video_path), mask_path, mask_is_video,
                              str(depth_path), str(idx_dir), str(net_dir),
                              serial, obj_name, str(mesh_path), idx_dir.name))
                found = True
                break
            # Also skip if old per-serial pose exists (backward compat)
            if not found:
                continue
    return tasks


# ── Process one video ────────────────────────────────────────────────────────

def _load_init_mask(mask_path, mask_is_video):
    """Load the init mask. Returns (H, W) uint8 array (0/1)."""
    if not mask_is_video:
        # Single PNG image
        img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return (img > 127).astype(np.uint8)
    else:
        # Read first valid frame from mask video
        cap = cv2.VideoCapture(mask_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            mask = (gray > 127).astype(np.uint8)
            if mask.sum() >= 100:
                cap.release()
                return mask
        cap.release()
        return None


def _to_4x4(T):
    if T.shape == (3, 4):
        T4 = np.eye(4, dtype=np.float64)
        T4[:3, :] = T
        return T4
    return T.astype(np.float64)


def _save_debug_images(rgb, depth, mask, out_dir):
    """Save debug visualizations of pose init inputs: depth.png, seg_grid.png."""
    # Depth: colorize with turbo colormap
    valid = depth > 0
    if valid.any():
        d_min, d_max = depth[valid].min(), depth[valid].max()
        depth_norm = np.zeros_like(depth, dtype=np.uint8)
        depth_norm[valid] = np.clip(
            (depth[valid] - d_min) / (d_max - d_min + 1e-6) * 255, 0, 255
        ).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
        depth_color[~valid] = 0
    else:
        depth_color = np.zeros((*depth.shape, 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(out_dir, "depth.png"), depth_color)

    # Seg mask: overlay mask contour on RGB
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = rgb_bgr.copy()
    mask_u8 = (mask * 255).astype(np.uint8)
    # Tint masked region green
    overlay[mask > 0, 1] = np.clip(
        overlay[mask > 0, 1].astype(np.int16) + 80, 0, 255
    ).astype(np.uint8)
    # Draw contour
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(out_dir, "seg_grid.png"), overlay)

    depth_path = os.path.join(out_dir, "depth.png")
    seg_path = os.path.join(out_dir, "seg_grid.png")
    print(f"  Debug saved: {depth_path} (range {depth[valid].min():.3f}-{depth[valid].max():.3f}m), {seg_path}"
          if valid.any() else f"  Debug saved: {depth_path} (EMPTY), {seg_path}", flush=True)


def _save_pose_overlay(rgb, pose_4x4, K, mesh_tensors, glctx, device, out_dir):
    """Render mesh overlay on RGB using estimated pose and save as overlay.png."""
    from Utils import nvdiffrast_render

    H, W = rgb.shape[:2]
    pose_t = torch.as_tensor(pose_4x4, device=device, dtype=torch.float32).reshape(1, 4, 4)

    render_color, _, _ = nvdiffrast_render(
        K=K, H=H, W=W, ob_in_cams=pose_t, glctx=glctx,
        mesh_tensors=mesh_tensors, use_light=False,
    )
    render_mask = render_color[0].detach().cpu().numpy().sum(axis=2) > 0

    overlay = rgb.copy()
    overlay[render_mask] = (
        overlay[render_mask].astype(np.float32) * (1.0 - OVERLAY_ALPHA)
        + MESH_COLOR.astype(np.float32) * OVERLAY_ALPHA
    ).astype(np.uint8)
    overlay_path = os.path.join(out_dir, "overlay.png")
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"  Debug saved: {overlay_path}", flush=True)


def process_one_video(tracker, video_path, mask_path, mask_is_video, depth_path,
                      cache_dir, net_dir, serial, obj_name, downscale, est_refine_iter,
                      mesh_tensors=None, glctx=None, device="cuda:0"):
    """Run FoundationPose tracking on one video. Saves pose/pose_world.npy (N,4,4)."""
    intrinsics, extrinsics = load_cam_param(Path(net_dir))
    K = intrinsics[serial].copy()
    T_cam = _to_4x4(extrinsics[serial])
    T_cam_inv = np.linalg.inv(T_cam)

    # Load init mask once
    init_mask = _load_init_mask(mask_path, mask_is_video)
    if init_mask is None:
        print("  No valid mask found, skipping", flush=True)
        return

    cap_rgb = cv2.VideoCapture(video_path)
    cap_depth = cv2.VideoCapture(depth_path)

    # Use RGB frame count only (FFV1 depth may report wrong count)
    n_frames = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  {W}x{H}, {n_frames} frames (mask: {'video' if mask_is_video else 'image'})", flush=True)

    # Downscale intrinsics if needed
    if downscale != 1.0:
        nW, nH = int(W * downscale), int(H * downscale)
        K[0] *= downscale
        K[1] *= downscale
    else:
        nW, nH = W, H

    # Prepare init mask at target resolution
    if downscale != 1.0:
        init_mask_scaled = cv2.resize(init_mask, (nW, nH), interpolation=cv2.INTER_NEAREST)
    else:
        init_mask_scaled = init_mask

    tracker.reset()
    initialized = False
    # Pre-allocate: NaN for frames without pose (world frame)
    all_poses_world = np.full((n_frames, 4, 4), np.nan, dtype=np.float32)
    n_saved = 0
    debug_saved = False

    for idx in tqdm(range(n_frames), desc="  pose", unit="f"):
        ret_rgb, bgr = cap_rgb.read()
        ret_depth, depth_bgr = cap_depth.read()
        if not ret_rgb or not ret_depth:
            break

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = decode_depth_uint16(depth_bgr)
        depth[(depth < 0.001) | (depth > 100.0)] = 0

        if downscale != 1.0:
            rgb = cv2.resize(rgb, (nW, nH))
            depth = cv2.resize(depth, (nW, nH), interpolation=cv2.INTER_NEAREST)

        try:
            if not initialized:
                pose_cam = tracker.init(rgb, depth, init_mask_scaled, K, iteration=est_refine_iter)
                initialized = True
            else:
                pose_cam = tracker.track(rgb, depth, K, iteration=2)

                # Save debug visualizations on second frame (idx==1)
                if not debug_saved:
                    debug_saved = True
                    pose_dir = os.path.join(cache_dir, "pose")
                    os.makedirs(pose_dir, exist_ok=True)
                    _save_debug_images(rgb, depth, init_mask_scaled, pose_dir)
                    if mesh_tensors is not None and glctx is not None:
                        try:
                            _save_pose_overlay(rgb, pose_cam.reshape(4, 4), K,
                                               mesh_tensors, glctx, device, pose_dir)
                        except Exception as e:
                            print(f"  Overlay debug failed: {e}", flush=True)

            # Convert to world frame: pose_world = inv(extrinsic) @ pose_cam
            pose_world = T_cam_inv @ pose_cam.reshape(4, 4)
            all_poses_world[idx] = pose_world
            n_saved += 1

        except Exception as e:
            tqdm.write(f"  Frame {idx}: {e}")
            continue

    cap_rgb.release()
    cap_depth.release()

    # Save world-frame poses (N,4,4)
    pose_dir = os.path.join(cache_dir, "pose")
    os.makedirs(pose_dir, exist_ok=True)
    out_path = os.path.join(pose_dir, "pose_world.npy")
    np.save(out_path, all_poses_world)
    print(f"  Saved {n_saved}/{n_frames} poses -> {out_path}", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base dir (network FS path)")
    parser.add_argument("--serials", nargs="+", required=True, help="Camera serial numbers")
    parser.add_argument("--mesh_dir", required=True, help="Root dir with {obj_name}/{obj_name}.obj")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--downscale", type=float, default=0.5,
                        help="Downscale factor for faster inference (default: 0.5)")
    parser.add_argument("--est_refine_iter", type=int, default=5,
                        help="Refinement iterations for registration (default: 5)")
    args = parser.parse_args()
    print("Starting batch pose estimation...", flush=True)

    torch.cuda.set_device(args.gpu)

    tasks = collect_tasks(args.base, args.serials, args.mesh_dir)
    if not tasks:
        print("Nothing to do (all videos already have poses).")
        return

    print(f"{len(tasks)} videos to process | GPU: {args.gpu}", flush=True)

    # Group tasks by object name to reuse PoseTracker
    from collections import defaultdict
    tasks_by_mesh = defaultdict(list)
    for t in tasks:
        mesh_path = t[8]  # mesh_path
        tasks_by_mesh[mesh_path].append(t)

    device = f"cuda:{args.gpu}"

    # Set up rendering for debug overlay
    fp_path = str(_FP_ROOT)
    if fp_path not in sys.path:
        sys.path.insert(0, fp_path)
    import trimesh
    import nvdiffrast.torch as dr
    from Utils import make_mesh_tensors

    glctx = dr.RasterizeCudaContext()

    done = 0
    total = len(tasks)
    for mesh_path, mesh_tasks in tasks_by_mesh.items():
        obj_name = mesh_tasks[0][7]
        print(f"\nLoading PoseTracker for {obj_name}: {mesh_path}", flush=True)
        tracker = PoseTracker(mesh_path, device_id=args.gpu)
        # Suppress FoundationPose logging (Utils.py reloads logging module on import)
        logging.getLogger().setLevel(logging.WARNING)

        # Load mesh for debug overlay
        mesh = trimesh.load(mesh_path, force="mesh")
        vertex_colors = np.tile(
            np.append(MESH_COLOR, 255).reshape(1, 4),
            (len(mesh.vertices), 1)
        )
        mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=vertex_colors)
        mesh_tensors = make_mesh_tensors(mesh, device=device)
        print("PoseTracker + mesh ready.", flush=True)

        for (video_path, mask_path, mask_is_video, depth_path, cache_dir,
             net_dir, serial, obj_name, _, idx_name) in mesh_tasks:
            done += 1
            print(f"[{done}/{total}] {obj_name}/{idx_name}/{serial}", flush=True)
            try:
                process_one_video(
                    tracker, video_path, mask_path, mask_is_video, depth_path,
                    cache_dir, net_dir, serial, obj_name,
                    args.downscale, args.est_refine_iter,
                    mesh_tensors=mesh_tensors, glctx=glctx, device=device,
                )
            except Exception as e:
                import traceback
                print(f"  Error: {e}", flush=True)
                traceback.print_exc()
                gc.collect()
                torch.cuda.empty_cache()

    print(f"\nAll done! {done}/{total} videos processed.", flush=True)


# Late import to avoid loading FoundationPose at module level
PoseTracker = None

def _get_pose_tracker():
    global PoseTracker
    if PoseTracker is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from autodex.perception.pose import PoseTracker as _PT
        PoseTracker = _PT
    return PoseTracker

# Override at import time
PoseTracker = None

if __name__ == "__main__":
    # Import PoseTracker at runtime
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from autodex.perception.pose import PoseTracker
    main()
