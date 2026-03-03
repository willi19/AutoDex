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
    """Collect tasks from local cache — videos with mask + depth but no pose."""
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
            for serial in serials:
                video_path = idx_dir / "videos" / f"{serial}.avi"
                depth_path = idx_dir / "depth" / f"{serial}.avi"
                if not (video_path.exists() and depth_path.exists()):
                    continue
                # Skip if depth is truncated (fewer frames than rgb)
                n_rgb = int(cv2.VideoCapture(str(video_path)).get(cv2.CAP_PROP_FRAME_COUNT))
                n_depth = int(cv2.VideoCapture(str(depth_path)).get(cv2.CAP_PROP_FRAME_COUNT))
                if n_depth < n_rgb:
                    continue
                mask_path, mask_is_video = _find_mask(idx_dir, serial)
                if mask_path is None:
                    continue
                # Skip if pose already exists
                if (idx_dir / "pose" / f"{serial}.npy").exists():
                    continue
                # Need cam_param from network FS
                rel = str(idx_dir.relative_to(cache_base))
                net_dir = Path(base_dir) / rel
                if not (net_dir / "cam_param").is_dir():
                    continue
                tasks.append((str(video_path), mask_path, mask_is_video,
                              str(depth_path), str(idx_dir), str(net_dir),
                              serial, obj_name, str(mesh_path), idx_dir.name))
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


def process_one_video(tracker, video_path, mask_path, mask_is_video, depth_path,
                      cache_dir, net_dir, serial, obj_name, downscale, est_refine_iter):
    """Run FoundationPose tracking on one video. Saves pose/{serial}.npy (N,4,4)."""
    intrinsics, extrinsics = load_cam_param(Path(net_dir))
    K = intrinsics[serial].copy()

    # Load init mask once
    init_mask = _load_init_mask(mask_path, mask_is_video)
    if init_mask is None:
        print("  No valid mask found, skipping", flush=True)
        return

    cap_rgb = cv2.VideoCapture(video_path)
    cap_depth = cv2.VideoCapture(depth_path)

    n_rgb = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
    n_depth = int(cap_depth.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(n_rgb, n_depth)
    W = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if n_rgb != n_depth:
        print(f"  {W}x{H}, rgb={n_rgb} depth={n_depth} -> using {n_frames} (mask: {'video' if mask_is_video else 'image'})", flush=True)
    else:
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
    # Pre-allocate: NaN for frames without pose
    all_poses = np.full((n_frames, 4, 4), np.nan, dtype=np.float32)
    n_saved = 0

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
                pose = tracker.init(rgb, depth, init_mask_scaled, K, iteration=est_refine_iter)
                initialized = True
            else:
                pose = tracker.track(rgb, depth, K, iteration=2)

            all_poses[idx] = pose.reshape(4, 4)
            n_saved += 1

        except Exception as e:
            tqdm.write(f"  Frame {idx}: {e}")
            continue

    cap_rgb.release()
    cap_depth.release()

    # Save single file: poses (N,4,4) — NaN for missing frames
    pose_dir = os.path.join(cache_dir, "pose")
    os.makedirs(pose_dir, exist_ok=True)
    out_path = os.path.join(pose_dir, f"{serial}.npy")
    np.save(out_path, all_poses)
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

    done = 0
    total = len(tasks)
    for mesh_path, mesh_tasks in tasks_by_mesh.items():
        obj_name = mesh_tasks[0][7]
        print(f"\nLoading PoseTracker for {obj_name}: {mesh_path}", flush=True)
        tracker = PoseTracker(mesh_path, device_id=args.gpu)
        # Suppress FoundationPose logging (Utils.py reloads logging module on import)
        logging.getLogger().setLevel(logging.WARNING)
        print("PoseTracker ready.", flush=True)

        for (video_path, mask_path, mask_is_video, depth_path, cache_dir,
             net_dir, serial, obj_name, _, idx_name) in mesh_tasks:
            done += 1
            print(f"[{done}/{total}] {obj_name}/{idx_name}/{serial}", flush=True)
            try:
                process_one_video(
                    tracker, video_path, mask_path, mask_is_video, depth_path,
                    cache_dir, net_dir, serial, obj_name,
                    args.downscale, args.est_refine_iter,
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
