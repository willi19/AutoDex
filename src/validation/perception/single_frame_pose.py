#!/usr/bin/env python3
"""Single-frame validation: depth overlay + pose + cross-view mesh overlay.

For each camera with depth+mask, estimates pose on frame 0, then reprojects
the mesh overlay onto ALL camera views. Produces:
  - Per source camera: a grid of 24 views with mesh overlay
  - A summary grid of all source cameras (self-view only)

Two-phase approach to avoid GPU OOM:
  Phase 1: Run FoundationPose on all cameras, collect poses, then free tracker.
  Phase 2: Load lightweight renderer, render overlays for all poses.

Usage:
    conda activate foundationpose
    python -u src/validation/perception/single_frame_pose.py \
        --capture_dir /home/mingi/shared_data/RSS2026_Mingi/experiment/selected_100/apple/20260206_181110 \
        --mesh /home/mingi/shared_data/object_6d/data/mesh/apple/apple.obj \
        --frame 0
"""

import argparse
import gc
import sys
import math
from pathlib import Path

import cv2
import numpy as np

AUTODEX_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(AUTODEX_ROOT))

_FP_ROOT = AUTODEX_ROOT / "autodex/perception/thirdparty/_object_6d_tracking/FoundationPose"

MESH_COLOR = np.array([128, 0, 128], dtype=np.uint8)
ALPHA = 0.6
DOWNSCALE = 0.5


def setup_fp_path():
    path = str(_FP_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
    mycpp_build = str(_FP_ROOT / "mycpp/build")
    if mycpp_build not in sys.path:
        sys.path.insert(0, mycpp_build)


def _to_4x4(T):
    if T.shape == (3, 4):
        T4 = np.eye(4, dtype=np.float64)
        T4[:3, :] = T
        return T4
    return T.astype(np.float64)


def render_mesh_overlay_bgr(bgr, pose_in_cam, K, mesh_tensors, glctx, device):
    """Render purple mesh overlay on BGR image. Returns BGR."""
    import torch
    from Utils import nvdiffrast_render

    H_orig, W_orig = bgr.shape[:2]
    H_r = int(H_orig * DOWNSCALE)
    W_r = int(W_orig * DOWNSCALE)
    K_r = K.copy()
    K_r[0] *= DOWNSCALE
    K_r[1] *= DOWNSCALE

    pose_t = torch.as_tensor(pose_in_cam, device=device, dtype=torch.float32).reshape(1, 4, 4)
    render_color, _, _ = nvdiffrast_render(
        K=K_r, H=H_r, W=W_r, ob_in_cams=pose_t, glctx=glctx,
        mesh_tensors=mesh_tensors, use_light=False,
    )
    render_mask = render_color[0].detach().cpu().numpy().sum(axis=2) > 0
    render_mask = cv2.resize(render_mask.astype(np.uint8), (W_orig, H_orig),
                             interpolation=cv2.INTER_NEAREST).astype(bool)

    overlay = bgr.copy()
    overlay[render_mask] = (
        overlay[render_mask].astype(np.float32) * (1.0 - ALPHA)
        + MESH_COLOR.astype(np.float32) * ALPHA
    ).astype(np.uint8)
    return overlay


def read_frame(video_path, frame_idx):
    """Read a single frame from video. Returns BGR or None."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def make_grid(images, labels, cols=6):
    """Arrange images into a grid with labels. All images must be same size."""
    n = len(images)
    rows = math.ceil(n / cols)
    h, w = images[0].shape[:2]
    grid = np.ones((rows * h, cols * w, 3), dtype=np.uint8) * 40
    for idx, (img, label) in enumerate(zip(images, labels)):
        r, c = divmod(idx, cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
        cv2.putText(grid, label, (c * w + 3, r * h + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    return grid


def main():
    parser = argparse.ArgumentParser(description="Single-frame depth+pose cross-view validation")
    parser.add_argument("--capture_dir", type=str, required=True)
    parser.add_argument("--mesh", type=str, required=True)
    parser.add_argument("--frame", type=int, default=0, help="Frame index to validate")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    capture_dir = Path(args.capture_dir)
    device = f"cuda:{args.gpu}"
    frame_idx = args.frame

    from autodex.perception.depth import load_cam_param, decode_depth_uint16

    intrinsics, extrinsics = load_cam_param(capture_dir)

    depth_dir = capture_dir / "depth"
    mask_dir = capture_dir / "obj_mask"
    video_dir = capture_dir / "videos"

    # Always use videos/ — depth and mask AVIs are aligned to distorted video frames.
    # Using images/ (undistorted) would create a coordinate mismatch.
    all_serials = sorted(p.stem for p in video_dir.glob("*.avi") if p.stem in intrinsics)
    print(f"Using video frames from {video_dir}")

    depth_serials = {p.stem for p in depth_dir.glob("*.avi")} if depth_dir.exists() else set()
    mask_serials = {p.stem for p in mask_dir.glob("*.avi")} if mask_dir.exists() else set()
    ready = sorted(depth_serials & mask_serials & set(all_serials))

    print(f"All cameras: {len(all_serials)}, Depth: {len(depth_serials)}, "
          f"Mask: {len(mask_serials)}, Ready for pose: {len(ready)}")

    if not ready:
        print("No cameras have depth + mask + video. Nothing to do.")
        return

    setup_fp_path()
    import torch
    from autodex.perception import PoseTracker

    torch.cuda.set_device(args.gpu)

    out_dir = capture_dir / "validation_single_frame"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-read frame from ALL cameras
    print(f"\nReading frames from all {len(all_serials)} cameras...")
    all_bgr = {}
    for serial in all_serials:
        bgr = read_frame(video_dir / f"{serial}.avi", frame_idx)
        if bgr is not None:
            all_bgr[serial] = bgr

    sample = next(iter(all_bgr.values()))
    H, W = sample.shape[:2]
    thumb_w, thumb_h = W // 3, H // 3

    # ── Phase 1: Pose estimation (heavy GPU usage) ──
    print("\n=== Phase 1: Pose estimation ===")
    tracker = PoseTracker(args.mesh, device_id=args.gpu)

    # {src_serial: ob_in_world}
    poses_world = {}

    for i, src_serial in enumerate(ready):
        K_src = intrinsics[src_serial].astype(np.float32)
        T_src = _to_4x4(extrinsics[src_serial])
        print(f"\n[{i+1}/{len(ready)}] Source: {src_serial}", flush=True)

        bgr_src = all_bgr.get(src_serial)
        if bgr_src is None:
            print(f"  Can't read frame {frame_idx}")
            continue

        rgb_src = cv2.cvtColor(bgr_src, cv2.COLOR_BGR2RGB)

        depth = np.zeros((H, W), dtype=np.float32)
        mask = np.zeros((H, W), dtype=bool)

        d_frame = read_frame(depth_dir / f"{src_serial}.avi", frame_idx)
        if d_frame is not None:
            depth = decode_depth_uint16(d_frame)

        m_frame = read_frame(mask_dir / f"{src_serial}.avi", frame_idx)
        if m_frame is not None:
            mask = cv2.cvtColor(m_frame, cv2.COLOR_BGR2GRAY) > 127

        has_depth = (depth > 0.001).sum() > 100
        has_mask = mask.sum() > 100

        print(f"  depth: {(depth > 0.001).sum()} valid px, mask: {mask.sum()} px", flush=True)

        if not has_depth or not has_mask:
            print(f"  Skipping: insufficient depth or mask")
            continue

        # Downscale inputs to reduce GPU memory (matches reference pipeline)
        ds = DOWNSCALE
        H_ds, W_ds = int(H * ds), int(W * ds)
        rgb_ds = cv2.resize(rgb_src, (W_ds, H_ds))
        depth_ds = cv2.resize(depth, (W_ds, H_ds), interpolation=cv2.INTER_NEAREST)
        mask_ds = cv2.resize(mask.astype(np.uint8), (W_ds, H_ds),
                             interpolation=cv2.INTER_NEAREST)
        K_ds = K_src.copy()
        K_ds[0] *= ds
        K_ds[1] *= ds

        tracker.reset()
        try:
            pose_in_cam = tracker.init(
                rgb_ds, depth_ds.astype(np.float32), mask_ds, K_ds)
            if pose_in_cam is None or np.isnan(pose_in_cam).any():
                print(f"  Pose returned None or NaN")
                continue
            print(f"  Pose OK: t=[{pose_in_cam[0,3]:.3f}, {pose_in_cam[1,3]:.3f}, {pose_in_cam[2,3]:.3f}]",
                  flush=True)
        except Exception as e:
            print(f"  Pose failed: {e}", flush=True)
            continue

        ob_in_world = np.linalg.inv(T_src) @ pose_in_cam
        poses_world[src_serial] = ob_in_world

    # Free tracker GPU memory
    del tracker
    torch.cuda.empty_cache()
    gc.collect()
    print(f"\nPhase 1 done: {len(poses_world)}/{len(ready)} poses estimated")

    if not poses_world:
        print("No poses succeeded. Nothing to render.")
        return

    # ── Phase 2: Render overlays (lightweight GPU usage) ──
    print("\n=== Phase 2: Rendering cross-view overlays ===")
    import trimesh
    import nvdiffrast.torch as dr
    from Utils import make_mesh_tensors

    mesh = trimesh.load(args.mesh, force="mesh")
    vertex_colors = np.tile(
        np.array([128, 0, 128, 255], dtype=np.uint8).reshape(1, 4),
        (len(mesh.vertices), 1)
    )
    mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=vertex_colors)
    mesh_tensors = make_mesh_tensors(mesh, device=device)
    glctx = dr.RasterizeCudaContext()

    summary_panels = []
    summary_labels = []

    for src_serial, ob_in_world in poses_world.items():
        print(f"\n  Rendering overlays for pose from {src_serial}...", flush=True)

        view_thumbs = []
        view_labels = []
        for tgt_serial in all_serials:
            bgr_tgt = all_bgr.get(tgt_serial)
            if bgr_tgt is None:
                thumb = np.ones((thumb_h, thumb_w, 3), dtype=np.uint8) * 40
            else:
                K_tgt = intrinsics[tgt_serial].astype(np.float32)
                T_tgt = _to_4x4(extrinsics[tgt_serial])
                ob_in_tgt_cam = (T_tgt @ ob_in_world).astype(np.float32)

                try:
                    overlay = render_mesh_overlay_bgr(
                        bgr_tgt, ob_in_tgt_cam, K_tgt, mesh_tensors, glctx, device)
                except Exception as e:
                    overlay = bgr_tgt.copy()

                thumb = cv2.resize(overlay, (thumb_w, thumb_h))

            is_self = (tgt_serial == src_serial)
            label = f"{'*' if is_self else ''}{tgt_serial}"
            view_thumbs.append(thumb)
            view_labels.append(label)

        grid = make_grid(view_thumbs, view_labels, cols=6)
        cv2.putText(grid, f"Pose from: {src_serial} frame={frame_idx}", (5, grid.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        grid_path = str(out_dir / f"crossview_{src_serial}.jpg")
        cv2.imwrite(grid_path, grid)
        print(f"  Saved: {grid_path}")

        self_idx = all_serials.index(src_serial) if src_serial in all_serials else 0
        summary_panels.append(view_thumbs[self_idx])
        summary_labels.append(src_serial)

    if summary_panels:
        summary = make_grid(summary_panels, summary_labels, cols=4)
        summary_path = str(out_dir / f"summary_f{frame_idx}.jpg")
        cv2.imwrite(summary_path, summary)
        print(f"\nSummary saved: {summary_path}")

    print(f"\nAll results in: {out_dir}")


if __name__ == "__main__":
    main()