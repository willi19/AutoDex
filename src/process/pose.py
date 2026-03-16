#!/usr/bin/env python3
"""6D pose estimation with FoundationPose tracking + optional mesh overlay.

Supports two modes:
  --capture_dir : process a single episode (all cameras with depth+mask)
  --base        : batch all episodes under a directory (with progress/ETA)

Usage:
    conda activate foundationpose

    # Single episode — tracking only
    python -u src/process/pose.py \
        --capture_dir /home/mingi/shared_data/RSS2026_Mingi/experiment/selected_100/apple/20260206_181110 \
        --mesh /home/mingi/shared_data/object_6d/data/mesh/apple/apple.obj

    # Single episode — tracking + overlay video
    python -u src/process/pose.py --overlay \
        --capture_dir /path/to/episode --mesh /path/to/mesh.obj

    # Batch all episodes (requires --mesh_dir)
    python -u src/process/pose.py \
        --base /home/mingi/shared_data/RSS2026_Mingi/experiment/selected_100 \
        --mesh_dir /home/mingi/shared_data/object_6d/data/mesh

    # Batch with overlay
    python -u src/process/pose.py --overlay \
        --base /path/to/selected_100 --mesh_dir /path/to/meshes
"""

import argparse
import gc
import logging
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

AUTODEX_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AUTODEX_ROOT))

_FP_ROOT = AUTODEX_ROOT / "autodex/perception/thirdparty/_object_6d_tracking/FoundationPose"

MESH_COLOR = np.array([128, 0, 128], dtype=np.uint8)  # purple
OVERLAY_ALPHA = 0.6
OVERLAY_DOWNSCALE = 0.5


# ── Mesh lookup ──────────────────────────────────────────────────────────────

def find_mesh(mesh_dir, obj_name):
    """Find mesh file for an object. Tries common patterns."""
    mesh_dir = Path(mesh_dir)
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
    found = list(mesh_dir.glob(f"{obj_name}/**/*.obj")) + \
            list(mesh_dir.glob(f"{obj_name}/**/*.ply"))
    return found[0] if found else None


# ── FoundationPose path setup ────────────────────────────────────────────────

def _setup_fp_path():
    path = str(_FP_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
    mycpp_build = str(_FP_ROOT / "mycpp/build")
    if mycpp_build not in sys.path:
        sys.path.insert(0, mycpp_build)


# ── Rendering ────────────────────────────────────────────────────────────────

def _render_overlay(rgb, pose_4x4, K, mesh_tensors, glctx, device):
    """Render mesh overlay on RGB frame. Returns overlay (H,W,3) uint8 RGB."""
    from Utils import nvdiffrast_render

    H_orig, W_orig = rgb.shape[:2]
    H_r = int(H_orig * OVERLAY_DOWNSCALE)
    W_r = int(W_orig * OVERLAY_DOWNSCALE)
    K_r = K.copy()
    K_r[0] *= OVERLAY_DOWNSCALE
    K_r[1] *= OVERLAY_DOWNSCALE

    pose_t = torch.as_tensor(pose_4x4, device=device, dtype=torch.float32).reshape(1, 4, 4)

    render_color, _, _ = nvdiffrast_render(
        K=K_r, H=H_r, W=W_r, ob_in_cams=pose_t, glctx=glctx,
        mesh_tensors=mesh_tensors, use_light=False,
    )
    render_mask = render_color[0].detach().cpu().numpy().sum(axis=2) > 0
    render_mask = cv2.resize(render_mask.astype(np.uint8), (W_orig, H_orig),
                             interpolation=cv2.INTER_NEAREST).astype(bool)

    overlay = rgb.copy()
    overlay[render_mask] = (
        overlay[render_mask].astype(np.float32) * (1.0 - OVERLAY_ALPHA)
        + MESH_COLOR.astype(np.float32) * OVERLAY_ALPHA
    ).astype(np.uint8)
    return overlay


# ── Helpers ──────────────────────────────────────────────────────────────────

def _to_4x4(T):
    if T.shape == (3, 4):
        T4 = np.eye(4, dtype=np.float64)
        T4[:3, :] = T
        return T4
    return T.astype(np.float64)


def _load_init_mask(capture_dir, serial):
    """Load init mask from obj_mask video (first valid frame) or obj_mask_first PNG."""
    capture_dir = Path(capture_dir)
    # Try single-image mask first
    first_mask = capture_dir / "obj_mask_first" / f"{serial}.png"
    if first_mask.exists():
        img = cv2.imread(str(first_mask), cv2.IMREAD_GRAYSCALE)
        return (img > 127).astype(np.uint8)
    # Fall back to mask video
    mask_video = capture_dir / "obj_mask" / f"{serial}.avi"
    if mask_video.exists():
        cap = cv2.VideoCapture(str(mask_video))
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


# ── Process one camera ───────────────────────────────────────────────────────

def process_camera(tracker, serial, capture_dir, K, T_cam,
                   do_overlay=False, mesh_tensors=None, glctx=None, device="cuda:0",
                   downscale=0.5):
    """Run pose tracking (+ optional overlay) for one camera.

    Returns (n_ok, n_fail, n_total) or None on skip.
    """
    from autodex.perception.depth import decode_depth_uint16

    capture_dir = Path(capture_dir)
    video_path = str(capture_dir / "videos" / f"{serial}.avi")
    depth_path = str(capture_dir / "depth" / f"{serial}.avi")

    # Load init mask
    init_mask = _load_init_mask(capture_dir, serial)
    if init_mask is None:
        print(f"    No valid mask for {serial}, skipping", flush=True)
        return None

    cap_rgb = cv2.VideoCapture(video_path)
    cap_depth = cv2.VideoCapture(depth_path)

    n_frames = min(int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT)),
                   int(cap_depth.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = cap_rgb.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Downscale
    K_ds = K.copy()
    if downscale != 1.0:
        nW, nH = int(W * downscale), int(H * downscale)
        K_ds[0] *= downscale
        K_ds[1] *= downscale
        init_mask_ds = cv2.resize(init_mask, (nW, nH), interpolation=cv2.INTER_NEAREST)
    else:
        nW, nH = W, H
        init_mask_ds = init_mask

    T_cam_inv = np.linalg.inv(_to_4x4(T_cam))

    # Setup overlay writer
    overlay_writer = None
    if do_overlay:
        out_dir = capture_dir / "pose_overlay"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / f"{serial}.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        overlay_writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H), True)

    # Pose output
    pose_dir = capture_dir / "pose"
    pose_dir.mkdir(parents=True, exist_ok=True)

    poses_cam = np.full((n_frames, 4, 4), np.nan, dtype=np.float32)
    poses_world = np.full((n_frames, 4, 4), np.nan, dtype=np.float32)
    tracker.reset()
    initialized = False
    n_ok = 0
    n_fail = 0

    for idx in range(n_frames):
        ret_rgb, bgr = cap_rgb.read()
        ret_depth, depth_bgr = cap_depth.read()
        if not ret_rgb:
            break

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if ret_depth:
            depth = decode_depth_uint16(depth_bgr)
        else:
            depth = np.zeros((H, W), dtype=np.float32)

        if downscale != 1.0:
            rgb_ds = cv2.resize(rgb, (nW, nH))
            depth_ds = cv2.resize(depth, (nW, nH), interpolation=cv2.INTER_NEAREST)
        else:
            rgb_ds = rgb
            depth_ds = depth

        has_depth = (depth_ds > 0.001).sum() > 100
        has_mask = init_mask_ds.sum() > 100

        if not has_depth or (not initialized and not has_mask):
            n_fail += 1
            if overlay_writer:
                overlay_writer.write(bgr)
            continue

        try:
            if not initialized:
                pose_cam = tracker.init(rgb_ds, depth_ds, init_mask_ds, K_ds)
                initialized = True
            else:
                pose_cam = tracker.track(rgb_ds, depth_ds, K_ds)

            pose_cam = pose_cam.reshape(4, 4)
            poses_cam[idx] = pose_cam
            poses_world[idx] = T_cam_inv @ pose_cam
            n_ok += 1

            if overlay_writer:
                overlay_rgb = _render_overlay(rgb, pose_cam, K, mesh_tensors, glctx, device)
                overlay_writer.write(cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

        except Exception as e:
            if idx < 3:
                print(f"    frame {idx}: {e}", flush=True)
            n_fail += 1
            if overlay_writer:
                overlay_writer.write(bgr)
            tracker.reset()
            initialized = False

    cap_rgb.release()
    cap_depth.release()
    if overlay_writer:
        overlay_writer.release()

    # Save poses
    np.save(str(pose_dir / f"{serial}.npy"), poses_cam)
    np.save(str(pose_dir / f"{serial}_world.npy"), poses_world)

    return n_ok, n_fail, n_frames


# ── Grid overlay ─────────────────────────────────────────────────────────────

def generate_pose_grid(capture_dir, intrinsics, extrinsics,
                       mesh_tensors, glctx, device):
    """Generate grid overlay VIDEO: for each source camera, render mesh on ALL cameras.

    Saves pose_overlay/grid_{src_serial}.avi
    Each grid cell = overlay | original side by side, labeled src->tgt.
    """
    capture_dir = Path(capture_dir)
    pose_dir = capture_dir / "pose"
    video_dir = capture_dir / "videos"
    overlay_dir = capture_dir / "pose_overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    # Find cameras with saved world poses
    pose_serials = []
    for p in sorted(pose_dir.glob("*_world.npy")):
        serial = p.stem.replace("_world", "")
        if serial in intrinsics and (video_dir / f"{serial}.avi").exists():
            pose_serials.append(serial)

    all_serials = sorted(s for s in intrinsics if (video_dir / f"{s}.avi").exists())

    if not pose_serials:
        print("No pose files found for grid overlay.", flush=True)
        return

    print(f"Pose grid video: {len(pose_serials)} sources, "
          f"{len(all_serials)} cameras", flush=True)

    for src_serial in pose_serials:
        poses_world = np.load(str(pose_dir / f"{src_serial}_world.npy"))
        n_pose_frames = len(poses_world)

        # Find first valid frame to check it works
        has_any_valid = any(not np.isnan(poses_world[i]).any()
                           for i in range(min(n_pose_frames, 5)))
        if not has_any_valid:
            print(f"  {src_serial}: no valid poses in first 5 frames, skipping", flush=True)
            continue

        # Open all target video captures
        tgt_caps = {}
        for s in all_serials:
            cap = cv2.VideoCapture(str(video_dir / f"{s}.avi"))
            if cap.isOpened():
                tgt_caps[s] = cap

        # Get video properties from first target
        first_cap = next(iter(tgt_caps.values()))
        fps = first_cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_video_frames = min(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in tgt_caps.values())
        n_frames = min(n_pose_frames, n_video_frames)

        # Read first frame to determine grid layout
        sample_ret, sample_bgr = first_cap.read()
        first_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind
        if not sample_ret:
            for c in tgt_caps.values():
                c.release()
            continue

        H_full, W_full = sample_bgr.shape[:2]
        cell_w = W_full // 4
        cell_h = H_full // 4
        n_cells = len(all_serials)
        cols = math.ceil(math.sqrt(n_cells))
        rows = math.ceil(n_cells / cols)
        grid_w = cols * cell_w
        grid_h = rows * cell_h

        # Precompute extrinsics
        T_tgts = {s: _to_4x4(extrinsics[s]) for s in all_serials}
        K_tgts = {s: intrinsics[s].astype(np.float32) for s in all_serials}

        out_path = str(overlay_dir / f"grid_{src_serial}.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (grid_w, grid_h), True)

        print(f"  {src_serial}: {n_frames} frames -> {out_path}", flush=True)

        for fidx in range(n_frames):
            pose_valid = not np.isnan(poses_world[fidx]).any()
            if pose_valid:
                pose_world = poses_world[fidx]

            grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

            for ci, tgt_serial in enumerate(all_serials):
                ret, tgt_bgr = tgt_caps[tgt_serial].read()
                if not ret:
                    continue

                r, c = divmod(ci, cols)

                if pose_valid:
                    tgt_rgb = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2RGB)
                    pose_cam = (T_tgts[tgt_serial] @ pose_world).astype(np.float32)
                    try:
                        overlay_rgb = _render_overlay(tgt_rgb, pose_cam,
                                                      K_tgts[tgt_serial],
                                                      mesh_tensors, glctx, device)
                        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                    except Exception:
                        overlay_bgr = tgt_bgr
                else:
                    overlay_bgr = tgt_bgr

                cell = cv2.resize(overlay_bgr, (cell_w, cell_h))

                y0, x0 = r * cell_h, c * cell_w
                grid[y0:y0 + cell_h, x0:x0 + cell_w] = cell

                if fidx == 0:
                    cv2.putText(grid, f"{src_serial}->{tgt_serial}",
                                (x0 + 5, y0 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Add labels on every frame
            for ci, tgt_serial in enumerate(all_serials):
                r, c = divmod(ci, cols)
                y0, x0 = r * cell_h, c * cell_w
                cv2.putText(grid, f"{src_serial}->{tgt_serial}",
                            (x0 + 5, y0 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            writer.write(grid)

        writer.release()
        for c in tgt_caps.values():
            c.release()
        print(f"  Saved: {out_path}", flush=True)


# ── Process one capture dir ──────────────────────────────────────────────────

def process_capture(capture_dir, mesh_path, gpu=0, downscale=0.5,
                    do_overlay=False, grid_only=False):
    """Process all ready cameras in a single capture directory."""
    from autodex.perception.depth import load_cam_param

    capture_dir = Path(capture_dir)
    intrinsics, extrinsics = load_cam_param(capture_dir)
    device = f"cuda:{gpu}"

    # Load mesh + setup rendering
    _setup_fp_path()
    import trimesh
    import nvdiffrast.torch as dr
    from Utils import make_mesh_tensors

    torch.cuda.set_device(gpu)

    mesh = trimesh.load(str(mesh_path), force="mesh")
    vertex_colors = np.tile(
        np.array([128, 0, 128, 255], dtype=np.uint8).reshape(1, 4),
        (len(mesh.vertices), 1)
    )
    mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=vertex_colors)
    mesh_tensors = make_mesh_tensors(mesh, device=device)
    glctx = dr.RasterizeCudaContext()

    if grid_only:
        # Just generate grid from existing poses, no tracking
        print(f"\n=== Generating pose grid overlay (grid_only) ===", flush=True)
        generate_pose_grid(capture_dir, intrinsics, extrinsics,
                           mesh_tensors, glctx, device)
        return

    # Find cameras with video + depth + mask
    video_dir = capture_dir / "videos"
    depth_dir = capture_dir / "depth"
    mask_dir = capture_dir / "obj_mask"

    video_serials = {p.stem for p in video_dir.glob("*.avi")}
    depth_serials = {p.stem for p in depth_dir.glob("*.avi")} if depth_dir.exists() else set()
    mask_serials = set()
    if mask_dir.exists():
        mask_serials |= {p.stem for p in mask_dir.glob("*.avi")}
    mask_first_dir = capture_dir / "obj_mask_first"
    if mask_first_dir.exists():
        mask_serials |= {p.stem for p in mask_first_dir.glob("*.png")}

    ready = sorted((video_serials & depth_serials & mask_serials) & set(intrinsics.keys()))
    print(f"Videos: {len(video_serials)}, Depth: {len(depth_serials)}, "
          f"Mask: {len(mask_serials)}, Ready: {len(ready)}", flush=True)

    if not ready:
        print("No cameras have all three (video + depth + mask). Nothing to do.")
        return

    # Load PoseTracker
    from autodex.perception import PoseTracker
    print(f"Loading PoseTracker with {mesh_path}...", flush=True)
    tracker = PoseTracker(str(mesh_path), device_id=gpu)
    logging.getLogger().setLevel(logging.WARNING)
    print("PoseTracker ready.", flush=True)

    t_start = time.perf_counter()
    for i, serial in enumerate(ready):
        K = intrinsics[serial].astype(np.float32)
        T_cam = extrinsics[serial]
        print(f"\n[{i+1}/{len(ready)}] {serial}", flush=True)

        t0 = time.perf_counter()
        result = process_camera(
            tracker, serial, capture_dir, K, T_cam,
            do_overlay=do_overlay, mesh_tensors=mesh_tensors,
            glctx=glctx, device=device, downscale=downscale,
        )
        dt = time.perf_counter() - t0

        if result is not None:
            n_ok, n_fail, n_total = result
            print(f"  {n_ok}/{n_total} tracked, {n_fail} failed, {dt:.1f}s", flush=True)

    # Generate grid overlay (frame 0)
    print(f"\n=== Generating pose grid overlay ===", flush=True)
    generate_pose_grid(capture_dir, intrinsics, extrinsics,
                       mesh_tensors, glctx, device)

    elapsed = time.perf_counter() - t_start
    print(f"\nDone! {len(ready)} cameras, {elapsed:.1f}s total", flush=True)


# ── Batch helpers ────────────────────────────────────────────────────────────

def _find_capture_dirs(base):
    """Find all capture directories (contain cam_param/ and videos/)."""
    base = Path(base)
    return sorted(
        p.parent for p in base.rglob("cam_param")
        if p.is_dir() and (p.parent / "videos").is_dir()
    )


def _format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h{m:02d}m"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="6D pose estimation with FoundationPose",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
modes:
  --capture_dir DIR    Process a single episode (all ready cameras)
  --base DIR           Batch all episodes under DIR (with progress/ETA)

examples:
  %(prog)s --capture_dir /path/to/episode --mesh /path/to/mesh.obj
  %(prog)s --overlay --capture_dir /path/to/episode --mesh /path/to/mesh.obj
  %(prog)s --base /path/to/selected_100 --mesh_dir /path/to/meshes
  %(prog)s --overlay --base /path/to/selected_100 --mesh_dir /path/to/meshes
""",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--capture_dir", type=str, help="Single episode directory")
    group.add_argument("--base", type=str, help="Batch: parent of all episodes")

    parser.add_argument("--mesh", type=str, default=None,
                        help="Path to mesh file (single episode mode)")
    parser.add_argument("--mesh_dir", type=str, default=None,
                        help="Root mesh dir with {obj_name}/{obj_name}.obj (batch mode)")
    parser.add_argument("--overlay", action="store_true",
                        help="Generate mesh overlay videos")
    parser.add_argument("--grid_only", action="store_true",
                        help="Skip tracking, only generate grid overlay from existing poses")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--downscale", type=float, default=0.5,
                        help="Downscale factor for inference (default: 0.5)")
    args = parser.parse_args()

    if args.capture_dir:
        # Single episode mode
        if not args.mesh:
            parser.error("--mesh is required for single episode mode")
        process_capture(
            args.capture_dir, args.mesh,
            gpu=args.gpu, downscale=args.downscale,
            do_overlay=args.overlay, grid_only=args.grid_only,
        )
    else:
        # Batch mode
        if not args.mesh_dir:
            parser.error("--mesh_dir is required for batch mode")

        dirs = _find_capture_dirs(args.base)
        total = len(dirs)
        print(f"Found {total} capture directories")
        print("=" * 60)

        failed = []
        t_global = time.perf_counter()

        for i, capture_dir in enumerate(dirs):
            obj_name = capture_dir.parent.name
            idx = capture_dir.name
            elapsed = time.perf_counter() - t_global

            if i > 0:
                avg = elapsed / i
                remaining = avg * (total - i)
                eta_str = f"ETA {_format_time(remaining)}"
            else:
                eta_str = "ETA --"

            # Find mesh for this object
            mesh_path = find_mesh(args.mesh_dir, obj_name)
            if mesh_path is None:
                print(f"\n[{i+1}/{total}] {obj_name}/{idx} SKIP (no mesh)", flush=True)
                continue

            print(f"\n{'=' * 60}")
            print(f"[{i+1}/{total}] {obj_name}/{idx}  "
                  f"(elapsed {_format_time(elapsed)}, {eta_str})")
            print("-" * 60)

            t_start = time.perf_counter()
            try:
                process_capture(
                    capture_dir, mesh_path,
                    gpu=args.gpu, downscale=args.downscale,
                    do_overlay=args.overlay, grid_only=args.grid_only,
                )
            except Exception as e:
                failed.append((capture_dir, str(e)))
                print(f"  FAILED: {e}", flush=True)
                gc.collect()
                torch.cuda.empty_cache()

            dt = time.perf_counter() - t_start
            print(f"  Completed in {_format_time(dt)}")

        total_time = time.perf_counter() - t_global
        print(f"\n{'=' * 60}")
        print(f"All done! {total} captures in {_format_time(total_time)}")
        if failed:
            print(f"\n{len(failed)} FAILED:")
            for d, reason in failed:
                print(f"  {d.parent.name}/{d.name}: {reason}")


if __name__ == "__main__":
    main()
