#!/usr/bin/env python3
"""Batch mesh overlay visualization for pose debugging.

Reads RGB videos and per-frame pose estimates from local cache,
renders the object mesh overlay on each frame using nvdiffrast,
and saves overlay videos.

Modes:
  --serials S1 S2    : overlay on cameras that have pose estimates (default)
  --all_serials      : overlay on ALL cameras (converts pose to world frame)

Pre-requisites:
    - Poses generated: python src/perception/batch_pose.py ...

Run:
    conda activate foundationpose
    python -u src/perception/batch_overlay.py \
        --base /home/mingi/paradex1/capture/eccv2026/inspire_f1 \
        --serials 22684755 23263780 \
        --mesh_dir /home/mingi/mesh

Upload results:
    python src/perception/upload_results.py --base ...
"""

import os
import sys
import gc
import json
import argparse
from pathlib import Path

import math
import cv2
import numpy as np
import torch
from tqdm import tqdm

CACHE_ROOT = os.path.expanduser("~/video_cache")
NETWORK_PREFIX = "/home/mingi/paradex1/capture"

# FoundationPose path for Utils.py (make_mesh_tensors, nvdiffrast_render)
_FP_ROOT = Path(__file__).resolve().parents[2] / "autodex/perception/thirdparty/_object_6d_tracking/FoundationPose"


def _setup_fp_path():
    path = str(_FP_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


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


# ── Mesh lookup ──────────────────────────────────────────────────────────────

def find_mesh(mesh_dir: Path, obj_name: str):
    """Find mesh file for an object."""
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
    found = list(mesh_dir.glob(f"{obj_name}/**/*.obj")) + list(mesh_dir.glob(f"{obj_name}/**/*.ply"))
    return found[0] if found else None


# ── Task collection ──────────────────────────────────────────────────────────

def collect_tasks(base_dir, serials, mesh_dir):
    """Collect tasks from local cache — videos with pose but no overlay."""
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
                pose_path = idx_dir / "pose" / f"{serial}.npy"
                if not video_path.exists():
                    continue
                if not pose_path.exists():
                    continue
                # Skip if overlay already exists
                overlay_path = idx_dir / "pose_overlay" / f"{serial}.avi"
                if overlay_path.exists():
                    continue
                # Need cam_param from network FS
                rel = str(idx_dir.relative_to(cache_base))
                net_dir = Path(base_dir) / rel
                if not (net_dir / "cam_param").is_dir():
                    continue
                tasks.append((str(video_path), str(pose_path), str(idx_dir),
                              str(net_dir), serial, obj_name, str(mesh_path),
                              idx_dir.name))
    return tasks


# ── Render one frame ─────────────────────────────────────────────────────────

MESH_COLOR = np.array([128, 0, 128], dtype=np.uint8)  # purple
ALPHA = 0.6


def render_overlay(rgb, pose_4x4, K, mesh_tensors, glctx, device, downscale):
    """Render mesh overlay on one RGB frame.

    Args:
        rgb: (H, W, 3) uint8 RGB image
        pose_4x4: (4, 4) object-in-camera pose
        K: (3, 3) intrinsics
        mesh_tensors: pre-computed mesh tensors
        glctx: nvdiffrast context
        device: torch device
        downscale: downscale factor for rendering (1.0 = full res)

    Returns:
        overlay: (H, W, 3) uint8 RGB image with mesh overlay
    """
    from Utils import nvdiffrast_render

    H_orig, W_orig = rgb.shape[:2]

    if downscale != 1.0:
        H_r = int(H_orig * downscale)
        W_r = int(W_orig * downscale)
        K_r = K.copy()
        K_r[0] *= downscale
        K_r[1] *= downscale
    else:
        H_r, W_r = H_orig, W_orig
        K_r = K

    pose_t = torch.as_tensor(pose_4x4, device=device, dtype=torch.float32).reshape(1, 4, 4)

    render_color, _, _ = nvdiffrast_render(
        K=K_r, H=H_r, W=W_r, ob_in_cams=pose_t, glctx=glctx,
        mesh_tensors=mesh_tensors, use_light=False,
    )
    render_mask = render_color[0].detach().cpu().numpy().sum(axis=2) > 0

    if downscale != 1.0:
        render_mask = cv2.resize(render_mask.astype(np.uint8), (W_orig, H_orig),
                                 interpolation=cv2.INTER_NEAREST).astype(bool)

    overlay = rgb.copy()
    overlay[render_mask] = (
        overlay[render_mask].astype(np.float32) * (1.0 - ALPHA)
        + MESH_COLOR.astype(np.float32) * ALPHA
    ).astype(np.uint8)
    return overlay


# ── Process one video ────────────────────────────────────────────────────────

def process_one_video(video_path, pose_path, cache_dir, net_dir, serial,
                      mesh_tensors, glctx, device, render_downscale):
    """Render mesh overlay video for one (obj, idx, serial)."""
    intrinsics, _ = load_cam_param(Path(net_dir))
    K = intrinsics[serial].copy()

    # Load all poses: (N, 4, 4), NaN for missing frames
    all_poses = np.load(pose_path)

    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  {W}x{H}, {n_frames} frames", flush=True)

    # Output
    out_dir = os.path.join(cache_dir, "pose_overlay")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{serial}.avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    n_overlaid = 0
    for idx in tqdm(range(n_frames), desc="  overlay", unit="f"):
        ret, bgr = cap.read()
        if not ret:
            break

        if idx < len(all_poses) and not np.isnan(all_poses[idx, 0, 0]):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            overlay_rgb = render_overlay(rgb, all_poses[idx], K, mesh_tensors, glctx, device, render_downscale)
            writer.write(cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
            n_overlaid += 1
        else:
            writer.write(bgr)

    cap.release()
    writer.release()
    print(f"  Saved {out_path} ({n_overlaid}/{n_frames} overlaid)", flush=True)


# ── Merge multi-camera overlay ────────────────────────────────────────────────

def merge_overlay_videos(cache_dir, serials):
    """Merge per-serial overlay videos into a single grid video.

    Saves: pose_overlay_merged/merged.avi
    """
    overlay_dir = Path(cache_dir) / "pose_overlay"
    paths = []
    for s in serials:
        p = overlay_dir / f"{s}.avi"
        if p.exists():
            paths.append((s, str(p)))
    if len(paths) < 2:
        return

    # Open all captures
    caps = [(s, cv2.VideoCapture(p)) for s, p in paths]
    n_cams = len(caps)

    # Get video properties from first capture
    fps = caps[0][1].get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(min(c.get(cv2.CAP_PROP_FRAME_COUNT) for _, c in caps))
    W_src = int(caps[0][1].get(cv2.CAP_PROP_FRAME_WIDTH))
    H_src = int(caps[0][1].get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Grid layout (same logic as paradex merge_image)
    grid_cols = math.ceil(math.sqrt(n_cams))
    grid_rows = math.ceil(n_cams / grid_cols)
    border_px = 10
    cell_W = 2048 // grid_cols
    cell_H = 1536 // grid_rows
    canvas_W = cell_W * grid_cols + border_px * (grid_cols - 1)
    canvas_H = 1536 + border_px * (grid_rows - 1)

    out_dir = Path(cache_dir) / "pose_overlay_merged"
    out_dir.mkdir(exist_ok=True)
    out_path = str(out_dir / "merged.avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (canvas_W, canvas_H))

    for _ in tqdm(range(n_frames), desc="  merge", unit="f"):
        canvas = np.ones((canvas_H, canvas_W, 3), dtype=np.uint8) * 255
        all_ok = True
        for idx, (serial, cap) in enumerate(caps):
            ret, bgr = cap.read()
            if not ret:
                all_ok = False
                break
            # Draw serial label
            font_scale = max(0.5, cell_W / 800)
            thickness = max(1, cell_W // 500)
            txt = serial
            ts = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            cv2.rectangle(bgr, (5, 5), (15 + ts[0], 15 + ts[1]), (0, 0, 0), -1)
            cv2.putText(bgr, txt, (10, 10 + ts[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), thickness)
            resized = cv2.resize(bgr, (cell_W, cell_H))
            r = idx // grid_cols
            c = idx % grid_cols
            y0 = r * (cell_H + border_px)
            x0 = c * (cell_W + border_px)
            canvas[y0:y0 + cell_H, x0:x0 + cell_W] = resized
        if not all_ok:
            break
        writer.write(canvas)

    for _, cap in caps:
        cap.release()
    writer.release()
    print(f"  Merged {n_cams} cameras -> {out_path}", flush=True)


def collect_merge_tasks(base_dir, serials):
    """Find episodes where all serials have overlays but no merged video yet."""
    cache_base = Path(_get_cache_base(base_dir))
    if not cache_base.is_dir():
        return []
    tasks = []
    for obj_dir in sorted(cache_base.iterdir()):
        if not obj_dir.is_dir():
            continue
        for idx_dir in sorted(obj_dir.iterdir()):
            if not idx_dir.is_dir():
                continue
            # Check all serials have overlay
            all_have = all((idx_dir / "pose_overlay" / f"{s}.avi").exists() for s in serials)
            if not all_have:
                continue
            # Skip if merged already exists
            if (idx_dir / "pose_overlay_merged" / "merged.avi").exists():
                continue
            tasks.append((str(idx_dir), obj_dir.name, idx_dir.name))
    return tasks


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base dir (network FS path)")
    parser.add_argument("--serials", nargs="+", required=True, help="Camera serial numbers")
    parser.add_argument("--mesh_dir", required=True, help="Root dir with {obj_name}/{obj_name}.obj")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--render_downscale", type=float, default=0.5,
                        help="Downscale for nvdiffrast rendering (default: 0.5)")
    args = parser.parse_args()
    print("Starting batch overlay visualization...", flush=True)

    _setup_fp_path()
    import trimesh
    import nvdiffrast.torch as dr
    from Utils import make_mesh_tensors

    torch.cuda.set_device(args.gpu)
    device = f"cuda:{args.gpu}"

    tasks = collect_tasks(args.base, args.serials, args.mesh_dir)
    if not tasks:
        print("Nothing to do (all videos already have overlays or no poses found).")
        return

    print(f"{len(tasks)} videos to process | GPU: {args.gpu}", flush=True)

    # Group by mesh to reuse mesh_tensors
    from collections import defaultdict
    tasks_by_mesh = defaultdict(list)
    for t in tasks:
        tasks_by_mesh[t[6]].append(t)  # group by mesh_path

    glctx = dr.RasterizeCudaContext()

    done = 0
    total = len(tasks)
    for mesh_path, mesh_tasks in tasks_by_mesh.items():
        obj_name = mesh_tasks[0][5]
        print(f"\nLoading mesh: {obj_name} ({mesh_path})", flush=True)

        mesh = trimesh.load(mesh_path, force="mesh")
        # Color mesh purple
        vertex_colors = np.tile(
            np.append(MESH_COLOR, 255).reshape(1, 4),
            (len(mesh.vertices), 1)
        )
        mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=vertex_colors)
        mesh_tensors = make_mesh_tensors(mesh, device=device)

        for (video_path, pose_dir, cache_dir, net_dir,
             serial, obj_name, _, idx_name) in mesh_tasks:
            done += 1
            print(f"[{done}/{total}] {obj_name}/{idx_name}/{serial}", flush=True)
            try:
                process_one_video(
                    video_path, pose_dir, cache_dir, net_dir, serial,
                    mesh_tensors, glctx, device, args.render_downscale,
                )
            except Exception as e:
                import traceback
                print(f"  Error: {e}", flush=True)
                traceback.print_exc()
                gc.collect()
                torch.cuda.empty_cache()

    print(f"\nOverlay done! {done}/{total} videos processed.", flush=True)

    # Merge multi-camera overlays
    merge_tasks = collect_merge_tasks(args.base, args.serials)
    if merge_tasks:
        print(f"\nMerging {len(merge_tasks)} episodes...", flush=True)
        for i, (cache_dir, obj_name, idx_name) in enumerate(merge_tasks, 1):
            print(f"[{i}/{len(merge_tasks)}] {obj_name}/{idx_name}", flush=True)
            try:
                merge_overlay_videos(cache_dir, args.serials)
            except Exception as e:
                print(f"  Merge error: {e}", flush=True)
        print(f"Merge done! {len(merge_tasks)} episodes merged.", flush=True)
    else:
        print("No episodes to merge (all merged or missing overlays).", flush=True)


if __name__ == "__main__":
    main()
