#!/usr/bin/env python3
"""Batch mesh overlay visualization (episode-wise).

For each episode with pose_world.npy, reads frame 0 from ALL cameras,
converts the world-frame pose to each camera frame, renders mesh overlay,
and saves per-camera PNGs + a grid image.

Pre-requisites:
    - Poses generated: python src/perception/batch_pose.py ...

Run:
    conda activate foundationpose
    python -u src/perception/batch_overlay.py \
        --base /home/mingi/paradex1/capture/eccv2026/inspire_f1 \
        --mesh_dir /home/mingi/mesh

Upload results:
    python src/perception/upload_results.py --base ...
"""

import os
import sys
import gc
import json
import math
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

CACHE_ROOT = os.path.expanduser("~/video_cache")
NETWORK_PREFIX = "/home/mingi/paradex1/capture"

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


# ── Episode collection ───────────────────────────────────────────────────────

def collect_episodes(base_dir, mesh_dir):
    """Collect episodes that have pose_world.npy."""
    cache_base = Path(_get_cache_base(base_dir))
    if not cache_base.is_dir():
        return []
    episodes = []
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
            pose_path = idx_dir / "pose" / "pose_world.npy"
            if not pose_path.exists():
                continue
            rel = str(idx_dir.relative_to(cache_base))
            net_dir = Path(base_dir) / rel
            if not (net_dir / "cam_param").is_dir():
                continue
            episodes.append((str(idx_dir), str(net_dir), obj_name,
                             str(mesh_path), idx_dir.name))
    return episodes


# ── Pose helpers ─────────────────────────────────────────────────────────────

def _to_4x4(T):
    if T.shape == (3, 4):
        T4 = np.eye(4, dtype=np.float64)
        T4[:3, :] = T
        return T4
    return T.astype(np.float64)


def get_first_valid_pose(pose_path):
    """Load pose_world.npy and return first non-NaN pose (4,4)."""
    all_poses = np.load(pose_path)
    for i in range(len(all_poses)):
        if not np.isnan(all_poses[i, 0, 0]):
            return all_poses[i]
    return None


# ── Render ───────────────────────────────────────────────────────────────────

MESH_COLOR = np.array([128, 0, 128], dtype=np.uint8)
ALPHA = 0.6


def render_overlay(rgb, pose_4x4, K, mesh_tensors, glctx, device, downscale):
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


def read_frame(video_path, frame_idx=1):
    """Read a specific frame from video, return RGB or None."""
    cap = cv2.VideoCapture(video_path)
    for _ in range(frame_idx):
        ret, _ = cap.read()
        if not ret:
            cap.release()
            return None
    ret, bgr = cap.read()
    cap.release()
    if not ret:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ── Grid ─────────────────────────────────────────────────────────────────────

def save_grid_image(overlay_images, serials, out_path):
    n = len(overlay_images)
    if n == 0:
        return
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    h, w = overlay_images[0].shape[:2]
    grid = np.ones((rows * h, cols * w, 3), dtype=np.uint8) * 255
    for idx, (img, serial) in enumerate(zip(overlay_images, serials)):
        font_scale = max(0.5, w / 800)
        thickness = max(1, w // 500)
        ts = cv2.getTextSize(serial, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        labeled = img.copy()
        cv2.rectangle(labeled, (5, 5), (15 + ts[0], 15 + ts[1]), (0, 0, 0), -1)
        cv2.putText(labeled, serial, (10, 10 + ts[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), thickness)
        r, c = divmod(idx, cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = labeled
    cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


# ── Process one episode ──────────────────────────────────────────────────────

def process_episode(cache_dir, net_dir, mesh_tensors, glctx, device, render_downscale):
    """Render overlay on frame 0 of all cameras for one episode.

    Returns number of cameras processed.
    """
    pose_path = os.path.join(cache_dir, "pose", "pose_world.npy")
    pose_world = get_first_valid_pose(pose_path)
    if pose_world is None:
        print("  No valid pose found, skipping", flush=True)
        return 0

    intrinsics, extrinsics = load_cam_param(Path(net_dir))

    video_dir = os.path.join(cache_dir, "videos")
    out_dir = os.path.join(cache_dir, "pose_overlay")
    os.makedirs(out_dir, exist_ok=True)

    overlay_images = []
    overlay_serials = []

    for vf in sorted(Path(video_dir).iterdir()):
        if vf.suffix != ".avi":
            continue
        serial = vf.stem
        if serial not in intrinsics:
            continue

        out_path = os.path.join(out_dir, f"{serial}.png")
        print(f"  {serial}: reading {vf}", flush=True)
        rgb = read_frame(str(vf))
        if rgb is None:
            continue

        K = intrinsics[serial].copy()
        T_cam = _to_4x4(extrinsics[serial])
        pose_cam = T_cam @ pose_world

        overlay_rgb = render_overlay(rgb, pose_cam, K, mesh_tensors, glctx, device, render_downscale)
        cv2.imwrite(out_path, cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
        overlay_images.append(overlay_rgb)
        overlay_serials.append(serial)

    # Save grid
    if overlay_images:
        grid_path = os.path.join(out_dir, "grid.png")
        save_grid_image(overlay_images, overlay_serials, grid_path)
        print(f"  {len(overlay_images)} cameras -> {grid_path}", flush=True)

    return len(overlay_images)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base dir (network FS path)")
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

    episodes = collect_episodes(args.base, args.mesh_dir)
    if not episodes:
        print("Nothing to do (all episodes have overlays or no poses found).")
        return

    print(f"{len(episodes)} episodes to process | GPU: {args.gpu}", flush=True)

    # Group by mesh to reuse mesh_tensors
    from collections import defaultdict
    episodes_by_mesh = defaultdict(list)
    for ep in episodes:
        episodes_by_mesh[ep[3]].append(ep)  # group by mesh_path

    glctx = dr.RasterizeCudaContext()

    done = 0
    total = len(episodes)
    for mesh_path, mesh_episodes in episodes_by_mesh.items():
        obj_name = mesh_episodes[0][2]
        print(f"\nLoading mesh: {obj_name} ({mesh_path})", flush=True)

        mesh = trimesh.load(mesh_path, force="mesh")
        vertex_colors = np.tile(
            np.append(MESH_COLOR, 255).reshape(1, 4),
            (len(mesh.vertices), 1)
        )
        mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=vertex_colors)
        mesh_tensors = make_mesh_tensors(mesh, device=device)

        for (cache_dir, net_dir, obj_name, _, idx_name) in mesh_episodes:
            done += 1
            print(f"[{done}/{total}] {obj_name}/{idx_name}", flush=True)
            try:
                process_episode(
                    cache_dir, net_dir, mesh_tensors, glctx, device,
                    args.render_downscale,
                )
            except Exception as e:
                import traceback
                print(f"  Error: {e}", flush=True)
                traceback.print_exc()
                gc.collect()
                torch.cuda.empty_cache()

    print(f"\nAll done! {done}/{total} episodes processed.", flush=True)


if __name__ == "__main__":
    main()
