#!/usr/bin/env python3
"""Run SAM3 mask + stereo depth on a single capture directory.

Caches videos/images from NFS to local disk, then runs perception.

Outputs (under {capture_dir}/perception_output/):
    obj_mask/{serial}.avi          — binary mask video (grayscale, 0/255)
    obj_mask_debug/{serial}.avi    — green overlay debug video
    depth/{serial}.png             — uint16 PNG depth (millimeters)
    depth_debug/
        colormap_{serial}.png      — depth colormap overlay on RGB
        grid.png                   — all cameras in one grid

Usage:
    # Step 1: Cache videos locally (fast local I/O for processing)
    python src/demo/run_perception.py \
        --capture_dir /home/mingi/shared_data/.../attached_container/20260121_163413 \
        --steps cache

    # Step 2: SAM3 mask
    conda activate sam3
    python src/demo/run_perception.py \
        --capture_dir /home/mingi/shared_data/.../attached_container/20260121_163413 \
        --steps mask

    # Step 3: Stereo depth
    conda activate foundation_stereo
    python src/demo/run_perception.py \
        --capture_dir /home/mingi/shared_data/.../attached_container/20260121_163413 \
        --steps depth

    # Or all at once:
    python src/demo/run_perception.py \
        --capture_dir /home/mingi/shared_data/.../attached_container/20260121_163413 \
        --steps cache mask depth
"""

import argparse
import math
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

AUTODEX_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AUTODEX_ROOT))

CACHE_ROOT = Path.home() / "video_cache"


# ── Cache step ───────────────────────────────────────────────────────────────

def get_cache_dir(capture_dir: Path) -> Path:
    """Map NFS capture_dir to a local cache path under ~/video_cache/."""
    # /home/mingi/shared_data/RSS2026_Mingi/.../obj/timestamp → video_cache/RSS2026_Mingi/.../obj/timestamp
    network_prefix = "/home/mingi/shared_data/"
    resolved = str(capture_dir.resolve())
    if resolved.startswith(network_prefix):
        rel = resolved[len(network_prefix):]
    else:
        # Already local or unknown mount — use last 2 path components
        rel = str(Path(capture_dir.parent.name) / capture_dir.name)
    return CACHE_ROOT / rel


def run_cache(capture_dir: Path) -> Path:
    """Copy videos, images, and cam_param from NFS to local cache. Returns cache dir."""
    cache_dir = get_cache_dir(capture_dir)
    print(f"[cache] {capture_dir} → {cache_dir}")

    for subdir in ["videos", "images", "cam_param"]:
        src = capture_dir / subdir
        dst = cache_dir / subdir
        if not src.exists():
            continue
        dst.mkdir(parents=True, exist_ok=True)

        files = list(src.iterdir())
        existing = set(p.name for p in dst.iterdir()) if dst.exists() else set()
        to_copy = [f for f in files if f.name not in existing]

        if not to_copy:
            print(f"[cache] {subdir}/: {len(files)} files (all cached)")
            continue

        print(f"[cache] {subdir}/: copying {len(to_copy)}/{len(files)} files ...", end=" ", flush=True)
        t0 = time.perf_counter()
        for f in to_copy:
            tmp = dst / (f.name + ".tmp")
            shutil.copy2(str(f), str(tmp))
            tmp.rename(dst / f.name)
        print(f"done ({time.perf_counter() - t0:.1f}s)")

    return cache_dir


# ── Mask step ────────────────────────────────────────────────────────────────

def run_mask(capture_dir: Path, prompt: str, gpu: int, output_dir: Path):
    from autodex.perception import Sam3Segmentor, save_mask_video

    video_dir = capture_dir / "videos"
    serials = sorted(p.stem for p in video_dir.glob("*.avi"))
    print(f"[mask] {len(serials)} cameras, prompt='{prompt}'")

    seg = Sam3Segmentor(gpu=gpu)

    for i, serial in enumerate(serials):
        video_path = str(video_dir / f"{serial}.avi")
        print(f"[mask] [{i+1}/{len(serials)}] {serial} ...", end=" ", flush=True)

        t0 = time.perf_counter()
        masks = seg.segment_video(
            video_path, prompt,
            fallback_prompts=["object"],
        )
        dt = time.perf_counter() - t0

        if masks is None:
            print(f"FAILED (no mask found) {dt:.1f}s")
            continue

        # Get FPS from source video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        save_mask_video(masks, video_path, str(output_dir), serial, fps, save_debug=True)
        print(f"OK ({len(masks)} frames, {dt:.1f}s)")

    print(f"[mask] Done. Output: {output_dir / 'obj_mask'}")


# ── Depth step ───────────────────────────────────────────────────────────────

def run_depth(capture_dir: Path, engine: str, output_dir: Path):
    from autodex.perception.depth import StereoDepthTRT, encode_depth_uint16

    print("[depth] Loading TRT engine...")
    t0 = time.perf_counter()
    stereo = StereoDepthTRT(engine)
    print(f"[depth] Engine loaded ({time.perf_counter() - t0:.1f}s)")

    stereo_debug_dir = str(output_dir / "depth_debug" / "stereo_pairs")
    print(f"[depth] Running estimate_capture (debug → {stereo_debug_dir}) ...")
    t0 = time.perf_counter()
    depths = stereo.estimate_capture(str(capture_dir), debug_dir=stereo_debug_dir)
    dt = time.perf_counter() - t0
    print(f"[depth] {len(depths)} cameras done ({dt:.1f}s)")

    # Save depth maps as uint16 PNG (mm)
    depth_dir = output_dir / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)

    for serial, depth_map in depths.items():
        depth_mm = np.clip(depth_map * 1000.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(depth_dir / f"{serial}.png"), depth_mm)

    print(f"[depth] Saved depth maps: {depth_dir}")

    # Debug: colormap overlay on RGB
    _save_depth_debug(capture_dir, depths, output_dir)


def _save_depth_debug(capture_dir: Path, depths: dict, output_dir: Path):
    """Save depth colormap overlays and a grid image."""
    debug_dir = output_dir / "depth_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    images_dir = capture_dir / "images"

    tiles = []
    serials = sorted(depths.keys())
    for serial in serials:
        depth = depths[serial]
        # Load RGB
        img_path = images_dir / f"{serial}.png"
        if not img_path.exists():
            continue
        bgr = cv2.imread(str(img_path))
        H, W = bgr.shape[:2]

        # Colormap
        valid = depth > 0.01
        if not valid.any():
            tiles.append(bgr)
            continue
        d_vis = depth.copy()
        d_min, d_max = depth[valid].min(), np.percentile(depth[valid], 98)
        d_vis = np.clip((d_vis - d_min) / (d_max - d_min + 1e-6), 0, 1)
        d_vis[~valid] = 0
        cmap = cv2.applyColorMap((d_vis * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        cmap[~valid] = 0

        overlay = cv2.addWeighted(bgr, 0.4, cmap, 0.6, 0)
        overlay[~valid] = bgr[~valid]

        cv2.imwrite(str(debug_dir / f"colormap_{serial}.png"), overlay)

        # Small tile for grid
        small = cv2.resize(overlay, (W // 4, H // 4))
        cv2.putText(small, serial, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        tiles.append(small)

    # Grid
    if tiles:
        n = len(tiles)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        h, w = tiles[0].shape[:2]
        grid = np.ones((rows * h, cols * w, 3), dtype=np.uint8) * 30
        for idx, tile in enumerate(tiles):
            r, c = divmod(idx, cols)
            th, tw = tile.shape[:2]
            grid[r * h : r * h + th, c * w : c * w + tw] = tile
        grid_path = debug_dir / "grid.png"
        cv2.imwrite(str(grid_path), grid)
        print(f"[depth] Debug grid: {grid_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run SAM3 mask + stereo depth on one capture")
    parser.add_argument("--capture_dir", type=str, required=True,
                        help="Capture directory with cam_param/, videos/, images/")
    parser.add_argument("--steps", nargs="+", default=["cache", "mask", "depth"],
                        choices=["cache", "mask", "depth"],
                        help="Which steps to run (default: cache mask depth)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="SAM3 text prompt (default: inferred from parent dir name)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--engine", type=str, default=None,
                        help="TRT engine path (default: thirdparty/FoundationStereo/output/foundation_stereo_448x672.engine)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: {capture_dir}/perception_output)")
    args = parser.parse_args()

    capture_dir = Path(args.capture_dir)

    # Cache step: copy NFS → local, use cached dir for processing
    work_dir = capture_dir
    if "cache" in args.steps:
        print(f"\n{'='*60}")
        print("Cache (NFS → local)")
        print(f"{'='*60}")
        work_dir = run_cache(capture_dir)
    elif capture_dir.resolve().as_posix().startswith("/home/mingi/shared_data"):
        # Auto-use existing cache if available
        cached = get_cache_dir(capture_dir)
        if cached.exists():
            work_dir = cached
            print(f"[cache] Using existing cache: {work_dir}")

    # Output goes alongside videos/, images/, cam_param/ — not in a subdirectory
    output_dir = Path(args.output_dir) if args.output_dir else work_dir

    prompt = args.prompt or "object on the checkerboard"
    print(f"Capture: {capture_dir}")
    print(f"Work:    {work_dir}")
    print(f"Output:  {output_dir}")
    print(f"Steps:   {args.steps}")

    if "mask" in args.steps:
        print(f"\n{'='*60}")
        print(f"SAM3 Mask (prompt='{prompt}')")
        print(f"{'='*60}")
        run_mask(work_dir, prompt, args.gpu, output_dir)

    if "depth" in args.steps:
        print(f"\n{'='*60}")
        print("Stereo Depth (TRT)")
        print(f"{'='*60}")
        run_depth(work_dir, args.engine, output_dir)

    print(f"\nAll done! Results in: {output_dir}")


if __name__ == "__main__":
    main()