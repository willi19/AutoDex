#!/usr/bin/env python3
"""Fix videos with mixed distorted/undistorted frames (raw_video_processor.py bug).

Per-frame detection:
  1. invalid_mask region (pixels outside undistort remap range) black?
     → already undistorted → compare with undistorted checkerboard template
       → match: dropped frame → replace with clean checkerboard
       → no match: normal undistorted frame → keep
  2. Not black → distorted → apply undistort

Flow:
  1. Collect all (experiment, serial) pairs
  2. NAS: rename videos/ → videos_tmp/ per experiment
  3. Global thread pool: download → fix → upload per video
  4. Clean local cache per video after upload

Usage:
    python src/process/fix_video_undistort.py \
        --exp ~/shared_data/AutoDex/experiment/selected_100/inspire/attached_container/20260405_235218

    python src/process/fix_video_undistort.py \
        --base ~/shared_data/AutoDex/experiment/selected_100/inspire/ --workers 28

    python src/process/fix_video_undistort.py --base ... --dry-run
"""
import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.expanduser("~/paradex"))
from paradex.image.undistort import precomute_undistort_map, apply_undistort_map

NAS_BASE = Path.home() / "shared_data" / "AutoDex" / "experiment"
CACHE_BASE = Path.home() / "cache" / "fix_video_undistort"


def prepare_camera(intrinsic, H=1536, W=2048):
    """Precompute undistort maps, invalid mask, and checkerboard template."""
    intr = dict(intrinsic)
    for k in ("original_intrinsics", "intrinsics_undistort", "dist_params"):
        if k in intr and not isinstance(intr[k], np.ndarray):
            intr[k] = np.array(intr[k])
    _, mapx, mapy = precomute_undistort_map(intr)

    invalid_mask = (mapx < 0) | (mapx >= W) | (mapy < 0) | (mapy >= H)

    checker_raw = np.zeros((H, W, 3), dtype=np.uint8)
    checker_raw[::2, :] = 255
    checker_undist = apply_undistort_map(checker_raw, mapx, mapy)

    return mapx, mapy, invalid_mask, checker_raw, checker_undist


def classify_frame(frame, invalid_mask, checker_undist):
    """Classify: 'distorted', 'undistorted', or 'dropped'."""
    inv_mean = frame[invalid_mask].mean() if invalid_mask.sum() > 0 else 999

    if inv_mean < 15:
        diff_checker = cv2.absdiff(frame, checker_undist).mean()
        if diff_checker < 30:
            return "dropped"
        return "undistorted"

    return "distorted"


def fix_video(src_path, dst_path, intrinsic, serial=""):
    """Fix one video. Returns (n_fixed, n_kept, n_dropped, n_total, dropped_frames)."""
    mapx, mapy, invalid_mask, checker_raw, checker_undist = prepare_camera(intrinsic)

    cap = cv2.VideoCapture(str(src_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    checker_raw_sized = np.zeros((H, W, 3), dtype=np.uint8)
    checker_raw_sized[::2, :] = 255

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(dst_path), fourcc, fps, (W, H))

    n_fixed = 0
    n_kept = 0
    n_dropped = 0
    dropped_frames = []

    for fidx in tqdm(range(n_frames), desc=f"  {serial}", unit="f", leave=False):
        ret, frame = cap.read()
        if not ret:
            break

        cls = classify_frame(frame, invalid_mask, checker_undist)

        if cls == "distorted":
            writer.write(apply_undistort_map(frame, mapx, mapy))
            n_fixed += 1
        elif cls == "dropped":
            writer.write(checker_raw_sized)
            n_dropped += 1
            dropped_frames.append(fidx)
        else:
            writer.write(frame)
            n_kept += 1

    cap.release()
    writer.release()
    return n_fixed, n_kept, n_dropped, n_frames, dropped_frames


def get_frame_count(vid_path):
    cap = cv2.VideoCapture(str(vid_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def find_experiments(base):
    base = Path(base)
    results = []
    for p in base.rglob("cam_param"):
        if not p.is_dir():
            continue
        exp = p.parent
        if (exp / "videos").is_dir() or (exp / "videos_tmp").is_dir():
            results.append(exp)
    return sorted(results)


def collect_jobs(experiments):
    """Collect all (exp_dir, serial, intrinsic) jobs and do renames.

    Returns list of (nas_src, nas_dst, cache_src, cache_dst, intrinsic, serial).
    """
    jobs = []
    for exp_dir in experiments:
        nas_vid_dir = exp_dir / "videos"
        nas_tmp_dir = exp_dir / "videos_tmp"
        param_path = exp_dir / "cam_param" / "intrinsics.json"

        if not param_path.exists():
            continue
        if not nas_vid_dir.exists() and not nas_tmp_dir.exists():
            continue

        with open(param_path) as f:
            intr_raw = json.load(f)

        # Rename videos/ → videos_tmp/
        if not nas_tmp_dir.exists():
            nas_vid_dir.rename(nas_tmp_dir)
        nas_vid_dir.mkdir(exist_ok=True)

        # Cache dir
        try:
            rel = exp_dir.relative_to(NAS_BASE)
        except ValueError:
            rel = Path(exp_dir.name)
        cache_dir = CACHE_BASE / rel
        cache_dir.mkdir(parents=True, exist_ok=True)

        for p in sorted(nas_tmp_dir.glob("*.avi")):
            serial = p.stem
            nas_src = nas_tmp_dir / f"{serial}.avi"
            nas_dst = nas_vid_dir / f"{serial}.avi"
            cache_src = cache_dir / f"{serial}_src.avi"
            cache_dst = cache_dir / f"{serial}_dst.avi"
            intrinsic = intr_raw.get(serial)

            jobs.append((nas_src, nas_dst, cache_src, cache_dst, intrinsic, serial))

    return jobs


def process_one(job):
    """Download → fix → upload one video."""
    nas_src, nas_dst, cache_src, cache_dst, intrinsic, serial = job

    # Skip: NAS already has fixed video with matching frame count
    if nas_dst.exists() and get_frame_count(nas_dst) == get_frame_count(nas_src):
        return None

    # No intrinsics → just copy
    if intrinsic is None:
        shutil.copy2(str(nas_src), str(nas_dst))
        return None

    n_expected = get_frame_count(nas_src)

    # Skip fix if local cache already has fixed video
    if cache_dst.exists() and get_frame_count(cache_dst) == n_expected:
        shutil.copy2(str(cache_dst), str(nas_dst))
        cache_src.unlink(missing_ok=True)
        cache_dst.unlink(missing_ok=True)
        return None

    # Download (NAS → local), skip if already downloaded
    if not cache_src.exists():
        shutil.copy2(str(nas_src), str(cache_src))

    # Fix (local CPU)
    n_fixed, n_kept, n_dropped, n_total, dropped_frames = fix_video(
        cache_src, cache_dst, intrinsic, serial,
    )

    # Upload (local → NAS)
    shutil.copy2(str(cache_dst), str(nas_dst))

    # Clean local cache
    cache_src.unlink(missing_ok=True)
    cache_dst.unlink(missing_ok=True)

    return (serial, str(nas_src.parent.parent.name), dropped_frames) if dropped_frames else None


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--exp", type=str)
    group.add_argument("--base", type=str)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    CACHE_BASE.mkdir(parents=True, exist_ok=True)

    if args.exp:
        experiments = [Path(args.exp).expanduser()]
    else:
        experiments = find_experiments(Path(args.base).expanduser())

    print(f"Found {len(experiments)} experiments")

    if args.dry_run:
        total = 0
        for exp in experiments:
            src = exp / "videos_tmp" if (exp / "videos_tmp").exists() else exp / "videos"
            if src.exists():
                total += len(list(src.glob("*.avi")))
        print(f"Total: {total} videos")
        return

    # Collect all jobs (and do renames)
    jobs = collect_jobs(experiments)
    print(f"Total: {len(jobs)} videos across {len(experiments)} experiments")

    all_dropped = []
    total_bar = tqdm(total=len(jobs), desc="total", unit="vid")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_one, job) for job in jobs]
        for future in futures:
            result = future.result()
            if result:
                all_dropped.append(result)
            total_bar.update(1)

    total_bar.close()

    if all_dropped:
        print(f"\nDROPPED FRAMES FOUND:")
        for serial, exp_name, frames in all_dropped:
            print(f"  {exp_name}/{serial}: frames {frames}")

    print(f"\nDone!")


if __name__ == "__main__":
    main()