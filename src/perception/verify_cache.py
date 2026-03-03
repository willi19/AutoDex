#!/usr/bin/env python3
"""Verify cache integrity — check frame counts match across pipeline stages.

Compares:
  - Cache video vs NAS video (file size)
  - Depth frames == RGB frames
  - Pose array length == RGB frames
  - Mask exists (video or first-frame image)

Usage:
    python src/perception/verify_cache.py \
        --base /home/mingi/paradex1/capture/eccv2026/inspire_f1 \
        --serials 22684755 23263780

    # Auto-delete truncated files:
    python src/perception/verify_cache.py \
        --base /home/mingi/paradex1/capture/eccv2026/inspire_f1 \
        --serials 22684755 23263780 --fix
"""

import os
import argparse
from pathlib import Path

import cv2
import numpy as np

CACHE_ROOT = os.path.expanduser("~/video_cache")
NETWORK_PREFIX = "/home/mingi/paradex1/capture"


def _get_cache_base(base_dir):
    base = str(Path(base_dir).resolve())
    if base.startswith(NETWORK_PREFIX):
        rel = base[len(NETWORK_PREFIX):].lstrip("/")
    else:
        rel = Path(base_dir).name
    return os.path.join(CACHE_ROOT, rel)


def _frame_count(path):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def verify(base_dir, serials, fix=False):
    cache_base = Path(_get_cache_base(base_dir))
    nas_base = Path(base_dir)

    if not cache_base.is_dir():
        print(f"Cache not found: {cache_base}")
        return

    n_ok = 0
    n_warn = 0
    n_err = 0
    deleted = []

    for obj_dir in sorted(cache_base.iterdir()):
        if not obj_dir.is_dir():
            continue
        for idx_dir in sorted(obj_dir.iterdir()):
            if not idx_dir.is_dir():
                continue
            obj_name = obj_dir.name
            idx_name = idx_dir.name
            rel = f"{obj_name}/{idx_name}"
            nas_dir = nas_base / obj_name / idx_name

            for serial in serials:
                issues = []

                # --- RGB video ---
                cache_vid = idx_dir / "videos" / f"{serial}.avi"
                nas_vid = nas_dir / "videos" / f"{serial}.avi"
                if not cache_vid.exists():
                    continue  # no video, nothing to check

                n_rgb = _frame_count(cache_vid)

                # Cache vs NAS size
                if nas_vid.exists():
                    c_sz = cache_vid.stat().st_size
                    n_sz = nas_vid.stat().st_size
                    if c_sz != n_sz:
                        issues.append(f"VIDEO SIZE MISMATCH: cache={c_sz} nas={n_sz}")

                # --- Depth ---
                depth_path = idx_dir / "depth" / f"{serial}.avi"
                if depth_path.exists():
                    n_dep = _frame_count(depth_path)
                    if n_dep < n_rgb:
                        issues.append(f"DEPTH TRUNCATED: {n_dep}/{n_rgb}")
                        if fix:
                            depth_path.unlink()
                            deleted.append(str(depth_path))
                    elif n_dep > n_rgb:
                        issues.append(f"DEPTH LONGER: {n_dep}/{n_rgb}")

                # --- Mask ---
                has_mask_vid = (idx_dir / "obj_mask" / f"{serial}.avi").exists()
                has_mask_img = (idx_dir / "obj_mask_first" / f"{serial}.png").exists()
                if not has_mask_vid and not has_mask_img:
                    if depth_path.exists():
                        issues.append("NO MASK")

                # Check mask video frame count
                if has_mask_vid:
                    mask_path = idx_dir / "obj_mask" / f"{serial}.avi"
                    n_mask = _frame_count(mask_path)
                    if n_mask < n_rgb:
                        issues.append(f"MASK TRUNCATED: {n_mask}/{n_rgb}")
                        if fix:
                            mask_path.unlink()
                            deleted.append(str(mask_path))

                # --- Pose ---
                pose_path = idx_dir / "pose" / f"{serial}.npy"
                if pose_path.exists():
                    poses = np.load(str(pose_path))
                    n_pose = poses.shape[0]
                    n_valid = int(np.sum(~np.isnan(poses[:, 0, 0])))
                    if n_pose < n_rgb:
                        issues.append(f"POSE SHORT: {n_pose}/{n_rgb} ({n_valid} valid)")
                        if fix:
                            pose_path.unlink()
                            deleted.append(str(pose_path))
                    elif n_valid == 0:
                        issues.append(f"POSE ALL NaN")
                        if fix:
                            pose_path.unlink()
                            deleted.append(str(pose_path))

                # --- NAS uploads ---
                nas_depth = nas_dir / "depth" / f"{serial}.avi"
                if nas_depth.exists() and depth_path.exists():
                    if nas_depth.stat().st_size != depth_path.stat().st_size:
                        issues.append("DEPTH NAS!=CACHE")

                nas_mask = nas_dir / "obj_mask" / f"{serial}.avi"
                if nas_mask.exists() and has_mask_vid:
                    cache_mask = idx_dir / "obj_mask" / f"{serial}.avi"
                    if nas_mask.stat().st_size != cache_mask.stat().st_size:
                        issues.append("MASK NAS!=CACHE")

                # --- Report ---
                if issues:
                    for issue in issues:
                        if "TRUNCATED" in issue or "MISMATCH" in issue or "SHORT" in issue or "NaN" in issue:
                            n_err += 1
                        else:
                            n_warn += 1
                    print(f"  {rel}/{serial}: {' | '.join(issues)}")
                else:
                    n_ok += 1

    print(f"\n{n_ok} OK, {n_err} errors, {n_warn} warnings")
    if deleted:
        print(f"Deleted {len(deleted)} truncated files:")
        for d in deleted:
            print(f"  {d}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base dir (network FS path)")
    parser.add_argument("--serials", nargs="+", required=True)
    parser.add_argument("--fix", action="store_true", help="Delete truncated files")
    args = parser.parse_args()
    verify(args.base, args.serials, args.fix)


if __name__ == "__main__":
    main()
