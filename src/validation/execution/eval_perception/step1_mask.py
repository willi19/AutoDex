#!/usr/bin/env python3
"""Step 1: Run SAM3 image model on all views. (conda: sam3)

Saves masks for subsequent steps.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

AUTODEX_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(AUTODEX_ROOT))


def load_cameras(capture_dir):
    """Load all camera images and calibration."""
    capture_dir = Path(capture_dir)
    img_dir = capture_dir / "images"
    if not img_dir.exists():
        img_dir = capture_dir / "raw" / "images"
    cam_dir = capture_dir / "cam_param"

    with open(cam_dir / "intrinsics.json") as f:
        intr_raw = json.load(f)
    with open(cam_dir / "extrinsics.json") as f:
        extr_raw = json.load(f)

    serials = sorted(p.stem for p in img_dir.glob("*.png"))
    cameras = []
    for s in serials:
        rgb = cv2.cvtColor(cv2.imread(str(img_dir / f"{s}.png")), cv2.COLOR_BGR2RGB)
        K = np.array(intr_raw[s]["intrinsics_undistort"], dtype=np.float32)
        T = np.array(extr_raw[s], dtype=np.float64)
        if T.shape == (3, 4):
            T = np.vstack([T, [0, 0, 0, 1]])
        cameras.append({"serial": s, "rgb": rgb, "K": K, "T": T})
    return cameras


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--episode", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="object on the checkerboard")
    args = parser.parse_args()

    capture_dir = Path(args.data_root) / args.obj / args.episode
    out_dir = capture_dir
    mask_dir = out_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Load cameras
    cameras = load_cameras(capture_dir)
    print(f"Loaded {len(cameras)} cameras from {capture_dir}")

    # Skip if all masks already exist
    existing = [c for c in cameras if (mask_dir / f"{c['serial']}.png").exists()]
    if len(existing) == len(cameras):
        print(f"  All {len(cameras)} masks already exist, skipping")
        return

    # Run SAM3 image model
    from autodex.perception import Sam3ImageSegmentor

    t0 = time.perf_counter()
    seg = Sam3ImageSegmentor(gpu=0)
    load_time = time.perf_counter() - t0
    print(f"SAM3 image model loaded in {load_time:.2f}s")

    timing = {"load": load_time, "per_view": {}}
    n_found = 0

    for cam in cameras:
        # Skip if mask already exists
        mask_path = mask_dir / f"{cam['serial']}.png"
        if mask_path.exists():
            n_found += 1
            print(f"  {cam['serial']}: exists, skip")
            continue

        t0 = time.perf_counter()
        mask = seg.segment(cam["rgb"], args.prompt)
        infer_time = time.perf_counter() - t0
        timing["per_view"][cam["serial"]] = infer_time

        if mask is not None:
            cv2.imwrite(str(mask_path), mask)
            n_found += 1
            print(f"  {cam['serial']}: mask found ({mask.sum() // 255} px) [{infer_time:.3f}s]")
        else:
            print(f"  {cam['serial']}: no mask [{infer_time:.3f}s]")

    timing["total"] = sum(timing["per_view"].values())
    with open(out_dir / "timing_mask.json", "w") as f:
        json.dump(timing, f, indent=2)

    print(f"\nMasks: {n_found}/{len(cameras)} views, total {timing['total']:.2f}s")


if __name__ == "__main__":
    main()