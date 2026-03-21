#!/usr/bin/env python3
"""Step 2b: Stereo depth for all valid pairs using existing pipeline code. (conda: foundation_stereo)

Uses find_all_stereo_pairs + process logic from src/process/depth.py.
Saves per-left-view depth and merged depth.
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
sys.path.insert(0, str(AUTODEX_ROOT / "src" / "process"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--episode", type=str, required=True)
    args = parser.parse_args()

    capture_dir = Path(args.data_root) / args.obj / args.episode
    depth_dir = capture_dir / "depth_stereo"
    depth_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = capture_dir / "depth_stereo_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    with open(capture_dir / "cam_param" / "intrinsics.json") as f:
        intr_raw = json.load(f)
    with open(capture_dir / "cam_param" / "extrinsics.json") as f:
        extr_raw = json.load(f)

    img_dir = capture_dir / "images"
    if not img_dir.exists():
        img_dir = capture_dir / "raw" / "images"
    serials = sorted(p.stem for p in img_dir.glob("*.png"))

    intrinsics = {s: np.array(intr_raw[s]["intrinsics_undistort"], dtype=np.float64) for s in serials}
    extrinsics = {}
    for s in serials:
        T = np.array(extr_raw[s], dtype=np.float64)
        if T.shape == (3, 4):
            T = np.vstack([T, [0, 0, 0, 1]])
        extrinsics[s] = T

    # Import from existing pipeline code
    from autodex.perception.depth import StereoDepthTRT, build_rectify_maps, disp_to_depth_left, find_all_stereo_pairs

    t0 = time.perf_counter()
    trt = StereoDepthTRT()
    H_trt, W_trt = trt.H_trt, trt.W_trt
    print(f"TRT loaded in {time.perf_counter() - t0:.2f}s")

    # Find all pairs using rig-based adjacency (same as src/process/depth.py)
    all_pairs = find_all_stereo_pairs(capture_dir, serials, intrinsics, extrinsics)
    print(f"Found {len(all_pairs)} stereo pairs")

    H_img, W_img = cv2.imread(str(img_dir / f"{serials[0]}.png")).shape[:2]
    merged = {s: np.zeros((H_img, W_img), dtype=np.float32) for s in serials}
    pair_info = []

    for left_s, right_s, baseline_m in all_pairs:
        left_rgb = cv2.cvtColor(cv2.imread(str(img_dir / f"{left_s}.png")), cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(cv2.imread(str(img_dir / f"{right_s}.png")), cv2.COLOR_BGR2RGB)

        K_left, K_right = intrinsics[left_s], intrinsics[right_s]
        T_left, T_right = extrinsics[left_s], extrinsics[right_s]

        try:
            result = build_rectify_maps(K_left, K_right, T_left, T_right, (W_img, H_img),
                                        capture_dir=capture_dir)
        except Exception as e:
            print(f"  {left_s}+{right_s}: rectify failed ({e})")
            continue

        map_left, map_right, R1, R2, f_rect, cx_rect, cy_rect, baseline, rect_size, disp_offset = result
        W_rect, H_rect = rect_size

        if baseline < 0.01 or f_rect <= 0:
            print(f"  {left_s}+{right_s}: SKIP tiny baseline")
            continue

        aspect = max(W_rect, H_rect) / max(min(W_rect, H_rect), 1)
        if aspect > 2.5:
            print(f"  {left_s}+{right_s}: SKIP bad aspect {aspect:.1f}")
            continue

        left_rect = cv2.remap(left_rgb, map_left[0], map_left[1], cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_rgb, map_right[0], map_right[1], cv2.INTER_LINEAR)

        # TRT inference
        disp = trt._run_trt(left_rect, right_rect)

        # Disparity → left-view depth (un-rectified, rz-corrected)
        depth_left = disp_to_depth_left(
            disp, f_rect, baseline,
            K_left, R1, cx_rect, cy_rect,
            W_img, H_img, W_rect, H_rect, H_trt, W_trt,
        )

        # Save raw left depth
        d_mm = (depth_left * 1000).clip(0, 65535).astype(np.uint16)
        cv2.imwrite(str(raw_dir / f"{left_s}.png"), d_mm)

        # Merge into left camera's depth (closest wins)
        new_valid = depth_left > 0.001
        old_valid = merged[left_s] > 0.001
        merged[left_s][new_valid & ~old_valid] = depth_left[new_valid & ~old_valid]
        both = new_valid & old_valid
        if both.any():
            closer = depth_left[both] < merged[left_s][both]
            idx = np.where(both)
            merged[left_s][idx[0][closer], idx[1][closer]] = depth_left[both][closer]

        valid = depth_left > 0.001
        print(f"  {left_s}+{right_s}: left {valid.sum()} px ({valid.sum()/depth_left.size*100:.0f}%), "
              f"range {depth_left[valid].min():.3f}-{depth_left[valid].max():.3f}m")
        pair_info.append({"pair": [left_s, right_s], "left": left_s})

    # Save merged depth
    for s in serials:
        d = merged[s]
        if (d > 0.001).any():
            d_mm = (d * 1000).clip(0, 65535).astype(np.uint16)
            cv2.imwrite(str(depth_dir / f"{s}.png"), d_mm)

    with open(capture_dir / "timing_depth_stereo.json", "w") as f:
        json.dump({"pairs": pair_info, "n_pairs": len(pair_info)}, f, indent=2)

    print(f"\nDone. {len(pair_info)} pairs processed")


if __name__ == "__main__":
    main()
