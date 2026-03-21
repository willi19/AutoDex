#!/usr/bin/env python3
"""Step 2a: Run DA3 monocular depth on all views. (conda: sam3)

Reads cam_param/ and raw images, saves depth maps.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--episode", type=str, required=True)
    args = parser.parse_args()

    capture_dir = Path(args.data_root) / args.obj / args.episode
    out_dir = capture_dir
    depth_dir = out_dir / "depth_da3"
    depth_dir.mkdir(parents=True, exist_ok=True)

    # Load camera data from cam_param/
    with open(capture_dir / "cam_param" / "intrinsics.json") as f:
        intr_raw = json.load(f)
    img_dir = capture_dir / "images"
    if not img_dir.exists():
        img_dir = capture_dir / "raw" / "images"
    serials = sorted(p.stem for p in img_dir.glob("*.png"))
    intrinsics = np.array([intr_raw[s]["intrinsics_undistort"] for s in serials], dtype=np.float32)

    with open(capture_dir / "cam_param" / "extrinsics.json") as f:
        extr_raw = json.load(f)
    extrinsics = []
    for s in serials:
        T = np.array(extr_raw[s], dtype=np.float64)
        if T.shape == (3, 4):
            T = np.vstack([T, [0, 0, 0, 1]])
        extrinsics.append(T)
    extrinsics = np.array(extrinsics, dtype=np.float32)

    images = []
    for s in serials:
        bgr = cv2.imread(str(img_dir / f"{s}.png"))
        images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    # Skip if all depths already exist
    existing = [s for s in serials if (depth_dir / f"{s}.png").exists()]
    if len(existing) == len(serials):
        print(f"  All {len(serials)} depths already exist, skipping")
        return

    print(f"Loaded {len(images)} images, running DA3 batch...")

    from autodex.perception import get_depth_da3

    t0 = time.perf_counter()
    depths = get_depth_da3(images, intrinsics=intrinsics, extrinsics=extrinsics)
    total_time = time.perf_counter() - t0

    print(f"DA3 batch inference: {total_time:.2f}s ({total_time/len(images):.3f}s/img)")

    # Save as uint16 mm PNG + colormap visualization (resize to original resolution)
    H_orig, W_orig = images[0].shape[:2]
    vis_dir = out_dir / "depth_da3_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    vis_images = []

    for i, s in enumerate(serials):
        d = depths[i]
        if d.shape[0] != H_orig or d.shape[1] != W_orig:
            d = cv2.resize(d, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)
        d_mm = (d * 1000).clip(0, 65535).astype(np.uint16)
        cv2.imwrite(str(depth_dir / f"{s}.png"), d_mm)
        print(f"  {s}: range {d.min():.3f}-{d.max():.3f}m")

        # Colormap
        d_norm = ((d - d.min()) / (d.max() - d.min() + 1e-6) * 255).astype(np.uint8)
        d_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)
        cv2.putText(d_color, f"{s} [{d.min():.2f}-{d.max():.2f}m]", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imwrite(str(vis_dir / f"{s}.png"), d_color)
        vis_images.append(d_color)

    # Grid
    if vis_images:
        cols = 4
        rows = (len(vis_images) + cols - 1) // cols
        scale = 0.25
        oh, ow = vis_images[0].shape[:2]
        th, tw = int(oh * scale), int(ow * scale)
        grid = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
        for idx, img in enumerate(vis_images):
            r, c = divmod(idx, cols)
            small = cv2.resize(img, (tw, th))
            grid[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = small
        cv2.imwrite(str(vis_dir / "grid.png"), grid)
        print(f"Depth colormap grid saved to {vis_dir / 'grid.png'}")

    timing = {"total": total_time, "per_img": total_time / len(images)}
    with open(out_dir / "timing_depth_da3.json", "w") as f:
        json.dump(timing, f, indent=2)


if __name__ == "__main__":
    main()