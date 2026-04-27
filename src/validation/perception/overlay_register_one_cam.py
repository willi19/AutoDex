#!/usr/bin/env python3
"""Subprocess worker: register one cam with FoundationPose, dump pose to file.

Used by overlay_compute_ref_pre.py to isolate each register call in its own
process so GPU memory is fully released between cams (avoids fragmentation OOM).

Usage:
    python overlay_register_one_cam.py \\
        --mesh /path/to/mesh.obj \\
        --image-path image.png \\
        --depth-path depth_array.npy \\
        --mask-path mask.png \\
        --K-json K.json \\
        --out pose_cam.npy
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--image-path", required=True)
    ap.add_argument("--depth-path", required=True)
    ap.add_argument("--mask-path", required=True)
    ap.add_argument("--K-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iteration", type=int, default=5)
    ap.add_argument("--downscale", type=float, default=0.5,
                    help="Downscale factor for inputs (matches PerceptionPipeline daemon).")
    args = ap.parse_args()

    bgr = cv2.imread(args.image_path)
    if bgr is None:
        raise FileNotFoundError(args.image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth = np.load(args.depth_path).astype(np.float32)
    H, W = rgb.shape[:2]
    if depth.shape[:2] != (H, W):
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
    mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(args.mask_path)
    K = np.asarray(json.load(open(args.K_json)), dtype=np.float32)

    # Downscale (matches PerceptionPipeline FPose daemon behavior).
    if args.downscale != 1.0:
        nH, nW = int(H * args.downscale), int(W * args.downscale)
        rgb = cv2.resize(rgb, (nW, nH), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (nW, nH), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (nW, nH), interpolation=cv2.INTER_NEAREST)
        K = K.copy()
        K[0, :] *= args.downscale
        K[1, :] *= args.downscale

    from autodex.perception.pose import PoseTracker
    tracker = PoseTracker(args.mesh, device_id=0)
    pose_cam = tracker.init(rgb=rgb, depth=depth, mask=mask, K=K, iteration=args.iteration)
    np.save(args.out, np.asarray(pose_cam, dtype=np.float64))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
