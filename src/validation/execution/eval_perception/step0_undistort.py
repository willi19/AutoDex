#!/usr/bin/env python3
"""Step 0: Undistort raw images if not already done.

raw/images/ → images/ (undistorted using intrinsics.json dist_params)
Skips if images/ already exists.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def undistort_img(img, intrinsic):
    return cv2.undistort(
        img,
        np.array(intrinsic["original_intrinsics"]),
        np.array(intrinsic["dist_params"]),
        None,
        np.array(intrinsic["intrinsics_undistort"]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--obj", type=str, default=None)
    parser.add_argument("--episode", type=str, default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root)

    # Build episode list
    episodes = []
    if args.obj and args.episode:
        episodes.append(data_root / args.obj / args.episode)
    elif args.obj:
        episodes = sorted(data_root / args.obj / ep for ep in (data_root / args.obj).iterdir() if ep.is_dir())
    else:
        for obj_dir in sorted(data_root.iterdir()):
            if not obj_dir.is_dir() or obj_dir.name == "eval_output":
                continue
            for ep_dir in sorted(obj_dir.iterdir()):
                if ep_dir.is_dir():
                    episodes.append(ep_dir)

    for ep_dir in episodes:
        raw_dir = ep_dir / "raw" / "images"
        out_dir = ep_dir / "images"

        if not raw_dir.exists():
            print(f"SKIP {ep_dir.relative_to(data_root)}: no raw/images/")
            continue

        if out_dir.exists() and any(out_dir.glob("*.png")):
            print(f"SKIP {ep_dir.relative_to(data_root)}: images/ already exists")
            continue

        # Load intrinsics
        intr_path = ep_dir / "cam_param" / "intrinsics.json"
        if not intr_path.exists():
            print(f"SKIP {ep_dir.relative_to(data_root)}: no intrinsics.json")
            continue

        with open(intr_path) as f:
            intrinsics = json.load(f)

        out_dir.mkdir(parents=True, exist_ok=True)

        raw_files = sorted(raw_dir.glob("*.png"))
        print(f"{ep_dir.relative_to(data_root)}: undistorting {len(raw_files)} images")

        for img_path in raw_files:
            serial = img_path.stem
            if serial not in intrinsics:
                print(f"  WARN: {serial} not in intrinsics.json, copying as-is")
                img = cv2.imread(str(img_path))
                cv2.imwrite(str(out_dir / img_path.name), img)
                continue

            img = cv2.imread(str(img_path))
            undistorted = undistort_img(img, intrinsics[serial])
            cv2.imwrite(str(out_dir / img_path.name), undistorted)

        print(f"  → saved to {out_dir}")


if __name__ == "__main__":
    main()