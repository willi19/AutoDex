#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from glob import glob

from PIL import Image


def _scale_matrix(mat, scale):
    scaled = [row[:] for row in mat]
    for r in range(2):
        for c in range(3):
            scaled[r][c] = scaled[r][c] * scale
    return scaled


def _select_intrinsics(value):
    for key in ("intrinsics", "intrinsics_adjusted", "intrinsics_undistort", "original_intrinsics"):
        if key in value:
            return value[key]
    return None


def _prepare_intrinsics(src_path, dst_path, scale):
    with open(src_path, "r") as f:
        data = json.load(f)

    out = {}
    for cam_id, value in data.items():
        base = _select_intrinsics(value)
        if base is None:
            raise ValueError(f"No intrinsics found for camera {cam_id}")

        new_value = dict(value)
        new_value["intrinsics"] = _scale_matrix(base, scale)
        for key in ("original_intrinsics", "intrinsics_adjusted", "intrinsics_undistort"):
            if key in value:
                new_value[key] = _scale_matrix(value[key], scale)
        for dim_key in ("height", "width", "dist_height"):
            if dim_key in value:
                new_value[dim_key] = int(round(value[dim_key] * scale))
        out[cam_id] = new_value

    with open(dst_path, "w") as f:
        json.dump(out, f, indent=4)


def _resize_images(src_dir, dst_dir, scale):
    os.makedirs(dst_dir, exist_ok=True)
    image_paths = sorted(glob(os.path.join(src_dir, "*.png")) + glob(os.path.join(src_dir, "*.jpg")))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {src_dir}")

    for path in image_paths:
        img = Image.open(path)
        new_size = (int(round(img.size[0] * scale)), int(round(img.size[1] * scale)))
        resized = img.resize(new_size, resample=Image.BILINEAR)
        out_path = os.path.join(dst_dir, os.path.basename(path))
        resized.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Prepare RSS2026 sequence for FoundationPose.")
    parser.add_argument("--src", required=True, help="Sequence path (contains cam_param/ and images/).")
    parser.add_argument("--dst", required=True, help="Output data directory.")
    parser.add_argument("--scale", type=float, default=0.5, help="Image scale factor.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite dst if it exists.")
    args = parser.parse_args()

    if os.path.exists(args.dst):
        if not args.overwrite:
            raise FileExistsError(f"Destination exists: {args.dst}")
        shutil.rmtree(args.dst)

    os.makedirs(args.dst, exist_ok=True)

    src_images = os.path.join(args.src, "images")
    src_cam = os.path.join(args.src, "cam_param")
    src_intr = os.path.join(src_cam, "intrinsics.json")
    src_extr = os.path.join(src_cam, "extrinsics.json")
    if not os.path.isfile(src_intr):
        raise FileNotFoundError(f"Missing intrinsics: {src_intr}")
    if not os.path.isfile(src_extr):
        raise FileNotFoundError(f"Missing extrinsics: {src_extr}")

    dst_images = os.path.join(args.dst, "images")
    _resize_images(src_images, dst_images, args.scale)
    _prepare_intrinsics(src_intr, os.path.join(args.dst, "intrinsics.json"), args.scale)
    shutil.copy2(src_extr, os.path.join(args.dst, "extrinsics.json"))


if __name__ == "__main__":
    main()
