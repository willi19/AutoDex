import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image


def _ensure_depth_anything_on_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    da3_path = os.path.join(repo_root, "Depth-Anything-3")
    if da3_path not in sys.path:
        sys.path.insert(0, da3_path)


def _load_depth_anything(model_id: str, device: str):
    _ensure_depth_anything_on_path()
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(model_id)
    model = model.to(device=device)
    model.eval()
    return model


def _collect_images(images_dir: str):
    image_paths = []
    for ext in ("*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG"):
        image_paths.extend(sorted(glob.glob(os.path.join(images_dir, ext))))
    return image_paths


def run_depth_anything(
    data_dir: str,
    model_id: str,
    device: str,
    export_dir: str,
    export_format: str,
):
    images_dir = os.path.join(data_dir, "images")
    image_path_list = _collect_images(images_dir)
    if not image_path_list:
        raise FileNotFoundError(f"No images found in {images_dir}")

    extrinsics_path = os.path.join(data_dir, "extrinsics.json")
    intrinsics_path = os.path.join(data_dir, "intrinsics.json")

    with open(extrinsics_path, "r") as f:
        extrinsics_dict = json.load(f)
    with open(intrinsics_path, "r") as f:
        intrinsics_dict = json.load(f)

    extrinsics_list = []
    intrinsics_list = []
    valid_image_paths = []
    original_sizes = []

    for image_path in image_path_list:
        image_key = os.path.splitext(os.path.basename(image_path))[0]
        if image_key in extrinsics_dict and image_key in intrinsics_dict:
            img = Image.open(image_path)
            orig_w, orig_h = img.size
            original_sizes.append((orig_h, orig_w))

            ext_3x4 = np.array(extrinsics_dict[image_key])
            ext_4x4 = np.vstack([ext_3x4, np.array([0, 0, 0, 1])])
            extrinsics_list.append(ext_4x4)

            intrinsics_list.append(np.array(intrinsics_dict[image_key]["intrinsics"]))
            valid_image_paths.append(image_path)
        else:
            print(f"Warning: No extrinsics/intrinsics found for {image_key}, skipping...")

    extrinsics_array = np.array(extrinsics_list)
    intrinsics_array = np.array(intrinsics_list)

    if not valid_image_paths:
        raise RuntimeError("No images with matching intrinsics/extrinsics found.")

    print(f"Loaded {len(valid_image_paths)} images with corresponding extrinsics and intrinsics")
    print(f"Extrinsics shape: {extrinsics_array.shape}")
    print(f"Intrinsics shape: {intrinsics_array.shape}")

    model = _load_depth_anything(model_id=model_id, device=device)
    infer_start = time.perf_counter()
    prediction = model.inference(
        image=valid_image_paths,
        extrinsics=extrinsics_array,
        intrinsics=intrinsics_array,
        export_dir=export_dir,
        export_format=export_format,
    )
    infer_elapsed = time.perf_counter() - infer_start
    print(f"[step2] inference: {infer_elapsed:.3f}s")

    depth_output_dir = os.path.join(data_dir, "depth")
    os.makedirs(depth_output_dir, exist_ok=True)

    print(f"\nSaving depth images to {depth_output_dir}...")
    save_start = time.perf_counter()
    for i, image_path in enumerate(valid_image_paths):
        frame_start = time.perf_counter()
        depth_map = prediction.depth[i]
        orig_h, orig_w = original_sizes[i]
        depth_map = cv2.resize(
            depth_map,
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST,
        )
        depth_mm = depth_map * 1000.0
        depth_uint16 = np.clip(depth_mm, 0, 65535).astype(np.uint16)

        image_key = os.path.splitext(os.path.basename(image_path))[0]
        depth_output_path = os.path.join(depth_output_dir, f"{image_key}.png")
        cv2.imwrite(depth_output_path, depth_uint16)
        elapsed = time.perf_counter() - frame_start
        print(
            f"Saved depth image: {depth_output_path} "
            f"(resized from {prediction.depth[i].shape} to {orig_h}x{orig_w}) "
            f"[{elapsed:.3f}s]"
        )

    save_elapsed = time.perf_counter() - save_start
    print(f"[step2] save: {save_elapsed:.3f}s for {len(valid_image_paths)} images")
    print(f"\nSuccessfully saved {len(valid_image_paths)} depth images to {depth_output_dir}")


def build_argparser():
    parser = argparse.ArgumentParser(description="Step 2: Depth Anything 3 depth generation")
    parser.add_argument(
        "--data-dir",
        default="/media/gunhee/DATA/robothome/FoundationPose/demo_data/baby_beaker_demo",
        help="Path to demo data directory containing images/",
    )
    parser.add_argument(
        "--model-id",
        default="depth-anything/DA3NESTED-GIANT-LARGE",
        help="Depth Anything 3 model identifier",
    )
    parser.add_argument("--device", default="cuda", help="Torch device for Depth Anything 3")
    parser.add_argument(
        "--export-dir",
        default="./output",
        help="Export directory for Depth Anything 3 outputs",
    )
    parser.add_argument(
        "--export-format",
        default="mini_npz-glb",
        help="Export format for Depth Anything 3 outputs",
    )
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    run_depth_anything(
        data_dir=args.data_dir,
        model_id=args.model_id,
        device=args.device,
        export_dir=args.export_dir,
        export_format=args.export_format,
    )


if __name__ == "__main__":
    main()
