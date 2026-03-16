#!/usr/bin/env python3
"""
Step 1: Mask generation validation (SAM3 / YOLO-E)

For each object in object_info.json, generates masks for all camera views
and saves debug overlays.

Output structure:
    output_dir/
    ├── images/                  # undistorted images + camera_data.npz
    ├── objects/{name}/
    │   ├── masks/               # binary masks per camera
    │   └── masks_debug/         # overlay images
    └── masks_combined/          # all objects colored on one image + grid

Usage (SAM3):
    conda activate sam3
    python src/validation/perception/step1_mask.py \
        --data_dir ~/shared_data/.../20260214_231802

Usage (YOLO-E):
    conda activate foundationpose
    python src/validation/perception/step1_mask.py \
        --data_dir ~/shared_data/.../20260214_231802 \
        --method yoloe
"""

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

AUTODEX_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(AUTODEX_ROOT))

from autodex.perception import get_mask_yoloe, get_mask_sam3

logging.basicConfig(level=logging.INFO, format="[mask] [%(levelname)s] %(message)s")

OBJECT_COLORS = [
    (255,  50,  50),
    ( 50, 255,  50),
    ( 50, 100, 255),
    (255, 220,  50),
    (255,  50, 255),
    ( 50, 255, 255),
    (255, 150,  50),
    (180,  50, 255),
]


def load_data(data_dir: Path):
    """Load images and camera params via paradex."""
    from paradex.image.image_dict import ImageDict

    img_dict = ImageDict.from_path(str(data_dir))
    img_dict = img_dict.undistort()
    serials = list(img_dict.images.keys())

    intrinsics, extrinsics = {}, {}
    for s in serials:
        intrinsics[s] = np.array(img_dict.intrinsic[s]["intrinsics_undistort"])
        ext = np.array(img_dict.extrinsic[s])
        extrinsics[s] = np.vstack([ext, [0, 0, 0, 1]])

    return img_dict, serials, intrinsics, extrinsics


def save_camera_data(output_dir: Path, serials, intrinsics, extrinsics, img_dict):
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    for s in serials:
        cv2.imwrite(str(images_dir / f"{s}.png"), img_dict.images[s])

    np.savez(
        str(output_dir / "camera_data.npz"),
        serials=np.array(serials),
        intrinsics=np.array([intrinsics[s] for s in serials]),
        extrinsics=np.array([extrinsics[s] for s in serials]),
    )
    logging.info(f"Saved {len(serials)} images + camera_data.npz")


def run_sam3(img_dict, serials, object_info, output_dir, device, confidence):
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    t_model = time.perf_counter()
    model = build_sam3_image_model(device=device)
    processor = Sam3Processor(model, device=device, confidence_threshold=confidence)
    model_load_s = time.perf_counter() - t_model

    obj_names = list(object_info.keys())
    prompts = [object_info[n]["text"] for n in obj_names]

    t_io_total = 0.0
    t_set_image_total = 0.0
    t_prompt_total = 0.0

    start = time.perf_counter()
    for serial in serials:
        t0 = time.perf_counter()

        t_io = time.perf_counter()
        image_bgr = img_dict.images[serial]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        t_io_total += time.perf_counter() - t_io

        t_si = time.perf_counter()
        state = processor.set_image(Image.fromarray(image_rgb))
        t_set_image_total += time.perf_counter() - t_si

        for obj_name, prompt in zip(obj_names, prompts):
            t_p = time.perf_counter()
            processor.reset_all_prompts(state)
            output = processor.set_text_prompt(state=state, prompt=prompt)
            t_prompt_total += time.perf_counter() - t_p

            masks_t = output.get("masks")
            scores_t = output.get("scores")
            if masks_t is None or masks_t.numel() == 0:
                H, W = image_bgr.shape[:2]
                mask_np = np.zeros((H, W), dtype=np.uint8)
            else:
                idx = int(torch.argmax(scores_t).item()) if scores_t is not None and scores_t.numel() > 0 else 0
                m = masks_t[idx]
                if m.ndim == 3:
                    m = m[0]
                mask_np = (m.to(torch.uint8) * 255).detach().cpu().numpy()

            obj_dir = output_dir / "objects" / obj_name
            cv2.imwrite(str(obj_dir / "masks" / f"{serial}.png"), mask_np)

            overlay = image_bgr.copy()
            if mask_np.sum() > 0:
                overlay[mask_np > 0] = (
                    overlay[mask_np > 0].astype(np.float32) * 0.5
                    + np.array([0, 255, 0], np.float32) * 0.5
                ).astype(np.uint8)
            cv2.imwrite(str(obj_dir / "masks_debug" / f"{serial}.png"), overlay)

        logging.info(f"  {serial}: {time.perf_counter() - t0:.3f}s")

    infer_s = time.perf_counter() - start
    logging.info(f"SAM3 done: {infer_s:.2f}s")
    return {
        "model_load_s": round(model_load_s, 2),
        "io_s": round(t_io_total, 2),
        "set_image_s": round(t_set_image_total, 2),
        "text_prompt_s": round(t_prompt_total, 2),
        "infer_s": round(infer_s, 2),
    }


def run_yoloe(img_dict, serials, object_info, output_dir, confidence):
    from ultralytics import YOLO

    obj_names = list(object_info.keys())
    prompts = [object_info[n]["text"] for n in obj_names]

    t_model = time.perf_counter()
    from autodex.perception.mask import YOLOE_WEIGHTS
    model = YOLO(YOLOE_WEIGHTS)
    model_load_s = time.perf_counter() - t_model

    t_set_classes_total = 0.0
    per_object_infer_s = {}

    start = time.perf_counter()
    for obj_name, prompt in zip(obj_names, prompts):
        t_sc = time.perf_counter()
        model.set_classes([prompt], model.get_text_pe([prompt]))
        t_set_classes_total += time.perf_counter() - t_sc

        t_obj = time.perf_counter()
        for serial in serials:
            image_bgr = img_dict.images[serial]
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            mask_np = get_mask_yoloe(image_rgb, prompt, model=model, conf_thr=confidence)
            if mask_np is None:
                H, W = image_bgr.shape[:2]
                mask_np = np.zeros((H, W), dtype=np.uint8)

            obj_dir = output_dir / "objects" / obj_name
            cv2.imwrite(str(obj_dir / "masks" / f"{serial}.png"), mask_np)

            overlay = image_bgr.copy()
            if mask_np.sum() > 0:
                overlay[mask_np > 0] = (
                    overlay[mask_np > 0].astype(np.float32) * 0.5
                    + np.array([0, 255, 0], np.float32) * 0.5
                ).astype(np.uint8)
            cv2.imwrite(str(obj_dir / "masks_debug" / f"{serial}.png"), overlay)

        per_object_infer_s[obj_name] = round(time.perf_counter() - t_obj, 2)
        logging.info(f"  {obj_name} done")

    infer_s = time.perf_counter() - start
    logging.info(f"YOLO-E done: {infer_s:.2f}s")
    return {
        "model_load_s": round(model_load_s, 2),
        "set_classes_s": round(t_set_classes_total, 2),
        "per_object_infer_s": per_object_infer_s,
        "infer_s": round(infer_s, 2),
    }


def _make_grid(images, ncols=6, label=True):
    """Simple grid layout from a dict of {label: bgr_image}."""
    items = list(images.items())
    if not items:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    h, w = items[0][1].shape[:2]
    ncols = min(ncols, len(items))
    nrows = (len(items) + ncols - 1) // ncols
    grid = np.zeros((nrows * h, ncols * w, 3), dtype=np.uint8)
    for idx, (name, img) in enumerate(items):
        r, c = divmod(idx, ncols)
        cell = img.copy()
        if label:
            cv2.putText(cell, str(name)[:20], (4, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(cell, str(name)[:20], (4, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = cell
    return grid


def save_combined_viz(img_dict, serials, object_info, output_dir):
    obj_names = list(object_info.keys())
    combined_dir = output_dir / "masks_combined"
    combined_dir.mkdir(exist_ok=True)

    grid_images = {}
    for serial in serials:
        image_bgr = img_dict.images[serial].copy()

        for i, obj_name in enumerate(obj_names):
            mask = cv2.imread(
                str(output_dir / "objects" / obj_name / "masks" / f"{serial}.png"),
                cv2.IMREAD_UNCHANGED,
            )
            if mask is None or mask.sum() == 0:
                continue
            if mask.ndim == 3:
                mask = mask[..., 0]
            color = np.array(OBJECT_COLORS[i % len(OBJECT_COLORS)], dtype=np.uint8)
            image_bgr[mask > 0] = (
                image_bgr[mask > 0].astype(np.float32) * 0.5
                + color.astype(np.float32) * 0.5
            ).astype(np.uint8)

        for i, obj_name in enumerate(obj_names):
            color = OBJECT_COLORS[i % len(OBJECT_COLORS)]
            y = 30 + i * 35
            cv2.rectangle(image_bgr, (10, y - 18), (30, y + 2), color, -1)
            text = f"{obj_name}: {object_info[obj_name]['text']}"
            cv2.putText(image_bgr, text, (38, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(image_bgr, text, (38, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imwrite(str(combined_dir / f"{serial}.png"), image_bgr)
        grid_images[serial] = image_bgr

    grid = _make_grid(grid_images, ncols=6, label=True)
    cv2.imwrite(str(combined_dir / "grid.png"), grid)
    logging.info(f"Combined grid saved to {combined_dir / 'grid.png'}")


def load_data_from_files(images_dir: Path):
    """Load images and camera params from already-saved output dir (no paradex needed)."""
    cam = np.load(str(images_dir.parent / "camera_data.npz"), allow_pickle=True)
    serials = list(cam["serials"])
    intrinsics = {s: cam["intrinsics"][i] for i, s in enumerate(serials)}
    extrinsics = {s: cam["extrinsics"][i] for i, s in enumerate(serials)}

    class _SimpleImageDict:
        def __init__(self):
            self.images = {}
    img_dict = _SimpleImageDict()
    for s in serials:
        img_dict.images[s] = cv2.imread(str(images_dir / f"{s}.png"))
    return img_dict, serials, intrinsics, extrinsics


def main(args):
    import json as _json

    data_dir = Path(args.data_dir) if args.data_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "6d_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine object_info source
    if args.reuse_images_from:
        src_dir = Path(args.reuse_images_from)
        object_info_path = src_dir / "object_info.json"
    else:
        object_info_path = data_dir / "object_info.json"
    with open(object_info_path) as f:
        object_info = json.load(f)
    shutil.copy2(str(object_info_path), str(output_dir / "object_info.json"))
    logging.info(f"Objects: {list(object_info.keys())}")

    if args.reuse_images_from:
        # Reuse already-saved images and camera_data (no paradex required)
        src_dir = Path(args.reuse_images_from)
        img_dict, serials, intrinsics, extrinsics = load_data_from_files(src_dir / "images")
        # Symlink or copy images + camera_data
        images_dir = output_dir / "images"
        if not images_dir.exists():
            images_dir.symlink_to((src_dir / "images").resolve())
        cam_npz = output_dir / "camera_data.npz"
        if not cam_npz.exists():
            shutil.copy2(str(src_dir / "camera_data.npz"), str(cam_npz))
        logging.info(f"Reusing {len(serials)} cameras from {src_dir}")
    else:
        img_dict, serials, intrinsics, extrinsics = load_data(data_dir)
        logging.info(f"Loaded {len(serials)} cameras")
        save_camera_data(output_dir, serials, intrinsics, extrinsics, img_dict)

    for obj_name in object_info:
        obj_dir = output_dir / "objects" / obj_name
        (obj_dir / "masks").mkdir(parents=True, exist_ok=True)
        (obj_dir / "masks_debug").mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    if args.method == "sam3":
        sub_timing = run_sam3(img_dict, serials, object_info, output_dir, args.device, args.confidence)
    else:
        sub_timing = run_yoloe(img_dict, serials, object_info, output_dir, args.confidence)
    elapsed = time.perf_counter() - t0

    save_combined_viz(img_dict, serials, object_info, output_dir)

    timing_path = output_dir / "timing.json"
    timing = _json.loads(timing_path.read_text()) if timing_path.exists() else {}
    timing["step1_mask"] = {
        "method": args.method,
        "total_s": round(elapsed, 2),
        "n_cameras": len(serials),
        "n_objects": len(object_info),
        **sub_timing,
    }
    timing_path.write_text(_json.dumps(timing, indent=2))
    logging.info(f"Step 1 done. ({elapsed:.1f}s) → {timing_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Raw data directory (required unless --reuse_images_from is set)")
    parser.add_argument("--reuse_images_from", type=str, default=None,
                        help="Reuse images/camera_data from an existing output dir (skips paradex)")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--method", type=str, default="sam3", choices=["sam3", "yoloe"])
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    if args.data_dir is None and args.reuse_images_from is None:
        parser.error("--data_dir is required unless --reuse_images_from is set")
    main(args)