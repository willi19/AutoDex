#!/usr/bin/env python3
"""
Step 4: Compare results across pose configurations.

Reads pose output dirs from output_base/pose/ and generates:
  - compare/{serial}_compare.png  : mask + pose overlay per camera, one row per config
  - compare/depth_compare.png     : depth colormap overlay per method, labeled with source
  - compare/summary.csv           : timing table across configs

New directory structure:
    output_base/
    ├── segmentation/{sam3,yoloe,...}/
    ├── depth/{da3,foundationstereo,...}/
    └── pose/{seg}_{depth}/
        ├── objects/{name}/visualizations/
        ├── sources.json   ← points back to seg_dir and depth_dir
        └── timing.json

Usage:
    conda activate foundationpose
    python src/validation/perception/step4_compare.py \\
        --output_base ~/shared_data/.../validation_output
"""

import argparse
import csv
import json
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="[compare] [%(levelname)s] %(message)s")


def load_timing(path: Path) -> dict:
    p = path / "timing.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load_sources(pose_dir: Path) -> dict:
    p = pose_dir / "sources.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load_source_info(depth_dir: Path) -> dict:
    p = depth_dir / "source_info.json"
    return json.loads(p.read_text()) if p.exists() else {}


def make_label(img, text, font_scale=0.5, color=(255, 255, 255), thickness=1):
    out = img.copy()
    cv2.putText(out, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(out, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return out


def make_label2(img, line1, line2, font_scale=0.4, color=(255, 255, 255), thickness=1):
    """Two-line label at top of image."""
    out = img.copy()
    for text, y in [(line1, 16), (line2, 32)]:
        cv2.putText(out, text, (5, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(out, text, (5, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return out


def resize_to(img, h, w):
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def load_image(path):
    img = cv2.imread(str(path))
    return img if img is not None else None


def make_depth_colormap_cell(serial, depth_dir, images_dir, source_info, th, tw):
    """Create a depth colormap overlay cell for one camera serial.

    Returns (cell_img, source_label) where cell_img is (th, tw, 3).
    source_label describes where the depth came from.
    """
    depth_path = depth_dir / "depth" / f"{serial}.png"
    orig = load_image(images_dir / f"{serial}.png")
    if orig is None:
        cell = np.zeros((th, tw, 3), dtype=np.uint8)
        return cell, "no image"

    orig = resize_to(orig, th, tw)

    method = source_info.get("method", "")
    pair_map = source_info.get("pairs", {})  # left_serial → right_serial
    # Build reverse map: right_serial → left_serial
    rev_map = {v: k for k, v in pair_map.items()}

    if not depth_path.exists():
        # No depth for this serial
        if serial in rev_map:
            label = f"right of {rev_map[serial]}"
        else:
            label = "no depth"
        cell = np.zeros((th, tw, 3), dtype=np.uint8)
        cell = make_label(cell, label, color=(80, 80, 80))
        return cell, label

    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    depth = cv2.resize(depth, (tw, th), interpolation=cv2.INTER_NEAREST)

    valid = depth > 0.001
    if valid.any():
        d_min, d_max = depth[valid].min(), depth[valid].max()
        depth_norm = np.zeros_like(depth)
        depth_norm[valid] = (depth[valid] - d_min) / (d_max - d_min + 1e-6)
        depth_color = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        depth_color[~valid] = 0
        mask3 = np.stack([valid, valid, valid], axis=-1)
        overlay = (orig.astype(np.float32) * 0.3 + depth_color.astype(np.float32) * 0.7).astype(np.uint8)
        cell = np.where(mask3, overlay, orig)
    else:
        cell = orig.copy()

    # Build source label
    source_serial = source_info.get("source_serial", "")
    if method == "stereo":
        if serial == source_serial:
            right_s = pair_map.get(serial, "?")
            source_label = f"{serial[-6:]}+{right_s[-6:]}"
        elif source_serial:
            source_label = f"reproj:{source_serial[-6:]}"
        else:
            source_label = serial[-8:]
    else:
        source_label = "mono"

    return cell, source_label


def make_depth_compare_grid(serials, depth_dirs, depth_names, images_dir, th, tw, cols=6):
    """Build a grid showing depth colormap overlays for all depth methods.

    Layout: one row-group per depth method, cols cameras per row.
    Each cell shows: depth colormap blended with original + source label.
    """
    all_row_groups = []

    for depth_dir, depth_name in zip(depth_dirs, depth_names):
        source_info = load_source_info(depth_dir)
        cells = []
        for serial in serials:
            cell, source_label = make_depth_colormap_cell(
                serial, depth_dir, images_dir, source_info, th, tw)
            line1 = serial[-8:]
            line2 = source_label
            cell = make_label2(cell, line1, line2)
            cells.append(cell)

        # Arrange into rows of `cols` cells
        rows = []
        for i in range(0, len(cells), cols):
            row_cells = cells[i:i + cols]
            # Pad last row
            while len(row_cells) < cols:
                row_cells.append(np.zeros((th, tw, 3), dtype=np.uint8))
            rows.append(np.hstack(row_cells))

        # Add method label strip on left of first row
        strip_w = 80
        labeled_rows = []
        for ri, row in enumerate(rows):
            strip = np.zeros((row.shape[0], strip_w, 3), dtype=np.uint8)
            label_text = depth_name if ri == 0 else ""
            cv2.putText(strip, label_text, (2, row.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            labeled_rows.append(np.hstack([strip, row]))

        all_row_groups.extend(labeled_rows)

        # Separator between methods
        sep_h = 4
        sep_w = labeled_rows[0].shape[1]
        all_row_groups.append(np.full((sep_h, sep_w, 3), 60, dtype=np.uint8))

    if not all_row_groups:
        return None

    # Pad all rows to same width
    max_w = max(r.shape[1] for r in all_row_groups)
    padded = []
    for r in all_row_groups:
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
            r = np.hstack([r, pad])
        padded.append(r)

    return np.vstack(padded)


def make_compare_grid(serial, pose_dirs, pose_names, seg_dirs, object_names, images_dir):
    """Build a comparison grid for one camera serial.

    Columns: original | mask_obj1 | ... | pose_obj1 | ...
    Rows: one per pose config
    """
    orig = load_image(images_dir / f"{serial}.png")
    if orig is None:
        return None
    H, W = orig.shape[:2]
    th, tw = H // 2, W // 2

    rows = []
    for pose_dir, pose_name, seg_dir in zip(pose_dirs, pose_names, seg_dirs):
        cells = []
        cells.append(make_label(resize_to(orig, th, tw), "original"))

        # Mask debug per object (from seg_dir)
        for obj_name in object_names:
            mask_path = Path(seg_dir) / "objects" / obj_name / "masks_debug" / f"{serial}.png"
            img = load_image(mask_path)
            if img is None:
                img = np.zeros((th, tw, 3), dtype=np.uint8)
            else:
                img = resize_to(img, th, tw)
            cells.append(make_label(img, f"mask:{obj_name[:10]}"))

        # Pose overlay per object (from pose_dir)
        for obj_name in object_names:
            vis_path = pose_dir / "objects" / obj_name / "visualizations" / f"{serial}_overlay.png"
            img = load_image(vis_path)
            if img is None:
                img = np.zeros((th, tw, 3), dtype=np.uint8)
                img = make_label(img, "no pose", color=(100, 100, 100))
            else:
                img = resize_to(img, th, tw)
            cells.append(make_label(img, f"pose:{obj_name[:8]}"))

        row = np.hstack(cells)
        rows.append(row)

    # Pad all rows to same width
    max_w = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
            r = np.hstack([r, pad])
        padded.append(r)

    # Add pose config label strip on left
    strip_w = 100
    strips = []
    for pose_name, r in zip(pose_names, padded):
        strip = np.zeros((r.shape[0], strip_w, 3), dtype=np.uint8)
        cv2.putText(strip, pose_name, (2, r.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        strips.append(np.hstack([strip, r]))

    return np.vstack(strips)


def write_summary_csv(compare_dir: Path, pose_dirs, pose_names):
    rows = []
    for pose_dir, pose_name in zip(pose_dirs, pose_names):
        sources = load_sources(pose_dir)
        seg_dir = Path(sources.get("seg_dir", "")) if sources.get("seg_dir") else None
        depth_dir = Path(sources.get("depth_dir", "")) if sources.get("depth_dir") else None

        row = {"config": pose_name}

        seg_timing = load_timing(seg_dir) if seg_dir and seg_dir.exists() else {}
        if "step1_mask" in seg_timing:
            s = seg_timing["step1_mask"]
            row["mask_method"] = s.get("method", "")
            row["mask_s"] = s.get("total_s", "")
            row["n_cameras"] = s.get("n_cameras", "")

        depth_timing = load_timing(depth_dir) if depth_dir and depth_dir.exists() else {}
        if "step2_depth" in depth_timing:
            s = depth_timing["step2_depth"]
            row["depth_method"] = s.get("method", "")
            row["depth_s"] = s.get("total_s", "")

        pose_timing = load_timing(pose_dir)
        if "step3_pose" in pose_timing:
            s = pose_timing["step3_pose"]
            row["pose_s"] = s.get("total_s", "")
            total = (seg_timing.get("step1_mask", {}).get("total_s", 0) +
                     depth_timing.get("step2_depth", {}).get("total_s", 0) +
                     s.get("total_s", 0))
            row["total_s"] = round(total, 2)

        rows.append(row)

    if not rows:
        return

    priority = ["config", "mask_method", "depth_method", "mask_s", "depth_s", "pose_s", "total_s", "n_cameras"]
    fieldnames = sorted({k for r in rows for k in r})
    ordered = [f for f in priority if f in fieldnames] + [f for f in fieldnames if f not in priority]

    csv_path = compare_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    logging.info(f"Timing summary → {csv_path}")
    for r in rows:
        logging.info(f"  {r.get('config')}: mask={r.get('mask_s','?')}s "
                     f"depth={r.get('depth_s','?')}s pose={r.get('pose_s','?')}s "
                     f"total={r.get('total_s','?')}s")


def main(args):
    output_base = Path(args.output_base)
    pose_base = output_base / "pose"

    # Auto-detect pose dirs
    if args.configs:
        pose_names = args.configs
        pose_dirs = [pose_base / c for c in pose_names]
    else:
        if pose_base.exists():
            pose_dirs = sorted([d for d in pose_base.iterdir()
                                if d.is_dir() and (d / "timing.json").exists()])
        else:
            # Fallback: old flat structure (sam3/, yoloe/ directly in output_base)
            pose_dirs = sorted([d for d in output_base.iterdir()
                                if d.is_dir() and (d / "timing.json").exists()
                                and d.name not in ("segmentation", "depth", "pose", "compare")])
        pose_names = [d.name for d in pose_dirs]

    if not pose_dirs:
        logging.error("No pose config dirs found.")
        return

    logging.info(f"Configs: {pose_names}")

    compare_dir = output_base / "compare"
    compare_dir.mkdir(exist_ok=True)

    # Resolve seg_dir for each pose config (for images + masks)
    seg_dirs = []
    for pose_dir in pose_dirs:
        sources = load_sources(pose_dir)
        seg_dir = sources.get("seg_dir", str(pose_dir))
        seg_dirs.append(seg_dir)

    # Get serials and object names from first seg_dir
    first_seg = Path(seg_dirs[0])
    cam_data = np.load(str(first_seg / "camera_data.npz"), allow_pickle=True)
    serials = list(cam_data["serials"])
    with open(first_seg / "object_info.json") as f:
        object_info = json.load(f)
    object_names = list(object_info.keys())
    images_dir = first_seg / "images"

    # --------------------------------------------------------
    # Depth comparison grid
    # --------------------------------------------------------
    depth_base = output_base / "depth"
    if depth_base.exists():
        depth_dirs = sorted([d for d in depth_base.iterdir() if d.is_dir()])
        depth_names = [d.name for d in depth_dirs]
        if depth_dirs:
            # Use image size from first available image
            sample_img = load_image(images_dir / f"{serials[0]}.png")
            if sample_img is not None:
                H, W = sample_img.shape[:2]
                th, tw = H // 3, W // 3
                depth_grid = make_depth_compare_grid(
                    serials, depth_dirs, depth_names, images_dir, th, tw, cols=5)
                if depth_grid is not None:
                    cv2.imwrite(str(compare_dir / "depth_compare.png"), depth_grid)
                    logging.info(f"Depth compare grid → {compare_dir}/depth_compare.png")

    # --------------------------------------------------------
    # Pose comparison grids (per camera)
    # --------------------------------------------------------
    for serial in serials:
        grid = make_compare_grid(serial, pose_dirs, pose_names, seg_dirs, object_names, images_dir)
        if grid is not None:
            cv2.imwrite(str(compare_dir / f"{serial}_compare.png"), grid)
    logging.info(f"Saved {len(serials)} comparison images to {compare_dir}")

    # Write timing summary
    write_summary_csv(compare_dir, pose_dirs, pose_names)
    logging.info("Step 4 done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_base", type=str, required=True,
                        help="Parent dir containing segmentation/, depth/, pose/ subdirs")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Pose config dir names to compare (default: auto-detect from pose/)")
    main(parser.parse_args())