"""Combine turntable grasp videos into a grid video for slides (1920x1080)."""

import argparse
import json
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
from tqdm import tqdm

CANVAS_W = 1920
CANVAS_H = 1080

ORDER_ROOT = os.path.expanduser("~/RSS_2026/order")
VERSIONS = ["revalidate", "v2", "v3"]


def _build_walk_order(data_dir, obj_name):
    """Reproduce the directory walk order used during rendering.
    Returns list of (scene_type, scene_id, grasp_name)."""
    root = os.path.join(data_dir, obj_name)
    # The turntable script used load_selected_grasps which walked
    # selected_100 in sorted order: scene_type → scene_id → grasp_name
    # But we don't have selected_100 here. Instead, we know each rank
    # maps to a directory, so just return rank indices.
    return None  # not needed with new approach


def _get_setcover_order(obj_name):
    """Load setcover order for an object. Returns list of (scene_type, scene_id, grasp_name)."""
    for v in VERSIONS:
        path = os.path.join(ORDER_ROOT, v, obj_name, "setcover_order.json")
        if os.path.exists(path):
            with open(path) as f:
                ordered_list = json.load(f)
            return [(item[2], item[3], item[4]) for item in ordered_list]
    return None


def _build_reorder_map(data_dir, obj_name):
    """Build old_rank -> new_rank mapping using setcover order.

    The existing data/ was rendered with directory walk order (sorted scene_type/scene_id/grasp_name).
    We want setcover order instead. This builds the mapping.
    """
    # Reconstruct what load_selected_grasps produced (sorted walk of selected_100)
    from autodex.utils.path import candidate_path
    selected_root = os.path.join(candidate_path, "selected_100", obj_name)
    if not os.path.exists(selected_root):
        return None

    walk_order = []
    for scene_type in sorted(os.listdir(selected_root)):
        st_path = os.path.join(selected_root, scene_type)
        if not os.path.isdir(st_path):
            continue
        for scene_id in sorted(os.listdir(st_path)):
            si_path = os.path.join(st_path, scene_id)
            if not os.path.isdir(si_path):
                continue
            for grasp_name in sorted(os.listdir(si_path),
                                     key=lambda x: int(x) if x.isdigit() else x):
                gp = os.path.join(si_path, grasp_name)
                if os.path.isdir(gp) and os.path.exists(os.path.join(gp, "wrist_se3.npy")):
                    walk_order.append((scene_type, scene_id, grasp_name))

    # walk_order[old_rank] = (scene_type, scene_id, grasp_name)
    walk_key_to_old_rank = {k: i for i, k in enumerate(walk_order)}

    # Get setcover order, filter to available
    setcover = _get_setcover_order(obj_name)
    if setcover is None:
        return None

    available = set(walk_key_to_old_rank.keys())
    setcover_filtered = [k for k in setcover if k in available]

    # setcover_filtered[new_rank] -> key -> old_rank
    # Return list where new_rank -> old_rank
    return [walk_key_to_old_rank[k] for k in setcover_filtered]


def make_grid_video(obj_name, data_dir, output_path, rows=4, cols=6):
    """Combine individual turntable videos into a 1920x1080 grid.
    Data is already in setcover order (000 = rank 1, etc.)."""
    n_cells = rows * cols

    videos = []
    for rank in range(n_cells):
        vpath = os.path.join(data_dir, obj_name, f"{rank:03d}", "turntable.mp4")
        videos.append(vpath if os.path.exists(vpath) else None)

    if not any(videos):
        print(f"No videos found for {obj_name}")
        return False

    # Cell size: divide canvas evenly
    cell_w = CANVAS_W // cols
    cell_h = CANVAS_H // rows

    # Open all video captures
    caps = [cv2.VideoCapture(vp) if vp else None for vp in videos]

    first_cap = next(c for c in caps if c is not None)
    n_frames = int(first_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(first_cap.get(cv2.CAP_PROP_FPS))

    temp_dir = tempfile.mkdtemp(prefix="grid_")

    for fi in tqdm(range(n_frames), desc=f"{obj_name} grid"):
        canvas = np.ones((CANVAS_H, CANVAS_W, 3), dtype=np.uint8) * 235

        for idx, cap in enumerate(caps):
            if cap is None:
                continue
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.resize(frame, (cell_w, cell_h))

            r, c = divmod(idx, cols)
            x = c * cell_w
            y = r * cell_h

            canvas[y:y + cell_h, x:x + cell_w] = frame

        cv2.imwrite(os.path.join(temp_dir, f"frame_{fi:04d}.png"), canvas)

    for cap in caps:
        if cap is not None:
            cap.release()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", os.path.join(temp_dir, "frame_%04d.png"),
        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-pix_fmt", "yuv420p", output_path,
    ], capture_output=True, text=True)

    shutil.rmtree(temp_dir)
    print(f"Saved: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Combine grasp turntable videos into 1920x1080 grid")
    parser.add_argument("--obj", default=None, help="Object name (omit for --batch-all)")
    parser.add_argument("--batch-all", action="store_true", help="Process all objects in data-dir")
    parser.add_argument("--data-dir", default="data", help="Data directory (default: data)")
    parser.add_argument("--output-dir", default="outputs/grid_video", help="Output dir for batch mode")
    parser.add_argument("--output", default=None, help="Output path for single object")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=6)
    args = parser.parse_args()

    if args.batch_all:
        objects = sorted(d for d in os.listdir(args.data_dir)
                         if os.path.isdir(os.path.join(args.data_dir, d)))
        print(f"Batch: {len(objects)} objects")
        os.makedirs(args.output_dir, exist_ok=True)
        for obj_name in objects:
            output = os.path.join(args.output_dir, f"{obj_name}.mp4")
            if os.path.exists(output):
                print(f"[skip] {obj_name}")
                continue
            make_grid_video(obj_name, args.data_dir, output,
                            rows=args.rows, cols=args.cols)
    else:
        if args.obj is None:
            print("Error: --obj required (or use --batch-all)")
            parser.print_help()
            return
        output = args.output or os.path.join(args.output_dir, f"{args.obj}.mp4")
        os.makedirs(os.path.dirname(output), exist_ok=True)
        make_grid_video(args.obj, args.data_dir, output,
                        rows=args.rows, cols=args.cols)


if __name__ == "__main__":
    main()
