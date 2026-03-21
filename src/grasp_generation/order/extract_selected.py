"""Extract top N grasps from setcover ordering into selected_N directory.

Usage:
    python src/grasp_generation/order/extract_selected.py
    python src/grasp_generation/order/extract_selected.py --hand allegro --version v3 --n 100
    python src/grasp_generation/order/extract_selected.py --obj attached_container
"""

import os
import json
import shutil
import argparse
from tqdm import tqdm

from autodex.utils.path import project_dir


def load_ordered_grasps(order_root, obj_name):
    order_path = os.path.join(order_root, obj_name, "setcover_order.json")
    if not os.path.exists(order_path):
        return []
    with open(order_path) as f:
        return json.load(f)


def extract_top_n(obj_name, candidate_root, order_root, output_root, n_grasps=100):
    ordered_grasps = load_ordered_grasps(order_root, obj_name)
    if not ordered_grasps:
        print(f"  No ordered grasps found")
        return

    top_n = ordered_grasps[:min(n_grasps, len(ordered_grasps))]
    print(f"  Extracting {len(top_n)} grasps")

    output_dir = os.path.join(output_root, obj_name)

    for grasp_info in tqdm(top_n, desc=f"  Copying", leave=False):
        _, _, scene_type, scene_id, grasp_name, _ = grasp_info

        src_dir = os.path.join(candidate_root, obj_name, scene_type, scene_id, grasp_name)
        if not os.path.exists(src_dir):
            print(f"  Warning: Source not found - {src_dir}")
            continue

        dst_dir = os.path.join(output_dir, scene_type, scene_id, grasp_name)
        os.makedirs(dst_dir, exist_ok=True)

        for filename in ['grasp_pose.npy', 'pregrasp_pose.npy', 'wrist_se3.npy']:
            src_file = os.path.join(src_dir, filename)
            dst_file = os.path.join(dst_dir, filename)
            if os.path.exists(src_file):
                shutil.copy2(src_file, dst_file)

    print(f"  Saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", type=str, default="allegro")
    parser.add_argument("--version", type=str, default="v3")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--obj", type=str, default=None)
    args = parser.parse_args()

    candidate_root = os.path.join(project_dir, "candidates", args.hand, args.version)
    order_root = os.path.join(project_dir, "candidates", args.hand, args.version + "_order")
    output_root = os.path.join(project_dir, "candidates", args.hand, f"selected_{args.n}")

    print(f"Hand: {args.hand}, Version: {args.version}, Top N: {args.n}")
    print(f"Candidates: {candidate_root}")
    print(f"Order: {order_root}")
    print(f"Output: {output_root}")

    if args.obj:
        obj_list = [args.obj]
    else:
        if not os.path.isdir(order_root):
            print(f"Order directory not found: {order_root}")
            exit(1)
        obj_list = sorted([d for d in os.listdir(order_root)
                          if os.path.isdir(os.path.join(order_root, d))])

    print(f"Objects: {len(obj_list)}")

    for obj_name in obj_list:
        print(f"\n{obj_name}:")
        extract_top_n(obj_name, candidate_root, order_root, output_root, args.n)

    print("\nDone!")