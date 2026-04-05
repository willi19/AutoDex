"""Extract 100 random grasps from each baseline zip into candidates/allegro/baseline_100/.

Process: for each zip, extract to temp dir → randomly pick 100 grasp dirs → copy to dest → delete temp.
Only copies wrist_se3.npy, pregrasp_pose.npy, grasp_pose.npy per grasp.
"""
import os
import sys
import glob
import shutil
import random
import zipfile
import tempfile
import numpy as np
from tqdm import tqdm

BASELINE_DIR = "/home/robot/shared_data/RSS2026_Mingi/candidates/baseline"
DEST_DIR = "/home/robot/shared_data/AutoDex/candidates/allegro/baseline_100"
SELECTED_100_DIR = "/home/robot/shared_data/AutoDex/candidates/allegro/selected_100"
N_SAMPLE = 100
SEED = 42

NEEDED_FILES = ["wrist_se3.npy", "pregrasp_pose.npy", "grasp_pose.npy"]


def get_common_objects():
    """Only process objects that exist in both baseline and selected_100."""
    baseline_objs = {f.replace(".zip", "") for f in os.listdir(BASELINE_DIR) if f.endswith(".zip")}
    selected_objs = set(os.listdir(SELECTED_100_DIR)) if os.path.isdir(SELECTED_100_DIR) else set()
    return sorted(baseline_objs & selected_objs)


def extract_and_sample(zip_path, obj_name, rng):
    """Extract zip to temp, sample 100 grasps, copy to dest, clean up temp."""
    tmp_dir = tempfile.mkdtemp(prefix=f"baseline_{obj_name}_")
    try:
        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(tmp_dir)

        # Find all grasp directories (have wrist_se3.npy)
        extracted_root = os.path.join(tmp_dir, obj_name)
        if not os.path.isdir(extracted_root):
            # Some zips might have different root
            subdirs = os.listdir(tmp_dir)
            if len(subdirs) == 1:
                extracted_root = os.path.join(tmp_dir, subdirs[0])
            else:
                print(f"  WARNING: unexpected zip structure for {obj_name}, skipping")
                return 0

        grasp_dirs = []
        for root, dirs, files in os.walk(extracted_root):
            if "wrist_se3.npy" in files:
                grasp_dirs.append(root)

        if len(grasp_dirs) == 0:
            print(f"  WARNING: no grasps found in {obj_name}")
            return 0

        # Sample
        n_pick = min(N_SAMPLE, len(grasp_dirs))
        selected = rng.sample(grasp_dirs, n_pick)

        # Copy to dest
        dest_obj = os.path.join(DEST_DIR, obj_name)
        for gdir in selected:
            # Get relative path: scene_type/scene_id/grasp_idx
            rel = os.path.relpath(gdir, extracted_root)
            dest_grasp = os.path.join(dest_obj, rel)
            os.makedirs(dest_grasp, exist_ok=True)
            for fname in NEEDED_FILES:
                src = os.path.join(gdir, fname)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(dest_grasp, fname))

        return n_pick
    finally:
        # Always clean up temp
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    rng = random.Random(SEED)
    obj_list = get_common_objects()
    print(f"Found {len(obj_list)} common objects between baseline and selected_100")

    os.makedirs(DEST_DIR, exist_ok=True)

    for obj_name in tqdm(obj_list, desc="Extracting baseline"):
        dest_obj = os.path.join(DEST_DIR, obj_name)
        if os.path.isdir(dest_obj):
            n_existing = sum(1 for _ in glob.iglob(os.path.join(dest_obj, "**/wrist_se3.npy"), recursive=True))
            if n_existing >= N_SAMPLE:
                print(f"  {obj_name}: already has {n_existing} grasps, skipping")
                continue

        zip_path = os.path.join(BASELINE_DIR, f"{obj_name}.zip")
        if not os.path.exists(zip_path):
            print(f"  {obj_name}: zip not found, skipping")
            continue

        n = extract_and_sample(zip_path, obj_name, rng)
        print(f"  {obj_name}: sampled {n} grasps")

    # Summary
    print("\n=== Summary ===")
    total = 0
    for obj_name in sorted(os.listdir(DEST_DIR)):
        n = sum(1 for _ in glob.iglob(os.path.join(DEST_DIR, obj_name, "**/wrist_se3.npy"), recursive=True))
        total += n
        print(f"  {obj_name}: {n}")
    print(f"Total: {total} grasps across {len(os.listdir(DEST_DIR))} objects")


if __name__ == "__main__":
    main()
