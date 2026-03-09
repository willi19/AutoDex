"""
Demo: plan grasps for an object and visualize results.

Usage:
    python src/demo/sim.py --obj attached_container
"""
import os
import random
import time
import argparse

import numpy as np
from scipy.spatial.transform import Rotation

from autodex.utils.path import obj_path
from autodex.utils.conversion import se32cart
from autodex.planner import GraspPlanner
from autodex.visualizer import GraspPlanningVisualizer

parser = argparse.ArgumentParser()
parser.add_argument('--obj', type=str, required=True)
parser.add_argument('--grasp_version', type=str, required=True,
                    help='Candidate version (e.g., scene_cover_table, selected_100)')
parser.add_argument('--pose_idx', type=str, default="000",
                    help='Tabletop pose index (e.g., 000, 001)')
args = parser.parse_args()


def load_tabletop_pose(obj_name, pose_idx):
    """Load a pre-recorded tabletop pose and adjust placement.

    x: random in [0.2, 0.4], y: 0, z: from original pose, random z-axis rotation.
    """
    pose_dir = os.path.join(obj_path, obj_name, "processed_data", "info", "tabletop")
    pose_file = os.path.join(pose_dir, f"{pose_idx}.npy")
    if not os.path.exists(pose_file):
        available = sorted(os.listdir(pose_dir))
        raise FileNotFoundError(f"Pose {pose_idx} not found. Available: {available}")
    pose = np.load(pose_file)
    # Adjust position: x in [0.2, 0.4], y=0, z from original + table surface offset
    table_surface_z = 0.037
    pose[0, 3] = random.uniform(0.4, 0.5)
    pose[1, 3] = 0.0
    pose[2, 3] = pose[2, 3] + table_surface_z
    # Apply random rotation around z-axis
    angle = random.uniform(0, 2 * np.pi)
    Rz = Rotation.from_euler('z', angle).as_matrix()
    pose[:3, :3] = Rz @ pose[:3, :3]
    return pose

def make_scene_cfg(obj_name, obj_pose):
    """Build scene_cfg from object name and SE3 pose (no cameras needed)."""
    return {
        "mesh": {
            "target": {
                "pose": se32cart(obj_pose).tolist(),
                "file_path": os.path.join(obj_path, obj_name, "processed_data", "mesh", "simplified.obj"),
                "urdf_path": os.path.join(obj_path, obj_name, "processed_data", "urdf", "coacd.urdf"),
            }
        },
        "cuboid": {
            "table": {
                "dims": [2, 3, 0.2],
                "pose": [1.1, 0, -0.1 + 0.037, 1, 0, 0, 0],
            }
        },
    }

if __name__ == "__main__":
    # 1. Load tabletop pose and build scene
    obj_pose = load_tabletop_pose(args.obj, args.pose_idx)
    scene_cfg = make_scene_cfg(args.obj, obj_pose)
    print(f"Scene built for '{args.obj}'")

    # 2. Plan all candidates
    t0 = time.time()
    planner = GraspPlanner()
    wrist_se3, pregrasp, grasp_pose, succ_mask, collision, traj_list = planner.plan_all(
        scene_cfg, args.obj, args.grasp_version
    )
    print(f"Planning done ({time.time() - t0:.2f}s) — {succ_mask.sum()}/{len(succ_mask)} succeeded")

    # 3. Sample subset for visualization
    num_samples = 5
    succ_idx = list(np.where(succ_mask)[0])
    fail_idx = list(np.where(~succ_mask & ~collision)[0])
    coll_idx = list(np.where(collision)[0])

    sample_idx = sorted(set(
        random.sample(succ_idx, min(num_samples, len(succ_idx)))
        + random.sample(fail_idx, min(num_samples, len(fail_idx)))
        + random.sample(coll_idx, min(num_samples, len(coll_idx)))
    ))

    if len(sample_idx) == 0:
        print("No candidates to visualize.")
        exit(0)

    # 4. Visualize
    vis = GraspPlanningVisualizer(
        scene_cfg=scene_cfg,
        wrist_se3=wrist_se3[sample_idx],
        pregrasp=pregrasp[sample_idx],
        grasp_pose=grasp_pose[sample_idx],
        collision=collision[sample_idx],
        succ=succ_mask[sample_idx],
        traj_list=[traj_list[i] for i in sample_idx],
    )
    vis.add_frame("base_frame", np.eye(4))
    vis.start_viewer(use_thread=True)

    input("Press Enter to exit...")
