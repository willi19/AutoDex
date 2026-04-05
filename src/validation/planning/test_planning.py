"""Test planning + visualize trajectory (no perception/execution).

Usage:
    python src/validation/planning/test_planning.py --obj attached_container --hand inspire --port 8080
    python src/validation/planning/test_planning.py --obj attached_container --hand allegro --x 0.4 --z_rot 30
"""
import argparse
import os
import sys
import logging
import numpy as np

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "paradex"))

from autodex.planner import GraspPlanner
from autodex.planner.visualizer import ScenePlanVisualizer
from autodex.utils.path import obj_path
from autodex.utils.conversion import se32cart


def load_scene(obj_name, pose_idx="000", x_offset=0.4, z_rotation=0.0):
    pose_dir = os.path.join(obj_path, obj_name, "processed_data", "info", "tabletop")
    pose_file = os.path.join(pose_dir, f"{pose_idx}.npy")
    if not os.path.exists(pose_file):
        available = sorted(f.replace(".npy", "") for f in os.listdir(pose_dir) if f.endswith(".npy"))
        print(f"Pose {pose_idx} not found. Available: {available}")
        pose_file = os.path.join(pose_dir, f"{available[0]}.npy")
        print(f"Using {available[0]}")

    obj_pose = np.load(pose_file)
    obj_pose[0, 3] += x_offset

    if z_rotation != 0.0:
        c, s = np.cos(z_rotation), np.sin(z_rotation)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        obj_pose[:3, :3] = Rz @ obj_pose[:3, :3]

    mesh_path = os.path.join(obj_path, obj_name, "processed_data", "mesh", "simplified.obj")
    if not os.path.exists(mesh_path):
        mesh_path = os.path.join(obj_path, obj_name, "raw_mesh", f"{obj_name}.obj")

    return {
        "mesh": {
            "target": {
                "pose": se32cart(obj_pose).tolist(),
                "file_path": mesh_path,
            },
        },
        "cuboid": {
            "table": {
                "dims": [2, 3, 0.2],
                "pose": [1.1, 0, -0.1 + 0.037, 1, 0, 0, 0],
            },
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--hand", type=str, default="allegro", choices=["allegro", "inspire"])
    parser.add_argument("--version", type=str, default="selected_100")
    parser.add_argument("--pose", type=str, default="000")
    parser.add_argument("--x", type=float, default=0.4)
    parser.add_argument("--z_rot", type=float, default=0.0, help="Z rotation (degrees)")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    logging.getLogger("curobo").setLevel(logging.ERROR)

    scene_cfg = load_scene(args.obj, args.pose, args.x, np.radians(args.z_rot))
    print(f"Object: {args.obj}, hand: {args.hand}, x={args.x}, z_rot={args.z_rot}°")

    planner = GraspPlanner(hand=args.hand)
    result = planner.plan(scene_cfg, args.obj, args.version, hand=args.hand)

    print(f"Planning: success={result.success}")
    if result.timing:
        for k, v in result.timing.items():
            print(f"  {k}: {v}")

    if result.success:
        print(f"Trajectory: {result.traj.shape}")
        print(f"traj[0] (deg): {np.degrees(result.traj[0])}")
        vis = ScenePlanVisualizer(scene_cfg, result, port=args.port, hand=args.hand)
    else:
        print("Planning failed — showing candidates")
        wrist_se3, _, grasp_pose, filtered = planner.get_candidates(
            scene_cfg, args.obj, args.version, hand=args.hand)
        vis = ScenePlanVisualizer(scene_cfg, None, port=args.port, hand=args.hand)
        vis.add_candidates(wrist_se3, grasp_pose, filtered)

    vis.start_viewer(use_thread=False)


if __name__ == "__main__":
    main()
