#!/usr/bin/env python3
"""View planning trajectory in viser.

Loads planning result (traj.npy, wrist_se3.npy) from eval pipeline output
and shows trajectory with ghost trail + object mesh + table.

Usage:
    python src/validation/execution/eval_perception/view_planning.py \
        --data_root ~/shared_data/mingi_object_test \
        --obj attached_container --episode 20260317_172712 --port 8080
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import trimesh

AUTODEX_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(AUTODEX_ROOT))
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "paradex"))

from paradex.visualization.visualizer.viser import ViserViewer
from autodex.utils.path import urdf_path
from autodex.utils.robot_config import INIT_STATE
from autodex.utils.conversion import se32cart

MESH_ROOT = Path.home() / "shared_data/object_6d/data/mesh"
TABLE_POSE = [1.1, 0, -0.1 + 0.037, 1, 0, 0, 0]
TABLE_DIMS = [2, 3, 0.2]


def find_mesh(obj_name):
    for name in [f"{obj_name}.obj", "simplified.obj", "coacd.obj"]:
        p = MESH_ROOT / obj_name / name
        if p.exists():
            return str(p)
    objs = list((MESH_ROOT / obj_name).glob("*.obj"))
    if objs:
        return str(objs[0])
    raise FileNotFoundError(f"No mesh for {obj_name}")


def pose7_to_se3(pose_list):
    from scipy.spatial.transform import Rotation
    se3 = np.eye(4)
    se3[:3, 3] = pose_list[:3]
    wxyz = pose_list[3:7]
    se3[:3, :3] = Rotation.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]]).as_matrix()
    return se3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--episode", type=str, required=True)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    capture_dir = Path(args.data_root) / args.obj / args.episode

    # Load results
    traj_path = capture_dir / "traj.npy"
    wrist_path = capture_dir / "wrist_se3.npy"
    pregrasp_path = capture_dir / "pregrasp_pose.npy"
    pose_world_path = capture_dir / "pose_world.npy"

    if not traj_path.exists():
        print(f"No trajectory at {traj_path}")
        return

    traj = np.load(str(traj_path))
    wrist_se3 = np.load(str(wrist_path))
    pregrasp_pose = np.load(str(pregrasp_path))
    pose_world = np.load(str(pose_world_path))

    # C2R for robot frame
    c2r_path = capture_dir / "cam_param" / "C2R.npy"
    if not c2r_path.exists():
        c2r_path = capture_dir / "C2R.npy"
    c2r = np.load(str(c2r_path)) if c2r_path.exists() else np.eye(4)
    r2c = np.linalg.inv(c2r)

    # Object pose in robot frame
    pose_robot = r2c @ pose_world
    obj_cart = se32cart(pose_robot)

    mesh_path = find_mesh(args.obj)
    print(f"Mesh: {mesh_path}")
    print(f"Trajectory: {len(traj)} frames")
    print(f"Object pose (robot): {pose_robot[:3, 3]}")

    # Build scene
    vis = ViserViewer(port=args.port)

    # Table
    table_se3 = pose7_to_se3(TABLE_POSE)
    table_mesh = trimesh.creation.box(extents=TABLE_DIMS)
    vis.add_object("table", table_mesh, table_se3)
    vis.change_color("table", [240/255, 240/255, 245/255, 0.5])

    # Object mesh
    obj_mesh = trimesh.load(mesh_path, process=False)
    if isinstance(obj_mesh, trimesh.Scene):
        obj_mesh = trimesh.util.concatenate([g for g in obj_mesh.geometry.values()])
    vis.add_trimesh("object", obj_mesh, pose_robot)

    # Grid
    vis.add_grid(size=12.0, cell_size=0.1, height=0.0)

    # Robot at init
    urdf_full = os.path.join(urdf_path, "xarm_allegro.urdf")
    vis.add_robot("xarm_init", urdf_full)
    vis.robot_dict["xarm_init"].update_cfg(INIT_STATE)
    vis.change_color("xarm_init", [0.7, 0.7, 0.7, 0.3])

    # Destination hand
    urdf_hand = os.path.join(urdf_path, "allegro_hand_description_right.urdf")
    vis.add_robot("dest_hand", urdf_hand, pose=wrist_se3)
    vis.robot_dict["dest_hand"].update_cfg(pregrasp_pose)
    vis.change_color("dest_hand", [0.3, 0.5, 1.0, 0.7])

    # Trajectory robot
    vis.add_robot("traj_robot", urdf_full)
    vis.robot_dict["traj_robot"].update_cfg(traj[0])
    vis.change_color("traj_robot", [0.7, 0.7, 0.7, 0.8])

    # Ghost trail
    ghost_spacing = max(1, len(traj) // 20)
    ghost_positions = list(range(0, len(traj), ghost_spacing))
    for i, pos in enumerate(ghost_positions):
        ghost_name = f"ghost_{i}"
        vis.add_robot(ghost_name, urdf_full)
        vis.robot_dict[ghost_name].update_cfg(traj[pos])
        vis.robot_dict[ghost_name].set_visibility(False)

    # GUI
    with vis.server.gui.add_folder("Planning"):
        vis.server.gui.add_text("Info", initial_value=f"Trajectory: {len(traj)} frames", disabled=True)
        timeline = vis.server.gui.add_slider("Timeline", min=0, max=len(traj)-1, step=1, initial_value=0)

        @timeline.on_update
        def _(_):
            frame = int(timeline.value)
            vis.robot_dict["traj_robot"].update_cfg(traj[frame])
            for i, ghost_pos in enumerate(ghost_positions):
                ghost_name = f"ghost_{i}"
                if frame >= ghost_pos:
                    vis.robot_dict[ghost_name].set_visibility(True)
                    vis.change_color(ghost_name, [0.7, 0.7, 0.7, 0.4])
                else:
                    vis.robot_dict[ghost_name].set_visibility(False)

    print(f"Viewer running at http://localhost:{args.port}")
    while True:
        import time
        time.sleep(1)


if __name__ == "__main__":
    main()
