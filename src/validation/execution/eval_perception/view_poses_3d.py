#!/usr/bin/env python3
"""Visualize per-view FPose results in 3D using viser.

Shows object mesh at each view's estimated pose_world, with texture and coordinate axes.
Also shows camera positions.

Usage:
    python view_poses_3d.py --data_root /path/to/mingi_object_test --obj attached_container --episode 20260317_172644 --port 8080
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from paradex.visualization.visualizer.viser import ViserViewer
from scipy.spatial.transform import Rotation as R

MESH_ROOT = Path("/home/mingi/shared_data/object_6d/data/mesh")

COLOR_AXIS_X = (1.0, 0.2, 0.2)
COLOR_AXIS_Y = (0.2, 0.85, 0.2)
COLOR_AXIS_Z = (0.2, 0.2, 1.0)
AXIS_LENGTH = 0.05
AXIS_WIDTH = 3.0


def find_mesh(obj_name):
    for name in [f"{obj_name}.obj", "simplified.obj", "coacd.obj"]:
        p = MESH_ROOT / obj_name / name
        if p.exists():
            return str(p)
    objs = list((MESH_ROOT / obj_name).glob("*.obj"))
    return str(objs[0]) if objs else None


def load_mesh(path):
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])
    return mesh


def add_axes(server, parent_path, length=0.05, width=3.0):
    origin = np.zeros(3)
    for axis_dir, color, label in [
        (np.array([1, 0, 0]), COLOR_AXIS_X, "axis_x"),
        (np.array([0, 1, 0]), COLOR_AXIS_Y, "axis_y"),
        (np.array([0, 0, 1]), COLOR_AXIS_Z, "axis_z"),
    ]:
        server.scene.add_spline_catmull_rom(
            name=f"{parent_path}/{label}",
            positions=np.array([origin, axis_dir * length]),
            color=color,
            line_width=width,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--episode", type=str, required=True)
    parser.add_argument("--depth_type", type=str, default="da3", choices=["da3", "stereo"])
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    capture_dir = Path(args.data_root) / args.obj / args.episode
    pose_dir = capture_dir / "pose" if args.depth_type == "da3" else capture_dir / "pose_stereo"

    # Load camera params
    with open(capture_dir / "cam_param" / "intrinsics.json") as f:
        intr_raw = json.load(f)
    with open(capture_dir / "cam_param" / "extrinsics.json") as f:
        extr_raw = json.load(f)

    img_dir = capture_dir / "images"
    if not img_dir.exists():
        img_dir = capture_dir / "raw" / "images"
    serials = sorted(p.stem for p in img_dir.glob("*.png"))

    extrinsics = {}
    for s in serials:
        T = np.array(extr_raw[s], dtype=np.float64)
        if T.shape == (3, 4):
            T = np.vstack([T, [0, 0, 0, 1]])
        extrinsics[s] = T

    # Load C2R
    c2r_path = capture_dir / "cam_param" / "C2R.npy"
    if c2r_path.exists():
        C2R = np.load(str(c2r_path))
        R2C = np.linalg.inv(C2R)
        print(f"C2R loaded, transforming to robot frame")
    else:
        C2R = np.eye(4)
        R2C = np.eye(4)
        print("No C2R found, using colmap frame")

    # Load mesh
    mesh_path = find_mesh(args.obj)
    mesh = load_mesh(mesh_path)
    print(f"Mesh: {mesh_path}, vertices: {len(mesh.vertices)}")

    # Load poses and transform to robot frame
    poses = {}
    for s in serials:
        p = pose_dir / f"{s}.npy"
        if p.exists():
            pose_colmap = np.load(str(p))
            poses[s] = R2C @ pose_colmap  # colmap → robot frame

    # Transform to robot frame
    # extrinsics[s] is used as: pose_world = inv(ext) @ pose_cam in FPose pipeline
    # So ext is W2C (world-to-camera), C2W = inv(ext)
    # C2R transforms colmap world → robot frame
    cam_poses_robot = {}
    for s in serials:
        C2W_colmap = np.linalg.inv(extrinsics[s])
        cam_poses_robot[s] = R2C @ C2W_colmap

    print(f"Loaded {len(poses)} poses (robot frame)")

    # Start viser
    vis = ViserViewer(port_number=args.port)

    # Lighting
    vis.server.scene.add_light_ambient("/lights/ambient", intensity=1.0, color=(255, 255, 255))
    vis.server.scene.add_light_point("/lights/point", color=(255, 255, 255), intensity=5.0,
                                      position=(0.0, 0.0, 3.0), cast_shadow=False)

    # Add object mesh at each pose_world
    colors = [
        (128, 0, 128),    # purple
        (0, 128, 128),    # teal
        (128, 128, 0),    # olive
        (0, 0, 200),      # blue
        (200, 0, 0),      # red
        (0, 200, 0),      # green
    ]

    for i, (s, pose_world) in enumerate(poses.items()):
        name = f"pose_{s}"
        vis.add_object(name, mesh, obj_T=pose_world)

        # Add large coordinate axes
        frame_path = f"/objects/{name}_frame"
        add_axes(vis.server, frame_path, length=AXIS_LENGTH, width=AXIS_WIDTH)

    # Add cameras using paradex's add_camera (in robot frame)
    with open(capture_dir / "cam_param" / "intrinsics.json") as f:
        intr_full = json.load(f)
    for s in serials:
        vis.add_camera(s, cam_poses_robot[s], intr_full[s], color=(0.0, 1.0, 0.0), size=0.05)

    # Add ground grid at z=0.037 (table surface in robot frame)
    table_z = 0.037
    grid_size = 1.5
    grid_step = 0.1
    grid_color = (0.3, 0.3, 0.3)
    for i in np.arange(-grid_size/2, grid_size/2 + grid_step, grid_step):
        vis.server.scene.add_spline_catmull_rom(
            f"/grid/x_{i:.2f}",
            positions=np.array([[i, -grid_size/2, table_z], [i, grid_size/2, table_z]]),
            color=grid_color, line_width=1.0,
        )
        vis.server.scene.add_spline_catmull_rom(
            f"/grid/y_{i:.2f}",
            positions=np.array([[-grid_size/2, i, table_z], [grid_size/2, i, table_z]]),
            color=grid_color, line_width=1.0,
        )

    # GT pose if exists
    gt_path = pose_dir / "gt.npy"
    if gt_path.exists():
        gt_pose = R2C @ np.load(str(gt_path))
        vis.add_object("gt_pose", mesh, obj_T=gt_pose)
        frame_path = "/objects/gt_pose_frame"
        add_axes(vis.server, frame_path, length=AXIS_LENGTH * 2, width=AXIS_WIDTH * 2)
        print("GT pose added (larger axes)")

    print(f"\nViser running on http://localhost:{args.port}")
    print(f"Showing {len(poses)} per-view poses + {len(serials)} cameras")

    vis.start_viewer()


if __name__ == "__main__":
    main()