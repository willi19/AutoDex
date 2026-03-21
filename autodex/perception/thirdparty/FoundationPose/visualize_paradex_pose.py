#!/usr/bin/env python3
"""
Viser visualization for a single posed mesh and camera frustums.
"""

import argparse
import glob
import json
import os
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation as R


def load_pose(pose_path: str):
    pose = np.loadtxt(pose_path).reshape(4, 4)
    return pose


def load_pose_dir(pose_dir: str):
    pose_files = sorted(glob.glob(os.path.join(pose_dir, "*.txt")))
    poses = {}
    for pose_file in pose_files:
        image_id = os.path.splitext(os.path.basename(pose_file))[0]
        poses[image_id] = load_pose(pose_file)
    return poses


def load_extrinsics(extrinsics_path: str):
    with open(extrinsics_path, "r") as f:
        extrinsics_data = json.load(f)
    extrinsics = {}
    for key, value in extrinsics_data.items():
        ext_3x4 = np.array(value)
        ext_4x4 = np.vstack([ext_3x4, np.array([0, 0, 0, 1])])
        extrinsics[key] = ext_4x4
    return extrinsics


def load_intrinsics(intrinsics_path: str):
    with open(intrinsics_path, "r") as f:
        intrinsics_data = json.load(f)
    intrinsics = {}
    image_sizes = {}
    for key, value in intrinsics_data.items():
        intrinsics[key] = np.array(value["intrinsics"]).reshape(3, 3)
        image_sizes[key] = (value["height"], value["width"])
    return intrinsics, image_sizes


def list_used_image_ids(data_dir: str):
    masks_dir = os.path.join(data_dir, "masks")
    if not os.path.isdir(masks_dir):
        return set()
    mask_files = glob.glob(os.path.join(masks_dir, "*.png")) + glob.glob(
        os.path.join(masks_dir, "*.jpg")
    )
    return {os.path.splitext(os.path.basename(p))[0] for p in mask_files}


def camera_pose_from_extrinsic(extrinsic):
    cam_to_world = np.linalg.inv(extrinsic)
    position = cam_to_world[:3, 3]
    rot = R.from_matrix(cam_to_world[:3, :3])
    quat_xyzw = rot.as_quat()
    wxyz = (quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2])
    return wxyz, position


def fov_from_intrinsics(K, width, height):
    fy = K[1, 1]
    fov_y = 2.0 * np.arctan2(0.5 * height, fy)
    aspect = float(width) / float(height)
    return fov_y, aspect


def load_mesh(mesh_file: str):
    loaded = trimesh.load(mesh_file)
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)
        ]
        if not meshes:
            raise RuntimeError("Scene contains no valid Trimesh objects")
        return trimesh.util.concatenate(meshes)
    return loaded


def main():
    parser = argparse.ArgumentParser(description="Viser view for posed mesh + cameras")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="demo_data/baby_beaker_demo",
        help="Data directory with extrinsics/intrinsics",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/baby_beaker_pose",
        help="Output directory containing pose files",
    )
    parser.add_argument(
        "--pose-file",
        type=str,
        default="optimized_pose_world.txt",
        help="Pose filename inside output-dir",
    )
    parser.add_argument(
        "--step3-dir",
        type=str,
        default="ob_in_world",
        help="Subdir for step3 world poses (inside output-dir)",
    )
    parser.add_argument(
        "--step4-file",
        type=str,
        default="selected_pose_world.txt",
        help="Step4 pose filename inside output-dir",
    )
    parser.add_argument(
        "--step5-file",
        type=str,
        default="optimized_pose_world.txt",
        help="Step5 pose filename inside output-dir",
    )
    parser.add_argument(
        "--mesh-file",
        type=str,
        default="demo_data/baby_beaker_demo/mesh/baby_beaker.obj",
        help="Mesh file path",
    )
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    args = parser.parse_args()

    code_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(code_dir, args.data_dir)
    output_dir = os.path.join(code_dir, args.output_dir)
    mesh_file = os.path.join(code_dir, args.mesh_file)
    pose_path = os.path.join(output_dir, args.pose_file)

    extrinsics_path = os.path.join(data_dir, "extrinsics.json")
    intrinsics_path = os.path.join(data_dir, "intrinsics.json")

    extrinsics = load_extrinsics(extrinsics_path)
    intrinsics, image_sizes = load_intrinsics(intrinsics_path)
    used_image_ids = list_used_image_ids(data_dir)

    mesh = load_mesh(mesh_file)

    print(f"Starting viser server on port {args.port}...")
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+y")

    server.scene.add_light_ambient("/lights/ambient", intensity=0.45)
    server.scene.add_light_directional(
        "/lights/key", intensity=1.0, position=(0.5, 0.8, 0.3)
    )
    server.scene.add_light_directional(
        "/lights/fill", intensity=0.4, position=(-0.6, 0.4, -0.5)
    )

    mesh_handles = []

    def add_mesh_handle(name, pose, color=(140, 180, 220)):
        mesh_copy = mesh.copy()
        mesh_copy.apply_transform(pose)
        try:
            return server.scene.add_mesh_trimesh(name, mesh_copy)
        except Exception:
            return server.scene.add_mesh_simple(
                name,
                vertices=mesh_copy.vertices,
                faces=mesh_copy.faces,
                color=color,
            )

    def clear_meshes():
        while mesh_handles:
            mesh_handles.pop().remove()

    def show_step(step_name):
        clear_meshes()
        if step_name == "step3":
            step3_dir = os.path.join(output_dir, args.step3_dir)
            poses = load_pose_dir(step3_dir) if os.path.isdir(step3_dir) else {}
            for image_id, pose in poses.items():
                mesh_handles.append(
                    add_mesh_handle(f"/meshes/step3/{image_id}", pose, color=(140, 180, 220))
                )
        elif step_name == "step4":
            step4_path = os.path.join(output_dir, args.step4_file)
            if os.path.exists(step4_path):
                pose = load_pose(step4_path)
                mesh_handles.append(add_mesh_handle("/meshes/step4", pose, color=(120, 200, 255)))
        elif step_name == "step5":
            step5_path = os.path.join(output_dir, args.step5_file)
            if os.path.exists(step5_path):
                pose = load_pose(step5_path)
                mesh_handles.append(add_mesh_handle("/meshes/step5", pose, color=(200, 160, 255)))

    try:
        step_selector = server.gui.add_dropdown(
            "Pose Step", options=["step3", "step4", "step5"], initial_value="step5"
        )

        @step_selector.on_update
        def _(_evt):
            show_step(step_selector.value)

        show_step(step_selector.value)
    except Exception as exc:
        print(f"GUI controls not available: {exc}")
        if os.path.exists(pose_path):
            mesh_handles.append(add_mesh_handle("/meshes/posed", load_pose(pose_path)))

    cam_palette = [
        (255, 120, 110),
        (120, 200, 255),
        (170, 230, 160),
        (255, 210, 120),
        (200, 150, 255),
        (255, 170, 220),
    ]

    cam_count = 0
    for idx, (image_id, extrinsic) in enumerate(extrinsics.items()):
        if used_image_ids and image_id not in used_image_ids:
            continue
        if image_id not in intrinsics or image_id not in image_sizes:
            continue
        K = intrinsics[image_id]
        height, width = image_sizes[image_id]
        wxyz, position = camera_pose_from_extrinsic(extrinsic)
        fov_y, aspect = fov_from_intrinsics(K, width, height)
        cam_color = cam_palette[idx % len(cam_palette)]

        image_path = os.path.join(data_dir, "images", f"{image_id}.jpg")
        if not os.path.exists(image_path):
            image_path = os.path.join(data_dir, "images", f"{image_id}.png")
        image = iio.imread(image_path) if os.path.exists(image_path) else None

        frustum = server.scene.add_camera_frustum(
            f"/cameras/{image_id}/frustum",
            fov=fov_y,
            aspect=aspect,
            scale=0.07,
            line_width=1.5,
            color=cam_color,
            wxyz=wxyz,
            position=tuple(position),
            image=image,
        )
        cam_count += 1

        @frustum.on_click
        def _(_, wxyz=wxyz, position=position):
            for client in server.get_clients().values():
                client.camera.wxyz = wxyz
                client.camera.position = position

    print("\nVisualization ready!")
    print(f"Pose default: {pose_path}")
    print(f"Added {cam_count} camera frustums")
    print(f"\nOpen your browser to: http://localhost:{args.port}")
    print("Press Ctrl+C to stop the server")

    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")


if __name__ == "__main__":
    main()
