#!/usr/bin/env python3
"""
Visualize baby_beaker pose estimation results using viser.
Shows world-space object poses, mesh, and camera positions.
"""

import numpy as np
import trimesh
import viser
import os
import json
import glob
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation as R


def load_poses(pose_dir):
    """Load all pose files from directory."""
    pose_files = sorted(glob.glob(os.path.join(pose_dir, "*.txt")))
    poses = {}
    for pose_file in pose_files:
        image_id = os.path.splitext(os.path.basename(pose_file))[0]
        pose = np.loadtxt(pose_file).reshape(4, 4)
        poses[image_id] = pose
    return poses


def load_extrinsics(extrinsics_path):
    """Load extrinsics from JSON file."""
    with open(extrinsics_path, 'r') as f:
        extrinsics_data = json.load(f)
    
    extrinsics = {}
    for key, value in extrinsics_data.items():
        # Convert 3x4 to 4x4
        ext_3x4 = np.array(value)
        ext_4x4 = np.vstack([ext_3x4, np.array([0, 0, 0, 1])])
        extrinsics[key] = ext_4x4
    return extrinsics


def camera_pose_from_extrinsic(extrinsic):
    """Return camera-to-world pose (wxyz, position) from world-to-camera extrinsic."""
    cam_to_world = np.linalg.inv(extrinsic)
    position = cam_to_world[:3, 3]
    rot = R.from_matrix(cam_to_world[:3, :3])
    quat_xyzw = rot.as_quat()
    wxyz = (quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2])
    return wxyz, position


def fov_from_intrinsics(K, width, height):
    """Compute vertical FOV (radians) and aspect from intrinsics."""
    fy = K[1, 1]
    fov_y = 2.0 * np.arctan2(0.5 * height, fy)
    aspect = float(width) / float(height)
    return fov_y, aspect


def main():
    parser = argparse.ArgumentParser(description="Visualize baby_beaker pose estimation results")
    parser.add_argument('--debug_dir', type=str, default='debug', help='Debug directory with pose outputs')
    parser.add_argument('--mesh_file', type=str, default='demo_data/baby_beaker/mesh/baby_beaker.obj', help='Mesh file path')
    parser.add_argument('--data_dir', type=str, default='demo_data/baby_beaker', help='Data directory with extrinsics')
    parser.add_argument('--port', type=int, default=8080, help='Viser server port')
    parser.add_argument(
        '--show_meshes',
        action='store_true',
        help='Also visualize meshes (default: cameras only).',
    )
    args = parser.parse_args()
    
    code_dir = os.path.dirname(os.path.realpath(__file__))
    debug_dir = os.path.join(code_dir, args.debug_dir)
    mesh_file = os.path.join(code_dir, args.mesh_file)
    data_dir = os.path.join(code_dir, args.data_dir)
    
    # Load world-space poses
    pose_dir = os.path.join(debug_dir, "ob_in_world")
    print(f"Loading poses from {pose_dir}...")
    poses = load_poses(pose_dir)
    print(f"Loaded {len(poses)} poses")

    used_image_ids = set(poses.keys())

    # Load mesh (optional)
    mesh = None
    if args.show_meshes:
        print(f"Loading mesh from {mesh_file}...")
        loaded = trimesh.load(mesh_file)
        if isinstance(loaded, trimesh.Scene):
            meshes = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)]
            if len(meshes) > 0:
                mesh = trimesh.util.concatenate(meshes)
            else:
                raise RuntimeError("Scene contains no valid Trimesh objects")
        else:
            mesh = loaded
    
    # Load extrinsics for camera visualization
    extrinsics_path = os.path.join(data_dir, "extrinsics.json")
    extrinsics = {}
    if os.path.exists(extrinsics_path):
        print(f"Loading extrinsics from {extrinsics_path}...")
        extrinsics = load_extrinsics(extrinsics_path)
        print(f"Loaded {len(extrinsics)} camera extrinsics")
    
    # Load intrinsics for camera frustum
    intrinsics_path = os.path.join(data_dir, "intrinsics_adjusted.json")
    intrinsics = {}
    image_sizes = {}
    if os.path.exists(intrinsics_path):
        print(f"Loading intrinsics from {intrinsics_path}...")
        with open(intrinsics_path, 'r') as f:
            intrinsics_data = json.load(f)
        for key, value in intrinsics_data.items():
            intrinsics[key] = np.array(value["intrinsics_adjusted"]).reshape(3, 3)
            image_sizes[key] = (value["height_adjusted"], value["width_adjusted"])
    
    # Start viser server
    print(f"Starting viser server on port {args.port}...")
    server = viser.ViserServer(port=args.port)
    
    # Add coordinate frame at origin
    server.scene.set_up_direction("+y")
    
    server.scene.add_light_ambient(
        "/lights/ambient",
        intensity=0.55,
    )
    server.scene.add_light_directional(
        "/lights/key",
        intensity=1.0,
        position=(0.5, 0.8, 0.3),
    )
    server.scene.add_light_directional(
        "/lights/fill",
        intensity=0.35,
        position=(-0.6, 0.4, -0.5),
    )
    # Add mesh at each pose
    mesh_handles = []
    if mesh is not None:
        for i, (image_id, pose) in enumerate(sorted(poses.items())):
            # Create a copy of mesh and transform it to world space
            mesh_copy = mesh.copy()
            mesh_copy.apply_transform(pose)

            # Add mesh - viser expects vertices and faces
            try:
                mesh_handle = server.scene.add_mesh_trimesh(
                    f"/meshes/{image_id}",
                    mesh_copy,
                )
                mesh_handles.append(mesh_handle)
            except Exception as e:
                print(f"Warning: Could not add mesh for {image_id}: {e}")
                # Fallback: send raw vertices/faces.
                try:
                    mesh_handle = server.scene.add_mesh_simple(
                        f"/meshes/{image_id}",
                        vertices=mesh_copy.vertices,
                        faces=mesh_copy.faces,
                        color=(100, 150, 200),  # Light blue
                    )
                    mesh_handles.append(mesh_handle)
                except Exception as e2:
                    print(f"Error adding mesh for {image_id}: {e2}")
    
    # Add camera frustums
    camera_handles = []
    cam_palette = [
        (255, 120, 110),
        (120, 200, 255),
        (170, 230, 160),
        (255, 210, 120),
        (200, 150, 255),
        (255, 170, 220),
    ]
    for idx, (image_id, extrinsic) in enumerate(extrinsics.items()):
        if used_image_ids and image_id not in used_image_ids:
            continue
        if image_id in intrinsics and image_id in image_sizes:
            K = intrinsics[image_id]
            height, width = image_sizes[image_id]
            
            try:
                wxyz, position = camera_pose_from_extrinsic(extrinsic)
                fov_y, aspect = fov_from_intrinsics(K, width, height)
                cam_color = cam_palette[idx % len(cam_palette)]

                server.scene.add_camera_frustum(
                    f"/cameras/{image_id}/frustum",
                    fov=fov_y,
                    aspect=aspect,
                    scale=0.07,
                    line_width=1.5,
                    color=cam_color,
                    wxyz=wxyz,
                    position=tuple(position),
                )
                
                camera_handles.append(image_id)
            except Exception as e:
                print(f"Warning: Could not create frustum for {image_id}: {e}")
    
    print(f"\nVisualization ready!")
    print(f"Added {len(mesh_handles)} mesh instances")
    print(f"Added {len(camera_handles)} camera frustums")
    print(f"\nOpen your browser to: http://localhost:{args.port}")
    print("Press Ctrl+C to stop the server")
    
    # Keep server running
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")


if __name__ == "__main__":
    main()
