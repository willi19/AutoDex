import numpy as np
import os
import transforms3d
import trimesh
import torch
from scipy.spatial.transform import Rotation as R
import json
import glob

from paradex.utils.path import shared_dir
from paradex.visualization.visualizer.viser import ViserViewer

from paradex.utils.path import shared_dir, home_path
from paradex.utils.file_io import find_latest_directory
from paradex.visualization.visualizer.viser import ViserViewer
from paradex.robot.utils import get_robot_urdf_path
from paradex.robot.robot_wrapper import RobotWrapper

from rsslib.path import robot_configs_path, get_object_mesh

obj_name = "attached_container"
mesh = get_object_mesh(obj_name)
        
vis = ViserViewer()

# Load OBB from simplified.json
simplified_json_path = os.path.join(shared_dir, "RSS2026_Mingi", "object", "paradex", obj_name, "processed_data", "info", "simplified.json")
with open(simplified_json_path, 'r') as f:
    simplified_data = json.load(f)
    obb_extents = np.array(simplified_data['obb'])  # [x, y, z]
    obb_transform = np.array(simplified_data['obb_transform'])  # 4x4 matrix
    # print(f"OBB extents: {obb_extents}")

table_pose_list = glob.glob(os.path.join(shared_dir, "RSS2026_Mingi", "object", "paradex", obj_name, "processed_data", "info", "debug", "*", "*_traj.npy"))
failed_pose_list = glob.glob(os.path.join(shared_dir, "RSS2026_Mingi", "object", "paradex", obj_name, "processed_data", "info", "debug", "False", "*_traj.npy"))
succ_pose_list = glob.glob(os.path.join(shared_dir, "RSS2026_Mingi", "object", "paradex", obj_name, "processed_data", "info", "debug", "True", "*_traj.npy"))

traj_dict = {}

# Grid layout params
grid_spacing = 0.2  # 20cm
grid_cols = int(np.ceil(np.sqrt(len(table_pose_list))))  # square grid

for idx, table_path in enumerate(table_pose_list):
    obj_traj = np.load(table_path)
    obj_traj_se3 = np.array([np.eye(4) for _ in range(obj_traj.shape[0])])
    traj_name = os.path.basename(table_path).removesuffix("_traj.npy")
    
    # Load face info
    info_path = table_path.replace("_traj.npy", "_info.json")
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            face_info = json.load(f)
    else:
        face_info = None

    # Grid offset
    row = idx // grid_cols - grid_cols // 2
    col = idx % grid_cols - grid_cols // 2
    offset = np.array([col * grid_spacing, row * grid_spacing, 0])

    for i in range(obj_traj.shape[0]):
        obj_traj_se3[i][:3, :3] = transforms3d.quaternions.quat2mat(obj_traj[i, 3:7])
        obj_traj_se3[i][:3, 3] = obj_traj[i, :3] + offset  # offset 추가
    
    traj_dict[traj_name] = obj_traj_se3
    vis.add_object(traj_name, mesh, obj_T=obj_traj_se3[0])
    if table_path in succ_pose_list:
        vis.change_color(name=traj_name, color=(0.0, 1.0, 0.0))  # Green for success
    else:
        vis.change_color(name=traj_name, color=(1.0, 0.0, 0.0))  # Red for failure
    
    # ========== ADD OBB BOX ==========
    # OBB is in object local frame, so we need to transform it
    init_pose = obj_traj_se3[0]  # 4x4 transform matrix
    init_R = init_pose[:3, :3]
    init_t = init_pose[:3, 3]
    
    # Create OBB corners in local frame (centered at origin)
    half_extents = obb_extents / 2
    obb_corners_local = np.array([
        [-half_extents[0], -half_extents[1], -half_extents[2]],
        [ half_extents[0], -half_extents[1], -half_extents[2]],
        [ half_extents[0],  half_extents[1], -half_extents[2]],
        [-half_extents[0],  half_extents[1], -half_extents[2]],
        [-half_extents[0], -half_extents[1],  half_extents[2]],
        [ half_extents[0], -half_extents[1],  half_extents[2]],
        [ half_extents[0],  half_extents[1],  half_extents[2]],
        [-half_extents[0],  half_extents[1],  half_extents[2]],
    ])
    obb_corners_local = (obb_transform[:3, :3] @ obb_corners_local.T).T + obb_transform[:3, 3]
    # Transform to world frame
    obb_corners_world = (init_R @ obb_corners_local.T).T + init_t
    
    # Draw OBB edges (12 edges of a box)
    edges = [
        # Bottom face
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Top face
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Vertical edges
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    
    for edge_idx, (i, j) in enumerate(edges):
        vis.add_arrow(
            name=f"{traj_name}_obb_edge_{edge_idx}",
            start=obb_corners_world[i],
            end=obb_corners_world[j],
            color=(255, 165, 0),  # Orange color for OBB
            shaft_radius=0.001,
        )
    
    # Optionally: Draw OBB axes (X, Y, Z in object local frame)
    obb_center = init_t
    axis_length = max(obb_extents) * 0.5
    
    # X-axis (red)
    vis.add_arrow(
        name=f"{traj_name}_obb_x",
        start=obb_center,
        end=obb_center + init_R[:, 0] * axis_length,
        color=(255, 0, 0),
        shaft_radius=0.002,
    )
    
    # Y-axis (green)
    vis.add_arrow(
        name=f"{traj_name}_obb_y",
        start=obb_center,
        end=obb_center + init_R[:, 1] * axis_length,
        color=(0, 255, 0),
        shaft_radius=0.002,
    )
    
    # Z-axis (blue)
    vis.add_arrow(
        name=f"{traj_name}_obb_z",
        start=obb_center,
        end=obb_center + init_R[:, 2] * axis_length,
        color=(0, 0, 255),
        shaft_radius=0.002,
    )
    
    # Face center & normal 표시
    if face_info is not None:
        face_center = np.array(face_info['surface_point'])
        face_normal = np.array(face_info['face_normal'])
        
        # Rotate by initial quat and add offset
        init_quat = obj_traj[0, 3:7]
        init_R_face = transforms3d.quaternions.quat2mat(init_quat)
        
        rotated_center = init_R_face @ face_center + obj_traj[0, :3] + offset
        rotated_normal = init_R_face @ face_normal
        
        # Add sphere at face center (red)
        vis.add_sphere(
            name=f"{traj_name}_face_center",
            position=rotated_center,
            radius=0.005,
            color=(1.0, 0.0, 0.0),
        )
        
        # Add arrow for normal (green)
        vis.add_arrow(
            name=f"{traj_name}_face_normal",
            start=rotated_center,
            end=rotated_center + rotated_normal * 0.05,
            color=(0, 255, 0),
            shaft_radius=0.002,
        )
        
        # print(f"{traj_name}: face_idx={face_info['face_idx']}, center={face_info['face_center']}, normal={face_info['face_normal']}, lowest_z={face_info['lowest_z']:.4f}, z_offset={face_info['z_offset']:.4f}")

vis.add_floor(0.0)
vis.add_traj("asdf", {}, traj_dict)
vis.start_viewer()