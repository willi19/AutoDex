import numpy as np
import os
import transforms3d
import trimesh
import torch
from scipy.spatial.transform import Rotation as R
import json
import glob
import subprocess
import shutil
import tempfile
import time

from paradex.utils.path import shared_dir
from paradex.visualization.visualizer.viser import ViserViewer

from paradex.utils.path import shared_dir, home_path
from paradex.utils.file_io import find_latest_directory
from paradex.visualization.visualizer.viser import ViserViewer
from paradex.robot.utils import get_robot_urdf_path
from paradex.robot.robot_wrapper import RobotWrapper

from rsslib.path import robot_configs_path, get_object_mesh

# ── Color palette ──────────────────────────────────────────────────────────────
COLOR_SUCCESS = (0.00, 1.00, 0.00)   # Green
COLOR_FAILURE = (1.00, 0.00, 0.00)   # Red
COLOR_OBB     = (0.27, 0.51, 1.00)   # Cyan
COLOR_AXIS_X  = (1.00, 0.20, 0.20)   # Red
COLOR_AXIS_Y  = (0.20, 0.85, 0.20)   # Green
COLOR_AXIS_Z  = (0.20, 0.20, 1.00)   # Blue

LINE_WIDTH_OBB    = 4.0
LINE_WIDTH_AXIS   = 3.0
LINE_WIDTH_NORMAL = 2.5
# ──────────────────────────────────────────────────────────────────────────────

# Find available objects
obj_base_dir = os.path.join(shared_dir, "RSS2026_Mingi", "object", "paradex")
available_objects = sorted([
    d for d in os.listdir(obj_base_dir)
    if os.path.isdir(os.path.join(obj_base_dir, d))
])

vis = ViserViewer()
vis.add_floor(0.0)
vis.add_view_save_gui()


def clear_scene():
    """Remove all dynamically added scene objects and trajectories."""
    for name in list(vis.obj_dict.keys()):
        try:
            vis.obj_dict[name]['frame'].remove()
        except Exception:
            pass
    vis.obj_dict.clear()
    vis.frame_nodes.clear()
    vis.clear_traj()


def load_object(obj_name):
    clear_scene()

    mesh = get_object_mesh(obj_name)

    # Load OBB from simplified.json
    simplified_json_path = os.path.join(
        shared_dir, "RSS2026_Mingi", "object", "paradex", obj_name,
        "processed_data", "info", "simplified.json"
    )
    with open(simplified_json_path, 'r') as f:
        simplified_data = json.load(f)
        obb_extents   = np.array(simplified_data['obb'])            # [x, y, z]
        obb_transform = np.array(simplified_data['obb_transform'])  # 4x4

    table_pose_list = glob.glob(os.path.join(
        shared_dir, "RSS2026_Mingi", "object", "paradex", obj_name,
        "processed_data", "info", "debug", "*", "*_traj.npy"
    ))
    succ_pose_list = glob.glob(os.path.join(
        shared_dir, "RSS2026_Mingi", "object", "paradex", obj_name,
        "processed_data", "info", "debug", "True", "*_traj.npy"
    ))

    traj_dict   = {}
    margin      = 0.05  # 5cm gap between bounding boxes
    grid_spacing = float(np.linalg.norm(obb_extents)) + margin
    grid_cols   = int(np.ceil(np.sqrt(max(len(table_pose_list), 1))))

    # OBB corners in object local frame (obb_transform only, no world transform).
    # Added as children of the object frame → they move with the object.
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
    obb_corners_obj = (obb_transform[:3, :3] @ obb_corners_local.T).T + obb_transform[:3, 3]

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),   # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),   # top face
        (0, 4), (1, 5), (2, 6), (3, 7),   # vertical edges
    ]

    # Axis length proportional to OBB size
    axis_len = float(np.max(obb_extents)) * 0.6

    for idx, table_path in enumerate(table_pose_list):
        obj_traj    = np.load(table_path)
        obj_traj_se3 = np.array([np.eye(4) for _ in range(obj_traj.shape[0])])
        traj_name   = os.path.basename(table_path).removesuffix("_traj.npy")

        info_path = table_path.replace("_traj.npy", "_info.json")
        face_info = None
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                face_info = json.load(f)

        row    = idx // grid_cols - grid_cols // 2
        col    = idx  % grid_cols - grid_cols // 2
        offset = np.array([col * grid_spacing, row * grid_spacing, 0])

        for i in range(obj_traj.shape[0]):
            obj_traj_se3[i][:3, :3] = transforms3d.quaternions.quat2mat(obj_traj[i, 3:7])
            obj_traj_se3[i][:3, 3]  = obj_traj[i, :3] + offset

        traj_dict[traj_name] = obj_traj_se3
        vis.add_object(traj_name, mesh, obj_T=obj_traj_se3[0])

        color = COLOR_SUCCESS if table_path in succ_pose_list else COLOR_FAILURE
        vis.change_color(name=traj_name, color=color)

        fp = f"/objects/{traj_name}_frame"  # parent frame path prefix

        # ── OBB edges as lines (child of object frame → moves with object) ──
        for edge_idx, (i, j) in enumerate(edges):
            vis.server.scene.add_spline_catmull_rom(
                name=f"{fp}/obb_edge_{edge_idx}",
                positions=np.array([obb_corners_obj[i], obb_corners_obj[j]]),
                color=COLOR_OBB,
                line_width=LINE_WIDTH_OBB,
            )

        # ── X / Y / Z axes at object origin (child of object frame) ──
        origin = np.zeros(3)
        for axis_dir, axis_color, axis_label in [
            (np.array([1, 0, 0]), COLOR_AXIS_X, "axis_x"),
            (np.array([0, 1, 0]), COLOR_AXIS_Y, "axis_y"),
            (np.array([0, 0, 1]), COLOR_AXIS_Z, "axis_z"),
        ]:
            vis.server.scene.add_spline_catmull_rom(
                name=f"{fp}/{axis_label}",
                positions=np.array([origin, axis_dir * axis_len]),
                color=axis_color,
                line_width=LINE_WIDTH_AXIS,
            )

        # ── Face center & normal (child of object frame) ──
        if face_info is not None:
            face_center = np.array(face_info['surface_point'])
            face_normal = np.array(face_info['face_normal'])

            vis.server.scene.add_icosphere(
                name=f"{fp}/face_center",
                radius=0.005,
                color=(255, 0, 0),
                position=face_center,
            )
            vis.server.scene.add_spline_catmull_rom(
                name=f"{fp}/face_normal",
                positions=np.array([face_center, face_center + face_normal * 0.05]),
                color=(0.0, 1.0, 0.0),
                line_width=LINE_WIDTH_NORMAL,
            )

    vis.add_traj("asdf", {}, traj_dict)
    print(f"Loaded: {obj_name}  ({len(table_pose_list)} trajectories)")


rec = {
    'active':   False,
    'view':     None,
    'temp_dir': None,
    'idx':      0,
    'total':    0,
    'width':    1280,
    'height':   720,
    'output':   'recording.mp4',
    'fps':      30,
}


def _encode_recording():
    n = rec['idx']
    if n == 0:
        print("No frames captured.")
        return
    out = rec['output']
    print(f"Encoding {n} frames → {out} …")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "warning",
        "-framerate", str(rec['fps']),
        "-i", os.path.join(rec['temp_dir'], "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        out,
    ], check=True)
    shutil.rmtree(rec['temp_dir'])
    rec['temp_dir'] = None
    print(f"Saved: {out}")


# ── GUI ────────────────────────────────────────────────────────────────────────

with vis.server.gui.add_folder("Object Selection"):
    obj_dropdown = vis.server.gui.add_dropdown(
        "Object",
        options=tuple(available_objects),
        initial_value=available_objects[0] if available_objects else "",
    )
    load_btn = vis.server.gui.add_button("Load Object")

@load_btn.on_click
def _(_) -> None:
    load_object(obj_dropdown.value)


with vis.server.gui.add_folder("Record Video"):
    gui_view_json  = vis.server.gui.add_text("View JSON",  initial_value="view.json")
    gui_output_mp4 = vis.server.gui.add_text("Output MP4", initial_value="recording.mp4")
    gui_vid_width  = vis.server.gui.add_number("Width",  initial_value=1280, min=640, max=3840)
    gui_vid_height = vis.server.gui.add_number("Height", initial_value=720,  min=480, max=2160)
    gui_vid_fps    = vis.server.gui.add_slider("FPS", min=10, max=60, step=1, initial_value=30)
    start_rec_btn  = vis.server.gui.add_button("Start Recording")
    stop_rec_btn   = vis.server.gui.add_button("Stop Recording", disabled=True)

@start_rec_btn.on_click
def _(_) -> None:
    view_json = gui_view_json.value
    if not os.path.exists(view_json):
        print(f"View file not found: {view_json}  — save a view first (Save/Load View folder).")
        return
    with open(view_json) as f:
        rec['view'] = json.load(f)
    rec['temp_dir'] = tempfile.mkdtemp()
    rec['idx']    = 0
    rec['total']  = vis.num_frames
    rec['width']  = int(gui_vid_width.value)
    rec['height'] = int(gui_vid_height.value)
    rec['output'] = gui_output_mp4.value
    rec['fps']    = int(gui_vid_fps.value)
    rec['active'] = True
    start_rec_btn.disabled = True
    stop_rec_btn.disabled  = False
    print(f"Recording started: {rec['total']} frames → {rec['output']}")

@stop_rec_btn.on_click
def _(_) -> None:
    rec['active'] = False   # run loop will encode on next iteration


# Load initial object
if available_objects:
    load_object(available_objects[0])

# ── Main loop (replaces vis.start_viewer) ─────────────────────────────────────
# When recording: drive update_scene + capture in the same thread → no race cond.
# When idle:      delegate to vis.update() as normal.
while True:
    if rec['active']:
        i = rec['idx']
        if i < rec['total']:
            clients = list(vis.server.get_clients().values())
            if clients:
                client = clients[0]
                vis.update_scene(i)
                vis.server.flush()
                client.camera.position = tuple(rec['view']['position'])
                client.camera.wxyz     = tuple(rec['view']['wxyz'])
                frame_path = os.path.join(rec['temp_dir'], f"frame_{i:05d}.png")
                vis.capture_scene_png(frame_path, height=rec['height'], width=rec['width'])
                rec['idx'] += 1
                if (i + 1) % 20 == 0:
                    print(f"  {i + 1}/{rec['total']}")
            else:
                time.sleep(0.1)   # wait for a client to connect
        else:
            # All frames captured — encode and finish
            rec['active'] = False
            start_rec_btn.disabled = False
            stop_rec_btn.disabled  = True
            _encode_recording()
    else:
        # Check if Stop was pressed mid-recording (frames already collected)
        if rec['temp_dir'] is not None:
            start_rec_btn.disabled = False
            stop_rec_btn.disabled  = True
            _encode_recording()
        vis.update()