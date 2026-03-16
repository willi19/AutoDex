import os
import json
import numpy as np
import trimesh

from paradex.visualization.visualizer.viser import ViserViewer
from autodex.utils.path import obj_path

# ── Color palette ──────────────────────────────────────────────────────────────
COLOR_OBB    = (0.27, 0.51, 1.00)
COLOR_AXIS_X = (1.00, 0.20, 0.20)
COLOR_AXIS_Y = (0.20, 0.85, 0.20)
COLOR_AXIS_Z = (0.20, 0.20, 1.00)

LINE_WIDTH_OBB  = 4.0
LINE_WIDTH_AXIS = 3.0
# ──────────────────────────────────────────────────────────────────────────────

available_objects = sorted([
    d for d in os.listdir(obj_path)
    if os.path.isdir(os.path.join(obj_path, d))
    and os.path.exists(os.path.join(obj_path, d, "raw_mesh", f"{d}.obj"))
])

print(f"Found {len(available_objects)} objects")

vis = ViserViewer()


def load_mesh(path):
    mesh = trimesh.load(path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh


def clear_scene():
    for name in list(vis.obj_dict.keys()):
        try:
            vis.obj_dict[name]['frame'].remove()
        except Exception:
            pass
    vis.obj_dict.clear()
    vis.frame_nodes.clear()


def add_obb(parent_frame, obb_extents, obb_transform, axis_len):
    """Add OBB wireframe and axes as children of parent_frame."""
    half = obb_extents / 2
    corners_local = np.array([
        [-half[0], -half[1], -half[2]],
        [ half[0], -half[1], -half[2]],
        [ half[0],  half[1], -half[2]],
        [-half[0],  half[1], -half[2]],
        [-half[0], -half[1],  half[2]],
        [ half[0], -half[1],  half[2]],
        [ half[0],  half[1],  half[2]],
        [-half[0],  half[1],  half[2]],
    ])
    corners = (obb_transform[:3, :3] @ corners_local.T).T + obb_transform[:3, 3]

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for ei, (i, j) in enumerate(edges):
        vis.server.scene.add_spline_catmull_rom(
            name=f"{parent_frame}/obb_edge_{ei}",
            positions=np.array([corners[i], corners[j]]),
            color=COLOR_OBB,
            line_width=LINE_WIDTH_OBB,
        )

    origin = np.zeros(3)
    for axis_dir, axis_color, label in [
        (np.array([1, 0, 0]), COLOR_AXIS_X, "axis_x"),
        (np.array([0, 1, 0]), COLOR_AXIS_Y, "axis_y"),
        (np.array([0, 0, 1]), COLOR_AXIS_Z, "axis_z"),
    ]:
        vis.server.scene.add_spline_catmull_rom(
            name=f"{parent_frame}/{label}",
            positions=np.array([origin, axis_dir * axis_len]),
            color=axis_color,
            line_width=LINE_WIDTH_AXIS,
        )


def load_object(obj_name):
    clear_scene()

    obj_dir = os.path.join(obj_path, obj_name)
    mesh_dir = os.path.join(obj_dir, "processed_data", "mesh")
    info_path = os.path.join(obj_dir, "processed_data", "info", "simplified.json")

    # Load OBB info
    with open(info_path, 'r') as f:
        info = json.load(f)
    obb_extents   = np.array(info['obb'])
    obb_transform = np.array(info['obb_transform'])
    axis_len = float(np.max(obb_extents)) * 0.6

    # Spacing between the three meshes
    spacing = float(np.linalg.norm(obb_extents)) * 1.5

    # Three mesh variants side by side
    variants = [
        ("raw",        os.path.join(obj_dir, "raw_mesh", f"{obj_name}.obj"), -spacing),
        ("simplified", os.path.join(mesh_dir, "simplified.obj"),              0.0),
        ("coacd",      os.path.join(mesh_dir, "coacd.obj"),                   spacing),
    ]

    identity = np.eye(4)

    for label, mesh_path, y_off in variants:
        if not os.path.exists(mesh_path):
            print(f"  Skipping {label}: {mesh_path} not found")
            continue

        mesh = load_mesh(mesh_path)
        pose = identity.copy()
        pose[1, 3] = y_off

        name = f"{obj_name}_{label}"
        vis.add_object(name, mesh, obj_T=pose)

        fp = f"/objects/{name}_frame"
        add_obb(fp, obb_extents, obb_transform, axis_len)


    print(f"Loaded: {obj_name}  (raw / simplified / coacd)")


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

if available_objects:
    load_object(available_objects[0])

vis.start_viewer()
