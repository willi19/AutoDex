"""
Reachability Set Viewer

Interactive viser viewer for reachability grid search results.

Two modes:
- Single: Navigate individual grid points with robot + object
- Heatmap: Color-coded spheres showing reachability across x_offset × z_rotation

Usage:
    python src/validation/planning/reachability_viewer.py
    python src/validation/planning/reachability_viewer.py --data_dir outputs/reachability
"""

import os
import sys
import json
import argparse
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "paradex"))

from paradex.visualization.visualizer.viser import ViserViewer
from autodex.utils.path import urdf_path, obj_path, load_candidate
from autodex.utils.conversion import cart2se3
from autodex.utils.robot_config import INIT_STATE


# ── Constants ─────────────────────────────────────────────────────────────────
COLOR_TABLE = (0.94, 0.94, 0.96)
TABLE_OPACITY = 0.9
TABLE_DIMS = [2, 3, 0.2]
TABLE_POSE_XYZ = [1.1, 0, -0.1]


def load_reachability_data(data_dir):
    """Load all reachability results, return {obj_name: {viz_data, grid_data}}."""
    objects = {}
    for obj_name in sorted(os.listdir(data_dir)):
        obj_dir = os.path.join(data_dir, obj_name)
        if not os.path.isdir(obj_dir):
            continue

        # Find viz json files
        viz_files = [f for f in os.listdir(obj_dir) if f.endswith("_viz.json")]
        if not viz_files:
            continue

        viz_path = os.path.join(obj_dir, viz_files[0])
        grid_path = viz_path.replace("_viz.json", ".json")

        with open(viz_path) as f:
            viz_data = json.load(f)

        grid_data = None
        if os.path.exists(grid_path):
            with open(grid_path) as f:
                grid_data = json.load(f)

        # Index viz_data by (pose_idx, x_offset, z_rotation_deg)
        viz_index = {}
        for entry in viz_data:
            key = (entry["pose_idx"], entry["x_offset"], entry["z_rotation_deg"])
            viz_index[key] = entry

        # Get unique pose indices, x_offsets, z_rotations
        pose_indices = sorted(set(e["pose_idx"] for e in viz_data))
        x_offsets = sorted(set(e["x_offset"] for e in viz_data))
        z_rotations = sorted(set(e["z_rotation_deg"] for e in viz_data))

        # Also include unreachable points from grid data
        if grid_data:
            pose_indices = grid_data["pose_indices"]
            x_offsets = grid_data["x_offsets"]
            z_rotations = grid_data["z_rotations_deg"]

        objects[obj_name] = {
            "viz_index": viz_index,
            "grid_data": grid_data,
            "pose_indices": pose_indices,
            "x_offsets": x_offsets,
            "z_rotations": z_rotations,
        }

    return objects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="outputs/reachability")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    data = load_reachability_data(args.data_dir)
    if not data:
        print(f"No reachability data found in {args.data_dir}")
        return

    obj_names = list(data.keys())
    print(f"Found {len(obj_names)} objects: {obj_names}")

    vis = ViserViewer(port_number=args.port)

    # Load full robot URDF
    urdf_full = os.path.join(urdf_path, "xarm_allegro.urdf")
    vis.add_robot("robot", urdf_full)

    # Add table
    table_mesh = trimesh.creation.box(extents=TABLE_DIMS)
    table_pose = np.eye(4)
    table_pose[:3, 3] = TABLE_POSE_XYZ
    vis.add_object("table", table_mesh, table_pose)
    vis.change_color("table", COLOR_TABLE + (TABLE_OPACITY,))

    # Add hand URDF for grasp candidate visualization
    N_HANDS = 5  # number of hands to show
    urdf_hand = os.path.join(urdf_path, "allegro_hand_description_right.urdf")
    for i in range(N_HANDS):
        vis.add_robot(f"hand_{i}", urdf_hand)
        vis.robot_dict[f"hand_{i}"].set_visibility(False)

    # State
    current = {
        "obj_name": None, "mesh_name": None, "heatmap_handles": [],
        "filtered_points": [],  # list of (pose_idx, x_off, z_deg) for current filter
    }

    def clear_object_mesh():
        if current["mesh_name"] and current["mesh_name"] in vis.obj_dict:
            try:
                vis.obj_dict[current["mesh_name"]]["frame"].remove()
            except Exception:
                pass
            del vis.obj_dict[current["mesh_name"]]
            if current["mesh_name"] in vis.frame_nodes:
                del vis.frame_nodes[current["mesh_name"]]
        current["mesh_name"] = None

    def clear_hands():
        for i in range(N_HANDS):
            vis.robot_dict[f"hand_{i}"].set_visibility(False)

    def clear_heatmap():
        for handle in current["heatmap_handles"]:
            try:
                handle.remove()
            except Exception:
                pass
        current["heatmap_handles"] = []

    def get_grid_result(obj_data, pose_idx, x_off, z_deg):
        """Get trials_with_ik / n_trials for a grid point from grid_data."""
        if obj_data["grid_data"] is None:
            # Fall back to viz_index (only has reachable points)
            key = (pose_idx, x_off, z_deg)
            return 1.0 if key in obj_data["viz_index"] else 0.0
        for r in obj_data["grid_data"]["grid"]:
            if (r["pose_idx"] == pose_idx and
                    abs(r["x_offset"] - x_off) < 1e-4 and
                    abs(r["z_rotation_deg"] - z_deg) < 1e-4):
                return r["trials_with_ik"] / r["n_trials"]
        return 0.0

    def build_filtered_list():
        """Build list of grid points matching the current filter."""
        obj_name = obj_dropdown.value
        obj_data = data[obj_name]
        if obj_data["grid_data"] is None:
            current["filtered_points"] = []
            return

        filt = filter_dropdown.value
        points = []
        for r in obj_data["grid_data"]["grid"]:
            rate = r["trials_with_ik"] / r["n_trials"]
            if filt == "All" or \
               (filt == "Reachable" and rate == 1.0) or \
               (filt == "Unreachable" and rate == 0.0) or \
               (filt == "Partial" and 0.0 < rate < 1.0):
                points.append((r["pose_idx"], r["x_offset"], r["z_rotation_deg"]))

        current["filtered_points"] = points
        nav_slider.max = max(len(points) - 1, 1)
        nav_slider.value = 0
        filter_count.value = f"{len(points)} points"

    def navigate_to_filtered():
        """Jump single view to the current filtered point."""
        points = current["filtered_points"]
        if not points:
            return
        idx = min(nav_slider.value, len(points) - 1)
        pose_idx, x_off, z_deg = points[idx]

        obj_data = data[obj_dropdown.value]
        pose_indices = obj_data["pose_indices"]
        x_offsets = obj_data["x_offsets"]
        z_rotations = obj_data["z_rotations"]

        # Set sliders to match (without triggering multiple updates)
        if pose_idx in pose_indices:
            pose_slider.value = pose_indices.index(pose_idx)
        if x_off in x_offsets:
            x_slider.value = x_offsets.index(x_off)
        if z_deg in z_rotations:
            z_slider.value = z_rotations.index(z_deg)

        update_single()

    def update_heatmap():
        """Show colored spheres for x_offset × z_rotation grid at selected pose.
        Also show robot at default pose and object mesh at the selected pose (x=0° rotation, middle x_offset)."""
        clear_heatmap()
        clear_object_mesh()
        clear_hands()

        obj_name = obj_dropdown.value
        obj_data = data[obj_name]
        pose_indices = obj_data["pose_indices"]
        x_offsets = obj_data["x_offsets"]
        z_rotations = obj_data["z_rotations"]

        pose_idx = pose_indices[min(heatmap_pose_slider.value, len(pose_indices) - 1)]

        # Show robot at default (INIT_STATE) pose
        vis.robot_dict["robot"].update_cfg(INIT_STATE)
        vis.robot_dict["robot"].set_visibility(True)

        z_deg_for_mesh = z_rotations[min(heatmap_z_slider.value, len(z_rotations) - 1)]

        # Show object mesh to the left of the grid (offset in y) for reference
        mid_x = x_offsets[len(x_offsets) // 2]
        obj_pose_se3 = get_obj_pose_se3(obj_name, pose_idx, mid_x, z_deg_for_mesh)
        if obj_pose_se3 is not None:
            # Shift in y so it's to the left of the grid
            obj_pose_se3[1, 3] -= 0.3
            mesh_path = os.path.join(
                obj_path, obj_name, "processed_data", "mesh", "simplified.obj")
            if os.path.exists(mesh_path):
                mesh = trimesh.load(mesh_path, force="mesh")
                mesh_name = f"target_{obj_name}"
                vis.add_object(mesh_name, mesh, obj_pose_se3)
                vis.change_color(mesh_name, [0.6, 0.6, 0.8, 0.5])
                current["mesh_name"] = mesh_name

        # Place spheres along x_offset axis (1D row), z_rotation from slider
        sphere_r = 0.015
        z_height = TABLE_POSE_XYZ[2] + TABLE_DIMS[2] / 2 + 0.05  # slightly above table

        z_deg = z_rotations[min(heatmap_z_slider.value, len(z_rotations) - 1)]

        for xi, x_off in enumerate(x_offsets):
            rate = get_grid_result(obj_data, pose_idx, x_off, z_deg)

            # Color: green(1.0) → yellow(0.5) → red(0.0)
            if rate >= 0.5:
                t = (rate - 0.5) * 2  # 0→1
                color = (1.0 - t, 1.0, 0.0)  # yellow → green
            else:
                t = rate * 2  # 0→1
                color = (1.0, t, 0.0)  # red → yellow

            pos = (x_off, 0.0, z_height)

            handle = vis.server.scene.add_icosphere(
                name=f"/heatmap/pt_{xi}",
                radius=sphere_r,
                color=color,
                position=pos,
            )
            current["heatmap_handles"].append(handle)

            # x_offset tick label
            tick = vis.server.scene.add_label(
                name=f"/heatmap/xtick_{xi}",
                text=f"{x_off:.2f}",
                position=(x_off, -0.03, z_height),
            )
            current["heatmap_handles"].append(tick)

        # Axis label
        label_x = vis.server.scene.add_label(
            name="/heatmap/label_x",
            text=f"X offset →",
            position=(np.mean(x_offsets), -0.06, z_height),
        )
        current["heatmap_handles"].append(label_x)

        # Count reachability stats for this pose + z_rotation
        n_reach = sum(1 for x in x_offsets
                      if get_grid_result(obj_data, pose_idx, x, z_deg) == 1.0)
        n_total = len(x_offsets)

        heatmap_info.value = (
            f"Object: {obj_name}  Pose: {pose_idx}  Z: {z_deg:.0f}°\n"
            f"Reachable: {n_reach}/{n_total}\n"
            f"Green=reachable  Red=unreachable  Yellow=partial"
        )

    def get_obj_pose_se3(obj_name, pose_idx, x_off, z_deg):
        """Compute 4x4 object pose from tabletop file + offsets."""
        tabletop_path = os.path.join(
            obj_path, obj_name, "processed_data", "info", "tabletop", f"{pose_idx}.npy")
        if not os.path.exists(tabletop_path):
            return None
        obj_pose = np.load(tabletop_path)
        obj_pose[0, 3] += x_off
        if z_deg != 0:
            z_rad = np.radians(z_deg)
            c, s = np.cos(z_rad), np.sin(z_rad)
            Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            obj_pose[:3, :3] = Rz @ obj_pose[:3, :3]
        return obj_pose

    def show_hands(obj_name, obj_pose_se3, grasp_version="selected_100"):
        """Load grasp candidates and show N_HANDS hands (mix of forward/backward)."""
        clear_hands()
        if obj_pose_se3 is None:
            return

        try:
            wrist_se3, pregrasp, grasp, scene_info = load_candidate(
                obj_name, obj_pose_se3, grasp_version, shuffle=False)
        except Exception:
            return

        if len(wrist_se3) == 0:
            return

        # Filter: backward facing (wrist z < 0.3)
        backward = wrist_se3[:, 0, 2] < 0.3
        forward_idx = np.where(~backward)[0]
        backward_idx = np.where(backward)[0]

        # Pick hands to show: up to 3 forward + 2 backward (or whatever's available)
        n_fwd = min(3, len(forward_idx))
        n_bwd = min(N_HANDS - n_fwd, len(backward_idx))
        show_idx = []
        if n_fwd > 0:
            step = max(1, len(forward_idx) // n_fwd)
            show_idx.extend(forward_idx[::step][:n_fwd].tolist())
        if n_bwd > 0:
            step = max(1, len(backward_idx) // n_bwd)
            show_idx.extend(backward_idx[::step][:n_bwd].tolist())

        for hi, ci in enumerate(show_idx[:N_HANDS]):
            name = f"hand_{hi}"
            robot = vis.robot_dict[name]
            robot.set_visibility(True)

            # Set hand pose (wrist SE3) and finger config (pregrasp)
            robot._visual_root_frame.position = wrist_se3[ci][:3, 3]
            wxyz = Rot.from_matrix(wrist_se3[ci][:3, :3]).as_quat()[[3, 0, 1, 2]]
            robot._visual_root_frame.wxyz = wxyz
            robot.update_cfg(pregrasp[ci])

            # Color: green=forward, red=backward
            if backward[ci]:
                vis.change_color(name, [1.0, 0.3, 0.3, 0.6])
            else:
                vis.change_color(name, [0.3, 1.0, 0.3, 0.6])

    def update_single():
        """Show robot + object at a single grid point."""
        clear_heatmap()

        obj_name = obj_dropdown.value
        obj_data = data[obj_name]

        pose_indices = obj_data["pose_indices"]
        x_offsets = obj_data["x_offsets"]
        z_rotations = obj_data["z_rotations"]

        pose_idx = pose_indices[min(pose_slider.value, len(pose_indices) - 1)]
        x_off = x_offsets[min(x_slider.value, len(x_offsets) - 1)]
        z_deg = z_rotations[min(z_slider.value, len(z_rotations) - 1)]

        key = (pose_idx, x_off, z_deg)
        entry = obj_data["viz_index"].get(key)

        # Update info text
        if entry:
            n_solutions = len(entry["qpos_list"])
            qpos_idx = min(qpos_slider.value, n_solutions - 1)
            info_text.value = (
                f"Object: {obj_name}\n"
                f"Pose: {pose_idx}  X: {x_off:.2f}  Z: {z_deg:.0f}deg\n"
                f"IK solutions: {n_solutions}  Showing: #{qpos_idx}"
            )
        else:
            info_text.value = (
                f"Object: {obj_name}\n"
                f"Pose: {pose_idx}  X: {x_off:.2f}  Z: {z_deg:.0f}deg\n"
                f"UNREACHABLE (no IK solution)"
            )

        # Clear old object mesh
        clear_object_mesh()

        # Compute object pose
        obj_pose_se3 = None
        if entry:
            obj_pose_se3 = np.array(entry["obj_pose"])
        else:
            obj_pose_se3 = get_obj_pose_se3(obj_name, pose_idx, x_off, z_deg)

        # Load object mesh
        mesh_path = os.path.join(
            obj_path, obj_name, "processed_data", "mesh", "simplified.obj")
        if obj_pose_se3 is not None and os.path.exists(mesh_path):
            mesh = trimesh.load(mesh_path, force="mesh")
            mesh_name = f"target_{obj_name}"
            vis.add_object(mesh_name, mesh, obj_pose_se3)
            current["mesh_name"] = mesh_name
            if not entry:
                vis.change_color(mesh_name, [1.0, 0.3, 0.3, 0.5])

        if entry:
            # Update qpos slider max
            n_solutions = len(entry["qpos_list"])
            qpos_slider.max = max(n_solutions - 1, 1)

            # Set robot qpos
            qpos_idx = min(qpos_slider.value, n_solutions - 1)
            qpos = np.array(entry["qpos_list"][qpos_idx])
            vis.robot_dict["robot"].update_cfg(qpos)
            vis.robot_dict["robot"].set_visibility(True)
        else:
            vis.robot_dict["robot"].set_visibility(False)

        # Show grasp candidate hands
        if show_hands_checkbox.value:
            show_hands(obj_name, obj_pose_se3)
        else:
            clear_hands()

    def on_mode_change():
        if mode_selector.value == "Single":
            update_single()
        else:
            update_heatmap()

    def on_object_change():
        obj_name = obj_dropdown.value
        obj_data = data[obj_name]

        # Update slider ranges for single mode
        pose_slider.max = max(len(obj_data["pose_indices"]) - 1, 1)
        pose_slider.value = 0
        x_slider.max = max(len(obj_data["x_offsets"]) - 1, 1)
        x_slider.value = len(obj_data["x_offsets"]) // 2  # middle x
        z_slider.max = max(len(obj_data["z_rotations"]) - 1, 1)
        z_slider.value = 0
        qpos_slider.value = 0

        # Update heatmap sliders
        heatmap_pose_slider.max = max(len(obj_data["pose_indices"]) - 1, 1)
        heatmap_pose_slider.value = 0
        heatmap_z_slider.max = max(len(obj_data["z_rotations"]) - 1, 1)
        heatmap_z_slider.value = 0

        current["obj_name"] = obj_name
        build_filtered_list()
        on_mode_change()

    # ── GUI ────────────────────────────────────────────────────────────────────
    with vis.server.gui.add_folder("Reachability"):
        obj_dropdown = vis.server.gui.add_dropdown(
            "Object", options=tuple(obj_names), initial_value=obj_names[0],
        )
        mode_selector = vis.server.gui.add_dropdown(
            "Mode", options=("Single", "Heatmap"), initial_value="Single",
        )

    with vis.server.gui.add_folder("Single View"):
        pose_slider = vis.server.gui.add_slider(
            "Pose Index", min=0, max=1, step=1, initial_value=0,
        )
        x_slider = vis.server.gui.add_slider(
            "X Offset", min=0, max=1, step=1, initial_value=0,
        )
        z_slider = vis.server.gui.add_slider(
            "Z Rotation", min=0, max=1, step=1, initial_value=0,
        )
        qpos_slider = vis.server.gui.add_slider(
            "IK Solution #", min=0, max=1, step=1, initial_value=0,
        )
        info_text = vis.server.gui.add_text("Info", initial_value="", disabled=True)
        show_hands_checkbox = vis.server.gui.add_checkbox("Show Grasp Candidates", initial_value=True)

    with vis.server.gui.add_folder("Filter & Navigate"):
        filter_dropdown = vis.server.gui.add_dropdown(
            "Filter", options=("All", "Reachable", "Unreachable", "Partial"),
            initial_value="All",
        )
        nav_slider = vis.server.gui.add_slider(
            "Point #", min=0, max=1, step=1, initial_value=0,
        )
        filter_count = vis.server.gui.add_text("Count", initial_value="", disabled=True)

    with vis.server.gui.add_folder("Heatmap"):
        heatmap_pose_slider = vis.server.gui.add_slider(
            "Pose Index", min=0, max=1, step=1, initial_value=0,
        )
        heatmap_z_slider = vis.server.gui.add_slider(
            "Z Rotation", min=0, max=1, step=1, initial_value=0,
        )
        heatmap_info = vis.server.gui.add_text("Heatmap Info", initial_value="", disabled=True)

    @obj_dropdown.on_update
    def _(_) -> None:
        on_object_change()

    @mode_selector.on_update
    def _(_) -> None:
        on_mode_change()

    @pose_slider.on_update
    def _(_) -> None:
        if mode_selector.value == "Single":
            update_single()

    @x_slider.on_update
    def _(_) -> None:
        if mode_selector.value == "Single":
            update_single()

    @z_slider.on_update
    def _(_) -> None:
        if mode_selector.value == "Single":
            update_single()

    @qpos_slider.on_update
    def _(_) -> None:
        if mode_selector.value == "Single":
            update_single()

    @show_hands_checkbox.on_update
    def _(_) -> None:
        if mode_selector.value == "Single":
            if show_hands_checkbox.value:
                # Need to recompute — call update_single
                update_single()
            else:
                clear_hands()

    @heatmap_pose_slider.on_update
    def _(_) -> None:
        if mode_selector.value == "Heatmap":
            update_heatmap()

    @heatmap_z_slider.on_update
    def _(_) -> None:
        if mode_selector.value == "Heatmap":
            update_heatmap()

    @filter_dropdown.on_update
    def _(_) -> None:
        build_filtered_list()
        if current["filtered_points"]:
            navigate_to_filtered()

    @nav_slider.on_update
    def _(_) -> None:
        navigate_to_filtered()

    # Initialize
    on_object_change()
    vis.add_floor(0.0)
    vis.start_viewer()


if __name__ == "__main__":
    main()
