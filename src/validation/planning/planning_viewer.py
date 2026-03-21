"""
Planning Success Viewer

Interactive viser viewer for planning success rate results.

Heatmap mode: Color-coded spheres showing planning result per grid point.
  - Green:  IK ok + plan always ok
  - Yellow: IK ok + plan stochastic
  - Red:    IK ok + plan always fail
  - Gray:   IK always fail

Usage:
    python src/validation/planning/planning_viewer.py
    python src/validation/planning/planning_viewer.py --data_dir outputs/planning_success_rate --port 8080
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
from autodex.utils.path import urdf_path, obj_path
from autodex.utils.robot_config import INIT_STATE


# ── Constants ─────────────────────────────────────────────────────────────────
COLOR_TABLE = (0.94, 0.94, 0.96)
TABLE_OPACITY = 0.9
TABLE_DIMS = [2, 3, 0.2]
TABLE_POSE_XYZ = [1.1, 0, -0.1]

# Category colors (RGBA)
CAT_COLORS = {
    "ik_fail":       (0.7, 0.7, 0.7),   # gray
    "plan_ok":       (0.3, 0.9, 0.3),   # green
    "plan_stoch":    (0.9, 0.9, 0.2),   # yellow
    "plan_fail":     (0.9, 0.2, 0.2),   # red
}

CAT_LABELS = {
    "ik_fail":    "IK fail",
    "plan_ok":    "IK+Plan ok",
    "plan_stoch": "Plan stochastic",
    "plan_fail":  "IK ok, Plan fail",
}


def categorize(result):
    ik_mean = result.get("ik_mean", 0)
    rate = result["success_rate"]
    if ik_mean == 0:
        return "ik_fail"
    elif rate == 1.0:
        return "plan_ok"
    elif rate == 0.0:
        return "plan_fail"
    else:
        return "plan_stoch"


def load_planning_data(data_dir):
    objects = {}
    for obj_name in sorted(os.listdir(data_dir)):
        obj_dir = os.path.join(data_dir, obj_name)
        if not os.path.isdir(obj_dir) or obj_name == "plots":
            continue

        json_files = [f for f in os.listdir(obj_dir)
                      if f.startswith("plan_vs_ik_") and f.endswith(".json")
                      and "partial" not in f and "prev" not in f]
        if not json_files:
            continue

        with open(os.path.join(obj_dir, json_files[0])) as f:
            data = json.load(f)

        results = data["results"]
        pose_indices = sorted(set(r["pose_idx"] for r in results))
        x_offsets = sorted(set(r["x_offset"] for r in results))
        z_rotations = sorted(set(r["z_rotation_deg"] for r in results))

        # Index by (pose_idx, x_offset, z_rotation_deg)
        result_index = {}
        for r in results:
            key = (r["pose_idx"], r["x_offset"], r["z_rotation_deg"])
            result_index[key] = r

        objects[obj_name] = {
            "data": data,
            "result_index": result_index,
            "pose_indices": pose_indices,
            "x_offsets": x_offsets,
            "z_rotations": z_rotations,
        }

    return objects


def get_obj_pose_se3(obj_name, pose_idx, x_off, z_deg):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="outputs/planning_success_rate")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    data = load_planning_data(args.data_dir)
    if not data:
        print(f"No planning data found in {args.data_dir}")
        return

    obj_names = list(data.keys())
    print(f"Found {len(obj_names)} objects: {obj_names}")

    vis = ViserViewer(port_number=args.port)

    # Load robot URDF
    urdf_full = os.path.join(urdf_path, "xarm_allegro.urdf")
    vis.add_robot("robot", urdf_full)

    # Add table
    table_mesh = trimesh.creation.box(extents=TABLE_DIMS)
    table_pose = np.eye(4)
    table_pose[:3, 3] = TABLE_POSE_XYZ
    vis.add_object("table", table_mesh, table_pose)
    vis.change_color("table", COLOR_TABLE + (TABLE_OPACITY,))

    # State
    current = {
        "obj_name": None, "mesh_name": None, "heatmap_handles": [],
        "filtered_points": [],
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

    def clear_heatmap():
        for handle in current["heatmap_handles"]:
            try:
                handle.remove()
            except Exception:
                pass
        current["heatmap_handles"] = []

    def update_heatmap():
        """Show colored spheres for x_offset row at selected pose + z_rotation."""
        clear_heatmap()
        clear_object_mesh()

        obj_name = obj_dropdown.value
        obj_data = data[obj_name]
        pose_indices = obj_data["pose_indices"]
        x_offsets = obj_data["x_offsets"]
        z_rotations = obj_data["z_rotations"]

        pose_idx = pose_indices[min(heatmap_pose_slider.value, len(pose_indices) - 1)]
        z_deg = z_rotations[min(heatmap_z_slider.value, len(z_rotations) - 1)]

        vis.robot_dict["robot"].update_cfg(INIT_STATE)
        vis.robot_dict["robot"].set_visibility(True)

        # Show object mesh for reference
        mid_x = x_offsets[len(x_offsets) // 2]
        obj_pose_se3 = get_obj_pose_se3(obj_name, pose_idx, mid_x, z_deg)
        if obj_pose_se3 is not None:
            obj_pose_se3[1, 3] -= 0.3
            mesh_path = os.path.join(
                obj_path, obj_name, "processed_data", "mesh", "simplified.obj")
            if os.path.exists(mesh_path):
                mesh = trimesh.load(mesh_path, force="mesh")
                mesh_name = f"target_{obj_name}"
                vis.add_object(mesh_name, mesh, obj_pose_se3)
                vis.change_color(mesh_name, [0.6, 0.6, 0.8, 0.5])
                current["mesh_name"] = mesh_name

        sphere_r = 0.015
        z_height = TABLE_POSE_XYZ[2] + TABLE_DIMS[2] / 2 + 0.05

        counts = {"ik_fail": 0, "plan_ok": 0, "plan_stoch": 0, "plan_fail": 0}

        for xi, x_off in enumerate(x_offsets):
            key = (pose_idx, x_off, z_deg)
            result = obj_data["result_index"].get(key)

            if result:
                cat = categorize(result)
            else:
                cat = "ik_fail"

            counts[cat] += 1
            color = CAT_COLORS[cat]

            handle = vis.server.scene.add_icosphere(
                name=f"/heatmap/pt_{xi}",
                radius=sphere_r,
                color=color,
                position=(x_off, 0.0, z_height),
            )
            current["heatmap_handles"].append(handle)

            tick = vis.server.scene.add_label(
                name=f"/heatmap/xtick_{xi}",
                text=f"{x_off:.2f}",
                position=(x_off, -0.03, z_height),
            )
            current["heatmap_handles"].append(tick)

        label_x = vis.server.scene.add_label(
            name="/heatmap/label_x",
            text="X offset →",
            position=(np.mean(x_offsets), -0.06, z_height),
        )
        current["heatmap_handles"].append(label_x)

        rate = obj_data["data"]["overall_mean_rate"]
        heatmap_info.value = (
            f"Object: {obj_name}  Pose: {pose_idx}  Z: {z_deg:.0f}°\n"
            f"Overall rate: {rate*100:.1f}%\n"
            f"OK={counts['plan_ok']}  Stoch={counts['plan_stoch']}  "
            f"Fail={counts['plan_fail']}  IK fail={counts['ik_fail']}"
        )

    def update_heatmap_2d():
        """Show 2D grid of spheres: x_offset × z_rotation at selected pose."""
        clear_heatmap()
        clear_object_mesh()

        obj_name = obj_dropdown.value
        obj_data = data[obj_name]
        pose_indices = obj_data["pose_indices"]
        x_offsets = obj_data["x_offsets"]
        z_rotations = obj_data["z_rotations"]

        pose_idx = pose_indices[min(heatmap_pose_slider.value, len(pose_indices) - 1)]

        vis.robot_dict["robot"].update_cfg(INIT_STATE)
        vis.robot_dict["robot"].set_visibility(True)

        sphere_r = 0.012
        z_height = TABLE_POSE_XYZ[2] + TABLE_DIMS[2] / 2 + 0.05
        y_spacing = 0.04  # spacing between z_rotation columns

        counts = {"ik_fail": 0, "plan_ok": 0, "plan_stoch": 0, "plan_fail": 0}

        for xi, x_off in enumerate(x_offsets):
            for zi, z_deg in enumerate(z_rotations):
                key = (pose_idx, x_off, z_deg)
                result = obj_data["result_index"].get(key)

                if result:
                    cat = categorize(result)
                else:
                    cat = "ik_fail"

                counts[cat] += 1
                color = CAT_COLORS[cat]

                y_pos = (zi - len(z_rotations) / 2) * y_spacing

                handle = vis.server.scene.add_icosphere(
                    name=f"/heatmap/pt_{xi}_{zi}",
                    radius=sphere_r,
                    color=color,
                    position=(x_off, y_pos, z_height),
                )
                current["heatmap_handles"].append(handle)

            # x_offset tick
            tick = vis.server.scene.add_label(
                name=f"/heatmap/xtick_{xi}",
                text=f"{x_off:.2f}",
                position=(x_off, (len(z_rotations) / 2 + 1) * y_spacing, z_height),
            )
            current["heatmap_handles"].append(tick)

        # z_rotation ticks
        for zi, z_deg in enumerate(z_rotations):
            if zi % 2 == 0:
                y_pos = (zi - len(z_rotations) / 2) * y_spacing
                tick = vis.server.scene.add_label(
                    name=f"/heatmap/ztick_{zi}",
                    text=f"{int(z_deg)}°",
                    position=(x_offsets[0] - 0.05, y_pos, z_height),
                )
                current["heatmap_handles"].append(tick)

        rate = obj_data["data"]["overall_mean_rate"]
        heatmap_info.value = (
            f"Object: {obj_name}  Pose: {pose_idx}\n"
            f"Overall rate: {rate*100:.1f}%\n"
            f"OK={counts['plan_ok']}  Stoch={counts['plan_stoch']}  "
            f"Fail={counts['plan_fail']}  IK fail={counts['ik_fail']}\n"
            f"X→ x_offset  Y→ z_rotation"
        )

    def update_single():
        """Show robot + object at a single grid point."""
        clear_heatmap()
        clear_object_mesh()

        obj_name = obj_dropdown.value
        obj_data = data[obj_name]
        pose_indices = obj_data["pose_indices"]
        x_offsets = obj_data["x_offsets"]
        z_rotations = obj_data["z_rotations"]

        pose_idx = pose_indices[min(pose_slider.value, len(pose_indices) - 1)]
        x_off = x_offsets[min(x_slider.value, len(x_offsets) - 1)]
        z_deg = z_rotations[min(z_slider.value, len(z_rotations) - 1)]

        key = (pose_idx, x_off, z_deg)
        result = obj_data["result_index"].get(key)

        # Object mesh
        obj_pose_se3 = get_obj_pose_se3(obj_name, pose_idx, x_off, z_deg)
        mesh_path = os.path.join(
            obj_path, obj_name, "processed_data", "mesh", "simplified.obj")
        if obj_pose_se3 is not None and os.path.exists(mesh_path):
            mesh = trimesh.load(mesh_path, force="mesh")
            mesh_name = f"target_{obj_name}"
            vis.add_object(mesh_name, mesh, obj_pose_se3)
            current["mesh_name"] = mesh_name
            if result:
                cat = categorize(result)
                vis.change_color(mesh_name, CAT_COLORS[cat] + (0.7,))
            else:
                vis.change_color(mesh_name, (0.5, 0.5, 0.5, 0.5))

        vis.robot_dict["robot"].update_cfg(INIT_STATE)
        vis.robot_dict["robot"].set_visibility(True)

        if result:
            cat = categorize(result)
            ik_counts = result.get("ik_counts", [])
            info_text.value = (
                f"Object: {obj_name}\n"
                f"Pose: {pose_idx}  X: {x_off:.2f}  Z: {z_deg:.0f}°\n"
                f"Category: {CAT_LABELS[cat]}\n"
                f"Success: {result['success_count']}/{result['n_trials']} "
                f"({result['success_rate']*100:.0f}%)\n"
                f"IK counts: {ik_counts}  mean={result.get('ik_mean', 0)}"
            )
        else:
            info_text.value = (
                f"Object: {obj_name}\n"
                f"Pose: {pose_idx}  X: {x_off:.2f}  Z: {z_deg:.0f}°\n"
                f"No data"
            )

    def build_filtered_list():
        obj_name = obj_dropdown.value
        obj_data = data[obj_name]
        filt = filter_dropdown.value
        points = []
        for r in obj_data["data"]["results"]:
            cat = categorize(r)
            if filt == "All" or \
               (filt == "IK+Plan ok" and cat == "plan_ok") or \
               (filt == "IK fail" and cat == "ik_fail") or \
               (filt == "Plan stochastic" and cat == "plan_stoch") or \
               (filt == "Plan fail" and cat == "plan_fail"):
                points.append((r["pose_idx"], r["x_offset"], r["z_rotation_deg"]))
        current["filtered_points"] = points
        nav_slider.max = max(len(points) - 1, 1)
        nav_slider.value = 0
        filter_count.value = f"{len(points)} points"

    def navigate_to_filtered():
        points = current["filtered_points"]
        if not points:
            return
        idx = min(nav_slider.value, len(points) - 1)
        pose_idx, x_off, z_deg = points[idx]

        obj_data = data[obj_dropdown.value]
        pose_indices = obj_data["pose_indices"]
        x_offsets = obj_data["x_offsets"]
        z_rotations = obj_data["z_rotations"]

        if pose_idx in pose_indices:
            pose_slider.value = pose_indices.index(pose_idx)
        if x_off in x_offsets:
            x_slider.value = x_offsets.index(x_off)
        if z_deg in z_rotations:
            z_slider.value = z_rotations.index(z_deg)

        update_single()

    def on_mode_change():
        mode = mode_selector.value
        if mode == "Single":
            update_single()
        elif mode == "Heatmap":
            update_heatmap()

    def on_object_change():
        obj_name = obj_dropdown.value
        obj_data = data[obj_name]

        pose_slider.max = max(len(obj_data["pose_indices"]) - 1, 1)
        pose_slider.value = 0
        x_slider.max = max(len(obj_data["x_offsets"]) - 1, 1)
        x_slider.value = len(obj_data["x_offsets"]) // 2
        z_slider.max = max(len(obj_data["z_rotations"]) - 1, 1)
        z_slider.value = 0

        heatmap_pose_slider.max = max(len(obj_data["pose_indices"]) - 1, 1)
        heatmap_pose_slider.value = 0
        heatmap_z_slider.max = max(len(obj_data["z_rotations"]) - 1, 1)
        heatmap_z_slider.value = 0

        current["obj_name"] = obj_name
        build_filtered_list()
        on_mode_change()

    # ── GUI ────────────────────────────────────────────────────────────────────
    with vis.server.gui.add_folder("Planning"):
        obj_dropdown = vis.server.gui.add_dropdown(
            "Object", options=tuple(obj_names), initial_value=obj_names[0],
        )
        mode_selector = vis.server.gui.add_dropdown(
            "Mode", options=("Single", "Heatmap"),
            initial_value="Heatmap",
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
        info_text = vis.server.gui.add_text("Info", initial_value="", disabled=True)

    with vis.server.gui.add_folder("Filter & Navigate"):
        filter_dropdown = vis.server.gui.add_dropdown(
            "Filter",
            options=("All", "IK+Plan ok", "IK fail", "Plan stochastic", "Plan fail"),
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
            "Z Rotation (1D only)", min=0, max=1, step=1, initial_value=0,
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

    @heatmap_pose_slider.on_update
    def _(_) -> None:
        on_mode_change()

    @heatmap_z_slider.on_update
    def _(_) -> None:
        on_mode_change()

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