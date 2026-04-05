"""Viser viewer for rebuttal test scenes + coverage results + grasp visualization.

Visualizes:
- Scene obstacles (cuboids as semi-transparent boxes)
- Object mesh (red)
- Table (gray)
- Grasp candidates: wrist axes + hand URDF for surviving grasps
- Coverage comparison: survive counts for selected_100 vs baseline_100

Usage:
    conda run -n planner python rebuttal/viewer.py
    conda run -n planner python rebuttal/viewer.py --port 8080
    conda run -n planner python rebuttal/viewer.py --obj attached_container
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import trimesh
import viser
import viser.transforms as tf
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from autodex.utils.path import obj_path, candidate_path, urdf_path
from autodex.utils.conversion import cart2se3

REBUTTAL_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_DIR = os.path.join(REBUTTAL_DIR, "scenes")
CACHE_DIR = os.path.join(REBUTTAL_DIR, "cache")
VERSIONS = ["selected_100", "baseline_100"]

ALLEGRO_URDF = os.path.join(urdf_path, "allegro_hand_description_right.urdf")


def _pose7d_to_mat(pose7d):
    """[x,y,z,qw,qx,qy,qz] -> 4x4."""
    T = np.eye(4)
    T[:3, 3] = pose7d[:3]
    qw, qx, qy, qz = pose7d[3], pose7d[4], pose7d[5], pose7d[6]
    T[:3, :3] = Rot.from_quat([qx, qy, qz, qw]).as_matrix()
    return T


def _mat_to_pos_wxyz(T):
    """4x4 -> (position, wxyz)."""
    pos = T[:3, 3].copy()
    wxyz = Rot.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]]
    return pos, wxyz


def _get_available_objects():
    if not os.path.isdir(SCENE_DIR):
        return []
    return sorted(d for d in os.listdir(SCENE_DIR) if os.path.isdir(os.path.join(SCENE_DIR, d)))


def _load_scene_list(obj_name, scene_type):
    type_dir = os.path.join(SCENE_DIR, obj_name, scene_type)
    if not os.path.isdir(type_dir):
        return []
    scenes = []
    for fname in sorted(os.listdir(type_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(type_dir, fname)) as f:
            scenes.append((fname, json.load(f)))
    return scenes


def _load_cache(obj_name, version, scene_type, idx):
    cache_file = os.path.join(CACHE_DIR, obj_name, version, f"{scene_type}_{idx:03d}.json")
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)
    return None


def _load_grasps(obj_name, version):
    """Load wrist_se3 (object-frame) and pregrasp for a version."""
    root = os.path.join(candidate_path, version, obj_name)
    if not os.path.isdir(root):
        return np.array([]), np.array([])
    wrist_list, pregrasp_list = [], []
    for st in sorted(os.listdir(root)):
        st_dir = os.path.join(root, st)
        if not os.path.isdir(st_dir):
            continue
        for sid in sorted(os.listdir(st_dir)):
            sid_dir = os.path.join(st_dir, sid)
            if not os.path.isdir(sid_dir):
                continue
            for gn in sorted(os.listdir(sid_dir)):
                gdir = os.path.join(sid_dir, gn)
                wf = os.path.join(gdir, "wrist_se3.npy")
                pf = os.path.join(gdir, "pregrasp_pose.npy")
                if os.path.exists(wf) and os.path.exists(pf):
                    wrist_list.append(np.load(wf))
                    pregrasp_list.append(np.load(pf))
    if not wrist_list:
        return np.array([]), np.array([])
    return np.array(wrist_list), np.array(pregrasp_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--obj", type=str, default=None)
    args = parser.parse_args()

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    print(f"Viser server at http://localhost:{args.port}")

    scene_handles = []
    _loading = False

    # Grasp data cache: {(obj, version): (wrist_obj, pregrasp)}
    _grasp_cache = {}

    avail_objs = _get_available_objects()
    if args.obj and args.obj in avail_objs:
        initial_obj = args.obj
    else:
        initial_obj = avail_objs[0] if avail_objs else ""

    with server.gui.add_folder("Scene"):
        gui_obj = server.gui.add_dropdown("Object", options=avail_objs, initial_value=initial_obj)
        gui_scene_type = server.gui.add_dropdown(
            "Scene Type", options=["wall", "shelf", "cluttered"], initial_value="wall")
        gui_idx = server.gui.add_slider("Scene Index", min=0, max=49, step=1, initial_value=0)

    with server.gui.add_folder("Coverage"):
        gui_sel_info = server.gui.add_text("selected_100", initial_value="--", disabled=True)
        gui_base_info = server.gui.add_text("baseline_100", initial_value="--", disabled=True)
        gui_diff_info = server.gui.add_text("difference", initial_value="--", disabled=True)
        gui_meta = server.gui.add_text("scene params", initial_value="--", disabled=True)

    with server.gui.add_folder("Grasp Display"):
        gui_show_grasps = server.gui.add_dropdown(
            "Show grasps", options=["none", "selected_100", "baseline_100"], initial_value="none")
        gui_grasp_idx = server.gui.add_slider("Grasp Index", min=0, max=0, step=1, initial_value=0)
        gui_grasp_info = server.gui.add_text("Grasp", initial_value="--", disabled=True)

    with server.gui.add_folder("Display"):
        gui_show_labels = server.gui.add_checkbox("Show obstacle names", initial_value=False)

    def clear_scene():
        nonlocal scene_handles
        for h in scene_handles:
            h.remove()
        scene_handles = []

    def _get_grasps(obj_name, version):
        key = (obj_name, version)
        if key not in _grasp_cache:
            _grasp_cache[key] = _load_grasps(obj_name, version)
        return _grasp_cache[key]

    def load_and_display():
        nonlocal _loading
        if _loading:
            return
        _loading = True
        try:
            _do_load()
        finally:
            _loading = False

    def _do_load():
        clear_scene()

        obj_name = gui_obj.value
        scene_type = gui_scene_type.value
        scenes = _load_scene_list(obj_name, scene_type)

        if not scenes:
            gui_sel_info.value = "No scenes"
            gui_base_info.value = ""
            gui_diff_info.value = ""
            gui_meta.value = ""
            return

        gui_idx.max = len(scenes) - 1
        idx = min(gui_idx.value, len(scenes) - 1)
        gui_idx.value = idx

        fname, cfg = scenes[idx]
        scene_idx = cfg.get("meta", {}).get("idx", idx)

        # --- Coverage info ---
        sel_cache = _load_cache(obj_name, "selected_100", scene_type, scene_idx)
        base_cache = _load_cache(obj_name, "baseline_100", scene_type, scene_idx)

        sel_surv = sel_cache["n_surviving"] if sel_cache else None
        base_surv = base_cache["n_surviving"] if base_cache else None

        if sel_cache:
            gui_sel_info.value = (
                f"Ours: {sel_cache['n_surviving']}/{sel_cache['n_grasps']} survive "
                f"(coll_pass={sel_cache['n_after_collision']})")
        else:
            gui_sel_info.value = "not computed"

        if base_cache:
            gui_base_info.value = (
                f"Base: {base_cache['n_surviving']}/{base_cache['n_grasps']} survive "
                f"(coll_pass={base_cache['n_after_collision']})")
        else:
            gui_base_info.value = "not computed"

        if sel_surv is not None and base_surv is not None:
            diff = sel_surv - base_surv
            sign = "+" if diff > 0 else ""
            gui_diff_info.value = f"Diff: {sign}{diff}  (ours {sel_surv} vs base {base_surv})"
        else:
            gui_diff_info.value = "--"

        # Meta
        meta = cfg.get("meta", {})
        meta_parts = [f"{k}={v}" for k, v in meta.items() if k not in ["type", "idx"]]
        gui_meta.value = "  ".join(meta_parts)[:150]

        # --- Draw scene ---
        # Object mesh
        mesh_info = cfg.get("mesh", {}).get("target", {})
        obj_pose_mat = np.eye(4)
        if mesh_info and os.path.exists(mesh_info.get("file_path", "")):
            mesh = trimesh.load(mesh_info["file_path"], force="mesh")
            obj_pose_mat = _pose7d_to_mat(mesh_info["pose"])
            pos, wxyz = _mat_to_pos_wxyz(obj_pose_mat)
            h = server.scene.add_mesh_trimesh("/scene/target", mesh=mesh, position=pos, wxyz=wxyz)
            scene_handles.append(h)

        # Cuboids
        cub_colors = {
            "table": (0.6, 0.6, 0.6),
            "wall": (0.3, 0.5, 0.8),
            "shelf_back": (0.3, 0.5, 0.8),
            "shelf_left": (0.4, 0.6, 0.8),
            "shelf_right": (0.4, 0.6, 0.8),
            "shelf_top": (0.5, 0.7, 0.9),
        }
        for cub_name, cub_info in cfg.get("cuboid", {}).items():
            T = _pose7d_to_mat(cub_info["pose"])
            pos, wxyz = _mat_to_pos_wxyz(T)
            color = cub_colors.get(cub_name, (0.7, 0.5, 0.3))
            if cub_name.startswith("clutter"):
                color = (0.8, 0.6, 0.2)

            box = trimesh.creation.box(extents=cub_info["dims"])
            box.visual.vertex_colors = np.array([*color, 0.4]) * 255
            h = server.scene.add_mesh_trimesh(f"/scene/{cub_name}", mesh=box, position=pos, wxyz=wxyz)
            scene_handles.append(h)

            if gui_show_labels.value and cub_name != "table":
                lh = server.scene.add_label(
                    f"/scene/label_{cub_name}", text=cub_name,
                    position=pos + np.array([0, 0, 0.05]))
                scene_handles.append(lh)

        # --- Grasp visualization ---
        grasp_ver = gui_show_grasps.value
        if grasp_ver != "none":
            wrist_obj, pregrasp = _get_grasps(obj_name, grasp_ver)
            if len(wrist_obj) > 0:
                # Transform to world frame
                wrist_world = np.einsum("ij,ajk->aik", obj_pose_mat, wrist_obj)
                n_grasps = len(wrist_world)
                gui_grasp_idx.max = n_grasps - 1
                gi = min(gui_grasp_idx.value, n_grasps - 1)
                gui_grasp_idx.value = gi

                # Draw current grasp: wrist frame axes
                wT = wrist_world[gi]
                pos, wxyz = _mat_to_pos_wxyz(wT)

                # Axes (x=red, y=green, z=blue)
                axis_len = 0.05
                for ax_i, ax_color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
                    end = pos + wT[:3, ax_i] * axis_len
                    pts = np.stack([pos, end])
                    lh = server.scene.add_line_segments(
                        f"/scene/grasp_axis_{ax_i}",
                        points=pts.reshape(1, 2, 3),
                        colors=np.array(ax_color).reshape(1, 3),
                        line_width=3.0,
                    )
                    scene_handles.append(lh)

                # Wrist position sphere
                sh = server.scene.add_icosphere(
                    "/scene/grasp_wrist", radius=0.008, position=pos, color=(255, 100, 0))
                scene_handles.append(sh)

                gui_grasp_info.value = f"#{gi}/{n_grasps}  pregrasp dims={pregrasp[gi].shape}"
            else:
                gui_grasp_info.value = f"No grasps for {grasp_ver}"
        else:
            gui_grasp_info.value = "--"

        # --- Indicator sphere ---
        if sel_surv is not None and base_surv is not None:
            if sel_surv > 0 and base_surv > 0:
                color, label = (0, 200, 0), f"Both ({sel_surv} vs {base_surv})"
            elif sel_surv > 0:
                color, label = (255, 200, 0), f"Ours only ({sel_surv})"
            elif base_surv > 0:
                color, label = (0, 100, 255), f"Base only ({base_surv})"
            else:
                color, label = (200, 0, 0), "Neither (0 vs 0)"

            obj_pos = obj_pose_mat[:3, 3]
            ih = server.scene.add_icosphere(
                "/scene/indicator", radius=0.012, position=obj_pos + [0, 0, 0.15], color=color)
            scene_handles.append(ih)
            lh = server.scene.add_label(
                "/scene/indicator_label", text=label, position=obj_pos + np.array([0, 0, 0.18]))
            scene_handles.append(lh)

        print(f"[viewer] {obj_name}/{scene_type}/{idx}  sel={sel_surv} base={base_surv}")

    # Callbacks
    @gui_obj.on_update
    def _(_): load_and_display()
    @gui_scene_type.on_update
    def _(_): load_and_display()
    @gui_idx.on_update
    def _(_): load_and_display()
    @gui_show_labels.on_update
    def _(_): load_and_display()
    @gui_show_grasps.on_update
    def _(_): load_and_display()
    @gui_grasp_idx.on_update
    def _(_): load_and_display()

    load_and_display()

    print("Viewer ready. Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
