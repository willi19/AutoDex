"""Browse BODex grasp generation outputs interactively.

Usage:
    python src/visualization/grasp_generation/view_bodex.py
"""

import os
import json
import argparse
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot

from paradex.visualization.visualizer.viser import ViserViewer
from autodex.utils.path import obj_path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BODEX_OUTPUT_ROOT = os.path.join(REPO_ROOT, "bodex_outputs")

HAND_URDFS = {
    "allegro": os.path.join(
        REPO_ROOT, "src", "grasp_generation", "BODex", "src", "curobo",
        "content", "assets", "robot", "allegro_description",
        "allegro_hand_description_right.urdf",
    ),
    "inspire": os.path.join(
        REPO_ROOT, "src", "grasp_generation", "BODex", "src", "curobo",
        "content", "assets", "robot", "inspire_description",
        "inspire_hand_right.urdf",
    ),
}


class BODexBrowser(ViserViewer):
    def __init__(self):
        super().__init__()

        self.obj_root = obj_path
        self.obj_pose = np.eye(4)

        self.current_hand = None
        self.current_version = None
        self.current_obj = None
        self.current_scene_type = None
        self.current_scene_idx = None

        self.all_grasp_dirs = []
        self.gui_playing.value = True

        hand_list = sorted(d for d in os.listdir(BODEX_OUTPUT_ROOT)
                           if os.path.isdir(os.path.join(BODEX_OUTPUT_ROOT, d)))

        with self.server.gui.add_folder("BODex Grasp Viewer"):
            self.hand_selector = self.server.gui.add_dropdown(
                "Hand", options=hand_list,
                initial_value=hand_list[0] if hand_list else "",
            )
            self.version_selector = self.server.gui.add_dropdown(
                "Version", options=[], initial_value="",
            )
            self.obj_selector = self.server.gui.add_dropdown(
                "Object", options=[], initial_value="",
            )
            self.scene_type_selector = self.server.gui.add_dropdown(
                "Scene Type", options=[], initial_value="",
            )
            self.scene_idx_selector = self.server.gui.add_dropdown(
                "Scene", options=[], initial_value="",
            )
            self.grasp_idx_slider = self.server.gui.add_slider(
                "Grasp Index", min=0, max=1, step=1, initial_value=0,
            )
            self.show_trajectory = self.server.gui.add_checkbox(
                "Show Trajectory", initial_value=False,
            )
            self.metric_text = self.server.gui.add_text(
                "Metrics", initial_value="No grasp loaded", disabled=True,
            )

        self._on_hand_change()

        @self.hand_selector.on_update
        def _(event):
            self.current_hand = self.hand_selector.value
            self._on_hand_visibility_change()

        @self.version_selector.on_update
        def _(event):
            self._on_version_change()

        @self.obj_selector.on_update
        def _(event):
            self._on_object_change()

        @self.scene_type_selector.on_update
        def _(event):
            self._on_scene_type_change()

        @self.scene_idx_selector.on_update
        def _(event):
            self._on_scene_idx_change()

        @self.grasp_idx_slider.on_update
        def _(event):
            self._load_current_grasp()

        @self.show_trajectory.on_update
        def _(event):
            self._load_current_grasp()

        self.squeeze_num = 10

    # --- helpers ---

    def _list_dirs(self, path):
        if not os.path.isdir(path):
            return []
        return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))

    def _bodex_root(self):
        if self.current_hand is None or self.current_version is None:
            return ""
        return os.path.join(BODEX_OUTPUT_ROOT, self.current_hand, self.current_version)

    def _get_hand_urdf(self):
        return HAND_URDFS.get(self.current_hand, HAND_URDFS["allegro"])

    # --- clear ---

    def clear_scene(self):
        for name in list(self.obj_dict.keys()):
            self.obj_dict[name]["frame"].remove()
            self.obj_dict[name]["handle"].remove()
            del self.obj_dict[name]

    # --- cascading updates ---

    def _on_hand_change(self):
        self.current_hand = self.hand_selector.value
        versions = self._list_dirs(os.path.join(BODEX_OUTPUT_ROOT, self.current_hand))
        self.version_selector.options = versions
        if versions:
            self.version_selector.value = versions[0]
            self._on_version_change()

    def _on_version_change(self):
        self.current_version = self.version_selector.value
        objs = self._list_dirs(self._bodex_root())
        self.obj_selector.options = objs
        if objs:
            self.obj_selector.value = objs[0]
            self._on_object_change()

    def _on_object_change(self):
        self.current_obj = self.obj_selector.value
        # Scene types from object data, not bodex output
        scene_types = self._list_dirs(os.path.join(self.obj_root, self.current_obj, "scene"))
        self.scene_type_selector.options = scene_types if scene_types else ["(none)"]
        if scene_types:
            self.scene_type_selector.value = scene_types[0]
            self._on_scene_type_change()

    def _on_scene_type_change(self):
        self.current_scene_type = self.scene_type_selector.value
        # Scene IDs from object data
        scene_dir = os.path.join(self.obj_root, self.current_obj, "scene", self.current_scene_type)
        if os.path.isdir(scene_dir):
            scenes = sorted(
                [f.split(".")[0] for f in os.listdir(scene_dir) if f.endswith(".json")],
                key=lambda x: int(x) if x.isdigit() else x,
            )
        else:
            scenes = []
        self.scene_idx_selector.options = scenes if scenes else ["(none)"]
        if scenes:
            self.scene_idx_selector.value = scenes[0]
            self._on_scene_idx_change()

    def _on_scene_idx_change(self):
        self.current_scene_idx = self.scene_idx_selector.value
        if not self.current_scene_idx or self.current_scene_idx == "(none)":
            return

        # Always load the scene (from object data, independent of hand)
        self._load_scene()

        # Collect grasp dirs from ALL hands that have this scene
        self.all_grasp_dirs = []
        for hand_name in HAND_URDFS:
            hand_root = os.path.join(BODEX_OUTPUT_ROOT, hand_name)
            if not os.path.isdir(hand_root):
                continue
            for version in os.listdir(hand_root):
                scene_path = os.path.join(
                    hand_root, version, self.current_obj,
                    self.current_scene_type, self.current_scene_idx,
                )
                if os.path.isdir(scene_path):
                    for d in os.listdir(scene_path):
                        if os.path.isdir(os.path.join(scene_path, d)) and d not in self.all_grasp_dirs:
                            self.all_grasp_dirs.append(d)

        self.all_grasp_dirs = sorted(set(self.all_grasp_dirs), key=lambda x: int(x) if x.isdigit() else x)

        if not self.all_grasp_dirs:
            self.grasp_idx_slider.disabled = True
            return

        self.grasp_idx_slider.disabled = False
        self.grasp_idx_slider.max = len(self.all_grasp_dirs) - 1
        self.grasp_idx_slider.value = 0
        self._load_current_grasp()

    # --- scene loading (from RSS_2026) ---

    def _load_scene(self):
        self.clear_scene()
        scene_json_path = os.path.join(
            self.obj_root, self.current_obj, "scene",
            self.current_scene_type, f"{self.current_scene_idx}.json",
        )

        if not os.path.exists(scene_json_path):
            print(f"[Warning] Scene json not found: {scene_json_path}")
            return

        with open(scene_json_path, "r") as f:
            cfg = json.load(f)

        scene = cfg["scene"]

        for mesh_name, info in scene.get("mesh", {}).items():
            mesh = trimesh.load(info["file_path"], force="mesh")

            pose = np.eye(4)
            pose[:3, 3] = np.array(info["pose"][:3])
            quat = info["pose"][3:]
            pose[:3, :3] = Rot.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()

            self.add_object(mesh_name, mesh, obj_T=pose)
            if mesh_name == "target":
                self.obj_pose = pose

        for cuboid_name, info in scene.get("cuboid", {}).items():
            box = trimesh.creation.box(extents=info["dims"])
            pose = np.eye(4)
            pose[:3, 3] = np.array(info["pose"][:3])
            quat = info["pose"][3:]
            pose[:3, :3] = Rot.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
            self.add_object(cuboid_name, box, obj_T=pose)

    # --- grasp loading (per-hand robots, visibility toggle) ---

    def _load_current_grasp(self):
        if not self.all_grasp_dirs:
            return

        # Remove all hand robots
        for robot_name in list(self.robot_dict.keys()):
            del self.robot_dict[robot_name]
        self.clear_traj()

        grasp_idx = self.grasp_idx_slider.value
        grasp_dir = self.all_grasp_dirs[grasp_idx]

        # Load and show each hand that has data for this scene
        active_hand = None
        for hand_name, hand_urdf in HAND_URDFS.items():
            grasp_path = os.path.join(
                BODEX_OUTPUT_ROOT, hand_name,
            )
            # Find matching version dir
            if not os.path.isdir(grasp_path):
                continue
            for version in os.listdir(grasp_path):
                full_path = os.path.join(
                    grasp_path, version, self.current_obj,
                    self.current_scene_type, self.current_scene_idx,
                    grasp_dir,
                )
                if not os.path.isdir(full_path):
                    continue

                wrist_se3 = np.load(os.path.join(full_path, "wrist_se3.npy"))
                pregrasp_pose = np.load(os.path.join(full_path, "pregrasp_pose.npy"))
                grasp_pose = np.load(os.path.join(full_path, "grasp_pose.npy"))

                robot_T = self.obj_pose @ wrist_se3
                robot_key = hand_name
                self.add_robot(robot_key, hand_urdf, pose=robot_T)

                if self.show_trajectory.value:
                    traj_list = [pregrasp_pose, grasp_pose]
                    for i in range(self.squeeze_num):
                        traj_list.append(grasp_pose * (i + 2) - pregrasp_pose * (i + 1))
                    traj_dict = {robot_key: np.stack(traj_list)}
                    self.add_traj(f"{robot_key}_traj", traj_dict)
                else:
                    traj_dict = {robot_key: np.stack([grasp_pose])}
                    self.add_traj(f"{robot_key}_traj", traj_dict)

                # Show only selected hand
                visible = (hand_name == self.current_hand)
                self.robot_dict[robot_key].set_visibility(visible)
                if visible:
                    active_hand = hand_name

                break  # only first matching version

        # Metrics for active hand
        info = f"{self.current_hand} | Seed {grasp_dir}"
        if active_hand:
            for version in os.listdir(os.path.join(BODEX_OUTPUT_ROOT, active_hand)):
                bodex_info_path = os.path.join(
                    BODEX_OUTPUT_ROOT, active_hand, version,
                    self.current_obj, self.current_scene_type,
                    self.current_scene_idx, grasp_dir, "bodex_info.npy",
                )
                if os.path.exists(bodex_info_path):
                    bi = np.load(bodex_info_path, allow_pickle=True).item()
                    ge = bi.get("grasp_error", np.array([0]))
                    de = bi.get("dist_error", np.array([0]))
                    info += f" | grasp_err={ge.max():.3f} | dist_err={de.max():.3f}"
                    break
        self.metric_text.value = info

    def _on_hand_visibility_change(self):
        """Toggle visibility when hand selector changes (no reload needed)."""
        for robot_name in list(self.robot_dict.keys()):
            visible = (robot_name == self.current_hand)
            self.robot_dict[robot_name].set_visibility(visible)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    vis = BODexBrowser()
    vis.start_viewer()