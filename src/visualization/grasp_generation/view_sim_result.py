"""Replay grasp simulation: hand + object motion + contact arrows.

Usage:
    python src/visualization/grasp_generation/view_sim_result.py --hand allegro --version v3
"""

import os
import json
import argparse
import numpy as np
import trimesh

from paradex.visualization.visualizer.viser import ViserViewer
from autodex.utils.path import obj_path as DEFAULT_OBJ_PATH

obj_path = DEFAULT_OBJ_PATH  # rebound from CLI in __main__

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BODEX_OUTPUT_ROOT = os.path.join(REPO_ROOT, "bodex_outputs")

HAND_URDFS = {
    "allegro": os.path.join(REPO_ROOT, "src", "grasp_generation", "BODex", "src",
                             "curobo", "content", "assets", "robot",
                             "allegro_description", "allegro_hand_description_right.urdf"),
    "inspire": os.path.join(REPO_ROOT, "src", "grasp_generation", "BODex", "src",
                             "curobo", "content", "assets", "robot",
                             "inspire_description", "inspire_hand_right.urdf"),
    "inspire_left": os.path.join(REPO_ROOT, "src", "grasp_generation", "BODex", "src",
                                  "curobo", "content", "assets", "robot",
                                  "inspire_description", "inspire_hand_left.urdf"),
    "inspire_f1": os.path.join(REPO_ROOT, "src", "grasp_generation", "BODex", "src",
                                "curobo", "content", "assets", "robot",
                                "inspire_f1_description", "inspire_f1_hand_right.urdf"),
}

CONTACT_ARROW_SCALE = 0.08


class SimResultViewer(ViserViewer):
    def __init__(self, hand, version):
        super().__init__()

        self.hand = hand
        self.version = version
        self.bodex_root = os.path.join(BODEX_OUTPUT_ROOT, hand, version)
        self.hand_urdf = HAND_URDFS[hand]
        self.sim_traj = None

        self.gui_playing.value = True

        obj_list = self._list_dirs(self.bodex_root)

        with self.server.gui.add_folder("Sim Result Viewer"):
            self.gui_obj = self.server.gui.add_dropdown(
                "Object", options=obj_list,
                initial_value=obj_list[0] if obj_list else "",
            )
            self.gui_scene_type = self.server.gui.add_dropdown(
                "Scene Type", options=[], initial_value="",
            )
            self.gui_scene_id = self.server.gui.add_dropdown(
                "Scene ID", options=[], initial_value="",
            )
            self.gui_filter = self.server.gui.add_dropdown(
                "Filter", options=["All", "Success Only", "Fail Only"],
                initial_value="All",
            )
            self.gui_grasp = self.server.gui.add_slider(
                "Grasp Index", min=0, max=1, step=1, initial_value=0,
            )
            self.gui_show_contacts = self.server.gui.add_checkbox(
                "Show Contacts", initial_value=True,
            )
            self.gui_info = self.server.gui.add_text(
                "Info", initial_value="", disabled=True,
            )

        @self.gui_obj.on_update
        def _(_): self._on_obj_change()
        @self.gui_scene_type.on_update
        def _(_): self._on_scene_type_change()
        @self.gui_scene_id.on_update
        def _(_): self._on_scene_id_change()
        @self.gui_filter.on_update
        def _(_): self._apply_filter()
        @self.gui_grasp.on_update
        def _(_): self._load_grasp()

        # Override update_scene for contact arrows
        self._original_update_scene = self.update_scene
        self.update_scene = self._update_scene_with_contacts

        if obj_list:
            self._on_obj_change()

    @staticmethod
    def _list_dirs(path):
        if not os.path.isdir(path):
            return []
        return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))

    def _clear_all(self):
        for name in list(self.obj_dict.keys()):
            self.obj_dict[name]["frame"].remove()
            self.obj_dict[name]["handle"].remove()
            del self.obj_dict[name]
        for rn in list(self.robot_dict.keys()):
            del self.robot_dict[rn]
        self.clear_traj()
        self._clear_contacts()

    def _clear_contacts(self):
        try:
            self.server.scene.remove("/contacts")
        except Exception:
            pass

    # --- cascading dropdown ---

    def _on_obj_change(self):
        obj = self.gui_obj.value
        types = self._list_dirs(os.path.join(self.bodex_root, obj))
        self.gui_scene_type.options = types if types else ["(none)"]
        if types:
            self.gui_scene_type.value = types[0]
            self._on_scene_type_change()

    def _on_scene_type_change(self):
        path = os.path.join(self.bodex_root, self.gui_obj.value, self.gui_scene_type.value)
        ids = self._list_dirs(path) if os.path.isdir(path) else []
        ids = sorted(ids, key=lambda x: int(x) if x.isdigit() else x)
        self.gui_scene_id.options = ids if ids else ["(none)"]
        if ids:
            self.gui_scene_id.value = ids[0]
            self._on_scene_id_change()

    def _on_scene_id_change(self):
        if self.gui_scene_id.value == "(none)":
            return
        self._apply_filter()

    def _apply_filter(self):
        scene_path = os.path.join(
            self.bodex_root, self.gui_obj.value,
            self.gui_scene_type.value, self.gui_scene_id.value,
        )
        if not os.path.isdir(scene_path):
            return

        filtered = []
        for d in self._list_dirs(scene_path):
            eval_path = os.path.join(scene_path, d, "sim_eval.json")
            if not os.path.exists(eval_path):
                continue
            r = json.load(open(eval_path))
            f = self.gui_filter.value
            if f == "All" or \
               (f == "Success Only" and r.get("success")) or \
               (f == "Fail Only" and not r.get("success")):
                filtered.append(d)

        self.all_grasp_dirs = sorted(filtered, key=lambda x: int(x) if x.isdigit() else x)
        if not self.all_grasp_dirs:
            self.gui_grasp.disabled = True
            self.gui_info.value = f"No grasps ({self.gui_filter.value})"
            return

        n = len(self.all_grasp_dirs)
        with self.server.atomic():
            self.gui_grasp.value = 0
            self.gui_grasp.max = max(n - 1, 1)  # Mantine slider needs max > min to avoid NaN
        self.gui_grasp.disabled = (n <= 1)
        self.gui_info.value = f"{n} grasps ({self.gui_filter.value})"
        self._load_grasp()

    # --- load grasp (no scene, just hand + object) ---

    def _load_grasp(self):
        if not hasattr(self, 'all_grasp_dirs') or not self.all_grasp_dirs:
            return

        was_playing = self.gui_playing.value
        self.gui_playing.value = False
        self._clear_all()

        idx = int(self.gui_grasp.value)
        grasp_dir = self.all_grasp_dirs[idx]
        seed_path = os.path.join(
            self.bodex_root, self.gui_obj.value,
            self.gui_scene_type.value, self.gui_scene_id.value, grasp_dir,
        )

        eval_result = json.load(open(os.path.join(seed_path, "sim_eval.json")))
        success = eval_result.get("success", False)
        reason = eval_result.get("reason", "")

        # Load object mesh at origin (prefer simplified for speed)
        obj_name = self.gui_obj.value
        candidates = [
            os.path.join(obj_path, obj_name, "processed_data", "mesh", "simplified.obj"),
            os.path.join(obj_path, obj_name, "raw_mesh", f"{obj_name}.obj"),
        ]
        mesh_path = next((p for p in candidates if os.path.exists(p)), None)
        if mesh_path:
            obj_mesh = trimesh.load(mesh_path, force="mesh")
            self.add_object("target", obj_mesh, obj_T=np.eye(4))
        else:
            print(f"[warning] no mesh found for {obj_name}")

        # Load hand
        wrist_se3 = np.load(os.path.join(seed_path, "wrist_se3.npy"))
        self.add_robot("robot", self.hand_urdf, pose=wrist_se3)

        # Load trajectory
        traj_path = os.path.join(seed_path, "sim_traj.json")
        self.sim_traj = None
        if os.path.exists(traj_path):
            self.sim_traj = json.load(open(traj_path))
            robot_qpos = np.array(self.sim_traj["robot_qpos"])
            hand_joints = robot_qpos[:, 7:] if robot_qpos.shape[1] > 16 else robot_qpos
            if self.hand in ("inspire", "inspire_left", "inspire_f1") and hand_joints.shape[1] == 12:
                # sim_traj stores mimic-expanded 12 joints; URDF expects 6 actuated
                hand_joints = hand_joints[:, [0, 1, 4, 6, 8, 10]]

            # Object pose trajectory: convert 7D [x,y,z,qw,qx,qy,qz] to 4x4
            obj_traj = {}
            if "object_pose" in self.sim_traj:
                from scipy.spatial.transform import Rotation as Rot
                poses_7d = np.array(self.sim_traj["object_pose"])  # (T, 7)
                T = len(poses_7d)
                poses_4x4 = np.zeros((T, 4, 4))
                poses_4x4[:, 3, 3] = 1.0
                poses_4x4[:, :3, 3] = poses_7d[:, :3]
                # quat wxyz -> scipy xyzw
                quat_xyzw = poses_7d[:, [4, 5, 6, 3]]
                poses_4x4[:, :3, :3] = Rot.from_quat(quat_xyzw).as_matrix()
                obj_traj["target"] = poses_4x4

            self.add_traj("sim_replay", {"robot": hand_joints}, obj_traj)
        else:
            grasp = np.load(os.path.join(seed_path, "grasp_pose.npy"))
            self.add_traj("static", {"robot": np.stack([grasp])})

        # Info
        status = "SUCCESS" if success else f"FAIL ({reason})" if reason else "FAIL"
        n_contacts_total = 0
        if self.sim_traj and "contacts" in self.sim_traj:
            n_contacts_total = sum(len(c) for c in self.sim_traj["contacts"])
        self.gui_info.value = f"{self.hand} | Seed {grasp_dir} | {status} | {n_contacts_total} contacts"

        if self.num_frames > 0:
            self.gui_playing.value = was_playing

    # --- update scene with contact arrows ---

    def _update_scene_with_contacts(self, timestep):
        self._original_update_scene(timestep)
        self._clear_contacts()

        if not self.gui_show_contacts.value or self.sim_traj is None:
            return

        contacts_list = self.sim_traj.get("contacts", [])
        if not contacts_list:
            return

        t = min(int(timestep), len(contacts_list) - 1)
        contacts = contacts_list[t]

        for ci, c in enumerate(contacts):
            pos = np.array(c["pos"])
            normal = np.array(c["normal"])
            # Arrow pointing outward (away from object)
            end = pos - normal * CONTACT_ARROW_SCALE

            # Red line = contact normal (outward)
            self.server.scene.add_spline_catmull_rom(
                f"/contacts/arrow_{ci}",
                positions=np.array([pos, end]),
                color=(1.0, 0.2, 0.2),
                line_width=5.0,
            )
            # Green sphere = contact point
            self.server.scene.add_icosphere(
                f"/contacts/point_{ci}",
                radius=0.005,
                color=(0.2, 1.0, 0.2),
                position=pos,
            )

        # Show external force direction during force phases
        phases = self.sim_traj.get("phase", [])
        if t < len(phases) and phases[t].startswith("force_"):
            fi = int(phases[t].split("_")[1])
            force_dirs = [
                [-1,0,0], [1,0,0],
                [0,-1,0], [0,1,0],
                [0,0,-1], [0,0,1],
            ]
            force_dir = np.array(force_dirs[fi])

            # Object CoM position
            obj_pose = self.sim_traj["object_pose"][t]
            obj_pos = np.array(obj_pose[:3])

            force_end = obj_pos + force_dir * 0.1  # 10cm arrow

            # Blue arrow = external force
            self.server.scene.add_spline_catmull_rom(
                "/contacts/ext_force",
                positions=np.array([obj_pos, force_end]),
                color=(0.2, 0.4, 1.0),
                line_width=5.0,
            )
            # Blue sphere at force origin
            self.server.scene.add_icosphere(
                "/contacts/ext_force_point",
                radius=0.005,
                color=(0.2, 0.4, 1.0),
                position=obj_pos,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", default="allegro")
    parser.add_argument("--version", default="v3")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--obj_path", default=DEFAULT_OBJ_PATH,
                        help="Object root dir (default: paradex)")
    args = parser.parse_args()

    globals()["obj_path"] = args.obj_path  # _load_grasp reads module-level obj_path

    vis = SimResultViewer(args.hand, args.version)
    vis.start_viewer()