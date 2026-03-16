import numpy as np
import os
import trimesh
import json
import glob
from scipy.spatial.transform import Rotation as Rot

from paradex.visualization.visualizer.viser import ViserViewer
from rsslib.path import obj_path, candidate_path, urdf_path


CANDIDATE_VERSION = "revalidate"


class SceneVisualizer(ViserViewer):
    def __init__(self):
        super().__init__()

        self.obj_name = None
        self.scene_root = None
        self.scene_files = []
        self.current_scene_type = None
        self.obj_pose = np.eye(4)
        self.grasp_dirs = []

        # GUI
        with self.server.gui.add_folder("Scene"):
            self.obj_selector = self.server.gui.add_dropdown(
                "Object",
                options=self._available_objects(),
                initial_value=self._available_objects()[0],
            )

            self.scene_type_selector = self.server.gui.add_dropdown(
                "Scene Type",
                options=[],
                initial_value="",
            )

            self.scene_idx_slider = self.server.gui.add_slider(
                "Scene Index",
                min=0,
                max=1,
                step=1,
                initial_value=0,
            )

        with self.server.gui.add_folder("Grasp"):
            self.grasp_idx_slider = self.server.gui.add_slider(
                "Grasp Index",
                min=0,
                max=0,
                step=1,
                initial_value=0,
            )
            self.grasp_info_text = self.server.gui.add_text(
                "Info",
                initial_value="No grasp",
                disabled=True,
            )

        @self.obj_selector.on_update
        def _(event):
            self._on_object_change()

        @self.scene_type_selector.on_update
        def _(event):
            self._on_scene_type_change()

        @self.scene_idx_slider.on_update
        def _(event):
            self._load_current_scene()

        @self.grasp_idx_slider.on_update
        def _(event):
            self._load_current_grasp()

        # Init
        self._on_object_change()
        self.add_floor(0.0)

        allegro_urdf = os.path.join(urdf_path, "allegro_hand_description_right.urdf")
        self.add_robot("robot", allegro_urdf)

    def _available_objects(self):
        objs = sorted(
            d for d in os.listdir(obj_path)
            if os.path.isdir(os.path.join(obj_path, d, "scene"))
        )
        return objs if objs else [""]

    def _available_scene_types(self):
        if self.scene_root is None or not os.path.exists(self.scene_root):
            return []
        return sorted(
            d for d in os.listdir(self.scene_root)
            if os.path.isdir(os.path.join(self.scene_root, d))
        )

    def _on_object_change(self):
        self.obj_name = self.obj_selector.value
        self.scene_root = os.path.join(obj_path, self.obj_name, "scene")

        scene_types = self._available_scene_types()
        self.scene_type_selector.options = scene_types
        if scene_types:
            self.scene_type_selector.value = scene_types[0]
        self._on_scene_type_change()

    def clear_scene(self):
        for name in list(self.obj_dict.keys()):
            self.obj_dict[name]["frame"].remove()
            self.obj_dict[name]["handle"].remove()
            del self.obj_dict[name]

    def _on_scene_type_change(self):
        self.current_scene_type = self.scene_type_selector.value
        if not self.current_scene_type:
            self.scene_files = []
            self.scene_idx_slider.max = 0
            self.scene_idx_slider.value = 0
            self.clear_scene()
            return

        scene_dir = os.path.join(self.scene_root, self.current_scene_type)
        self.scene_files = sorted(
            glob.glob(os.path.join(scene_dir, "*.json")),
            key=lambda x: int(os.path.basename(x).split('.')[0]),
        )

        if len(self.scene_files) == 0:
            self.scene_idx_slider.max = 0
            self.scene_idx_slider.value = 0
            self.clear_scene()
            return

        self.scene_idx_slider.max = len(self.scene_files) - 1
        self.scene_idx_slider.value = 0
        self._load_current_scene()

    def _load_current_scene(self):
        self.clear_scene()
        self.clear_traj()

        if len(self.scene_files) == 0:
            return

        scene_path = self.scene_files[self.scene_idx_slider.value]
        scene_idx_str = os.path.basename(scene_path).replace(".json", "")

        with open(scene_path, "r") as f:
            cfg = json.load(f)

        scene = cfg["scene"]

        # mesh
        for mesh_name, info in scene.get("mesh", {}).items():
            mesh = trimesh.load(info["file_path"], force="mesh")

            pose = np.eye(4)
            pose[:3, 3] = np.array(info["pose"][:3])
            quat = info["pose"][3:]  # wxyz
            pose[:3, :3] = Rot.from_quat(
                [quat[1], quat[2], quat[3], quat[0]]
            ).as_matrix()

            self.add_object(mesh_name, mesh, obj_T=pose)
            if mesh_name == "target":
                self.change_color(name=mesh_name, color=(0.8, 0.2, 0.2))
                self.obj_pose = pose.copy()

        # cuboid
        for cuboid_name, info in scene.get("cuboid", {}).items():
            box = trimesh.creation.box(extents=info["dims"])

            pose = np.eye(4)
            pose[:3, 3] = np.array(info["pose"][:3])
            quat = info["pose"][3:]
            pose[:3, :3] = Rot.from_quat(
                [quat[1], quat[2], quat[3], quat[0]]
            ).as_matrix()

            self.add_object(cuboid_name, box, obj_T=pose)

        print(
            f"[SceneVisualizer] {self.current_scene_type} "
            f"{self.scene_idx_slider.value + 1}/{len(self.scene_files)} "
            f"{os.path.basename(scene_path)}"
        )

        # Update grasp list for this scene
        self._update_grasp_list(scene_idx_str)

    def _update_grasp_list(self, scene_idx_str):
        grasp_root = os.path.join(
            candidate_path, CANDIDATE_VERSION,
            self.obj_name, self.current_scene_type, scene_idx_str,
        )

        if os.path.isdir(grasp_root):
            self.grasp_dirs = sorted(
                [d for d in os.listdir(grasp_root)
                 if os.path.isdir(os.path.join(grasp_root, d))],
                key=lambda x: int(x),
            )
        else:
            self.grasp_dirs = []

        if self.grasp_dirs:
            self.grasp_idx_slider.max = len(self.grasp_dirs) - 1
            self.grasp_idx_slider.value = 0
            self.grasp_info_text.value = f"{len(self.grasp_dirs)} grasps"
            self._load_current_grasp()
        else:
            self.grasp_idx_slider.max = 0
            self.grasp_idx_slider.value = 0
            self.grasp_info_text.value = "No grasps"
            # Hide robot
            if "robot" in self.robot_dict:
                self.robot_dict["robot"].show_visual = False

    def _load_current_grasp(self):
        if not self.grasp_dirs:
            return

        self.clear_traj()

        grasp_dir = self.grasp_dirs[self.grasp_idx_slider.value]
        scene_idx_str = os.path.basename(
            self.scene_files[self.scene_idx_slider.value]
        ).replace(".json", "")

        grasp_path = os.path.join(
            candidate_path, CANDIDATE_VERSION,
            self.obj_name, self.current_scene_type,
            scene_idx_str, grasp_dir,
        )

        # Load wrist SE3 and grasp pose
        wrist_se3 = np.load(os.path.join(grasp_path, "wrist_se3.npy"))
        grasp_pose = np.load(os.path.join(grasp_path, "grasp_pose.npy"))

        # Transform wrist to world frame
        robot_T = self.obj_pose @ wrist_se3

        # Position robot
        if "robot" in self.robot_dict:
            robot = self.robot_dict["robot"]
            robot.show_visual = True
            robot._visual_root_frame.position = robot_T[:3, 3]
            robot._visual_root_frame.wxyz = Rot.from_matrix(
                robot_T[:3, :3]
            ).as_quat()[[3, 0, 1, 2]]

        # Set joint angles via single-frame traj
        self.add_traj("grasp", {"robot": grasp_pose.reshape(1, -1)})

        # Info
        info_str = f"Grasp {grasp_dir} ({self.grasp_idx_slider.value + 1}/{len(self.grasp_dirs)})"

        coll_path = os.path.join(grasp_path, "coll_valid.npy")
        if os.path.exists(coll_path):
            coll_valid = np.load(coll_path).item()
            info_str += f" | coll: {coll_valid}"

        eval_path = os.path.join(grasp_path, "sim_eval", "eval_results.json")
        if os.path.exists(eval_path):
            with open(eval_path, "r") as f:
                eval_results = json.load(f)
            succ = eval_results.get("succ_flag", False)
            if isinstance(succ, list):
                succ = succ[0]
            info_str += f" | sim: {succ}"

        self.grasp_info_text.value = info_str
        print(f"[SceneVisualizer] {info_str}")


vis = SceneVisualizer()
vis.start_viewer()