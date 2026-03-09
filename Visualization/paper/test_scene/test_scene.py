import numpy as np
import os
import transforms3d
import trimesh
import json
import glob
from scipy.spatial.transform import Rotation as Rot

from paradex.visualization.visualizer.viser import ViserViewer

from rsslib.path import obj_path, urdf_path
from rsslib.curobo_util import xarm_init_pose, allegro_init_pose

obj_name = "blue_vase"
allegro_links = [  '/robot/robot/visual/link_base/link1/link2/link3/link4/link5/link6/wrist/base_link/base_link.obj', 
                   '/robot/robot/visual/link_base/link1/link2/link3/link4/link5/link6/wrist/base_link/link_0.0/link_0.0.obj', '/robot/robot/visual/link_base/link1/link2/link3/link4/link5/link6/wrist/base_link/link_0.0/link_1.0/link_1.0.obj', '/robot/robot/visual/link_base/link1/link2/link3/link4/link5/link6/wrist/base_link/link_0.0/link_1.0/link_2.0/link_2.0.obj', '/robot/robot/visual/link_base/link1/link2/link3/link4/link5/link6/wrist/base_link/link_0.0/link_1.0/link_2.0/link_3.0/link_3.0.obj', '/robot/robot/visual/link_base/link1/link2/link3/link4/link5/link6/wrist/base_link/link_0.0/link_1.0/link_2.0/link_3.0/link_3.0_tip/link_3.0_tip.obj', '/robot/robot/visual/link_base/link1/link2/link3/link4/link5/link6/wrist/base_link/link_4.0/link_0.0.obj_1', '/robot/robot/visual/link_base/link1/link2/link3/link4/link5/link6/wrist/base_link/link_4.0/link_5.0/link_1.0.obj_1', '/robot/robot/visual/link_base/link1/link2/link3/link4/link5/link6/wrist/base_link/link_4.0/link_5.0/link_6.0/link_2.0.obj_1', '/robot/robot/visual/link_base/link1/link2/link3/link4/link5/link6/wrist/base_link/link_4.0/link_5.0/link_6.0/link_7.0/link_3.0.obj_1', '/robot/robot/visual/link_base/link1/link2/link3/link4/link5/link6/wrist/base_link/link_4.0/link_5.0/link_6.0/link_7.0/link_7.0_tip/link_3.0_tip.obj_1', '/robot/robot/visual/link_base/link1/link2/link3/link4/link5/link6/wrist/base_link/link_8.0/link_0.0.obj_2', '/robot/robot/visual/link_base/link1/link2/link3/link4/link5/link6/wrist/base_link/link_8.0/link_9.0/link_1.0.obj_2', '/robot/robot/visual/link_base/link1/link2/link3/link4/link5/link6/wrist/base_link/link_8.0/link_9.0/link_10.0/link_2.0.obj_2', '/robot/robot/visual/link_base/link1/link2/link3/link4/link5/link6/wrist/base_link/link_8.0/link_9.0/link_10.0/link_11.0/link_3.0.obj_2', '/robot/robot/visual/link_base/link1/link2/link3/link4/link5/link6/wrist/base_link/link_8.0/link_9.0/link_10.0/link_11.0/link_11.0_tip/link_3.0_tip.obj_2', '/robot/robot/visual/link_base/link1/link2/link3/link4/link5/link6/wrist/base_link/link_12.0/link_12.0_right.obj', '/robot/robot/visual/link_base/link1/link2/link3/link4/link5/link6/wrist/base_link/link_12.0/link_13.0/link_13.0.obj', '/robot/robot/visual/link_base/link1/link2/link3/link4/link5/link6/wrist/base_link/link_12.0/link_13.0/link_14.0/link_14.0.obj', '/robot/robot/visual/link_base/link1/link2/link3/link4/link5/link6/wrist/base_link/link_12.0/link_13.0/link_14.0/link_15.0/link_15.0.obj', '/robot/robot/visual/link_base/link1/link2/link3/link4/link5/link6/wrist/base_link/link_12.0/link_13.0/link_14.0/link_15.0/link_15.0_tip/link_15.0_tip.obj']
class SceneVisualizer(ViserViewer):
    def __init__(self, obj_name):
        super().__init__()

        self.obj_name = obj_name
        self.scene_root = os.path.join(obj_path, obj_name, "test_scene_voxel")  # ← 경로 수정
        self.scene_files = []
        self.current_scene_type = None

        # GUI
        with self.server.gui.add_folder("Scene"):
            self.scene_type_selector = self.server.gui.add_dropdown(
                "Scene Type (N)",  # ← 이름 수정
                options=self._available_scene_types(),
                initial_value=self._available_scene_types()[0] if self._available_scene_types() else "N_10",
            )

            self.scene_idx_slider = self.server.gui.add_slider(
                "Scene Index",
                min=0,
                max=1,
                step=1,
                initial_value=0,
            )

        @self.scene_type_selector.on_update
        def _(event):
            self._on_scene_type_change()

        @self.scene_idx_slider.on_update
        def _(event):
            self._load_current_scene()

        # ㄷ기화
        self._on_scene_type_change()
        allegro_urdf_path = os.path.join(urdf_path, "xarm_allegro.urdf")
        self.add_robot("robot", allegro_urdf_path)
        self.change_color("robot", (0.6, 0.5, 0.88), allegro_links)  # Gray
        self.add_floor()

    def _available_scene_types(self):
        if not os.path.exists(self.scene_root):
            return []
        return sorted(
            d for d in os.listdir(self.scene_root)
            if os.path.isdir(os.path.join(self.scene_root, d))
        )

    def clear_scene(self):
        for name in list(self.obj_dict.keys()):
            self.obj_dict[name]["frame"].remove()
            self.obj_dict[name]["handle"].remove()
            del self.obj_dict[name]
        self.clear_traj()

    def _on_scene_type_change(self):
        self.current_scene_type = self.scene_type_selector.value
        scene_dir = os.path.join(self.scene_root, self.current_scene_type)
        self.scene_files = sorted(
            glob.glob(os.path.join(scene_dir, "*.json")),
            key=lambda x: os.path.basename(x).replace('.json', '')  # ← {idx}.json 형식
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
        
        if len(self.scene_files) == 0:
            return

        scene_path = self.scene_files[self.scene_idx_slider.value]
        print(os.path.basename(scene_path))
        with open(scene_path, "r") as f:
            cfg = json.load(f)
        
        scene = cfg["scene"]
        meta = cfg.get("meta", {})
        
        # mesh
        for mesh_name, info in scene.get("mesh", {}).items():
            mesh = trimesh.load((os.path.join(
                                obj_path, self.obj_name, "raw_mesh", f"{self.obj_name}.obj"
                            )))            
        
            pose = np.eye(4)
            pose[:3, 3] = np.array(info["pose"][:3])
            quat = info["pose"][3:]  # wxyz
            pose[:3, :3] = Rot.from_quat(
                [quat[1], quat[2], quat[3], quat[0]]
            ).as_matrix()

            self.add_trimesh(mesh_name, mesh, pose)
            # if mesh_name == "target":
            #     self.change_color(name=mesh_name, color=(0.2, 0.8, 0.2))  # Green

        # cuboid (장애물 spheres)
        for cuboid_name, info in scene.get("cuboid", {}).items():
            if cuboid_name == "table":
                info["dims"][0] = 10.0
                info["dims"][1] = 10.0
                
            box = trimesh.creation.box(extents=info["dims"])

            pose = np.eye(4)
            pose[:3, 3] = np.array(info["pose"][:3])
            quat = info["pose"][3:]
            pose[:3, :3] = Rot.from_quat(
                [quat[1], quat[2], quat[3], quat[0]]
            ).as_matrix()

            self.add_object(cuboid_name, box, obj_T=pose)
            
            # 장애물은 빨간색
            if cuboid_name.startswith("obs_"):
                self.change_color(name=cuboid_name, color=(70/255, 150/255, 240/255))

            if cuboid_name == "table":
                self.change_color(name=cuboid_name, color=(200/255, 205/255, 210/255))  # Wheat

        print(
            f"[SceneVisualizer] {self.current_scene_type} "
            f"Scene {self.scene_idx_slider.value + 1}/{len(self.scene_files)} "
            f"| N={meta.get('N', 'N/A')} obstacles"
        )
        init_pose = np.concatenate([xarm_init_pose, allegro_init_pose]).reshape(1, -1)
        init_pose[0, 0] = 0.0  # reset base rotation
        # init_pose[0,0] -= np.pi/4
        self.add_traj("tmp", {"robot": init_pose})


vis = SceneVisualizer(obj_name)
vis.start_viewer()