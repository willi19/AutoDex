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

obj_name = "attached_container"

class SceneVisualizer(ViserViewer):
    def __init__(self, obj_name):
        super().__init__()

        self.obj_name = obj_name
        self.scene_root = os.path.join(obj_path, obj_name, "test_scene_test")
        self.scene_files = []
        self.current_scene_type = None

        # GUI
        with self.server.gui.add_folder("Scene"):
            self.scene_type_selector = self.server.gui.add_dropdown(
                "Scene Type",
                options=self._available_scene_types(),
                initial_value=self._available_scene_types()[0],
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

        # 초기화
        self._on_scene_type_change()
        # self.add_floor(0.0)
        allegro_urdf_path = os.path.join(urdf_path, "xarm_allegro.urdf")
        self.add_robot("robot", allegro_urdf_path)
        init_pose = np.concatenate([xarm_init_pose, allegro_init_pose]).reshape(1, -1)
        self.add_traj("tmp", {"robot": init_pose})

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

    def _on_scene_type_change(self):
        self.current_scene_type = self.scene_type_selector.value
        scene_dir = os.path.join(self.scene_root, self.current_scene_type)
        self.scene_files = sorted(glob.glob(os.path.join(scene_dir, "*.json")), key=lambda x: int(os.path.basename(x).split('.')[0]))

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
    
        with open(scene_path, "r") as f:
            cfg = json.load(f)

        scene = cfg["scene"]
        pose_idx = cfg["meta"].get("pose_idx", None)
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
                self.change_color(name=mesh_name, color=(0.8, 0.2, 0.2))  # Green

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


vis = SceneVisualizer(obj_name)

# # Load scene JSON files
# scene_dir = os.path.join(obj_path, obj_name, "scene")
# scene_types = ["table", "packed", "wall"]  # 보고 싶은 scene type

# all_scenes = []
# for scene_type in scene_types:
#     scene_path = os.path.join(scene_dir, scene_type)
#     if os.path.exists(scene_path):
#         for scene_file in os.listdir(scene_path):
#             if scene_file.endswith('.json'):
#                 all_scenes.append(os.path.join(scene_path, scene_file))

# print(f"Found {len(all_scenes)} scenes")

# # Grid layout
# grid_spacing = 0.5  # 50cm spacing
# grid_cols = int(np.ceil(np.sqrt(len(all_scenes))))

# for scene_idx, scene_path in enumerate(all_scenes[:20]):  # 처음 20개만
#     with open(scene_path, 'r') as f:
#         scene_cfg = json.load(f)
    
#     scene = scene_cfg['scene']
#     meta = scene_cfg['meta']
    
#     # Grid offset
#     row = scene_idx // grid_cols
#     col = scene_idx % grid_cols
#     offset = np.array([col * grid_spacing, row * grid_spacing, 0])
    
#     scene_name = os.path.basename(scene_path).replace('.json', '')
    
#     # Add meshes
#     if 'mesh' in scene:
#         for mesh_name, mesh_info in scene['mesh'].items():
#             # Load mesh
#             mesh_obj = trimesh.load(mesh_info['file_path'], force='mesh')
            
#             # Parse pose [x, y, z, qw, qx, qy, qz]
#             pose_data = mesh_info['pose']
#             pose = np.eye(4)
#             pose[:3, 3] = np.array(pose_data[:3]) + offset
#             quat_wxyz = pose_data[3:]  # [w, x, y, z]
#             quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
#             pose[:3, :3] = transforms3d.quaternions.quat2mat(quat_wxyz)
            
#             # Add to visualizer
#             obj_name_full = f"{scene_name}_{mesh_name}"
#             vis.add_object(obj_name_full, mesh_obj, obj_T=pose)
            
#             # Color coding
#             if mesh_name == "target":
#                 vis.change_color(name=obj_name_full, color=(0.2, 0.8, 0.2))  # Green
#             else:
#                 vis.change_color(name=obj_name_full, color=(0.8, 0.2, 0.2))  # Red (obstacles)
    
#     # Add cuboids
#     if 'cuboid' in scene:
#         for cuboid_name, cuboid_info in scene['cuboid'].items():
#             # Create box mesh
#             dims = cuboid_info['dims']
#             box_mesh = trimesh.creation.box(extents=dims)
            
#             # Parse pose
#             pose_data = cuboid_info['pose']
#             pose = np.eye(4)
#             pose[:3, 3] = np.array(pose_data[:3]) + offset
#             quat_wxyz = pose_data[3:]
#             quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
#             pose[:3, :3] = transforms3d.quaternions.quat2mat(quat_wxyz)
            
#             # Add to visualizer
#             cuboid_name_full = f"{scene_name}_{cuboid_name}"

#             if cuboid_name == "table":
#                 continue
#             vis.add_object(cuboid_name_full, box_mesh, obj_T=pose)
            
#             # Color coding
#             if cuboid_name == "table":
#                 vis.change_color(name=cuboid_name_full, color=(0.5, 0.5, 0.5))  # Gray
#             elif cuboid_name == "wall":
#                 vis.change_color(name=cuboid_name_full, color=(0.6, 0.6, 0.8))  # Blue-gray
    
#     # Add text label for metadata
#     label_pos = offset + np.array([0, 0, 0.3])
#     param_str = ", ".join([f"{k}={v}" for k, v in meta.get('param', {}).items()])
#     print(f"Scene {scene_idx}: {scene_name}, pose_idx={meta.get('pose_idx')}, {param_str}")

# vis.add_floor(0.0)
vis.start_viewer()