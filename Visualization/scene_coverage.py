import os
import glob
import json
import numpy as np
import trimesh
import transforms3d
from scipy.spatial.transform import Rotation as Rot

from paradex.utils.path import shared_dir
from paradex.visualization.visualizer.viser import ViserViewer

from rsslib.conversion import se32cart
from rsslib.path import bodex_path, obj_path, code_path, urdf_path

class Visualizer(ViserViewer):
    def __init__(self):
        super().__init__()
        self.add_robot("xarm_allegro", os.path.join(urdf_path, "xarm_allegro.urdf"))

        with self.server.gui.add_folder("BODex Grasp Viewer"):
            self.test_scene_version = None
            self.test_scene_version_list = sorted(os.listdir(os.path.join(code_path, 'coverage_scene')))
            
            self.test_scene_version_selector = self.server.gui.add_dropdown(
                "Test_Scene_Version",
                options=self.test_scene_version_list,
                initial_value=self.test_scene_version_list[0] if self.test_scene_version_list else "",
            )

            self.grasp_version = None
            self.grasp_version_list = []
            self.grasp_version_selector = self.server.gui.add_dropdown(
                "Grasp_Version",
                options=self.grasp_version_list,
                initial_value=self.grasp_version_list[0] if self.grasp_version_list else "",
            )

            self.obj_name = None
            self.obj_list = []
            self.obj_selector = self.server.gui.add_dropdown(
                "Object",
                options=self.obj_list,
                initial_value=self.obj_list[0] if self.obj_list else "",
            )

            self.scene_type = None
            self.scene_type_list = []
            self.scene_type_selector = self.server.gui.add_dropdown(
                "Scene_Type",
                options=self.scene_type_list,
                initial_value=self.scene_type_list[0] if self.scene_type_list else "",
            )

            self.scene_name = None
            self.scene_name_list = []
            self.scene_name_selector = self.server.gui.add_dropdown(
                "Scene_Name",
                options=self.scene_name_list,
                initial_value=self.scene_name_list[0] if self.scene_name_list else "",
            )

            
        @self.test_scene_version_selector.on_update
        def _(event):
            self._on_version_change()
        
        @self.grasp_version_selector.on_update
        def _(event):
            self._on_grasp_version_change()

        @self.scene_type_selector.on_update
        def _(event):
            self._on_scene_type_change()
        
        # Callbacks
        @self.obj_selector.on_update
        def _(event):
            self._on_object_change()
        
        @self.scene_name_selector.on_update
        def _(event):
            self._on_scene_name_change()
        
        self._on_version_change()

    def _on_version_change(self):
        self.test_scene_version = self.test_scene_version_selector.value
        if not self.test_scene_version:
            return
        
        self.grasp_version_list = os.listdir(os.path.join(code_path, 'coverage_scene', self.test_scene_version))
        self.grasp_version_selector.options = self.grasp_version_list
        
        if self.grasp_version is None or self.grasp_version not in self.grasp_version_list:
            self.grasp_version = self.grasp_version_list[0]
            self.grasp_version_selector.value = self.grasp_version
            self._on_grasp_version_change()

        self.obj_list = os.listdir(os.path.join(code_path, 'coverage_scene', self.test_scene_version, self.grasp_version))
        self.obj_selector.options = self.obj_list
        if self.obj_name is not None and self.obj_name in self.obj_list:
            self.obj_selector.value = self.obj_name
            
        self.scene_type_list = os.listdir(os.path.join(code_path, 'coverage_scene', self.test_scene_version, self.grasp_version, self.obj_name))
        self.scene_type_selector.options =self.scene_type_list
        if self.scene_type is not None and self.scene_type in self.scene_type_list:
            self.scene_type_selector.value = self.scene_type

        self.scene_name_list = os.listdir(os.path.join(code_path, 'coverage_scene', self.test_scene_version, self.grasp_version, self.obj_name, self.scene_type))
        self.scene_name_selector.options =self.scene_name_list

    def _on_grasp_version_change(self):
        self.grasp_version = self.grasp_version_selector.value
        print(f"Grasp version changed to: {self.grasp_version}")
        self.obj_list = os.listdir(os.path.join(code_path, 'coverage_scene', self.test_scene_version, self.grasp_version))
        self.obj_selector.options = self.obj_list

        if self.obj_name is None or self.obj_name not in self.obj_list:
            self.obj_name = self.obj_list[0]
            self.obj_selector.value = self.obj_name
            self._on_object_change()
        
        else:
            self.grasp_root_path = os.path.join(code_path, 'candidates', self.grasp_version, self.obj_name)
            
        self.scene_type_list = os.listdir(os.path.join(code_path, 'coverage_scene', self.test_scene_version, self.grasp_version, self.obj_name))
        self.scene_type_selector.options =self.scene_type_list
        if self.scene_type is None or self.scene_type not in self.scene_type_list:
            self.scene_type = self.scene_type_list[0]
            self.scene_type_selector.value = self.scene_type
            self._on_scene_type_change()

        self.scene_name_list = os.listdir(os.path.join(code_path, 'coverage_scene', self.test_scene_version, self.grasp_version, self.obj_name, self.scene_type))
        self.scene_name_selector.options =self.scene_name_list

        if self.scene_name is not None and self.scene_name in self.scene_name_list:
            self.scene_name_selector.value = self.scene_name
        self._on_scene_name_change()

    def _on_object_change(self):
        self.obj_name = self.obj_selector.value
        self.grasp_root_path = os.path.join(code_path, 'candidates', self.grasp_version, self.obj_name)
        self.scene_type_list = os.listdir(os.path.join(code_path, 'coverage_scene', self.test_scene_version, self.grasp_version, self.obj_name))
        self.scene_type_selector.options = self.scene_type_list
        if self.scene_type is None or self.scene_type not in self.scene_type_list:
            self.scene_type = self.scene_type_list[0]
            self.scene_type_selector.value = self.scene_type
        
        self.scene_name_list = os.listdir(os.path.join(code_path, 'coverage_scene', self.test_scene_version, self.grasp_version, self.obj_name, self.scene_type))
        self.scene_name_selector.options =self.scene_name_list
        self.scene_name = self.scene_name_list[0]
        self.scene_name_selector.value = self.scene_name
        self._on_scene_name_change()

    def _on_scene_type_change(self):
        self.scene_type = self.scene_type_selector.value
        self.scene_name_list = os.listdir(os.path.join(code_path, 'coverage_scene', self.test_scene_version, self.grasp_version, self.obj_name, self.scene_type))
        self.scene_name_selector.options = self.scene_name_list
        self.scene_name = self.scene_name_list[0]
        self.scene_name_selector.value = self.scene_name
        self._on_scene_name_change()


    def clear_scene(self):
        """Clear all objects except floor"""
        for name in list(self.obj_dict.keys()):
            self.obj_dict[name]["frame"].remove()
            self.obj_dict[name]["handle"].remove()
            del self.obj_dict[name]
        print(self.obj_dict, "after clear")

    def _load_scene(self):
        print(f"[BODexVisualizer] Loading scene...")
        """Load object mesh from scene json"""
        self.clear_scene()
        scene_json_path = os.path.join(
            obj_path,
            self.obj_name, 
            self.test_scene_version, 
            self.scene_type,
            f"{self.scene_name}.json"
        )
        
        if not os.path.exists(scene_json_path):
            print(f"[Warning] Scene json not found: {scene_json_path}")
            return
    
        with open(scene_json_path, "r") as f:
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
            # Save object pose for grasp transformation
            if mesh_name == "target":
                self.obj_pose = pose
                print(f"[BODexVisualizer] Loaded object pose for {mesh_name}")

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

        print(f"[BODexVisualizer] Loaded scene: {scene_json_path}")
        
    def _on_scene_name_change(self):
        self.coverage_scene_path = os.path.join(code_path, 'coverage_scene', self.test_scene_version, self.grasp_version, self.obj_name, self.scene_type)
        
        self.scene_name = self.scene_name_selector.value
        self._load_scene()
        
        self.valid_grasp_list = json.load(open(os.path.join(self.coverage_scene_path, self.scene_name, 'scene_info.json'), 'r'))
        self.qpos = np.load(os.path.join(self.coverage_scene_path, self.scene_name, 'qpos.npy'))

        self.clear_traj()

        for i in range(len(self.valid_grasp_list)):
            grasp_info = self.valid_grasp_list[i]
            hand_pose = np.load(os.path.join(self.grasp_root_path, grasp_info[2], grasp_info[3], grasp_info[4], 'pregrasp_pose.npy')).reshape(1, -1)
            action = np.concatenate([self.qpos[i:i+1], hand_pose], axis=1)
            self.add_traj(str(i), {"xarm_allegro":action})


if __name__ == "__main__":
    vis = Visualizer()
    vis.start_viewer()