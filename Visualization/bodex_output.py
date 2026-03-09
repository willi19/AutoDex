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
from rsslib.path import candidate_path, obj_path

class BODexVisualizer(ViserViewer):
    def __init__(self):
        super().__init__()

        self.bodex_root = candidate_path
        self.obj_root = obj_path
        
        self.cur_version = None
        self.current_obj = None
        self.current_scene_type = None
        self.current_scene_idx = None

        self.version_list = os.listdir(self.bodex_root)
        self.obj_list = []
        self.scene_type_list = []
        self.scene_idx_list = []
        self.grasp_list = []
        self.all_grasp_dirs = []  # 모든 grasp
        self.filtered_grasp_dirs = []  # 필터링된 grasp
        self.gui_playing.value = True
        
        # GUI
        with self.server.gui.add_folder("BODex Grasp Viewer"):
            self.version_selector = self.server.gui.add_dropdown(
                "Version",
                options=self.version_list,
                initial_value=self.version_list[0] if self.version_list else "",
            )

            self.obj_selector = self.server.gui.add_dropdown(
                "Object",
                options=self._available_objects(),
                initial_value=self._available_objects()[0] if self._available_objects() else "",
            )

            self.scene_type_selector = self.server.gui.add_dropdown(
                "Scene Type",
                options=[],
                initial_value="",
            )

            self.scene_idx_selector = self.server.gui.add_dropdown(
                "Scene",
                options=[],
                initial_value="",
            )
            
            # Success Filter
            self.success_filter = self.server.gui.add_dropdown(
                "Success Filter",
                options=["All", "Success Only", "Fail Only"],
                initial_value="All",
            )

            self.grasp_idx_slider = self.server.gui.add_slider(
                "Grasp Index",
                min=0,
                max=1,
                step=1,
                initial_value=0,
            )
            
            self.show_contact = self.server.gui.add_checkbox("Show Contact Points", initial_value=True)
            self.show_trajectory = self.server.gui.add_checkbox("Show Trajectory", initial_value=False)
            
            self.metric_text = self.server.gui.add_text(
                "Metrics",
                initial_value="No grasp loaded",
                disabled=True,
            )

        self._on_version_change()
        
        # Callbacks
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
            
        @self.success_filter.on_update
        def _(event):
            self._apply_filter()

        @self.grasp_idx_slider.on_update
        def _(event):
            self._load_current_grasp()
            
        @self.show_contact.on_update
        def _(event):
            self._load_current_grasp()
            
        @self.show_trajectory.on_update
        def _(event):
            self._load_current_grasp()
        
        self.squeeze_num = 10

    def _on_version_change(self):
        self.cur_version = self.version_selector.value
        if not self.cur_version:
            return
        
        objs = self._available_objects()
        if not objs:
            self.obj_selector.disabled = True
            return
        
        self.obj_selector.disabled = False
        self.obj_selector.options = objs
        self.obj_selector.value = objs[0]
        self._on_object_change()

    def _available_objects(self):
        if self.cur_version is None or not os.path.exists(os.path.join(self.bodex_root, self.cur_version)):
            return []
        return sorted(
            d for d in os.listdir(os.path.join(self.bodex_root, self.cur_version))
            if os.path.isdir(os.path.join(self.bodex_root, self.cur_version, d))
        )

    def _available_scene_types(self):
        if self.current_obj is None:
            return []
        obj_candidate_path = os.path.join(self.bodex_root, self.cur_version, self.current_obj)
        if not os.path.exists(obj_candidate_path):
            return []
        return sorted(
            d for d in os.listdir(obj_candidate_path)
            if os.path.isdir(os.path.join(obj_candidate_path, d))
        )

    def _available_scenes(self):
        if self.current_obj is None or self.current_scene_type is None:
            return []
        scene_path = os.path.join(self.bodex_root, self.cur_version, self.current_obj, self.current_scene_type)
        if not os.path.exists(scene_path):
            return []
        return sorted(
            d for d in os.listdir(scene_path)
            if os.path.isdir(os.path.join(scene_path, d))
        )

    def clear_scene(self):
        """Clear all objects except floor"""
        for name in list(self.obj_dict.keys()):
            self.obj_dict[name]["frame"].remove()
            self.obj_dict[name]["handle"].remove()
            del self.obj_dict[name]

    def _on_object_change(self):
        self.current_obj = self.obj_selector.value
        scene_types = self._available_scene_types()
        
        if not scene_types:
            self.scene_type_selector.disabled = True
            self.scene_idx_selector.disabled = True
            self.grasp_idx_slider.disabled = True
            self.clear_scene()
            return
        
        self.scene_type_selector.disabled = False
        self.scene_type_selector.options = scene_types
        self.scene_type_selector.value = scene_types[0]
        self._on_scene_type_change()

    def _on_scene_type_change(self):
        self.current_scene_type = self.scene_type_selector.value
        
        if not self.current_scene_type:
            return
            
        scenes = self._available_scenes()
        
        if not scenes:
            self.scene_idx_selector.disabled = True
            self.grasp_idx_slider.disabled = True
            self.clear_scene()
            return
        
        self.scene_idx_selector.disabled = False
        self.scene_idx_selector.options = scenes
        self.scene_idx_selector.value = scenes[0]
        self._on_scene_idx_change()

    def _on_scene_idx_change(self):
        self.current_scene_idx = self.scene_idx_selector.value
        
        if not self.current_scene_idx:
            return
            
        scene_path = os.path.join(
            self.bodex_root,
            self.cur_version,
            self.current_obj, 
            self.current_scene_type, 
            self.current_scene_idx
        )
        
        if not os.path.exists(scene_path):
            self.grasp_idx_slider.disabled = True
            self.clear_scene()
            return
        
        # Get all grasp directories
        self.all_grasp_dirs = sorted(
            [d for d in os.listdir(scene_path) 
            if os.path.isdir(os.path.join(scene_path, d))],
            key=lambda x: int(x)
        )
        
        if not self.all_grasp_dirs:
            self.grasp_idx_slider.disabled = True
            self.clear_scene()
            return
        
        # Load scene first
        self._load_scene()
        
        # Apply filter
        self._apply_filter()

    def _apply_filter(self):
        """Apply success filter to grasp directories"""
        if not self.all_grasp_dirs:
            return
        
        filter_mode = self.success_filter.value
        
        if filter_mode == "All":
            self.filtered_grasp_dirs = self.all_grasp_dirs
        else:
            # Load bodex_info for each grasp to check success
            filtered = []
            for grasp_dir in self.all_grasp_dirs:
                grasp_path = os.path.join(
                    self.bodex_root,
                    self.cur_version,
                    self.current_obj,
                    self.current_scene_type,
                    self.current_scene_idx,
                    grasp_dir
                )
                
                bodex_info_path = os.path.join(grasp_path, "bodex_info.npy")
                if not os.path.exists(bodex_info_path):
                    continue
                
                bodex_info = np.load(bodex_info_path, allow_pickle=True).item()
                # success = bodex_info.get('success', False)
                success = bodex_info['grasp_error'].max() < 0.2 and bodex_info['dist_error'].max() < 0.01


                if filter_mode == "Success Only" and success:
                    filtered.append(grasp_dir)
                elif filter_mode == "Fail Only" and not success:
                    filtered.append(grasp_dir)
            
            self.filtered_grasp_dirs = filtered
        
        if not self.filtered_grasp_dirs:
            self.grasp_idx_slider.disabled = True
            print(f"[BODexVisualizer] No grasps match filter: {filter_mode}")
            return
        
        self.grasp_idx_slider.disabled = False
        self.grasp_idx_slider.max = len(self.filtered_grasp_dirs) - 1
        self.grasp_idx_slider.value = 0
        
        print(f"[BODexVisualizer] Filter: {filter_mode}, {len(self.filtered_grasp_dirs)}/{len(self.all_grasp_dirs)} grasps")
        self._load_current_grasp()

    def _load_scene(self):
        print(f"[BODexVisualizer] Loading scene...")
        """Load object mesh from scene json"""
        self.clear_scene()
        scene_json_path = os.path.join(
            self.obj_root, 
            self.current_obj, 
            "scene", 
            self.current_scene_type, 
            f"{self.current_scene_idx}.json"
        )
        
        if not os.path.exists(scene_json_path):
            print(f"[Warning] Scene json not found: {scene_json_path}")
            return
    
        with open(scene_json_path, "r") as f:
            cfg = json.load(f)

        scene = cfg["scene"]
        
        # mesh
        for mesh_name, info in scene.get("mesh", {}).items():
            info['file_path'] = info['file_path'].replace('mingi', 'robot')
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

        print(f"[BODexVisualizer] Loaded scene: {self.current_scene_type}/{self.current_scene_idx}")
        
    def _load_current_grasp(self):
        """Load and visualize current grasp"""
        if not self.filtered_grasp_dirs:
            return
        
        for robot_name in list(self.robot_dict.keys()):
            del self.robot_dict[robot_name]

        self.clear_traj()
        
        grasp_idx = self.grasp_idx_slider.value
        grasp_dir = self.filtered_grasp_dirs[grasp_idx]
        grasp_path = os.path.join(
            self.bodex_root,
            self.cur_version,
            self.current_obj,
            self.current_scene_type,
            self.current_scene_idx,
            grasp_dir
        )
        
        # Load grasp data
        wrist_se3 = np.load(os.path.join(grasp_path, "wrist_se3.npy"))
        pregrasp_pose = np.load(os.path.join(grasp_path, "pregrasp_pose.npy"))
        grasp_pose = np.load(os.path.join(grasp_path, "grasp_pose.npy"))
        
        # Transform to world frame
        robot_T = self.obj_pose @ wrist_se3
        
        urdf_path = "BODex/src/curobo/content/assets/robot/allegro_description/allegro_hand_description_right.urdf"
        self.add_robot("robot", urdf_path, pose=robot_T)
        
        
        # Show trajectory
        if self.show_trajectory.value:
            traj_list = [pregrasp_pose, grasp_pose]
            for i in range(self.squeeze_num):
                traj_list.append(grasp_pose * (i+2) - pregrasp_pose * (i+1))

            traj_dict = {
                "robot": np.stack(traj_list)
            }
            self.add_traj("robot_traj", traj_dict)
        
        else:
            traj_list = [grasp_pose]
            traj_dict = {
                "robot": np.stack(traj_list)
            }   
            self.add_traj("robot_traj", traj_dict)

if __name__ == "__main__":
    vis = BODexVisualizer()
    vis.start_viewer()