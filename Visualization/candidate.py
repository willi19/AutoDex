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
from rsslib.path import candidate_path, obj_path, urdf_path

class CandidateVisualizer(ViserViewer):
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
        self.all_grasp_dirs = []
        self.filtered_grasp_dirs = []
        self.gui_playing.value = True
        
        # Animation state
        self.viz_data = None
        self.is_playing = False
        self.current_frame = 0
        
        # R_delta inverse for coordinate transform
        q_delta = np.array([0, 1, 0, 1], dtype=np.float64)
        q_delta = q_delta / np.linalg.norm(q_delta)
        self.R_delta_inv = transforms3d.quaternions.quat2mat(q_delta).T
        
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
            
            # Filters as checkboxes
            self.filter_sim_success = self.server.gui.add_checkbox(
                "Sim Success Only", 
                initial_value=False
            )
            
            self.filter_sim_fail = self.server.gui.add_checkbox(
                "Sim Fail Only", 
                initial_value=False
            )
            
            self.filter_coll_valid = self.server.gui.add_checkbox(
                "Collision Valid Only", 
                initial_value=False
            )
            
            self.filter_coll_invalid = self.server.gui.add_checkbox(
                "Collision Invalid Only", 
                initial_value=False
            )

            self.grasp_idx_slider = self.server.gui.add_slider(
                "Grasp Index",
                min=0,
                max=1,
                step=1,
                initial_value=0,
            )
            
            # Animation controls
            self.frame_slider = self.server.gui.add_slider(
                "Frame",
                min=0,
                max=1,
                step=1,
                initial_value=0,
                disabled=True,
            )
            
            self.play_button = self.server.gui.add_button("Play/Pause", disabled=True)
            
            self.show_contact = self.server.gui.add_checkbox(
                "Show Contact Points", 
                initial_value=True
            )
            
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
            
        @self.filter_sim_success.on_update
        def _(event):
            self._apply_filter()
            
        @self.filter_sim_fail.on_update
        def _(event):
            self._apply_filter()
            
        @self.filter_coll_valid.on_update
        def _(event):
            self._apply_filter()
            
        @self.filter_coll_invalid.on_update
        def _(event):
            self._apply_filter()

        @self.grasp_idx_slider.on_update
        def _(event):
            self._load_current_grasp()
            
        @self.show_contact.on_update
        def _(event):
            if self.viz_data is not None:
                self._update_sim_frame()
                
        @self.frame_slider.on_update
        def _(event):
            self._update_sim_frame()
                
        @self.play_button.on_click
        def _(event):
            self.is_playing = not self.is_playing
            if self.is_playing:
                self._play_animation()
        
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
        obj_bodex_path = os.path.join(self.bodex_root, self.cur_version, self.current_obj)
        if not os.path.exists(obj_bodex_path):
            return []
        return sorted(
            d for d in os.listdir(obj_bodex_path)
            if os.path.isdir(os.path.join(obj_bodex_path, d))
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
        """Apply filters to grasp directories"""
        if not self.all_grasp_dirs:
            return
        
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
            
            # Check sim success/fail filter
            if self.filter_sim_success.value or self.filter_sim_fail.value:
                eval_path = os.path.join(grasp_path, "sim_eval", "eval_results.json")
                if not os.path.exists(eval_path):
                    continue
                eval_results = json.load(open(eval_path, "r"))
                succ_flag = eval_results.get("succ_flag", False)
                if isinstance(succ_flag, list):
                    succ_flag = succ_flag[0]
                
                if self.filter_sim_success.value and not succ_flag:
                    continue
                if self.filter_sim_fail.value and succ_flag:
                    continue
            
            # Check collision valid/invalid filter
            if self.filter_coll_valid.value or self.filter_coll_invalid.value:
                coll_path = os.path.join(grasp_path, "coll_valid.npy")
                if not os.path.exists(coll_path):
                    continue
                coll_valid = np.load(coll_path).item()
                
                if self.filter_coll_valid.value and not coll_valid:
                    continue
                if self.filter_coll_invalid.value and coll_valid:
                    continue
            
            filtered.append(grasp_dir)
        
        self.filtered_grasp_dirs = filtered
        
        if not self.filtered_grasp_dirs:
            self.grasp_idx_slider.disabled = True
            print(f"[BODexVisualizer] No grasps match filters")
            return
        
        self.grasp_idx_slider.disabled = False
        self.grasp_idx_slider.max = len(self.filtered_grasp_dirs) - 1
        self.grasp_idx_slider.value = 0
        
        print(f"[BODexVisualizer] {len(self.filtered_grasp_dirs)}/{len(self.all_grasp_dirs)} grasps after filtering")
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
            mesh = trimesh.load(os.path.join(obj_path, self.current_obj, "raw_mesh", f"{self.current_obj}.obj"), force="mesh")

            pose = np.eye(4)
            pose[:3, 3] = np.array(info["pose"][:3])
            quat = info["pose"][3:]  # wxyz
            pose[:3, :3] = Rot.from_quat(
                [quat[1], quat[2], quat[3], quat[0]]
            ).as_matrix()

            self.add_object(mesh_name, mesh, obj_T=pose)
            if mesh_name == "target":
                self.obj_pose = pose

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
        self._clear_contacts()
        
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
        
        # Load viz_data
        viz_data_path = os.path.join(grasp_path, "sim_eval", "viz_data.json")
        if os.path.exists(viz_data_path):
            with open(viz_data_path, "r") as f:
                self.viz_data = json.load(f)
        else:
            self.viz_data = None
            
        # Load bodex_info for contact points
        bodex_info_path = os.path.join(grasp_path, "bodex_info.npy")
        if os.path.exists(bodex_info_path):
            self.bodex_info = np.load(bodex_info_path, allow_pickle=True).item()
        else:
            self.bodex_info = None
        
        # Transform to world frame
        robot_T = self.obj_pose @ wrist_se3
        
        allegro_path = os.path.join(urdf_path, "allegro_hand_description_right.urdf")
        self.add_robot("robot", allegro_path, pose=robot_T)
        
        if self.viz_data is not None:
            # Extract joint trajectory
            qpos_traj = np.array(self.viz_data["qpos_trajectory"])
            joint_traj = qpos_traj[:, 7:23]  # Only joint angles
            
            robot_traj = {"robot": joint_traj}
            self.add_traj("sim_traj", robot_traj)
        
        # Update metrics
        self._update_metrics(grasp_dir, grasp_idx, grasp_path)

    def _clear_contacts(self):
        """Clear all contact point spheres"""
        for name in list(self.obj_dict.keys()):
            if name.startswith("cp_"):
                self.obj_dict[name]["frame"].remove()
                self.obj_dict[name]["handle"].remove()
                del self.obj_dict[name]

    def _update_contacts_frame(self, frame_idx):
        """Update contact points for specific simulation frame"""
        self._clear_contacts()
        
        if not self.show_contact.value or self.viz_data is None:
            return
            
        contact_frame = self.viz_data["contact_frames"][frame_idx]
        
        # Show hand-object contacts
        for cidx, contact in enumerate(contact_frame.get("ho_contacts", [])):
            pos = np.array(contact["pos"])
            self.add_sphere(f"cp_ho_{cidx}", position=pos, radius=0.002, color=(1, 0, 0))

    def _update_metrics(self, grasp_dir, grasp_idx, grasp_path):
        """Update metrics text"""
        # Load evaluation results
        eval_path = os.path.join(grasp_path, "sim_eval", "eval_results.json")
        if os.path.exists(eval_path):
            eval_results = json.load(open(eval_path, "r"))
            sim_success = eval_results.get("succ_flag", False)
            if isinstance(sim_success, list):
                sim_success = sim_success[0]
        else:
            sim_success = None
            
        # Load collision valid
        coll_path = os.path.join(grasp_path, "coll_valid.npy")
        if os.path.exists(coll_path):
            coll_valid = np.load(coll_path).item()
        else:
            coll_valid = None
        
        # Build metrics string
        metrics_str = f"Grasp {grasp_dir} ({grasp_idx+1}/{len(self.filtered_grasp_dirs)})\n"
        
        if sim_success is not None:
            metrics_str += f"Sim Success: {sim_success}\n"
        
        if coll_valid is not None:
            metrics_str += f"Collision Valid: {coll_valid}\n"
            
        if self.bodex_info is not None:
            grasp_error = self.bodex_info.get('grasp_error', np.array([0]))
            dist_error = self.bodex_info.get('dist_error', np.array([0]))
            metrics_str += f"Grasp Error: {grasp_error.mean():.4f}\n"
            metrics_str += f"Dist Error: {dist_error.mean():.4f}"
        
        self.metric_text.value = metrics_str
        
        print(f"[BODexVisualizer] Loaded grasp {grasp_dir}: sim_success={sim_success}, coll_valid={coll_valid}")

if __name__ == "__main__":
    vis = CandidateVisualizer()
    vis.start_viewer()