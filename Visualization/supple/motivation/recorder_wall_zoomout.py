import os
import glob
import json
import numpy as np
import trimesh
import transforms3d
from scipy.spatial.transform import Rotation as R
import shapely.geometry as geom
import tempfile
import subprocess
import shutil
from datetime import datetime
from scipy.spatial.transform import Slerp, Rotation

from paradex.utils.path import shared_dir
from paradex.visualization.visualizer.viser import ViserViewer

from rsslib.conversion import se32cart, cart2se3
from rsslib.path import candidate_path, obj_path, urdf_path


xarm_init_pose = np.array([
    -0.21991149, -0.20245819, -1.13620934,  2.33175988,  0.31939525, 2.36492114]
) 

allegro_init_pose = np.array( [0.0, 1.5707,  0.0,  0.0,# 0.00331845,  1.24899578, 0.11299501,  0.54854941,  
                               0.0,  1.5707,  0.0, 0.0, 
                               0.0,  1.5707,  0.0, 0.0, 
                               1.24565697,  0.05513508,  0.23153956, -0.02217758])
init_pose = np.concatenate([xarm_init_pose, allegro_init_pose])

COLORS = {
    "target_obj":   (0, 100, 0),         # 짙은 초록 (타겟)
    "obstacle":     (119, 136, 153),     # 쿨 슬레이트 그레이 (장애물)
    "table":        (205, 210, 215),     # 밝은 쿨 화이트 (테이블)
    "robot":        (250, 250, 250),     # 순백색 (로봇)
}
def rotz(theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.eye(4)
    R[0, 0] = c
    R[0, 1] = -s
    R[1, 0] = s
    R[1, 1] = c
    return R

def transl(xyz):
    T = np.eye(4)
    T[:3, 3] = xyz
    return T

def mat4_to_pose(T):
    """
    [x, y, z, qw, qx, qy, qz]
    """
    q = R.from_matrix(T[:3, :3]).as_quat()  # x y z w
    return [
        T[0, 3], T[1, 3], T[2, 3],
        q[3], q[0], q[1], q[2]
    ]

def get_mesh_dict(obj_name, pose):
    ret = {
        "scale": [1.0, 1.0, 1.0],
        "pose": [
            pose[0, 3], pose[1, 3], pose[2, 3],
            *transforms3d.quaternions.mat2quat(pose[:3, :3])
        ],
        "file_path": os.path.join(
            obj_path,
            obj_name,
            "processed_data",
            "mesh",
            "simplified.obj"
        ),
        "urdf_path": os.path.join(
            obj_path,
            obj_name,
            "processed_data",
            "urdf",
            "coacd.urdf"
        ),
    }
    return ret

def get_wall_scene(obj_name, tabletop_pose, obb_info, z_rotation_deg, gap, t, scene_z_rot_deg):
    """
    Create single wall scene with given rotation
    Args:
        z_rotation_deg: Z축 회전 각도
        gap: wall과의 거리
    """
    # Z축 회전
    angle_rad = np.radians(z_rotation_deg)
    z_rotation = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0, 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    rotated_pose = z_rotation @ tabletop_pose
    
    # OBB world transform
    obb_transform = np.array(obb_info['obb_transform'])
    R_obb = obb_transform[:3, :3]
    obb_extents = np.array(obb_info['obb'])
    
    R_world = rotated_pose[:3, :3]
    t_world = rotated_pose[:3, 3]
    obb_world = R_world @ R_obb
    
    # 8 corners
    corners_local = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                corner = np.array([i, j, k]) * obb_extents / 2
                corners_local.append(corner)
    
    corners_world = np.array([(obb_world @ c + t_world) for c in corners_local])
    
    # Object의 최고점
    max_z = corners_world[:, 2].max()
    
    # Wall dimensions - adaptive height
    wall_height = max_z + 0.1  # object 최고점 + 10cm (hand clearance)
    
    # Wall을 robot(0,0,0)과 object 사이에 배치 (x축 40cm)
    wall_x = 0.4  # robot에서 x축으로 40cm
    
    # Scene
    scene = get_tabletop_scene(obj_name, rotated_pose)
    scene["cuboid"]["wall"] = {
        "dims": [0.02, 0.6, wall_height],  # x방향으로 얇은 벽
        "pose": [wall_x, 0.0, wall_height/2, 1, 0, 0, 0],
    }
    
    return scene
    
def get_tabletop_scene(obj_name, tabletop_pose):
    ret = {
        "mesh": {},
        "cuboid": {
            "table": {
                "dims": [66.0, 6.0, 0.2],
                "pose": [0.0, 0.0, -0.1, 1, 0, 0, 0.0],
            }   
        }
    }
    ret["mesh"]["target"] = get_mesh_dict(obj_name, tabletop_pose)
    return ret


class Renderer(ViserViewer):
    def __init__(self, version, obj_name, scene_type, scene_idx):
        super().__init__()
        self.init_pose = init_pose.copy().reshape(1, -1)

        # Set white background
        self.server.scene.set_background_image(
            np.ones((1, 1, 3), dtype=np.uint8) * 255
        )

        # Animation state
        self.viz_data = None
        self.is_playing = True
        
        # R_delta inverse for coordinate transform
        q_delta = np.array([0, 1, 0, 1], dtype=np.float64)
        q_delta = q_delta / np.linalg.norm(q_delta)
        self.R_delta_inv = transforms3d.quaternions.quat2mat(q_delta).T
        
        self.cur_version = version
        self.current_obj = obj_name
        self.current_scene_type = scene_type
        self.current_scene_idx = scene_idx
        self.add_robot("asdf", os.path.join(urdf_path, "xarm_allegro.urdf"))

        self.squeeze_num = 10
        self.grasp_robots = []  # grasp 로봇 이름 추적 (load_scene 전에 초기화)
        self.load_scene()

        self.add_video_capture_gui()
        self.add_view_save_gui()  # view 저장/로드 GUI
        self.add_appearance_gui()
        self.add_animation_gui()

        # 초기에 arm과 obstacle 숨기기
        self._hide_arm_and_obstacles()

    def add_appearance_gui(self):
        """Add appearance control GUI"""
        with self.server.gui.add_folder("Appearance"):
            # Wall controls
            wall_color = self.server.gui.add_rgb("Wall Color", COLORS["obstacle"])
            wall_opacity = self.server.gui.add_slider(
                "Wall Opacity", min=0.0, max=1.0, step=0.05, initial_value=0.3
            )
            
            # Table controls
            table_color = self.server.gui.add_rgb("Table Color", COLORS["table"])
            table_opacity = self.server.gui.add_slider(
                "Table Opacity", min=0.0, max=1.0, step=0.05, initial_value=0.9
            )
            
            # Robot controls - 각 로봇별로
            self.robot_color_controls = {}
            for robot_name in self.robot_dict.keys():
                with self.server.gui.add_folder(f"Robot: {robot_name}"):
                    visible_checkbox = self.server.gui.add_checkbox("Visible", initial_value=True)
                    color_btn = self.server.gui.add_button_group(
                        "Color", 
                        ("Red", "Green", "Blue", "Yellow", "Orange", "Purple", "Gray")
                    )
                    opacity_slider = self.server.gui.add_slider(
                        "Opacity", min=0.0, max=1.0, step=0.05, initial_value=1.0
                    )
                    
                    self.robot_color_controls[robot_name] = {
                        'visible': visible_checkbox,
                        'color': color_btn,
                        'opacity': opacity_slider
                    }
            
            # 색상 매핑
            self.robot_color_map = {
                "Red": (230, 100, 100),
                "Green": (100, 255, 100),
                "Blue": (100, 150, 230),
                "Yellow": (230, 180, 80),
                "Orange": (255, 165, 0),
                "Purple": (200, 100, 255),
                "Gray": (150, 150, 150),
            }
            
        # 각 로봇별 콜백 등록
        for robot_name in self.robot_dict.keys():
            controls = self.robot_color_controls[robot_name]
            
            @controls['visible'].on_update
            def _(_, name=robot_name):
                self.robot_dict[name].set_visibility(self.robot_color_controls[name]['visible'].value)
            
            @controls['color'].on_click
            def _(_, name=robot_name):
                color_rgb = self.robot_color_map[self.robot_color_controls[name]['color'].value]
                opacity = self.robot_color_controls[name]['opacity'].value
                self._update_single_robot_appearance(name, color_rgb, opacity)
            
            @controls['opacity'].on_update
            def _(_, name=robot_name):
                color_rgb = self.robot_color_map[self.robot_color_controls[name]['color'].value]
                opacity = self.robot_color_controls[name]['opacity'].value
                self._update_single_robot_appearance(name, color_rgb, opacity)

        @wall_color.on_update
        def _(_):
            color_rgb = wall_color.value
            opacity = wall_opacity.value
            self._update_wall_appearance(color_rgb, opacity)
        @wall_opacity.on_update
        def _(_):
            color_rgb = wall_color.value
            opacity = wall_opacity.value
            self._update_wall_appearance(color_rgb, opacity)
        @table_color.on_update
        def _(_):
            color_rgb = table_color.value
            opacity = table_opacity.value
            self._update_table_appearance(color_rgb, opacity)
        @table_opacity.on_update
        def _(_):
            color_rgb = table_color.value
            opacity = table_opacity.value
            self._update_table_appearance(color_rgb, opacity)

    def add_video_capture_gui(self):
        """Add video capture GUI controls"""
        with self.server.gui.add_folder("Video Capture"):
            # Resolution controls
            self.capture_width = self.server.gui.add_slider(
                "Width",
                min=480,
                max=1920,
                step=80,
                initial_value=1280
            )
            
            self.capture_height = self.server.gui.add_slider(
                "Height",
                min=480,
                max=1920,
                step=80,
                initial_value=720
            )
            
            # Video parameters
            self.video_fps = self.server.gui.add_slider(
                "Video FPS",
                min=10,
                max=60,
                step=1,
                initial_value=30
            )

            self.video_duration = self.server.gui.add_slider(
                "Duration (sec)",
                min=1.0,
                max=10.0,
                step=0.5,
                initial_value=3.0
            )
            
            # Output path
            self.output_path = self.server.gui.add_text(
                "Output Path",
                initial_value=""
            )
            
            # View controls
            self.set_start_view_btn = self.server.gui.add_button("Set Start View")
            self.set_end_view_btn = self.server.gui.add_button("Set End View")
            self.record_video_btn = self.server.gui.add_button("Record Video")
            
            # Screenshot
            self.capture_png_btn = self.server.gui.add_button("Capture PNG")

        # Callbacks
        @self.set_start_view_btn.on_click
        def _(_) -> None:
            if len(self.server.get_clients()) == 0:
                print("❌ No client connected!")
                return
            client = next(iter(self.server.get_clients().values()))
            self.start_view = {
                'position': client.camera.position,
                'wxyz': client.camera.wxyz
            }
            print("✓ Start view set:", self.start_view['position'])

        @self.set_end_view_btn.on_click
        def _(_) -> None:
            if len(self.server.get_clients()) == 0:
                print("❌ No client connected!")
                return
            client = next(iter(self.server.get_clients().values()))
            self.end_view = {
                'position': client.camera.position,
                'wxyz': client.camera.wxyz
            }
            print("✓ End view set:", self.end_view['position'])

        @self.record_video_btn.on_click
        def _(_) -> None:
            self._record_interpolated_video()
        
        @self.capture_png_btn.on_click
        def _(_) -> None:
            # output_path가 비어있으면 타임스탬프로 생성
            if self.output_path.value.strip() == "":
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = f"capture_{timestamp}.png"
            else:
                out_path = self.output_path.value
            
            print(f"Capturing current view to {out_path}...")
            
            # GUI에서 설정한 해상도 사용
            width = int(self.capture_width.value)
            height = int(self.capture_height.value)
            
            # PNG 캡처
            self.capture_scene_png(out_path, height=height, width=width)
            print(f"✅ Saved: {out_path}")

    def _record_interpolated_video(self):
        """Record interpolated video between start and end views"""
        # Validation
        if not hasattr(self, 'start_view') or not hasattr(self, 'end_view'):
            print("❌ Please set both start and end views first!")
            return
        
        # Client 확인
        if len(self.server.get_clients()) == 0:
            print("❌ No client connected!")
            return
        
        client = next(iter(self.server.get_clients().values()))
        
        # Output path
        if self.output_path.value.strip() == "":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = f"video_{timestamp}.mp4"
        else:
            out_path = self.output_path.value
            if not out_path.endswith('.mp4'):
                out_path += '.mp4'
        
        print(f"\n🎬 Recording video to {out_path}...")
        
        # Parameters
        fps = int(self.video_fps.value)
        duration = float(self.video_duration.value)
        n_frames = int(fps * duration)
        width = int(self.capture_width.value)
        height = int(self.capture_height.value)
        
        # Position interpolation
        start_pos = np.array(self.start_view['position'])
        end_pos = np.array(self.end_view['position'])
        
        # Rotation interpolation (slerp)
        start_quat = np.array(self.start_view['wxyz'])  # [w, x, y, z]
        end_quat = np.array(self.end_view['wxyz'])
        
        # scipy는 [x, y, z, w] 순서
        start_rot = Rotation.from_quat([start_quat[1], start_quat[2], start_quat[3], start_quat[0]])
        end_rot = Rotation.from_quat([end_quat[1], end_quat[2], end_quat[3], end_quat[0]])
        
        slerp = Slerp([0, 1], Rotation.concatenate([start_rot, end_rot]))
        
        # Temp directory for frames
        temp_dir = tempfile.mkdtemp()
        
        print(f"📸 Rendering {n_frames} frames ({width}x{height} @ {fps}fps)...")
        
        for i in range(n_frames):
            t = i / (n_frames - 1) if n_frames > 1 else 0
            
            # Linear position interpolation
            interp_pos = start_pos * (1 - t) + end_pos * t
            
            # Slerp rotation
            interp_rot = slerp(t)
            interp_quat_xyzw = interp_rot.as_quat()
            interp_quat_wxyz = [interp_quat_xyzw[3], interp_quat_xyzw[0], 
                            interp_quat_xyzw[1], interp_quat_xyzw[2]]
            
            # Update camera
            client.camera.position = tuple(interp_pos)
            client.camera.wxyz = tuple(interp_quat_wxyz)
            
            # Capture
            frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
            self.capture_scene_png(frame_path, height=height, width=width)
            
            if (i + 1) % 30 == 0 or i == n_frames - 1:
                print(f"  Progress: {i + 1}/{n_frames} frames")
        
        # Encode with ffmpeg
        print("🎞️  Encoding video...")
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-loglevel', 'warning',
            '-framerate', str(fps),
            '-i', os.path.join(temp_dir, 'frame_%04d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',
            out_path
        ]
        
        subprocess.run(ffmpeg_cmd, check=True)
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        print(f"✅ Video saved: {out_path}\n")

    def _update_wall_appearance(self, color_rgb, opacity):
        """Update wall appearance"""
        color = tuple(c / 255.0 for c in color_rgb)
        wall_names = ["back", "side_pos", "side_neg", "up", "wall", 
                    "box_front", "box_back", "box_left", "box_right"]

        for name in wall_names:
            if name in self.obj_dict:
                self.change_color(name, (*color, opacity))

    def _update_table_appearance(self, color_rgb, opacity):
        """Update table appearance"""
        color = tuple(c / 255.0 for c in color_rgb)
        if "table" in self.obj_dict:
            self.change_color("table", (*color, opacity))

    def _update_single_robot_appearance(self, robot_name, color_rgb, opacity):
        """Update single robot appearance"""
        color = tuple(c / 255.0 for c in color_rgb)
        self.robot_dict[robot_name].change_color([], (*color, opacity))
    
    def load_scene(self):
        """Load scene from procedural get_*_scene functions"""
        
        # --------------------------------------------------
        # 1. object pose (tabletop 기준)
        # --------------------------------------------------
        scene_json_path = os.path.join(
            obj_path, 
            self.current_obj, 
            "scene", 
            self.current_scene_type, 
            f"{self.current_scene_idx}.json"
        )

        cfg = json.load(open(scene_json_path, "r"))

        tabletop_pose = cart2se3(cfg['scene']['mesh']['target']['pose'])
        tabletop_pose[0, 3] += 0.5  # x축으로 30cm 이동
        # --------------------------------------------------
        # 2. OBB info (필요한 scene에서만 사용)
        # --------------------------------------------------
        obb_path = os.path.join(
            obj_path,
            self.current_obj,
            "processed_data",
            "info",
            "simplified.json",
        )
        with open(obb_path, "r") as f:
            obb_info = json.load(f)
            
        if self.current_scene_type == "wall":
            scene = get_wall_scene(
                obj_name=self.current_obj,
                tabletop_pose=tabletop_pose,
                obb_info=obb_info,
                z_rotation_deg=0.0,
                gap=0.00,
                t=None,
                scene_z_rot_deg=0.0,
            )
            print("Wall scene created.")

        for mesh_name, info in scene.get("mesh", {}).items():
            if mesh_name != "target":
                mesh = trimesh.load(os.path.join(
                    obj_path, mesh_name, "raw_mesh", f"{mesh_name}.obj"
                ))
            else:
                mesh = trimesh.load((os.path.join(
                    obj_path, self.current_obj, "raw_mesh", f"{self.current_obj}.obj"
                )))

            pose = np.eye(4)
            pose[0, 3] += 0.3
            pose[:3, 3] = info["pose"][:3]
            quat = info["pose"][3:]  # wxyz
            pose[:3, :3] = R.from_quat(
                [quat[1], quat[2], quat[3], quat[0]]
            ).as_matrix()

            self.add_trimesh(mesh_name, mesh, pose=pose)

            if mesh_name == "target":
                self.obj_pose = pose

        # --------------------------------------------------
        # 5. cuboid 렌더링
        # --------------------------------------------------
        for name, info in scene.get("cuboid", {}).items():
            box = trimesh.creation.box(extents=info["dims"])

            pose = np.eye(4)
            pose[:3, 3] = info["pose"][:3]
            quat = info["pose"][3:]
            pose[:3, :3] = R.from_quat(
                [quat[1], quat[2], quat[3], quat[0]]
            ).as_matrix()

            self.add_object(name, box, obj_T=pose)
            if name == "table":
                self.change_color(name, COLORS["table"] + (0.9,))
            else:
                print(name)
                self.change_color(name, COLORS["obstacle"] + (0.9,))

        print(f"[Visualizer] Procedural scene loaded: {self.current_scene_type}")
        self._load_current_grasp()
        
    def _load_current_grasp(self):
        candidate_root_path = os.path.join(
            candidate_path,
            "baseline",
            obj_name,
            "float",
            "0",
        )

        candidate_idx_list = ["3", "13", "6"]
        print(f"Loading {len(candidate_idx_list)} grasps: {candidate_idx_list}")

        wrist_se3_list = []
        grasp_pose_list = []
        traj_dict = {}
        for candidate_idx in candidate_idx_list:
            grasp_path = os.path.join(
                candidate_root_path,
                candidate_idx
            )
            grasp_name = candidate_idx
            wrist_se3 = np.load(os.path.join(grasp_path, "wrist_se3.npy"))
            grasp_pose = np.load(os.path.join(grasp_path, "grasp_pose.npy")).reshape(1, -1)
            
            robot_T = self.obj_pose @ wrist_se3
            
            allegro_path = os.path.join(urdf_path, "allegro_hand_description_right.urdf")
            self.add_robot(f"robot_{grasp_name}", allegro_path, pose=robot_T)
            self.grasp_robots.append(f"robot_{grasp_name}")  # 추적

            wrist_se3_list.append(robot_T)
            grasp_pose_list.append(grasp_pose)
            traj_dict[f"robot_{grasp_name}"] = grasp_pose
        traj_dict["asdf"] = self.init_pose

        self.add_traj("asdf", traj_dict)

    def add_animation_gui(self):
        """Add animation control GUI"""
        with self.server.gui.add_folder("Animation"):
            # View file paths
            self.start_view_path = self.server.gui.add_text(
                "Start View JSON", initial_value="start.json"
            )
            self.arm_view_path = self.server.gui.add_text(
                "Arm View JSON", initial_value="arm.json"
            )

            # Grasp appearance timing
            self.grasp_appear_duration = self.server.gui.add_slider(
                "Grasp Appear (sec)",
                min=0.3,
                max=2.0,
                step=0.1,
                initial_value=0.5
            )

            self.grasp_fade_duration = self.server.gui.add_slider(
                "Fade-in Duration (sec)",
                min=0.1,
                max=1.0,
                step=0.1,
                initial_value=0.3
            )

            self.hold_duration = self.server.gui.add_slider(
                "Hold Duration (sec)",
                min=0.5,
                max=5.0,
                step=0.5,
                initial_value=2.0
            )

            self.camera_move_duration = self.server.gui.add_slider(
                "Camera Move (sec)",
                min=1.0,
                max=5.0,
                step=0.5,
                initial_value=2.0
            )

            self.filter_duration = self.server.gui.add_slider(
                "Filter Duration (sec)",
                min=0.5,
                max=3.0,
                step=0.5,
                initial_value=1.0
            )

            # Preview and record
            self.preview_anim_btn = self.server.gui.add_button("Preview Animation")
            self.record_anim_btn = self.server.gui.add_button("Record Animation")

            # Reset scene
            self.reset_scene_btn = self.server.gui.add_button("Reset Scene")

        @self.preview_anim_btn.on_click
        def _(_) -> None:
            self._play_animation(record=False)

        @self.record_anim_btn.on_click
        def _(_) -> None:
            self._play_animation(record=True)

        @self.reset_scene_btn.on_click
        def _(_) -> None:
            self._reset_scene()

    def _hide_arm_and_obstacles(self):
        """Hide arm and obstacles initially"""
        # Hide arm (asdf)
        if "asdf" in self.robot_dict:
            self.robot_dict["asdf"].set_visibility(False)

        # Hide obstacles (table, wall)
        obstacle_names = ["table", "wall"]
        for name in obstacle_names:
            if name in self.obj_dict:
                self.obj_dict[name]['handle'].visible = False

        # Hide grasp robots
        self._reset_grasp_visibility()

    def _reset_scene(self):
        """Reset scene - hide all grasps, arm, obstacles"""
        self._hide_arm_and_obstacles()
        print("Scene reset")

    # Grasp 색상 팔레트 (각 grasp 마다 다른 색)
    GRASP_COLORS = [
        (0.9, 0.3, 0.3),   # Red
        (0.3, 0.7, 0.3),   # Green
        (0.3, 0.4, 0.9),   # Blue
        (0.9, 0.6, 0.2),   # Orange
        (0.7, 0.3, 0.8),   # Purple
        (0.2, 0.8, 0.8),   # Cyan
    ]

    def _reset_grasp_visibility(self):
        """Hide all grasp robots"""
        for robot_name in self.grasp_robots:
            if robot_name in self.robot_dict:
                self.robot_dict[robot_name].set_visibility(False)

    def _set_grasp_visible(self, robot_name, visible):
        """Set visibility and color for a single grasp robot"""
        if robot_name in self.robot_dict:
            self.robot_dict[robot_name].set_visibility(visible)
            if visible:
                # 색상 적용 (index에 따라 다른 색, opacity 0.4)
                idx = self.grasp_robots.index(robot_name)
                color = self.GRASP_COLORS[idx % len(self.GRASP_COLORS)]
                self.robot_dict[robot_name].change_color([], (*color, 0.4))

    def _set_arm_visible(self, visible):
        """Set arm visibility"""
        if "asdf" in self.robot_dict:
            self.robot_dict["asdf"].set_visibility(visible)

    def _set_obstacle_visible(self, visible):
        """Set obstacles (table, wall) visibility"""
        for name in ["table", "wall"]:
            if name in self.obj_dict:
                self.obj_dict[name]['handle'].visible = visible

    def _play_animation(self, record=False):
        """Play or record the animation sequence

        Sequence:
        1. Start from start.json view (grasp only visible)
        2. Phase 1: Grasps appear one by one with fade-in
        3. Phase 2: Hold
        4. Phase 3: Camera moves to arm.json view + arm/obstacles appear
        """
        import time

        # Load views from JSON files
        start_path = self.start_view_path.value.strip() or "start.json"
        arm_path = self.arm_view_path.value.strip() or "arm.json"

        if not os.path.exists(start_path):
            print(f"Start view file not found: {start_path}")
            return
        if not os.path.exists(arm_path):
            print(f"Arm view file not found: {arm_path}")
            return

        with open(start_path, "r") as f:
            start_view = json.load(f)
        with open(arm_path, "r") as f:
            arm_view = json.load(f)

        print(f"Start view: {start_view['position']}")
        print(f"Arm view: {arm_view['position']}")

        if len(self.grasp_robots) == 0:
            print("No grasp robots loaded!")
            return

        # Client check
        if len(self.server.get_clients()) == 0:
            print("No client connected!")
            return

        client = next(iter(self.server.get_clients().values()))

        # Parameters
        fps = int(self.video_fps.value)
        grasp_appear_sec = float(self.grasp_appear_duration.value)
        fade_sec = float(self.grasp_fade_duration.value)
        hold_sec = float(self.hold_duration.value)
        camera_move_sec = float(self.camera_move_duration.value)
        filter_sec = float(self.filter_duration.value)

        # Calculate total frames
        n_grasps = len(self.grasp_robots)
        phase1_frames = int((grasp_appear_sec * n_grasps + fade_sec) * fps)
        phase2_frames = int(hold_sec * fps)
        phase3_frames = int(camera_move_sec * fps)
        phase4_frames = int(filter_sec * fps)
        total_frames = phase1_frames + phase2_frames + phase3_frames + phase4_frames

        print(f"\nAnimation: {n_grasps} grasps")
        print(f"  Phase 1 (grasp appear): {phase1_frames} frames")
        print(f"  Phase 2 (hold): {phase2_frames} frames")
        print(f"  Phase 3 (camera + arm/obstacle): {phase3_frames} frames")
        print(f"  Phase 4 (filter - opacity up then hide): {phase4_frames} frames")
        print(f"  Total: {total_frames} frames")

        # Recording setup
        temp_dir = None
        if record:
            temp_dir = tempfile.mkdtemp()
            width = int(self.capture_width.value)
            height = int(self.capture_height.value)
            print(f"Recording {width}x{height} @ {fps}fps...")

        # Reset scene - hide everything
        self._reset_scene()

        # Set initial camera position from start.json
        client.camera.position = tuple(start_view['position'])
        client.camera.wxyz = tuple(start_view['wxyz'])

        # Prepare camera interpolation for phase 3
        start_pos = np.array(start_view['position'])
        arm_pos = np.array(arm_view['position'])

        start_quat = np.array(start_view['wxyz'])
        arm_quat = np.array(arm_view['wxyz'])

        start_rot = Rotation.from_quat([start_quat[1], start_quat[2], start_quat[3], start_quat[0]])
        arm_rot = Rotation.from_quat([arm_quat[1], arm_quat[2], arm_quat[3], arm_quat[0]])
        slerp_cam = Slerp([0, 1], Rotation.concatenate([start_rot, arm_rot]))

        frame_idx = 0

        # PHASE 1: Grasps appear one by one
        print("Phase 1: Grasps appearing...")
        for grasp_idx, robot_name in enumerate(self.grasp_robots):
            grasp_start_frame = int(grasp_idx * grasp_appear_sec * fps)
            grasp_end_frame = int((grasp_idx + 1) * grasp_appear_sec * fps)

            while frame_idx < min(grasp_end_frame, phase1_frames):
                # Show this grasp at its start frame
                if frame_idx == grasp_start_frame:
                    self._set_grasp_visible(robot_name, True)

                if record:
                    frame_path = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")
                    self.capture_scene_png(frame_path, height=height, width=width)
                else:
                    time.sleep(1.0 / fps)

                frame_idx += 1

        # Ensure all grasps are visible
        for robot_name in self.grasp_robots:
            self._set_grasp_visible(robot_name, True)

        # Fill remaining phase 1 frames
        while frame_idx < phase1_frames:
            if record:
                frame_path = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")
                self.capture_scene_png(frame_path, height=height, width=width)
            else:
                time.sleep(1.0 / fps)
            frame_idx += 1

        # PHASE 2: Hold
        print("Phase 2: Holding...")
        hold_start = frame_idx
        while frame_idx < hold_start + phase2_frames:
            if record:
                frame_path = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")
                self.capture_scene_png(frame_path, height=height, width=width)
            else:
                time.sleep(1.0 / fps)
            frame_idx += 1

        # PHASE 3: Camera moves to arm view + arm/obstacles appear
        print("Phase 3: Camera moving + arm/obstacles appearing...")
        # Show arm and obstacles at start of phase 3
        self._set_arm_visible(True)
        self._set_obstacle_visible(True)

        cam_start = frame_idx
        while frame_idx < cam_start + phase3_frames:
            t = (frame_idx - cam_start) / phase3_frames if phase3_frames > 0 else 1.0

            # Smooth easing (ease-in-out)
            t_smooth = t * t * (3 - 2 * t)

            # Camera interpolation
            interp_pos = start_pos * (1 - t_smooth) + arm_pos * t_smooth
            interp_rot = slerp_cam(t_smooth)
            interp_quat_xyzw = interp_rot.as_quat()
            interp_quat_wxyz = [interp_quat_xyzw[3], interp_quat_xyzw[0],
                                interp_quat_xyzw[1], interp_quat_xyzw[2]]

            client.camera.position = tuple(interp_pos)
            client.camera.wxyz = tuple(interp_quat_wxyz)

            if record:
                frame_path = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")
                self.capture_scene_png(frame_path, height=height, width=width)
            else:
                time.sleep(1.0 / fps)

            frame_idx += 1

        # PHASE 4: Each grasp: opacity 0.4 → 1.0, then hide (one by one)
        print("Phase 4: Filtering grasps one by one...")
        frames_per_grasp = (phase4_frames * 2) // n_grasps if n_grasps > 0 else phase4_frames

        for grasp_idx, robot_name in enumerate(self.grasp_robots):
            color = self.GRASP_COLORS[grasp_idx % len(self.GRASP_COLORS)]

            for f in range(frames_per_grasp):
                t = f / frames_per_grasp if frames_per_grasp > 0 else 1.0

                # Opacity: 0.4 → 1.0
                opacity = 0.4 + 0.6 * t
                self.robot_dict[robot_name].change_color([], (*color, opacity))

                if record:
                    frame_path = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")
                    self.capture_scene_png(frame_path, height=height, width=width)
                else:
                    time.sleep(1.0 / fps)

                frame_idx += 1

            # Hide this grasp after it reaches full opacity
            self.robot_dict[robot_name].set_visibility(False)

        print(f"Animation complete! Total frames: {frame_idx}")

        # Encode video if recording
        if record:
            if self.output_path.value.strip() == "":
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = f"animation_{timestamp}.mp4"
            else:
                out_path = self.output_path.value
                if not out_path.endswith('.mp4'):
                    out_path += '.mp4'

            print("Encoding video...")
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-loglevel', 'warning',
                '-framerate', str(fps),
                '-i', os.path.join(temp_dir, 'frame_%04d.png'),
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', '18',
                out_path
            ]

            subprocess.run(ffmpeg_cmd, check=True)
            shutil.rmtree(temp_dir)
            print(f"Video saved: {out_path}")


if __name__ == "__main__":#)
    version = "baseline"
    obj_name = "banana"
    scene_type = "wall"
    scene_idx = "0"
    vis = Renderer(version, obj_name, scene_type, scene_idx)
    vis.start_viewer()