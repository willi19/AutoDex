import os
import json
import numpy as np
import trimesh
import matplotlib.cm as cm
from scipy.spatial.transform import Rotation as R # add_arrow용
from tqdm import tqdm

from rsslib.conversion import cart2se3
from rsslib.path import candidate_path, obj_path, code_path, urdf_path
from paradex.visualization.visualizer.viser import ViserViewer

# -----------------------------------------------------------------------------
# 색상 설정 (0.0 ~ 1.0 범위)
# -----------------------------------------------------------------------------
COLOR_MAP = {
    "robot": (153/255, 128/255, 224/255),       
    "arrow_high": (1.0, 0.0, 0.0),  
    "arrow_mid": (1.0, 1.0, 0.0),   
    "arrow_low": (0.0, 0.0, 1.0),   
    "arrow_zero": (0.8, 0.8, 0.8),  
}

class SequentialRenderer(ViserViewer):
    def __init__(self, version, obj_name):
        super().__init__()
        self.version = version
        self.obj_name = obj_name
        
        self.base_dir = os.path.join(code_path, "order", version, obj_name)
        self.json_path = os.path.join(self.base_dir, "setcover_order.json")
        self.npy_path = os.path.join(self.base_dir, "valid_array.npy")
        
        self._load_data()
        self.load_object()
        self.add_gui()
        self.update_state(0)

    def add_arrow(self, name, start, end, color=(0,1,0), shaft_radius=0.002, head_radius=0.005, head_length=0.015, opacity=1.0):
        """
        [수정됨] 
        1. Color 입력을 0~1 범위로 가정 (COLOR_MAP과 통일)
        2. dict 리턴 유지
        """
        start = np.array(start)
        end = np.array(end)
        direction = end - start
        total_length = np.linalg.norm(direction)
        
        if total_length < 1e-6: return None # 길이가 너무 짧으면 패스
        
        direction_norm = direction / total_length
        
        # Shaft length
        shaft_length = max(0, total_length - head_length)
        shaft_end = end - direction_norm * head_length
        
        # Create shaft cylinder
        shaft_mesh = trimesh.creation.cylinder(radius=shaft_radius, height=shaft_length)
        
        # Rotate and position shaft
        default_dir = np.array([0, 0, 1])
        # 방향 벡터 정렬 로직 (Cross product 사용이 더 안전할 수 있으나 기존 로직 존중)
        try:
            rotation = R.align_vectors([direction_norm], [default_dir])[0]
            rotation_matrix = rotation.as_matrix()
        except:
            rotation_matrix = np.eye(3)

        shaft_center = (start + shaft_end) / 2
        transform_shaft = np.eye(4)
        transform_shaft[:3, :3] = rotation_matrix
        transform_shaft[:3, 3] = shaft_center
        shaft_mesh.apply_transform(transform_shaft)
        
        # Add shaft mesh (Viser)
        # [주의] color는 이미 0~1 범위이므로 /255 하지 않음
        shaft_handle = self.server.scene.add_mesh_simple(
            name=f"/arrows/{name}_shaft",
            vertices=shaft_mesh.vertices,
            faces=shaft_mesh.faces,
            color=color, 
            opacity=opacity,
        )
        
        # Create cone mesh
        cone_mesh = trimesh.creation.cone(radius=head_radius, height=head_length)
        cone_center = shaft_end
        
        # Transform cone
        transform_cone = np.eye(4)
        transform_cone[:3, :3] = rotation_matrix
        transform_cone[:3, 3] = cone_center
        cone_mesh.apply_transform(transform_cone)
        
        # Add cone mesh
        head_handle = self.server.scene.add_mesh_simple(
            name=f"/arrows/{name}_head",
            vertices=cone_mesh.vertices,
            faces=cone_mesh.faces,
            color=color,
            opacity=opacity,
        )
        
        return {'shaft': shaft_handle, 'head': head_handle}

    def _load_data(self):
        if not os.path.exists(self.json_path) or not os.path.exists(self.npy_path):
            raise FileNotFoundError(f"데이터 파일이 없습니다. {self.base_dir} 확인해주세요.")

        print("Loading data...")
        with open(self.json_path, 'r') as f:
            self.ordered_list = json.load(f)  
        self.valid_array = np.load(self.npy_path)
        
        all_orig_indices = sorted([item[-1] for item in self.ordered_list])
        self.orig_to_col = {orig: i for i, orig in enumerate(all_orig_indices)}
        
        for item in self.ordered_list:
            orig = item[-1]
            if orig in self.orig_to_col:
                item.append(self.orig_to_col[orig]) 
            else:
                item.append(0)
        print(f"Data loaded. Total steps: {len(self.ordered_list)}")

        # Top 6 grasps 정보 출력
        print("\n=== Top 6 Grasps (Set Cover Order) ===")
        for i in range(min(6, len(self.ordered_list))):
            item = self.ordered_list[i]
            print(f"  Grasp {i+1}: {item[2]}/{item[3]}/{item[4]} (idx={item[-2]})")
        print("=" * 40 + "\n")

        # 수동 선택된 6개 grasp (for demo)
        self.selected_grasps = [
            ("shelf", "1", "11"),
            ("shelf", "1", "13"),
            ("shelf", "1", "24"),
            ("wall", "0", "6"),
            ("shelf", "104", "0"),
            ("shelf", "104", "8"),
        ]
        # ordered_list에서 해당 grasp의 인덱스 찾기
        self.selected_indices = []
        for scene_type, scene_id, grasp_name in self.selected_grasps:
            for idx, item in enumerate(self.ordered_list):
                if item[2] == scene_type and item[3] == scene_id and item[4] == grasp_name:
                    self.selected_indices.append(idx)
                    break
        print(f"\n=== Selected 6 Grasps (for Demo) ===")
        for i, (scene_type, scene_id, grasp_name) in enumerate(self.selected_grasps):
            print(f"  Grasp {i+1}: {scene_type}/{scene_id}/{grasp_name}")
        print("=" * 40 + "\n")

    def load_object(self):
        scene_json_path = os.path.join(obj_path, self.obj_name, "scene", "table", "4.json")
        if os.path.exists(scene_json_path):
            cfg = json.load(open(scene_json_path, "r"))
            self.obj_pose = cart2se3(cfg['scene']['mesh']['target']['pose'])
        else:
            self.obj_pose = np.eye(4)
            
        mesh_path = os.path.join(obj_path, self.obj_name, "raw_mesh", f"{self.obj_name}.obj")
        if os.path.exists(mesh_path):
            mesh = trimesh.load(mesh_path)
            self.add_trimesh(self.obj_name, mesh, pose=self.obj_pose)

    def _init_arrows(self):
        """화살표 핸들 생성 및 초기 숨김 처리"""
        self.arrow_handles = [] 
        print("Creating arrows...")
        max_arrows = min(len(self.ordered_list), 300)
        
        for i in range(len(self.ordered_list)):
            if i >= max_arrows:
                self.arrow_handles.append(None)
                continue
                
            item = self.ordered_list[i]
            scene_type, scene_id, grasp_name = item[2], item[3], item[4]
            grasp_path = os.path.join(candidate_path, self.version, self.obj_name, scene_type, scene_id, grasp_name)
            
            if not os.path.exists(grasp_path):
                self.arrow_handles.append(None)
                continue
                
            wrist_se3 = np.load(os.path.join(grasp_path, "wrist_se3.npy"))
            world_wrist = self.obj_pose @ wrist_se3
            
            start = world_wrist[:3, 3]
            direction = world_wrist[:3, 0] 
            end = start + direction * 0.05
            
            # [수정] color 범위 0~1 그대로 전달
            handle_dict = self.add_arrow(
                f"arrow_{i}", 
                start=start, end=end, 
                color=COLOR_MAP["arrow_zero"], 
                shaft_radius=0.002, 
                head_radius=0.005
            )
            
            # [핵심 수정] dict 내부의 mesh handle들에 접근해서 visible 설정
            if handle_dict:
                handle_dict['shaft'].visible = False
                handle_dict['head'].visible = False
                
            self.arrow_handles.append(handle_dict)

    def update_state(self, current_idx):
        # 1. 로봇 그리기
        current_item = self.ordered_list[current_idx]
        self._draw_robot(current_item)

    def _draw_robot(self, item):
        for name in list(self.robot_dict.keys()):
            del self.robot_dict[name]
        self.clear_traj()

        scene_type, scene_id, grasp_name = item[2], item[3], item[4]
        grasp_path = os.path.join(candidate_path, self.version, self.obj_name, scene_type, scene_id, grasp_name)
        
        if not os.path.exists(grasp_path): return

        wrist_se3 = np.load(os.path.join(grasp_path, "wrist_se3.npy"))
        grasp_pose = np.load(os.path.join(grasp_path, "grasp_pose.npy")).reshape(1, -1)
        robot_T = self.obj_pose @ wrist_se3
        
        allegro_urdf = os.path.join(urdf_path, "allegro_hand_description_right.urdf")
        self.add_robot("current_robot", allegro_urdf, pose=robot_T)
        self.robot_dict["current_robot"].change_color([], (*COLOR_MAP["robot"], 1.0))
        self.add_traj("traj", {"current_robot": grasp_pose})
        self.gui_playing.value = True

    def add_gui(self):
        with self.server.gui.add_folder("Sequential Control"):
            self.slider = self.server.gui.add_slider(
                "Step Index", min=0, max=len(self.ordered_list)-1, step=1, initial_value=0
            )
            self.btn_prev = self.server.gui.add_button("Prev Step")
            self.btn_next = self.server.gui.add_button("Next Step")
            self.info_text = self.server.gui.add_text("Info", initial_value="Loading...")

        # Selected 6 Grasps 바로가기 버튼
        with self.server.gui.add_folder("Selected 6 Grasps (for Demo)"):
            self.grasp_btns = {}
            for i, idx in enumerate(self.selected_indices):
                scene_type, scene_id, grasp_name = self.selected_grasps[i]
                label = f"Grasp {i+1}: {scene_type}/{scene_id}/{grasp_name}"
                btn = self.server.gui.add_button(label)
                self.grasp_btns[i+1] = btn

                def make_callback(order_idx):
                    def callback(_):
                        self.slider.value = order_idx
                    return callback
                btn.on_click(make_callback(idx))

            # 현재 선택된 grasp 정보 출력
            self.grasp_info_text = self.server.gui.add_text("Current Grasp", initial_value="")

        # Color Control
        with self.server.gui.add_folder("Color Control"):
            self.robot_color = self.server.gui.add_rgb("Robot Color", initial_value=(153, 128, 224))
            self.apply_color_btn = self.server.gui.add_button("Apply Color")

        @self.apply_color_btn.on_click
        def _(_):
            r, g, b = self.robot_color.value
            color = (r/255, g/255, b/255, 1.0)
            if "current_robot" in self.robot_dict:
                self.robot_dict["current_robot"].change_color([], color)
            print(f"Applied color: RGB({r}, {g}, {b})")

        @self.slider.on_update
        def _(_):
            self.update_state(self.slider.value)
            # 현재 인덱스가 0~5면 Grasp 1~6으로 표시
            idx = self.slider.value
            if idx < 6:
                item = self.ordered_list[idx]
                self.grasp_info_text.value = f"Grasp {idx+1}: {item[2]}/{item[3]}/{item[4]}"
            else:
                self.grasp_info_text.value = f"Step {idx}"

        @self.btn_prev.on_click
        def _(_):
            v = self.slider.value - 1
            if v >= 0: self.slider.value = v

        @self.btn_next.on_click
        def _(_):
            v = self.slider.value + 1
            if v < len(self.ordered_list):
                self.slider.value = v

        # Turntable Control
        with self.server.gui.add_folder("Turntable"):
            self.turntable_frames = self.server.gui.add_slider(
                "Frames", min=4, max=36, step=1, initial_value=12
            )
            self.turntable_radius = self.server.gui.add_slider(
                "Radius", min=0.1, max=1.0, step=0.05, initial_value=0.4
            )
            self.turntable_height = self.server.gui.add_slider(
                "Height", min=0.0, max=0.5, step=0.02, initial_value=0.15
            )
            self.turntable_angle = self.server.gui.add_slider(
                "Angle (deg)", min=0, max=360, step=5, initial_value=0
            )
            self.turntable_preview_btn = self.server.gui.add_button("Preview Angle")
            self.turntable_capture_btn = self.server.gui.add_button("Capture Turntable")

        @self.turntable_preview_btn.on_click
        def _(_):
            self._set_turntable_camera(np.radians(self.turntable_angle.value))

        @self.turntable_capture_btn.on_click
        def _(_):
            self._capture_turntable()

    def _set_turntable_camera(self, angle_rad):
        """카메라를 z축 기준으로 회전시킴"""
        if len(self.server.get_clients()) == 0:
            print("No client connected!")
            return

        client = next(iter(self.server.get_clients().values()))

        # Object 중심 (obj_pose의 translation)
        center = self.obj_pose[:3, 3]

        radius = self.turntable_radius.value
        height = self.turntable_height.value

        # 카메라 위치 (원형 경로)
        cam_x = center[0] + radius * np.cos(angle_rad)
        cam_y = center[1] + radius * np.sin(angle_rad)
        cam_z = center[2] + height

        # 카메라가 center를 바라보도록 방향 계산
        forward = center - np.array([cam_x, cam_y, cam_z])
        forward = forward / np.linalg.norm(forward)

        # Up vector (world z)
        world_up = np.array([0, 0, 1])

        # Right = forward x up
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1, 0, 0])
        right = right / np.linalg.norm(right)

        # Recompute up = right x forward
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        # Rotation matrix (카메라는 -z를 바라봄)
        rot_mat = np.stack([right, up, -forward], axis=1)

        # Quaternion 변환
        quat = R.from_matrix(rot_mat).as_quat()  # x, y, z, w
        wxyz = (quat[3], quat[0], quat[1], quat[2])

        client.camera.position = (cam_x, cam_y, cam_z)
        client.camera.wxyz = wxyz

    def _capture_turntable(self):
        """Turntable 캡쳐"""
        import time

        n_frames = int(self.turntable_frames.value)
        angles = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)

        # 현재 grasp 정보
        idx = self.slider.value
        item = self.ordered_list[idx]
        grasp_name = f"{item[2]}_{item[3]}_{item[4]}"

        output_dir = f"turntable_{grasp_name}"
        os.makedirs(output_dir, exist_ok=True)

        print(f"Capturing {n_frames} frames for {grasp_name}...")

        for i, angle in enumerate(angles):
            self._set_turntable_camera(angle)
            time.sleep(0.3)  # 렌더링 대기

            # 스크린샷 (수동으로 찍어야 함 - 자동 캡쳐 API 없으면)
            print(f"  Frame {i+1}/{n_frames}: angle={np.degrees(angle):.1f}°")

        print(f"Turntable preview complete. Use browser screenshot to capture each frame.")
        print(f"Or press 'Preview Angle' and adjust 'Angle (deg)' slider to capture manually.")

if __name__ == "__main__":
    vis = SequentialRenderer("revalidate", "soap_dispenser")
    vis.start_viewer()