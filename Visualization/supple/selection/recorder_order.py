import os
import json
import numpy as np
import cv2
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

        # 흰색 배경 설정
        self.server.gui.configure_theme(dark_mode=False)

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
        self.add_video_capture_gui()

        with self.server.gui.add_folder("Sequential Control"):
            self.slider = self.server.gui.add_slider(
                "Step Index", min=0, max=len(self.ordered_list)-1, step=1, initial_value=0
            )
            self.btn_prev = self.server.gui.add_button("Prev Step")
            self.btn_next = self.server.gui.add_button("Next Step")
            self.info_text = self.server.gui.add_text("Info", initial_value="Loading...")

        # Save/Load View
        with self.server.gui.add_folder("Save/Load View"):
            self.view_file_path = self.server.gui.add_text("File Path", initial_value="view.json")
            self.save_view_btn = self.server.gui.add_button("Save View")
            self.load_view_btn = self.server.gui.add_button("Load View")

        @self.save_view_btn.on_click
        def _(_):
            if len(self.server.get_clients()) == 0:
                print("No client connected!")
                return
            client = next(iter(self.server.get_clients().values()))
            view = {
                "position": list(client.camera.position),
                "wxyz": list(client.camera.wxyz),
            }
            path = self.view_file_path.value.strip() or "view.json"
            with open(path, "w") as f:
                json.dump(view, f, indent=2)
            print(f"View saved: {path}")

        @self.load_view_btn.on_click
        def _(_):
            if len(self.server.get_clients()) == 0:
                print("No client connected!")
                return
            path = self.view_file_path.value.strip() or "view.json"
            if not os.path.exists(path):
                print(f"File not found: {path}")
                return
            with open(path, "r") as f:
                view = json.load(f)
            client = next(iter(self.server.get_clients().values()))
            client.camera.position = tuple(view["position"])
            client.camera.wxyz = tuple(view["wxyz"])
            print(f"View loaded: {path}")

        @self.slider.on_update
        def _(_):
            self.update_state(self.slider.value)

        @self.btn_prev.on_click
        def _(_):
            v = self.slider.value - 1
            if v >= 0: self.slider.value = v

        @self.btn_next.on_click
        def _(_):
            v = self.slider.value + 1
            if v < len(self.ordered_list):
                self.slider.value = v

        # Turntable Control (Object 회전)
        with self.server.gui.add_folder("Turntable"):
            self.turntable_frames = self.server.gui.add_slider(
                "Frames", min=12, max=60, step=1, initial_value=30
            )
            self.turntable_angle = self.server.gui.add_slider(
                "Angle (deg)", min=0, max=360, step=5, initial_value=0
            )
            self.turntable_format = self.server.gui.add_dropdown(
                "Output Format",
                options=["mp4 (no alpha)", "mov (ProRes alpha)", "png sequence"],
                initial_value="mp4 (no alpha)"
            )
            self.turntable_preview_btn = self.server.gui.add_button("Preview Angle")
            self.turntable_capture_btn = self.server.gui.add_button("Capture Turntable")
            self.turntable_capture_all_btn = self.server.gui.add_button("Capture All (0~15)")

        @self.turntable_preview_btn.on_click
        def _(_):
            self._set_turntable_angle(np.radians(self.turntable_angle.value))

        @self.turntable_capture_btn.on_click
        def _(_):
            self._capture_turntable(output_format=self.turntable_format.value)

        @self.turntable_capture_all_btn.on_click
        def _(_):
            self._capture_all_turntables()

    def _set_turntable_angle(self, angle_rad):
        """Object와 robot을 z축 기준으로 회전시킴"""
        # Z축 회전 행렬
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        rot_z = np.array([
            [c, -s, 0, 0],
            [s,  c, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1]
        ])

        # 회전된 object pose
        rotated_obj_pose = rot_z @ self.obj_pose

        # Object mesh 업데이트
        if self.obj_name in self.obj_dict:
            frame = self.obj_dict[self.obj_name]['frame']
            pos = rotated_obj_pose[:3, 3]
            quat = R.from_matrix(rotated_obj_pose[:3, :3]).as_quat()  # x, y, z, w
            frame.position = tuple(pos)
            frame.wxyz = (quat[3], quat[0], quat[1], quat[2])

        # Robot 업데이트
        if "current_robot" in self.robot_dict:
            idx = self.slider.value
            item = self.ordered_list[idx]
            scene_type, scene_id, grasp_name = item[2], item[3], item[4]
            grasp_path = os.path.join(candidate_path, self.version, self.obj_name, scene_type, scene_id, grasp_name)

            if os.path.exists(grasp_path):
                wrist_se3 = np.load(os.path.join(grasp_path, "wrist_se3.npy"))
                robot_T = rotated_obj_pose @ wrist_se3
                robot = self.robot_dict["current_robot"]
                if hasattr(robot, '_visual_root_frame'):
                    robot._visual_root_frame.position = tuple(robot_T[:3, 3])
                    quat = R.from_matrix(robot_T[:3, :3]).as_quat()  # x, y, z, w
                    robot._visual_root_frame.wxyz = (quat[3], quat[0], quat[1], quat[2])

    def _remove_white_background(self, img_path, threshold=250):
        """흰색 배경을 투명으로 변환하고 덮어쓰기"""
        import imageio.v2 as imageio

        img = imageio.imread(img_path)

        # RGB인 경우 RGBA로 변환
        if img.shape[2] == 3:
            rgba = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
            rgba[:, :, :3] = img
            rgba[:, :, 3] = 255
        else:
            rgba = img.copy()

        # 흰색에 가까운 픽셀 찾기 (R, G, B 모두 threshold 이상)
        white_mask = np.all(rgba[:, :, :3] >= threshold, axis=2)
        rgba[white_mask, 3] = 0  # 투명하게

        imageio.imwrite(img_path, rgba)

    def _capture_turntable(self, output_format="mp4 (no alpha)"):
        """Turntable 캡쳐 -> 다양한 포맷으로 저장"""
        import time
        import subprocess
        import tempfile
        import shutil

        if len(self.server.get_clients()) == 0:
            print("No client connected!")
            return

        # trajectory 재생 멈추기
        self.gui_playing.value = False

        n_frames = int(self.turntable_frames.value)
        angles = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)

        # 현재 grasp 정보
        idx = self.slider.value
        item = self.ordered_list[idx]
        grasp_name = f"{item[2]}_{item[3]}_{item[4]}"

        # 영상 설정
        width, height = 540, 540
        fps = 10

        # 투명 배경 여부
        need_alpha = "ProRes" in output_format or "png" in output_format

        # 출력 경로 결정
        if "mov" in output_format:
            output_path = os.path.abspath(f"turntable_{grasp_name}.mov")
        elif "png" in output_format:
            output_path = os.path.abspath(f"turntable_{grasp_name}_frames")
            os.makedirs(output_path, exist_ok=True)
        else:
            output_path = os.path.abspath(f"turntable_{grasp_name}.mp4")

        # 임시 디렉토리에 프레임 저장
        if "png" in output_format:
            temp_dir = output_path  # PNG 시퀀스는 바로 저장
        else:
            temp_dir = tempfile.mkdtemp()

        print(f"Capturing {n_frames} frames for {grasp_name}... (format={output_format}, alpha={need_alpha})")

        for i, angle in enumerate(angles):
            self._set_turntable_angle(angle)
            time.sleep(0.3)  # 렌더링 대기

            # 스크린샷 캡쳐
            frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
            self.capture_scene_png(frame_path, height=height, width=width)

            # 투명 배경 필요시 흰색 → 투명 변환
            if need_alpha:
                self._remove_white_background(frame_path)

            print(f"  Frame {i+1}/{n_frames}: angle={np.degrees(angle):.1f}°")

        # PNG 시퀀스면 여기서 종료
        if "png" in output_format:
            print(f"PNG sequence saved to: {output_path}")
            return

        # ffmpeg로 영상 생성
        print("Encoding video with ffmpeg...")

        if "ProRes" in output_format:
            # ProRes 4444 with alpha (MOV container) - Keynote 지원
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", os.path.join(temp_dir, "frame_%04d.png"),
                "-c:v", "prores_ks",
                "-profile:v", "4444",
                "-pix_fmt", "yuva444p10le",
                output_path
            ]
        else:
            # MP4 with h264 (no alpha)
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", os.path.join(temp_dir, "frame_%04d.png"),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                output_path
            ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ffmpeg error: {result.stderr}")

        # 임시 파일 정리
        shutil.rmtree(temp_dir)
        print(f"Turntable video saved: {output_path}")

    def _capture_all_turntables(self):
        """0~15 스텝 전체 turntable 캡쳐"""
        import time

        output_format = self.turntable_format.value
        print(f"=== Capturing all turntables (0~15), format={output_format} ===")
        for step_idx in range(16):
            print(f"\n--- Step {step_idx}/15 ---")
            self.slider.value = step_idx
            time.sleep(0.5)  # 상태 업데이트 대기
            self._capture_turntable(output_format=output_format)

        print("\n=== All turntables captured! ===")

if __name__ == "__main__":
    vis = SequentialRenderer("revalidate", "soap_dispenser")
    vis.start_viewer()