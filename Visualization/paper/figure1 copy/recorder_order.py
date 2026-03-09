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
    "robot": (0.2, 0.6, 1.0),       
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
        self._init_arrows()
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
        
        # 2. Coverage 마스크
        history_indices = []
        for i in range(current_idx):
            col_idx = self.ordered_list[i][-1] 
            history_indices.append(col_idx)
            
        if history_indices:
            covered_mask = np.any(self.valid_array[:, history_indices], axis=1)
        else:
            covered_mask = np.zeros(self.valid_array.shape[0], dtype=bool)
        
        uncovered_mask = ~covered_mask
        current_uncovered_count = np.sum(uncovered_mask)
        
        # 3. 점수 계산
        subset = self.valid_array[uncovered_mask] 
        if subset.shape[0] > 0:
            scores = np.sum(subset, axis=0)
        else:
            scores = np.zeros(self.valid_array.shape[1])
            
        max_score = np.max(scores) if np.max(scores) > 0 else 1.0
        colormap = cm.get_cmap('RdYlBu_r') 

        # 4. 화살표 업데이트 Loop
        for i, handle_dict in enumerate(self.arrow_handles):
            if handle_dict is None: continue
            
            # handle_dict는 {'shaft': ..., 'head': ...} 형태
            
            # 현재/과거 화살표 숨김
            if i <= current_idx:
                handle_dict['shaft'].visible = False
                handle_dict['head'].visible = False
                continue
            
            # 미래 화살표 표시
            col_idx = self.ordered_list[i][-1]
            score = scores[col_idx]
            
            handle_dict['shaft'].visible = True
            handle_dict['head'].visible = True
            
            if score == 0:
                c = COLOR_MAP["arrow_zero"]
                opacity = 0.1
            else:
                ratio = score / max_score
                rgba = colormap(ratio)
                c = rgba[:3]
                opacity = 0.5
                
            # [핵심 수정] dict 접근하여 속성 업데이트
            handle_dict['shaft'].color = (int(c[0]*255), int(c[1]*255), int(c[2]*255))
            handle_dict['head'].color = (int(c[0]*255), int(c[1]*255), int(c[2]*255))
            handle_dict['shaft'].opacity = opacity
            handle_dict['head'].opacity = opacity
                
        # 5. GUI 텍스트
        col_idx_curr = current_item[-1]
        info_str = (
            f"Step: {current_idx} / {len(self.ordered_list)-1}\n"
            f"Scenes Left: {current_uncovered_count}\n"
            f"This Grasp Score: {int(scores[col_idx_curr])}\n"
            f"Scene: {current_item[2]}/{current_item[3]}"
        )
        self.info_text.value = info_str

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

if __name__ == "__main__":
    vis = SequentialRenderer("v2", "attached_container")
    vis.start_viewer()