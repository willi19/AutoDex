import os
import json
import random
import time
import subprocess
import tempfile
import cv2
random.seed(42)

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot
from paradex.visualization.visualizer.viser import ViserViewer
from rsslib.path import obj_path, candidate_path, urdf_path
import transforms3d
import shapely.geometry as geom

COLORS = {
    "target_obj":   (0, 100, 0),         # 짙은 초록 (타겟)
    "obstacle":     (119, 136, 153),     # 쿨 슬레이트 그레이 (장애물)
    "table":        (255, 255, 255),     # 밝은 쿨 화이트 (테이블)
    "robot":        (250, 250, 250),     # 순백색 (로봇)
}
OBJ_NAME = "soap_dispenser"

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
    q = Rot.from_matrix(T[:3, :3]).as_quat()  # x y z w
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
    
    # 가장 왼쪽 (Y 최솟값)
    min_y = corners_world[:, 1].min()
    max_z = corners_world[:, 2].max()
    
    # Wall dimensions - adaptive height
    wall_height = max_z + 0.1  # object 최고점 + 10cm (hand clearance)
    wall_y = min_y - gap
    
    # Scene
    scene = get_tabletop_scene(obj_name, rotated_pose)
    scene["cuboid"]["wall"] = {
        "dims": [0.3, 0.02, wall_height],
        "pose": [0.0, wall_y - 0.01, wall_height/2, 1, 0, 0, 0],
    }
    return scene
    
def get_tabletop_scene(obj_name, tabletop_pose):
    ret = {
        "mesh": {},
        "cuboid": {
            # "table": {
            #     "dims": [6.0, 6.0, 0.2],
            #     "pose": [0.0, 0.0, -0.1, 1, 0, 0, 0.0],
            # }   
        }
    }
    ret["mesh"]["target"] = get_mesh_dict(obj_name, tabletop_pose)
    return ret

def get_box_scene(
    obj_name,
    tabletop_pose,
    height_offset,
    z_scene_theta=0.0,
    x_offset=0.0,
):
    """
    Object를 감싸는 tight box 생성 (Union용 Overlap 적용)
    """

    # Load mesh
    mesh_path = os.path.join(obj_path, obj_name, "processed_data", "mesh", "simplified.obj")
    mesh = trimesh.load(mesh_path, force="mesh")

    # Scene transform
    T_scene = transl([x_offset, 0, 0]) @ rotz(z_scene_theta)

    # Object vertices in world frame
    verts = mesh.vertices
    R_obj = tabletop_pose[:3, :3]
    t_obj = tabletop_pose[:3, 3]
    verts_w = (R_obj @ verts.T).T + t_obj

    # XY projection → minimum rotated rectangle
    points_xy = verts_w[:, :2]
    poly = geom.MultiPoint(points_xy).convex_hull
    rect = poly.minimum_rotated_rectangle
    rect_pts = np.array(rect.exterior.coords)[:4]
    
    cx, cy = rect.centroid.coords[0]
    edge = rect_pts[1] - rect_pts[0]
    
    # Box Orientation
    yaw = np.arctan2(edge[1], edge[0])
    
    # Dimensions (Inner Cavity)
    width = np.linalg.norm(rect_pts[1] - rect_pts[0])
    depth = np.linalg.norm(rect_pts[2] - rect_pts[1])

    # Wall height
    max_z = verts_w[:, 2].max()
    wall_height = max_z - height_offset
    if wall_height <= 0:
        return None

    THICK = 0.02

    # Object scene
    scene = get_tabletop_scene(obj_name, T_scene @ tabletop_pose)

    # Box center transform
    T_box_center = np.eye(4)
    T_box_center[:3, :3] = rotz(yaw)[:3, :3]
    T_box_center[:3, 3] = [cx, cy, wall_height / 2]
    
    def add_wall(name, local_x, local_y, w, d):
        T_local = np.eye(4)
        T_local[:3, 3] = [local_x, local_y, 0]
        T_wall = T_box_center @ T_local
        scene["cuboid"][name] = {
            "dims": [w, d, wall_height],
            "pose": mat4_to_pose(T_scene @ T_wall),
        }

    # [FIX] Overlapping Joints Strategy
    # Front/Back Walls: Cover full depth + corners
    # Left/Right Walls: Cover full width + corners (Intersecting)
    # 이렇게 하면 모서리(Corner) 부분에 THICK x THICK 만큼의 교집합이 생겨 Union이 완벽해짐
    
    hw, hd = width/2, depth/2
    
    # Front/Back (X축 방향 오프셋, Y축 방향으로 긴 벽)
    add_wall("box_front", hw + THICK/2, 0, THICK, depth)
    add_wall("box_back", -hw - THICK/2, 0, THICK, depth)
    
    # Left/Right (Y축 방향 오프셋, X축 방향으로 긴 벽)
    # [수정] width -> width + 2*THICK (모서리 겹치게 확장)
    add_wall("box_right", 0, hd + THICK/2, width + 2*THICK, THICK)
    add_wall("box_left", 0, -hd - THICK/2, width + 2*THICK, THICK)

    return scene

def get_shelf_scene(obj_name, tabletop_pose, obb_info,
                    z_rotation_deg, gap, z_scene_theta, t,
                    up=True, side=True, back=True):
    """
    Shelf scene (Union용 Overlap 적용)
    """
    # Z rotation
    angle = np.radians(z_rotation_deg)
    z_rot = np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [np.sin(angle),  np.cos(angle), 0, 0],
        [0,              0,             1, 0],
        [0,              0,             0, 1],
    ])
    pose = z_rot @ tabletop_pose

    # OBB world calculation
    obb_tf = np.array(obb_info["obb_transform"])
    R_obb = obb_tf[:3, :3]
    ext = np.array(obb_info["obb"]) / 2.0

    Rw = pose[:3, :3]
    tw = pose[:3, 3]
    axes = Rw @ R_obb

    # Get AABB of OBB in World Frame
    corners = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                local = np.array([sx, sy, sz]) * ext
                corners.append(axes @ local + tw)
    corners = np.array(corners)

    THICK = 0.02  # 2cm
    
    # Inner Void Dimensions (물체가 들어갈 공간)
    min_x, max_x = corners[:, 0].min(), corners[:, 0].max()
    min_y, max_y = corners[:, 1].min(), corners[:, 1].max()
    max_z = corners[:, 2].max()
    
    # 1. Back Wall (-Y 방향) 기준 위치 설정
    # Shelf의 "뒤쪽"을 OBB의 min_y 쪽으로 설정
    inner_y_back = min_y - gap
    inner_y_front = max_y + gap
    
    inner_x_min = min_x - gap
    inner_x_max = max_x + gap
    
    inner_z_top = max_z + gap
    
    # Dimensions for Walls
    wall_h = inner_z_top  # 바닥(0)부터 천장 아래까지
    
    scene = get_tabletop_scene(obj_name, pose)
    
    # [FIX] Overlapping Logic
    # 모든 벽이 만나는 지점에서 서로 관통하도록 치수 확장
    
    # ---------------- Back Wall (-Y) ----------------
    # X축 전체 커버 (좌우 벽 두께 포함 + 겹침)
    full_width = (inner_x_max - inner_x_min) + 2*THICK
    
    if back:
        scene["cuboid"]["back"] = {
            "dims": [full_width, THICK, wall_h],
            "pose": [
                (inner_x_min + inner_x_max) / 2, # Center X
                inner_y_back - THICK/2,          # Center Y (Inside face at inner_y_back)
                wall_h / 2,                      # Center Z
                1, 0, 0, 0
            ],
        }
    
    # ---------------- Side Walls (±X) ----------------
    # Y축 전체 커버 (뒷벽 두께 포함 + 겹침)
    # [수정] Back Wall을 뚫고 지나가도록 길이 확장
    full_depth = (inner_y_front - inner_y_back) + THICK 
    
    if side:
        # Positive X Side
        scene["cuboid"]["side_pos"] = {
            "dims": [THICK, full_depth-THICK, wall_h],
            "pose": [
                inner_x_max + THICK/2,                    # Center X
                (inner_y_back - THICK/2 + inner_y_front)/2, # Center Y (Covers back wall area)
                wall_h / 2,
                1, 0, 0, 0
            ],
        }
        # Negative X Side
        scene["cuboid"]["side_neg"] = {
            "dims": [THICK, full_depth-THICK, wall_h],
            "pose": [
                inner_x_min - THICK/2,
                (inner_y_back - THICK/2 + inner_y_front)/2,
                wall_h / 2,
                1, 0, 0, 0
            ],
        }
    
    # ---------------- Ceiling (+Z) ----------------
    # XY 평면 전체 커버 (벽 두께들 모두 포함 + 겹침)
    if up:
        ceil_w = full_width 
        ceil_d = full_depth 
        ceil_z_center = inner_z_top + THICK/2
        
        scene["cuboid"]["up"] = {
            "dims": [ceil_w, ceil_d, THICK],
            "pose": [
                (inner_x_min + inner_x_max) / 2,
                (inner_y_back - THICK + inner_y_front)/2,
                ceil_z_center,
                1, 0, 0, 0
            ],
        }
        
    return scene

def get_scene(scene_path):
    # JSON 로드
    with open(scene_path, "r") as f:
        data = json.load(f)

    scene_json = data["scene"]
    meta = data.get("meta", {})
    params = meta.get("param", {})

    # 핵심 정보들
    target = scene_json["mesh"]["target"]

    # 7D pose → 4x4 matrix로 변환
    pose7 = target["pose"]
    trans = np.array(pose7[:3])
    quat = np.array([pose7[4], pose7[5], pose7[6], pose7[3]])  # x y z w
    tabletop_pose = np.eye(4)
    tabletop_pose[:3, :3] = Rot.from_quat(quat).as_matrix()
    tabletop_pose[:3, 3] = trans

    # optional params
    obj_dir = os.path.join(obj_path, OBJ_NAME)
    obb_info = json.load(open(os.path.join(obj_dir, "processed_data", "info", "simplified.json"), 'r'))
    height_offset = params.get("height_offset", 0.05)
    z_rotation = params.get("z_rotation", 0.0)
    gap = params.get("gap", 0.05)
    z_scene_theta = params.get("z_scene_theta", 0.0)
    t = params.get("t", 0.0)
    up = params.get("up", True)
    side = params.get("side", True)
    back = params.get("back", True)
    
    # scene_type 자동 판별
    filename = os.path.basename(os.path.dirname(scene_path)).lower()
    
    if "wall" in filename:
        scene_type = "wall"
    elif "box" in filename:
        scene_type = "box"
    elif "shelf" in filename:
        scene_type = "shelf"
    else:
        scene_type = "tabletop"

    obj_name = target["file_path"].split("/")[-4]
    # ---------------- generate scene ----------------
    if scene_type == "tabletop":
        return get_tabletop_scene(obj_name, tabletop_pose)

    if scene_type == "wall":
        return get_wall_scene(
            obj_name,
            tabletop_pose,
            obb_info,
            z_rotation_deg=z_rotation,
            gap=gap,
            t=t,
            scene_z_rot_deg=z_scene_theta,
        )

    if scene_type == "box":
        return get_box_scene(
            obj_name,
            tabletop_pose,
            height_offset=height_offset,
            z_scene_theta=z_scene_theta,
            x_offset=t
        )

    if scene_type == "shelf":
        return get_shelf_scene(
            obj_name,
            tabletop_pose,
            obb_info,
            z_rotation_deg=z_rotation,
            gap=gap,
            z_scene_theta=z_scene_theta,
            t=t,
            up=up, side=side, back=back
        )

    raise ValueError(f"Unknown scene type: {scene_type}")

 
class GridSceneVisualizer(ViserViewer):
    def __init__(self, obj_name, grid_spacing=0.5, max_scenes=None, version="revalidate", max_grasps_per_scene=4):
        super().__init__()

        # 흰색 배경 설정
        self.server.scene.set_background_image(
            np.ones((1, 1, 3), dtype=np.uint8) * 255
        )
        self.server.gui.configure_theme(dark_mode=False)

        self.obj_name = obj_name
        self.grid_spacing = grid_spacing
        self.version = version
        self.max_scenes = max_scenes
        self.max_grasps_per_scene = max_grasps_per_scene
        self.scene_root = os.path.join(obj_path, obj_name, "scene")

        self.scene_files = []
        for scene_type in sorted(os.listdir(self.scene_root)):
            if scene_type == "float" or scene_type == "table":
                continue

            folder = os.path.join(self.scene_root, scene_type)
            if not os.path.isdir(folder):
                continue
            files = sorted(
                f for f in os.listdir(folder) if f.endswith(".json")
            )
            if scene_type != "box":
                for f in files:
                    self.scene_files.append(os.path.join(folder, f))
            else:
                for f in files:
                    if "4" in f or "5" in f:
                        self.scene_files.append(os.path.join(folder, f))

        random.shuffle(self.scene_files)

        self.scene_handles = {}  # name -> {"frames": [...], "robots": [...], "offset": np.array}
        self._load_all_scenes()
        self.add_video_capture_gui()
        self.add_view_save_gui()
        self.add_scene_select_gui()

    def _load_all_scenes(self):
        # grasp가 있는 scene만 남기기
        filtered = []
        for scene_path in self.scene_files:
            scene_type = os.path.basename(os.path.dirname(scene_path)).lower()
            scene_idx = os.path.basename(scene_path).replace(".json", "")
            grasp_root = os.path.join(candidate_path, self.version, self.obj_name, scene_type, scene_idx)

            if os.path.isdir(grasp_root) and any(
                os.path.isdir(os.path.join(grasp_root, d))
                and os.path.exists(os.path.join(grasp_root, d, "wrist_se3.npy"))
                for d in os.listdir(grasp_root)
            ):
                filtered.append(scene_path)
        # max_scenes 적용 (filtering 후에!)
        if self.max_scenes is not None and len(filtered) > self.max_scenes:
            filtered = filtered[:self.max_scenes-1]

        # shelf28 무조건 추가 (테스트용)
        shelf28_path = os.path.join(self.scene_root, "shelf", "28.json")
        if shelf28_path not in filtered:
            filtered.append(shelf28_path)

        self.scene_files = filtered

        n = len(self.scene_files)
        grid_cols = int(np.ceil(np.sqrt(n)))

        print(f"Loading {n} scenes (grasp available) into a {grid_cols}x{grid_cols} grid")

        target_mesh = trimesh.load(
            os.path.join(obj_path, self.obj_name, "raw_mesh", f"{self.obj_name}.obj"),
            force="mesh"
        )

        for idx, scene_path in enumerate(self.scene_files):
            row = idx // grid_cols
            col = idx % grid_cols
            offset = np.array([col * self.grid_spacing, row * self.grid_spacing, 0])

            scene = get_scene(scene_path)
            scene_type = os.path.basename(os.path.dirname(scene_path)).lower()
            scene_idx = os.path.basename(scene_path).replace(".json", "")
            name = scene_type + scene_idx
            
            scene_frame_names = []
            scene_robot_names = []

            # =============================
            # Mesh
            # =============================
            scene_obj_pose = None
            for mesh_name, info in scene.get("mesh", {}).items():
                mesh = target_mesh

                pose = np.eye(4)
                pose[:3, 3] = np.array(info["pose"][:3]) + offset
                quat = info["pose"][3:]  # w xyz
                pose[:3, :3] = Rot.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()

                obj_id = f"{name}_{mesh_name}"
                self.add_trimesh(obj_id, mesh, pose=pose)
                scene_frame_names.append(obj_id)

                if mesh_name == "target":
                    scene_obj_pose = pose.copy()

            # =============================
            # Cuboid
            # =============================
            for cuboid_name, info in scene.get("cuboid", {}).items():
                box = trimesh.creation.box(extents=info["dims"])

                pose = np.eye(4)
                pose[:3, 3] = np.array(info["pose"][:3]) + offset
                quat = info["pose"][3:]
                pose[:3, :3] = Rot.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()

                obj_id = f"{name}_{cuboid_name}"
                self.add_object(obj_id, box, obj_T=pose)
                scene_frame_names.append(obj_id)

                if cuboid_name == "table":
                    self.change_color(obj_id, (0.5, 0.5, 0.5))
                else:
                    self.change_color(obj_id, (119/255, 136/255, 153/255, 0.55))

            # =============================
            # Grasp trajectories (여러 개 로드)
            # =============================
            if scene_obj_pose is not None:
                grasp_root = os.path.join(
                    candidate_path, self.version, self.obj_name, scene_type, scene_idx
                )
                if os.path.isdir(grasp_root):
                    loaded_count = 0
                    for grasp_name in sorted(os.listdir(grasp_root)):
                        if loaded_count >= self.max_grasps_per_scene:
                            break
                        grasp_path = os.path.join(grasp_root, grasp_name)
                        if not os.path.isdir(grasp_path):
                            continue
                        wrist_path = os.path.join(grasp_path, "wrist_se3.npy")
                        grasp_pose_path = os.path.join(grasp_path, "grasp_pose.npy")
                        if not os.path.exists(wrist_path):
                            continue

                        # 초기 포즈 설정
                        wrist_se3 = np.load(wrist_path)
                        robot_T = scene_obj_pose @ wrist_se3

                        robot_id = f"{name}_robot_{grasp_name}"
                        allegro_path = os.path.join(urdf_path, "allegro_hand_description_right.urdf")
                        self.add_robot(robot_id, allegro_path, pose=robot_T)
                        scene_robot_names.append(robot_id)

                        if os.path.exists(grasp_pose_path):
                            grasp_pose = np.load(grasp_pose_path).flatten()
                            self.robot_dict[robot_id].update_cfg(grasp_pose)

                        # 보라색으로 통일
                        self.robot_dict[robot_id].change_color([], (0.6, 0.5, 0.88, 1.0))

                        print(f"[Grid] Loaded grasp {grasp_name} for {name}")
                        loaded_count += 1

            self.scene_handles[name] = {
                "frames": scene_frame_names,
                "robots": scene_robot_names,
                "offset": offset.copy(),
            }

        # self.add_floor(0.0)

    def add_scene_select_gui(self):
        self._focused = False
        # 가시성 상태 추적용 변수
        self._show_obstacles = True
        self._show_robots = True

        with self.server.gui.add_folder("Scene Selection"):
            self.focus_scene_input = self.server.gui.add_text(
                "Scene", initial_value="shelf28"
            )
            self.focus_btn = self.server.gui.add_button("Focus Scene")
            
            # --- 추가된 토글 버튼들 ---
            self.toggle_obstacle_btn = self.server.gui.add_button("Toggle Obstacles")
            self.toggle_robot_btn = self.server.gui.add_button("Toggle Robots")

        @self.focus_btn.on_click
        def _(_) -> None:
            scene_name = self.focus_scene_input.value.strip()
            if scene_name not in self.scene_handles:
                print(f"Scene '{scene_name}' not found.")
                return
            if self._focused:
                self._set_scene_visibility(list(self.scene_handles.keys()))
                self._focused = False
            else:
                self._set_scene_visibility([scene_name])
                self._focused = True

        # --- Obstacle 토글 로직 ---
        @self.toggle_obstacle_btn.on_click
        def _(_) -> None:
            scene_name = self.focus_scene_input.value.strip()
            if scene_name not in self.scene_handles:
                return
            
            self._show_obstacles = not self._show_obstacles
            handles = self.scene_handles[scene_name]
            
            for frame_name in handles["frames"]:
                # 'target'이 아닌 cuboid들(wall, side, back 등)만 필터링
                if "target" not in frame_name and frame_name in self.frame_nodes:
                    self.frame_nodes[frame_name].visible = self._show_obstacles
            print(f"Obstacles in {scene_name} visible: {self._show_obstacles}")

        # --- Robot 토글 로직 ---
        @self.toggle_robot_btn.on_click
        def _(_) -> None:
            scene_name = self.focus_scene_input.value.strip()
            if scene_name not in self.scene_handles:
                return
            
            self._show_robots = not self._show_robots
            handles = self.scene_handles[scene_name]
            
            for robot_name in handles["robots"]:
                if robot_name in self.robot_dict:
                    self.robot_dict[robot_name].show_visual = self._show_robots
            print(f"Robots in {scene_name} visible: {self._show_robots}")

    def _set_scene_visibility(self, visible_scenes):
        visible_set = set(visible_scenes)
        for name, handles in self.scene_handles.items():
            vis = name in visible_set
            for frame_name in handles["frames"]:
                if frame_name in self.frame_nodes:
                    self.frame_nodes[frame_name].visible = vis
            for robot_name in handles["robots"]:
                if robot_name in self.robot_dict:
                    self.robot_dict[robot_name].show_visual = vis

    def add_auto_capture_gui(self):
        """Add automation GUI for collision-free grasping visualization"""
        self.auto_capture_dir = os.path.dirname(os.path.abspath(__file__))

        with self.server.gui.add_folder("Auto Capture"):
            self.auto_capture_btn = self.server.gui.add_button("Auto Capture Sequence")
            self.grasp_repeat_input = self.server.gui.add_number(
                "Grasp Repeats", initial_value=3, min=1, max=10, step=1
            )
            self.fade_steps_input = self.server.gui.add_number(
                "Fade Steps", initial_value=10, min=5, max=30, step=1
            )

        @self.auto_capture_btn.on_click
        def _(_) -> None:
            self._auto_capture_sequence()

    def _interpolate_camera(self, start_view, end_view, t):
        """Interpolate camera position between two views (t: 0->1)"""
        # Position interpolation (linear)
        start_pos = np.array(start_view.get("position", [0, 0, 1]))
        end_pos = np.array(end_view.get("position", [0, 0, 1]))
        pos = start_pos * (1 - t) + end_pos * t

        # Orientation interpolation (SLERP)
        start_wxyz = np.array(start_view.get("wxyz", [1, 0, 0, 0]))
        end_wxyz = np.array(end_view.get("wxyz", [1, 0, 0, 0]))

        # Convert wxyz to xyzw for scipy
        start_quat = [start_wxyz[1], start_wxyz[2], start_wxyz[3], start_wxyz[0]]
        end_quat = [end_wxyz[1], end_wxyz[2], end_wxyz[3], end_wxyz[0]]

        from scipy.spatial.transform import Slerp, Rotation
        rots = Rotation.from_quat([start_quat, end_quat])
        slerp = Slerp([0, 1], rots)
        interp_rot = slerp([t])[0]
        interp_quat = interp_rot.as_quat()  # xyzw
        wxyz = [interp_quat[3], interp_quat[0], interp_quat[1], interp_quat[2]]

        return pos, wxyz

    def _set_robot_opacity(self, robot_name, opacity):
        """Set opacity for a robot (0 = invisible, 1 = fully visible)"""
        if robot_name in self.robot_dict:
            robot = self.robot_dict[robot_name]
            robot.show_visual = opacity > 0.01
            # Note: Full opacity control may need mesh material changes

    def _load_additional_grasps(self, scene_name, max_grasps=4):
        """특정 scene에 대해 추가 grasp들을 동적으로 로드"""
        if scene_name not in self.scene_handles:
            return

        handles = self.scene_handles[scene_name]
        existing_robots = set(handles["robots"])

        # scene_name에서 scene_type, scene_idx 추출 (예: "shelf28" -> "shelf", "28")
        scene_type = ''.join(c for c in scene_name if c.isalpha())
        scene_idx = ''.join(c for c in scene_name if c.isdigit())

        # scene의 object pose 찾기
        target_frame = f"{scene_name}_target"
        if target_frame not in self.frame_nodes:
            print(f"Target frame not found for {scene_name}")
            return

        scene_obj_pose = self.frame_nodes[target_frame].wxyz  # 실제 pose 가져오기 필요
        # 대신 offset을 사용
        offset = handles["offset"]

        grasp_root = os.path.join(
            candidate_path, self.version, self.obj_name, scene_type, scene_idx
        )

        if not os.path.isdir(grasp_root):
            return

        loaded_count = len(existing_robots)
        for grasp_name in sorted(os.listdir(grasp_root)):
            if loaded_count >= max_grasps:
                break

            grasp_path = os.path.join(grasp_root, grasp_name)
            if not os.path.isdir(grasp_path):
                continue

            robot_id = f"{scene_name}_robot_{grasp_name}"
            if robot_id in existing_robots:
                continue  # 이미 로드됨

            wrist_path = os.path.join(grasp_path, "wrist_se3.npy")
            grasp_pose_path = os.path.join(grasp_path, "grasp_pose.npy")
            if not os.path.exists(wrist_path):
                continue

            # scene file에서 object pose 다시 읽기
            scene_file = None
            for sf in self.scene_files:
                if scene_type in sf and scene_idx in os.path.basename(sf):
                    scene_file = sf
                    break

            if scene_file:
                scene = get_scene(scene_file)
                target_info = scene.get("mesh", {}).get("target", {})
                if target_info:
                    pose7 = target_info["pose"]
                    scene_obj_pose = np.eye(4)
                    quat = [pose7[4], pose7[5], pose7[6], pose7[3]]  # xyzw
                    scene_obj_pose[:3, :3] = Rot.from_quat(quat).as_matrix()
                    scene_obj_pose[:3, 3] = np.array(pose7[:3]) + offset

                    wrist_se3 = np.load(wrist_path)
                    robot_T = scene_obj_pose @ wrist_se3

                    allegro_path = os.path.join(urdf_path, "allegro_hand_description_right.urdf")
                    self.add_robot(robot_id, allegro_path, pose=robot_T)
                    handles["robots"].append(robot_id)

                    if os.path.exists(grasp_pose_path):
                        grasp_pose = np.load(grasp_pose_path).flatten()
                        self.robot_dict[robot_id].update_cfg(grasp_pose)

                    # 보라색으로 통일 + 처음에는 숨기기
                    self.robot_dict[robot_id].change_color([], (0.6, 0.5, 0.88, 1.0))
                    self.robot_dict[robot_id].show_visual = False

                    print(f"[Dynamic] Loaded additional grasp {grasp_name} for {scene_name}")
                    loaded_count += 1

    def _auto_capture_sequence(self):
        """
        Automated capture sequence:
        1. Object only (no obstacles, no robot)
        2. Obstacles appear
        3. Grasp fade in/out (repeat N times)
        4. Transition to global view (end.json)
        """
        start_json_path = os.path.join(self.auto_capture_dir, "start.json")
        end_json_path = os.path.join(self.auto_capture_dir, "end.json")

        if not os.path.exists(start_json_path) or not os.path.exists(end_json_path):
            print(f"Error: Need start.json and end.json in {self.auto_capture_dir}")
            return

        with open(start_json_path, 'r') as f:
            start_view = json.load(f)
        with open(end_json_path, 'r') as f:
            end_view = json.load(f)

        scene_name = self.focus_scene_input.value.strip()
        if scene_name not in self.scene_handles:
            print(f"Scene '{scene_name}' not found. Set scene name first.")
            return

        # 해당 scene에 대해 추가 grasp 로드 (최대 4개)
        grasp_repeats = int(self.grasp_repeat_input.value)
        self._load_additional_grasps(scene_name, max_grasps=grasp_repeats)

        handles = self.scene_handles[scene_name]
        robot_names = handles["robots"]
        frame_names = handles["frames"]

        if not robot_names:
            print(f"No robots found in scene {scene_name}")
            return

        # Parameters
        grasp_repeats = int(self.grasp_repeat_input.value)
        fade_steps = int(self.fade_steps_input.value)
        width, height = 1024, 800
        fps = 15

        # Prepare frame storage
        temp_dir = tempfile.mkdtemp()
        frame_idx = 0

        print(f"Starting auto capture for scene {scene_name}")
        print(f"  - {grasp_repeats} grasp cycles, {fade_steps} fade steps")

        # Focus on the scene (hide all other scenes)
        self._set_scene_visibility([scene_name])
        self._focused = True

        # Hide all robots initially
        base_color = (0.6, 0.5, 0.88)  # 보라색 계열
        for robot_name in robot_names:
            if robot_name in self.robot_dict:
                robot = self.robot_dict[robot_name]
                robot.show_visual = False

        # Hide obstacles initially (show only target object)
        for frame_name in frame_names:
            if "target" not in frame_name and frame_name in self.frame_nodes:
                self.frame_nodes[frame_name].visible = False

        # Set initial camera (start view)
        self._apply_camera_view(start_view)
        time.sleep(0.5)  # 렌더링 반영 대기 (0.3 -> 0.5)

        # ===== Phase 1: Object only =====
        print("  Phase 1: Object only")
        for _ in range(fps):  # 1 second
            frame_idx = self._capture_frame(temp_dir, frame_idx, width, height)

        # ===== Phase 2: Obstacles appear =====
        print("  Phase 2: Obstacles appear")
        for frame_name in frame_names:
            if "target" not in frame_name and frame_name in self.frame_nodes:
                self.frame_nodes[frame_name].visible = True
        time.sleep(0.2)

        for _ in range(fps):  # 1 second hold
            frame_idx = self._capture_frame(temp_dir, frame_idx, width, height)

        # ===== Phase 3: Grasp fade in/out (다른 grasp들 순환) =====
        if not robot_names:
            print("No robots to show")
            return

        base_color = (0.6, 0.5, 0.88)  # 보라색 계열
        num_grasps = min(grasp_repeats, len(robot_names))

        for cycle in range(num_grasps):
            robot_name = robot_names[cycle % len(robot_names)]
            robot = self.robot_dict[robot_name]
            print(f"  Phase 3: Grasp {cycle + 1}/{num_grasps} - {robot_name}")

            # 숨긴 상태에서 opacity 0 설정 후 visibility ON
            robot.show_visual = False
            robot.change_color([], (*base_color, 0.0))
            time.sleep(0.1)  # 색상 적용 대기
            robot.show_visual = True
            time.sleep(0.05)

            # Fade in (opacity 0 -> 1)
            for step in range(fade_steps):
                opacity = step / (fade_steps - 1)
                robot.change_color([], (*base_color, opacity))
                time.sleep(0.05)
                frame_idx = self._capture_frame(temp_dir, frame_idx, width, height)

            # Hold full opacity
            for _ in range(fps):  # 1 second hold
                frame_idx = self._capture_frame(temp_dir, frame_idx, width, height)

            # Fade out (opacity 1 -> 0)
            for step in range(fade_steps):
                opacity = 1.0 - step / (fade_steps - 1)
                robot.change_color([], (*base_color, opacity))
                time.sleep(0.05)
                frame_idx = self._capture_frame(temp_dir, frame_idx, width, height)

            # Visibility OFF
            robot.show_visual = False
            time.sleep(0.1)

        # ===== Phase 3.5: 마지막 grasp 다시 보이기 (opacity 1) =====
        last_robot = self.robot_dict[robot_names[0]]
        last_robot.change_color([], (*base_color, 1.0))
        last_robot.show_visual = True
        time.sleep(0.2)

        for _ in range(fps):  # 1 second hold
            frame_idx = self._capture_frame(temp_dir, frame_idx, width, height)

        # ===== Phase 4: Transition to global view =====
        print("  Phase 4: Transitioning to global view...")
        transition_frames = fps * 2  # 2 seconds
        for i in range(transition_frames):
            t = i / (transition_frames - 1)
            pos, wxyz = self._interpolate_camera(start_view, end_view, t)

            # Gradually show all scenes
            if t > 0.3:
                self._set_scene_visibility(list(self.scene_handles.keys()))
                self._focused = False

            self._set_camera_position(pos, wxyz)
            time.sleep(0.03)
            frame_idx = self._capture_frame(temp_dir, frame_idx, width, height)

        # Hold final view
        for _ in range(fps):  # 1 second
            frame_idx = self._capture_frame(temp_dir, frame_idx, width, height)

        # Encode video with ffmpeg
        output_path = os.path.join(self.auto_capture_dir, f"auto_capture_{scene_name}.mp4")
        frame_pattern = os.path.join(temp_dir, "frame_%05d.png")

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            output_path
        ]

        subprocess.run(cmd, capture_output=True)

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

        print(f"Auto capture saved to: {output_path}")

    def _apply_camera_view(self, view):
        """Apply camera view from JSON"""
        pos = view.get("position", [0, 0, 1])
        wxyz = view.get("wxyz", [1, 0, 0, 0])
        self._set_camera_position(pos, wxyz)

    def _set_camera_position(self, pos, wxyz):
        """Set camera position and orientation"""
        for client in self.server.get_clients().values():
            client.camera.position = tuple(pos)
            client.camera.wxyz = tuple(wxyz)

    def _capture_frame(self, temp_dir, frame_idx, width, height):
        """Capture a single frame and save to temp directory"""
        for client in self.server.get_clients().values():
            img = client.get_render(height=height, width=width)
            frame_path = os.path.join(temp_dir, f"frame_{frame_idx:05d}.png")
            cv2.imwrite(frame_path, cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
            break
        return frame_idx + 1


# ----------------------------
# 실행
# ----------------------------
vis = GridSceneVisualizer(OBJ_NAME, grid_spacing=0.5, max_scenes=81, max_grasps_per_scene=1)
vis.add_auto_capture_gui()
vis.start_viewer()
