import os
import json
import random
random.seed(42)

import numpy as np
import cv2
import trimesh
from scipy.spatial.transform import Rotation as Rot
from paradex.visualization.visualizer.viser import ViserViewer
from rsslib.path import obj_path, code_path
import transforms3d
import shapely.geometry as geom

COLORS = {
    "target_obj":   (0, 100, 0),         # 짙은 초록 (타겟)
    "obstacle":     (119, 136, 153),     # 쿨 슬레이트 그레이 (장애물)
    "table":        (240, 240, 245),     # 밝은 쿨 화이트 (테이블)
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
    def __init__(self, obj_name, grid_spacing=0.5, max_scenes=60):
        super().__init__()
        self.obj_name = obj_name
        self.grid_spacing = grid_spacing
        self.scene_root = os.path.join(obj_path, obj_name, "scene")

        # 선택된 6개 grasp
        self.selected_grasps = [
            ("shelf", "1", "11"),
            ("shelf", "1", "13"),
            ("shelf", "1", "24"),
            ("wall", "0", "6"),
            ("shelf", "104", "0"),
            ("shelf", "104", "8"),
        ]

        # valid_array, setcover_order 미리 로드
        version = "revalidate"
        order_path = os.path.join(code_path, "order", version, obj_name)
        self.valid_array = np.load(os.path.join(order_path, "valid_array.npy"))
        with open(os.path.join(order_path, "setcover_order.json"), "r") as f:
            self.setcover_order = json.load(f)

        # 선택된 grasp들의 인덱스 찾기
        self.selected_grasp_indices = []
        for scene_type, scene_id, grasp_name in self.selected_grasps:
            for entry in self.setcover_order:
                if entry[2] == scene_type and entry[3] == scene_id and entry[4] == grasp_name:
                    self.selected_grasp_indices.append(entry[-1])
                    break

        # Scene index -> scene name 매핑 구축
        self.scene_idx_to_name = {}
        self.scene_name_to_idx = {}
        idx = 0
        for scene_type in sorted(os.listdir(self.scene_root)):
            if scene_type in ["float", "table"]:
                continue
            folder = os.path.join(self.scene_root, scene_type)
            if not os.path.isdir(folder):
                continue
            for scene_name in sorted(os.listdir(folder)):
                if not scene_name.endswith(".json"):
                    continue
                scene_key = scene_type + scene_name.replace(".json", "")
                if idx < self.valid_array.shape[0]:
                    self.scene_idx_to_name[idx] = scene_key
                    self.scene_name_to_idx[scene_key] = idx
                    idx += 1

        # 6개 grasp 중 하나라도 커버하는 scene만 필터링
        covered_scene_indices = set()
        for grasp_idx in self.selected_grasp_indices:
            covered = np.where(self.valid_array[:, grasp_idx])[0]
            covered_scene_indices.update(covered)

        print(f"Total scenes covered by 6 grasps: {len(covered_scene_indices)}")

        # 커버되는 scene만 scene_files에 추가 (box는 5번만 허용)
        self.scene_files = []
        for scene_type in sorted(os.listdir(self.scene_root)):
            if scene_type in ["float", "table"]:
                continue
            folder = os.path.join(self.scene_root, scene_type)
            if not os.path.isdir(folder):
                continue
            for f in sorted(os.listdir(folder)):
                if not f.endswith(".json"):
                    continue
                # box는 5번만 허용
                if scene_type == "box" and f != "5.json":
                    continue
                scene_key = scene_type + f.replace(".json", "")
                if scene_key in self.scene_name_to_idx:
                    sidx = self.scene_name_to_idx[scene_key]
                    if sidx in covered_scene_indices:
                        self.scene_files.append(os.path.join(folder, f))

        random.shuffle(self.scene_files)

        # 60개가 안 되면 커버 안 되는 scene으로 채우기 (grasp가 있는 scene만)
        self.filler_scene_names = []  # filler scene 이름 저장
        if len(self.scene_files) < max_scenes:
            extra_needed = max_scenes - len(self.scene_files)
            extra_scenes = []
            for scene_type in sorted(os.listdir(self.scene_root)):
                if scene_type in ["float", "table"]:
                    continue
                folder = os.path.join(self.scene_root, scene_type)
                if not os.path.isdir(folder):
                    continue
                for f in sorted(os.listdir(folder)):
                    if not f.endswith(".json"):
                        continue
                    # box는 5번만 허용
                    if scene_type == "box" and f != "5.json":
                        continue
                    scene_key = scene_type + f.replace(".json", "")
                    if scene_key in self.scene_name_to_idx:
                        sidx = self.scene_name_to_idx[scene_key]
                        # 6개 grasp에 커버 안 되지만, 최소 1개 grasp는 있는 scene만
                        if sidx not in covered_scene_indices and np.any(self.valid_array[sidx, :]):
                            extra_scenes.append((os.path.join(folder, f), scene_key))
            random.shuffle(extra_scenes)
            added_extras = extra_scenes[:extra_needed]
            for path, scene_key in added_extras:
                self.scene_files.append(path)
                self.filler_scene_names.append(scene_key)
            print(f"Added {len(added_extras)} uncovered scenes as filler: {self.filler_scene_names}")

        if max_scenes is not None:
            self.scene_files = self.scene_files[:max_scenes]

        print(f"Filtered to {len(self.scene_files)} scenes")

        self.scene_handles = {}  # name -> {"frames": [...], "robots": [...], "offset": np.array}
        self._load_all_scenes()
        self.add_video_capture_gui()
        self.add_view_save_gui()
        self.add_scene_select_gui()
        self.add_set_cover_gui()

    def _load_all_scenes(self):
        n = len(self.scene_files)
        grid_cols = 6  # 6x10 그리드 (6 columns)
        grid_rows = int(np.ceil(n / grid_cols))

        print(f"Loading {n} scenes into a {grid_rows}x{grid_cols} grid")

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

            # =============================
            # Mesh
            # =============================
            for mesh_name, info in scene.get("mesh", {}).items():
                mesh = target_mesh

                pose = np.eye(4)
                pose[:3, 3] = np.array(info["pose"][:3]) + offset
                quat = info["pose"][3:]  # w xyz
                pose[:3, :3] = Rot.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()

                obj_id = f"{name}_{mesh_name}"
                self.add_trimesh(obj_id, mesh, pose=pose)
                scene_frame_names.append(obj_id)

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
                    self.change_color(obj_id, (119/255, 136/255, 153/255, 0.4))

            self.scene_handles[name] = {
                "frames": scene_frame_names,
                "robots": [],
                "offset": offset.copy(),
            }

        # self.add_floor(0.0)  # 바닥 제거

    def add_view_save_gui(self):
        with self.server.gui.add_folder("Save/Load View"):
            self.view_file_path = self.server.gui.add_text(
                "File Path", initial_value="view.json"
            )
            self.save_view_btn = self.server.gui.add_button("Save View")
            self.load_view_btn = self.server.gui.add_button("Load View")
            self.level_view_btn = self.server.gui.add_button("Level View")

        @self.save_view_btn.on_click
        def _(_) -> None:
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
        def _(_) -> None:
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

        @self.level_view_btn.on_click
        def _(_) -> None:
            if len(self.server.get_clients()) == 0:
                print("No client connected!")
                return
            
            client = next(iter(self.server.get_clients().values()))
            pos = np.array(client.camera.position)
            w, x, y, z = client.camera.wxyz

            # 1. 현재 카메라가 바라보는 방향(Forward vector) 계산
            r_current = Rot.from_quat([x, y, z, w])
            # 기본적으로 카메라는 -z(또는 라이브러리에 따라 +x)를 바라봅니다. 
            # Viser/Three.js 기준으로는 보통 [0, 0, -1]이 정면입니다.
            forward = r_current.as_matrix() @ np.array([0, 0, -1])

            # 2. 새로운 Right, Up, Forward 축 계산 (Gram-Schmidt 유사 방식)
            # 월드의 Up은 [0, 0, 1]
            world_up = np.array([0, 0, 1])
            
            # Right = Forward x World_Up (수평 오른쪽 방향 추출)
            right = np.cross(forward, world_up)
            
            # 만약 카메라가 수직으로 하늘/땅을 보고 있어서 cross product가 0이 되는 경우 처리
            if np.linalg.norm(right) < 1e-6:
                print("Camera is looking straight up/down, cannot level Roll.")
                return

            right /= np.linalg.norm(right)
            
            # New Up = Right x Forward (이제 New Up은 월드 Z축 쪽으로 최대한 정렬됨)
            new_up = np.cross(-right, forward)
            new_up /= np.linalg.norm(new_up)

            # 3. 새로운 Rotation Matrix 생성
            # Matrix columns: [Right, Up, -Forward] (Viser/Three.js 좌표계 기준)
            new_R = np.stack([-right, new_up, -forward], axis=1)
            
            # 4. 쿼터니언 변환 및 적용
            new_q = Rot.from_matrix(new_R).as_quat() # x, y, z, w
            client.camera.wxyz = (new_q[3], new_q[0], new_q[1], new_q[2])
            
            print("View leveled: Up vector is now aligned with World +Z")

    def add_scene_select_gui(self):
        self._focused = False
        # 가시성 상태 추적용 변수
        self._show_obstacles = True
        self._show_robots = True

        with self.server.gui.add_folder("Scene Selection"):
            self.focus_scene_input = self.server.gui.add_text(
                "Scene", initial_value="shelf102"
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

    def add_set_cover_gui(self):
        """Set cover algorithm visualization GUI"""
        # 전체 scene 이름 리스트
        all_scenes = list(self.scene_handles.keys())

        # 이미 __init__에서 로드한 데이터 사용
        self.grasp_coverage = {}
        self.grasp_info = {}
        for i, grasp_idx in enumerate(self.selected_grasp_indices):
            scene_type, scene_id, grasp_name = self.selected_grasps[i]
            grasp_key = f"{scene_type}/{scene_id}/{grasp_name}"

            # valid_array에서 이 grasp가 cover하는 scene들 찾기
            covered_scene_indices = np.where(self.valid_array[:, grasp_idx])[0]
            covered_scenes = set()
            for sidx in covered_scene_indices:
                if sidx in self.scene_idx_to_name:
                    sname = self.scene_idx_to_name[sidx]
                    if sname in all_scenes:
                        covered_scenes.add(sname)

            self.grasp_coverage[i + 1] = covered_scenes
            self.grasp_info[i + 1] = grasp_key
            print(f"Grasp {i + 1} ({grasp_key}): covers {len(covered_scenes)} scenes")

        # 현재 남아있는 scene들
        self.remaining_scenes = set(all_scenes)

        # 기본 색상 저장
        self.original_colors = {}
        for name in all_scenes:
            self.original_colors[name] = (119/255, 136/255, 153/255, 0.4)  # obstacle color

        # 현재 하이라이트된 grasp
        self._current_highlight = None

        with self.server.gui.add_folder("Set Cover Simulation"):
            # Grasp 선택 버튼들
            self.grasp_btns = {}
            for i in range(1, 7):
                btn = self.server.gui.add_button(f"Show Grasp {i} Coverage")
                self.grasp_btns[i] = btn

                def make_callback(grasp_id):
                    def callback(_):
                        self._highlight_grasp_coverage(grasp_id)
                    return callback
                btn.on_click(make_callback(i))

            # 리셋 버튼
            self.reset_highlight_btn = self.server.gui.add_button("Reset Highlight")
            self.remove_best_btn = self.server.gui.add_button("Remove Best Grasp's Scenes")
            self.full_reset_btn = self.server.gui.add_button("Full Reset")

            # 자동 스크린샷
            self.auto_screenshot_btn = self.server.gui.add_button("Auto Screenshot (Set Cover)")

        @self.reset_highlight_btn.on_click
        def _(_):
            self._reset_highlight()

        @self.remove_best_btn.on_click
        def _(_):
            self._remove_best_grasp_scenes()

        @self.full_reset_btn.on_click
        def _(_):
            self._full_reset()

        @self.auto_screenshot_btn.on_click
        def on_auto_screenshot(_):
            print("Button clicked!")
            self._auto_screenshot_set_cover()

    def _highlight_grasp_coverage(self, grasp_id):
        """특정 grasp가 cover하는 scene들을 초록색으로 하이라이트"""
        # 먼저 모든 scene을 기본 색상으로
        self._reset_highlight()

        covered = self.grasp_coverage.get(grasp_id, set())
        # remaining_scenes와 교집합
        covered_remaining = covered & self.remaining_scenes

        print(f"Grasp {grasp_id}: covers {len(covered_remaining)} remaining scenes")

        # 초록색으로 하이라이트
        for name in covered_remaining:
            handles = self.scene_handles[name]
            for frame_name in handles["frames"]:
                if "target" in frame_name:
                    # target mesh는 진한 초록
                    self.change_color(frame_name, (0/255, 200/255, 0/255, 1.0))
                else:
                    # obstacle은 연한 초록
                    self.change_color(frame_name, (100/255, 200/255, 100/255, 0.6))

        self._current_highlight = grasp_id

    def _reset_highlight(self):
        """하이라이트 리셋 - 모든 남은 scene을 기본 색상으로"""
        for name in self.remaining_scenes:
            handles = self.scene_handles[name]
            for frame_name in handles["frames"]:
                if "target" in frame_name:
                    self.change_color(frame_name, (0.5, 0.5, 0.5, 1.0))
                else:
                    self.change_color(frame_name, (119/255, 136/255, 153/255, 0.4))
        self._current_highlight = None

    def _remove_best_grasp_scenes(self):
        """가장 많이 cover하는 grasp의 scene들을 제거"""
        # 각 grasp별로 remaining과의 교집합 크기 계산
        best_grasp = None
        best_count = 0

        for grasp_id, covered in self.grasp_coverage.items():
            count = len(covered & self.remaining_scenes)
            if count > best_count:
                best_count = count
                best_grasp = grasp_id

        if best_grasp is None:
            print("No more scenes to remove")
            return

        # best grasp가 cover하는 scene들 제거 (visibility off)
        to_remove = self.grasp_coverage[best_grasp] & self.remaining_scenes
        print(f"Selected Grasp {best_grasp} (covers {len(to_remove)} scenes) - Removing...")

        for name in to_remove:
            handles = self.scene_handles[name]
            for frame_name in handles["frames"]:
                if frame_name in self.frame_nodes:
                    self.frame_nodes[frame_name].visible = False

        # remaining에서 제거
        self.remaining_scenes -= to_remove

        # 해당 grasp도 제거
        del self.grasp_coverage[best_grasp]

        print(f"Remaining scenes: {len(self.remaining_scenes)}")
        print(f"Remaining grasps: {list(self.grasp_coverage.keys())}")

    def _full_reset(self):
        """전체 리셋"""
        all_scenes = list(self.scene_handles.keys())

        # 모든 scene 다시 표시
        for name in all_scenes:
            handles = self.scene_handles[name]
            for frame_name in handles["frames"]:
                if frame_name in self.frame_nodes:
                    self.frame_nodes[frame_name].visible = True

        # remaining 리셋
        self.remaining_scenes = set(all_scenes)

        # grasp coverage 다시 계산 (이미 로드된 데이터 사용)
        self.grasp_coverage = {}
        self.grasp_info = {}
        for i, grasp_idx in enumerate(self.selected_grasp_indices):
            scene_type, scene_id, grasp_name = self.selected_grasps[i]
            grasp_key = f"{scene_type}/{scene_id}/{grasp_name}"

            covered_scene_indices = np.where(self.valid_array[:, grasp_idx])[0]
            covered_scenes = set()
            for sidx in covered_scene_indices:
                if sidx in self.scene_idx_to_name:
                    sname = self.scene_idx_to_name[sidx]
                    if sname in all_scenes:
                        covered_scenes.add(sname)

            self.grasp_coverage[i + 1] = covered_scenes
            self.grasp_info[i + 1] = grasp_key

        self._reset_highlight()
        print("Full reset complete")

    def _capture_png(self, client, filepath, width, height):
        """client.get_render()로 스크린샷 찍기"""
        import time
        time.sleep(0.2)  # 렌더링 대기
        img = client.get_render(height=height, width=width)
        cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def _auto_screenshot_set_cover(self):
        """자동으로 Set Cover 과정 스크린샷 찍기"""
        import time
        import traceback

        print("=== Auto Screenshot Started ===")

        try:
            # 1. view.json에서 view 로드
            view_path = self.view_file_path.value.strip() or "view.json"
            print(f"Looking for view file: {view_path}")
            if not os.path.exists(view_path):
                print(f"view.json not found: {view_path}")
                return

            with open(view_path, "r") as f:
                view = json.load(f)
            print(f"View file loaded successfully")

            clients = self.server.get_clients()
            print(f"Connected clients: {len(clients)}")
            if len(clients) == 0:
                print("No client connected!")
                return

            client = next(iter(clients.values()))
            client.camera.position = tuple(view["position"])
            client.camera.wxyz = tuple(view["wxyz"])
            print(f"Camera view set")

            # 스크린샷 설정
            width = int(self.capture_width.value)
            height = int(self.capture_height.value)
            output_dir = "setcover_screenshots"
            os.makedirs(output_dir, exist_ok=True)
            print(f"Output dir: {output_dir}, Resolution: {width}x{height}")

            # 2. Full reset
            self._full_reset()
            time.sleep(0.5)

            # 3. 초기 상태 스크린샷
            print("Capturing initial state...")
            self._capture_png(client, f"{output_dir}/00_initial.png", width, height)
            print(f"Saved: 00_initial.png")

            # 4. 각 grasp의 개별 coverage 스크린샷 (지우기 전)
            for grasp_id in range(1, 7):
                self._highlight_grasp_coverage(grasp_id)
                time.sleep(0.3)
                self._capture_png(client, f"{output_dir}/01_grasp{grasp_id}_coverage.png", width, height)
                print(f"Saved: 01_grasp{grasp_id}_coverage.png")
                self._reset_highlight()
                time.sleep(0.2)

            # 5. Set cover 과정 스크린샷
            step = 1
            while self.grasp_coverage:
                # 가장 많이 커버하는 grasp 찾기
                best_grasp = None
                best_count = 0
                for grasp_id, covered in self.grasp_coverage.items():
                    count = len(covered & self.remaining_scenes)
                    if count > best_count:
                        best_count = count
                        best_grasp = grasp_id

                if best_grasp is None:
                    break

                # 하이라이트
                self._highlight_grasp_coverage(best_grasp)
                time.sleep(0.3)
                self._capture_png(client, f"{output_dir}/step{step:02d}_a_highlight_grasp{best_grasp}.png", width, height)
                print(f"Saved: step{step:02d}_a_highlight_grasp{best_grasp}.png")

                # Scene 제거
                self._remove_best_grasp_scenes()
                time.sleep(0.3)
                self._capture_png(client, f"{output_dir}/step{step:02d}_b_removed.png", width, height)
                print(f"Saved: step{step:02d}_b_removed.png")

                # 남은 grasp들의 개별 coverage 스크린샷
                remaining_grasps = sorted(self.grasp_coverage.keys())
                for grasp_id in remaining_grasps:
                    self._highlight_grasp_coverage(grasp_id)
                    time.sleep(0.3)
                    self._capture_png(client, f"{output_dir}/step{step:02d}_c_grasp{grasp_id}_coverage.png", width, height)
                    print(f"Saved: step{step:02d}_c_grasp{grasp_id}_coverage.png")
                    self._reset_highlight()
                    time.sleep(0.2)

                step += 1

            print(f"Auto screenshot complete! Images saved to {output_dir}/")

        except Exception as e:
            print(f"ERROR in auto screenshot: {e}")
            traceback.print_exc()

# ----------------------------
# 실행
# ----------------------------
vis = GridSceneVisualizer(OBJ_NAME, grid_spacing=0.5, max_scenes=60)
vis.start_viewer()
