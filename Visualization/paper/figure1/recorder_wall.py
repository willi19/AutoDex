import os
import glob
import json
import numpy as np
import trimesh
import transforms3d
from scipy.spatial.transform import Rotation as R
import shapely.geometry as geom

from paradex.utils.path import shared_dir
from paradex.visualization.visualizer.viser import ViserViewer

from rsslib.conversion import se32cart, cart2se3
from rsslib.path import candidate_path, obj_path, urdf_path

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
    
    # 가장 왼쪽 (Y 최솟값)
    min_y = corners_world[:, 1].min()
    max_z = corners_world[:, 2].max()
    
    # Wall dimensions - adaptive height
    wall_height = max_z + 0.1  # object 최고점 + 10cm (hand clearance)
    wall_y = min_y - gap
    
    # Scene
    scene = get_tabletop_scene(obj_name, rotated_pose)
    scene["cuboid"]["wall"] = {
        "dims": [0.6, 0.02, wall_height],
        "pose": [0.0, wall_y - 0.01, wall_height/2, 1, 0, 0, 0],
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
    add_wall("box_front", hw + THICK/2, 0, THICK, depth + 2*THICK)
    add_wall("box_back", -hw - THICK/2, 0, THICK, depth + 2*THICK)
    
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

class Renderer(ViserViewer):
    def __init__(self, version, obj_name, scene_type, scene_idx):
        super().__init__()

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
        self.load_scene()
        
        self.squeeze_num = 10

        self.add_appearance_gui()

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

    def _update_object_appearance(self, color_rgb, opacity):
        """Update object appearance"""
        color = tuple(c / 255.0 for c in color_rgb)
        if "target" in self.obj_dict:
            self.change_color("target", (*color, opacity))

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

        # --------------------------------------------------
        # 3. scene_type에 따라 scene 생성
        # --------------------------------------------------
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

        elif self.current_scene_type == "box":
            scene = get_box_scene(
                obj_name=self.current_obj,
                tabletop_pose=tabletop_pose,
                height_offset=0.1,
                z_scene_theta=0.0,
            )

        elif self.current_scene_type == "shelf":
            scene = get_shelf_scene(
                obj_name=self.current_obj,
                tabletop_pose=tabletop_pose,
                obb_info=obb_info,
                z_rotation_deg=0.0,
                gap=0.03,
                z_scene_theta=0.0,
                t=None,
                up=True,
                side=True,
                back=True,
            )

        else:
            raise ValueError(f"Unknown scene type: {self.current_scene_type}")

        # --------------------------------------------------
        # 4. mesh 렌더링
        # --------------------------------------------------
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
        """Load and visualize current grasp"""
        for robot_name in list(self.robot_dict.keys()):
            del self.robot_dict[robot_name]

        grasp_root_path = os.path.join(
            candidate_path,
            self.cur_version,
            self.current_obj,
            self.current_scene_type,
            self.current_scene_idx
        )
        
        traj_dict = {}
        grasp_list = ["0", "1", "2", "7", "10", "15"]
        for grasp_name in grasp_list:
            grasp_path = os.path.join(grasp_root_path, grasp_name)
            
            wrist_se3 = np.load(os.path.join(grasp_path, "wrist_se3.npy"))
            grasp_pose = np.load(os.path.join(grasp_path, "grasp_pose.npy")).reshape(1, -1)
            
            robot_T = self.obj_pose @ wrist_se3
            
            allegro_path = os.path.join(urdf_path, "allegro_hand_description_right.urdf")
            self.add_robot(f"robot_{grasp_name}", allegro_path, pose=robot_T)
            # self.robot_dict[f"robot_{grasp_name}"].set_visibility(False)

            traj_dict[f"robot_{grasp_name}"] = grasp_pose
        self.add_traj("asdf", traj_dict)
            

if __name__ == "__main__":
    version = "visualization"
    obj_name = "attached_container"
    scene_type = "wall"
    scene_idx = "44"
    vis = Renderer(version, obj_name, scene_type, scene_idx)
    vis.start_viewer()