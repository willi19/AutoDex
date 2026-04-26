"""Viser viewer for robothome FR3 + Inspire(left) planning.

GUI:
  - Mode toggle: Joint sliders | EEF xyz/rpy (IK via pinocchio)
  - Hand: 6 actuated sliders (mimic resolved by yourdfpy)
  - Scene mesh (table/collision) loaded as fixed background
  - Object: dropdown of robothome objects → Add → per-object pose sliders + Remove
"""
import argparse
import json
import re
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import trimesh
import viser
import yourdfpy
from scipy.spatial.transform import Rotation as R

from curobo.geom.types import Mesh as CuroboMesh
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose as CuroboPose
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig


HERE = Path(__file__).resolve().parent
URDF_PATH = HERE / "fr3_inspire_left.urdf"
SCENE_MESH_PATH = HERE / "scene_mesh.obj"
SCENE_VOXEL_PATH = Path("/home/mingi/Downloads/scene_voxel.npz")
HANDEYE_PATH = HERE / "hand_eye_result.pkl"
# Robot region (in world frame) to mask out from the scene SDF — same region
# we cut from the mesh to remove the robot's own embedding in the scan.
ROBOT_MASK_X_MAX = 0.2
ROBOT_MASK_Z_MIN = 0.0
OBJECT_DIRS = [
    Path("/home/mingi/shared_data/AutoDex/object/robothome"),
    Path("/home/mingi/shared_data/AutoDex/object/paradex"),
]
ROBOT_CFG_NAME = "fr3_inspire_left.yml"
SCENE_MESH_KEY = "scene_collision"


def _camel(name: str) -> str:
    parts = name.strip().split()
    if not parts:
        return name
    return parts[0] + "".join(p[:1].upper() + p[1:] for p in parts[1:])


def parse_tracking_key(key: str):
    """`'table:paper cup_1'` -> ('paper cup', 1). idx defaults to 1."""
    after = key.split(":", 1)[-1]
    m = re.match(r"^(.*?)_(\d+)$", after)
    if m:
        return m.group(1).strip(), int(m.group(2))
    return after.strip(), 1


def find_mesh_dir(base_name: str):
    """Resolve `'paper cup'` -> Path to mesh dir under OBJECT_DIRS.

    Tries: camelCase form, raw, first-word, case-insensitive substring.
    """
    candidates = [_camel(base_name), base_name.replace(" ", "_"),
                  base_name.replace(" ", ""), base_name.split()[0] if base_name else ""]
    for root in OBJECT_DIRS:
        if not root.exists():
            continue
        # Exact-name try
        for cand in candidates:
            if not cand:
                continue
            p = root / cand
            if p.is_dir():
                return p
        # Case-insensitive contains (first word)
        first = base_name.split()[0].lower() if base_name else ""
        if first:
            for sub in root.iterdir():
                if sub.is_dir() and first in sub.name.lower():
                    return sub
    return None

EEF_LINK = "hand_tcp"
ARM_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
HAND_ACTUATED = [
    "left_thumb_1_joint", "left_thumb_2_joint",
    "left_index_1_joint", "left_middle_1_joint",
    "left_ring_1_joint", "left_little_1_joint",
]
ARM_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])


def mat_to_pos_wxyz(T):
    pos = tuple(float(v) for v in T[:3, 3])
    qxyzw = R.from_matrix(T[:3, :3]).as_quat()
    return pos, (float(qxyzw[3]), float(qxyzw[0]), float(qxyzw[1]), float(qxyzw[2]))


def euler_xyz_to_mat(rx, ry, rz):
    return R.from_euler("xyz", [rx, ry, rz]).as_matrix()


# ----- cuRobo collision wrapper -----
class CuroboChecker:
    """Owns RobotWorld, allows updating world meshes (scene + dynamic objects)
    and querying per-sphere collision distance for a given joint config.
    Joint order matches the cspace.joint_names of the loaded yml.
    """

    def __init__(self, robot_cfg_name: str, scene_mesh_path: Path | None,
                 scene_voxel_path: Path | None = None,
                 collision_activation_distance: float = 0.0):
        self.tensor_args = TensorDeviceType()
        self.cfg_dict = load_yaml(join_path(get_robot_configs_path(), robot_cfg_name))
        self._collision_activation_distance = collision_activation_distance
        self._world_meshes = {}  # name -> CuroboMesh
        # Per-object trimesh proxies for objects added at runtime (proximity
        # query is fine for small clean meshes); the static scene uses the SDF
        # grid below instead.
        self._world_proxy = {}
        self._lock = threading.Lock()

        # Static scene SDF (preferred when available — direct, no mesh needed).
        # Robot region is masked out so the robot's own scan doesn't collide.
        self._scene_sdf = None  # dict with 'sdf','origin','vs','shape'
        if scene_voxel_path is not None and scene_voxel_path.exists():
            self._load_scene_sdf(scene_voxel_path)

        if scene_mesh_path is not None and scene_mesh_path.exists():
            self._world_meshes[SCENE_MESH_KEY] = self._make_curobo_mesh(
                SCENE_MESH_KEY, scene_mesh_path, np.eye(4),
            )
            self._set_proxy_mesh(SCENE_MESH_KEY, scene_mesh_path, np.eye(4))

        self._build_robot_world()

        self.curobo_joint_names = self.rw.kinematics.joint_names
        self.total_spheres = int(self.rw.kinematics.kinematics_config.total_spheres)
        self.link_sphere_idx_map = (
            self.rw.kinematics.kinematics_config.link_sphere_idx_map.detach().cpu().numpy().astype(int)
        )
        # Self-collision ignore set (link-level) from yml. Build name<->idx
        # using kinematics_config.link_name_to_idx_map (whose values are what
        # link_sphere_idx_map indexes into — NOT rw.kinematics.link_names).
        kc = self.rw.kinematics.kinematics_config
        self._link_name_to_idx = dict(kc.link_name_to_idx_map)
        self._link_idx_to_name = {v: k for k, v in self._link_name_to_idx.items()}
        ignore = self.cfg_dict["robot_cfg"]["kinematics"].get("self_collision_ignore", {})
        self._self_ignore_pairs = set()
        for a, bs in ignore.items():
            for b in bs:
                self._self_ignore_pairs.add((a, b))
                self._self_ignore_pairs.add((b, a))

    def _make_curobo_mesh(self, name: str, mesh_path: Path, T: np.ndarray) -> CuroboMesh:
        rot = T[:3, :3]
        qxyzw = R.from_matrix(rot).as_quat()
        # CuroboMesh pose: [x,y,z,qw,qx,qy,qz]
        return CuroboMesh(
            name=name,
            file_path=str(mesh_path),
            pose=[float(T[0, 3]), float(T[1, 3]), float(T[2, 3]),
                  float(qxyzw[3]), float(qxyzw[0]), float(qxyzw[1]), float(qxyzw[2])],
        )

    def _build_robot_world(self):
        wc = WorldConfig(mesh=list(self._world_meshes.values()))
        rw_cfg = RobotWorldConfig.load_from_config(
            self.cfg_dict, wc, tensor_args=self.tensor_args,
            collision_activation_distance=self._collision_activation_distance,
        )
        self.rw = RobotWorld(rw_cfg)
        # IK solver shares the same robot/world config and is rebuilt whenever
        # we update the world (so it sees the latest obstacle set).
        # use_cuda_graph=False: viser slider callbacks fire from a worker thread
        # while collision queries happen on the main thread; sharing a captured
        # CUDA graph across threads breaks stream capture (cudaErrorStreamCaptureInvalidated).
        ik_cfg = IKSolverConfig.load_from_robot_config(
            self.cfg_dict, wc, tensor_args=self.tensor_args,
            num_seeds=32,
            collision_activation_distance=0.005,
            use_cuda_graph=False,
        )
        self.ik_solver = IKSolver(ik_cfg)
        self.ee_link = self.cfg_dict["robot_cfg"]["kinematics"]["ee_link"]

    def _set_proxy_mesh(self, name: str, mesh_path: Path, T: np.ndarray):
        m = trimesh.load(str(mesh_path), force="mesh", process=False)
        m.apply_transform(T)
        from trimesh.proximity import ProximityQuery
        self._world_proxy[name] = (m, ProximityQuery(m))

    def _load_scene_sdf(self, npz_path: Path):
        """Load voxel SDF, mask out the robot region, store for fast queries."""
        d = np.load(str(npz_path))
        sdf = d["feature_tensor"].astype(np.float32).copy()
        origin = np.asarray(d["origin"], dtype=np.float32)
        vs = float(d["voxel_size"])
        # Robot region: x < 0.2 AND z > 0 → set to free (large positive SDF).
        ix_hi = max(0, int(np.ceil((ROBOT_MASK_X_MAX - origin[0]) / vs)))
        iz_lo = max(0, int(np.ceil((ROBOT_MASK_Z_MIN - origin[2]) / vs)))
        sdf[:ix_hi, :, iz_lo:] = 1.0
        self._scene_sdf = {
            "sdf": sdf,         # (Nx, Ny, Nz) — negative = inside obstacle
            "origin": origin,
            "vs": vs,
            "shape": np.asarray(sdf.shape),
        }
        print(f"[scene_sdf] loaded: shape={sdf.shape}, vs={vs}, "
              f"x_range=[{origin[0]:.2f},{origin[0] + sdf.shape[0] * vs:.2f}]")

    def _scene_sdf_lookup(self, points: np.ndarray) -> np.ndarray:
        """Trilinear-interpolate SDF at given world-frame points (N, 3).

        Returns SDF (N,) — points outside the grid get +inf (free).
        """
        info = self._scene_sdf
        if info is None:
            return np.full(len(points), np.inf, dtype=np.float32)
        sdf = info["sdf"]
        origin = info["origin"]
        vs = info["vs"]
        nx, ny, nz = sdf.shape

        # Continuous grid coordinates
        gc = (points - origin) / vs
        i0 = np.floor(gc).astype(np.int32)
        f = gc - i0  # fractional part

        out = np.full(len(points), np.inf, dtype=np.float32)
        in_grid = ((i0[:, 0] >= 0) & (i0[:, 0] < nx - 1) &
                   (i0[:, 1] >= 0) & (i0[:, 1] < ny - 1) &
                   (i0[:, 2] >= 0) & (i0[:, 2] < nz - 1))
        if not in_grid.any():
            return out

        ig = i0[in_grid]
        fg = f[in_grid]
        # 8-corner trilinear
        c000 = sdf[ig[:, 0],     ig[:, 1],     ig[:, 2]]
        c001 = sdf[ig[:, 0],     ig[:, 1],     ig[:, 2] + 1]
        c010 = sdf[ig[:, 0],     ig[:, 1] + 1, ig[:, 2]]
        c011 = sdf[ig[:, 0],     ig[:, 1] + 1, ig[:, 2] + 1]
        c100 = sdf[ig[:, 0] + 1, ig[:, 1],     ig[:, 2]]
        c101 = sdf[ig[:, 0] + 1, ig[:, 1],     ig[:, 2] + 1]
        c110 = sdf[ig[:, 0] + 1, ig[:, 1] + 1, ig[:, 2]]
        c111 = sdf[ig[:, 0] + 1, ig[:, 1] + 1, ig[:, 2] + 1]
        fx, fy, fz = fg[:, 0], fg[:, 1], fg[:, 2]
        c00 = c000 * (1 - fx) + c100 * fx
        c01 = c001 * (1 - fx) + c101 * fx
        c10 = c010 * (1 - fx) + c110 * fx
        c11 = c011 * (1 - fx) + c111 * fx
        c0 = c00 * (1 - fy) + c10 * fy
        c1 = c01 * (1 - fy) + c11 * fy
        out[in_grid] = c0 * (1 - fz) + c1 * fz
        return out

    def add_object_mesh(self, name: str, mesh_path: Path, T: np.ndarray):
        self._world_meshes[name] = self._make_curobo_mesh(name, mesh_path, T)
        self._set_proxy_mesh(name, mesh_path, T)
        self._rebuild_world()

    def update_object_pose(self, name: str, T: np.ndarray):
        if name not in self._world_meshes:
            return
        m = self._world_meshes[name]
        path = Path(m.file_path)
        self._world_meshes[name] = self._make_curobo_mesh(name, path, T)
        self._set_proxy_mesh(name, path, T)
        self._rebuild_world()

    def remove_object_mesh(self, name: str):
        if name in self._world_meshes:
            del self._world_meshes[name]
        if name in self._world_proxy:
            del self._world_proxy[name]
        self._rebuild_world()

    def world_collide(self, x_sph: np.ndarray) -> np.ndarray:
        """Per-sphere world collision (N,) using *unsigned* distance to mesh
        surfaces. A sphere collides if its center is within `radius` of any
        env mesh surface.

        Why unsigned: the static scene mesh (extracted from a truncated SDF)
        is not watertight, so trimesh `signed_distance` gives wrong signs and
        was producing false-positive 'inside' hits far from any surface.
        Distance-to-surface is reliable regardless of watertightness.
        """
        n = x_sph.shape[0]
        coll = np.zeros(n, dtype=bool)
        if not self._world_proxy:
            return coll
        centers = x_sph[:, :3]
        radii = x_sph[:, 3]
        valid = radii > 0
        if not valid.any():
            return coll

        for _, pq in self._world_proxy.values():
            # closest_point returns (closest_pts, distances, triangle_ids).
            _, dist, _ = pq.on_surface(centers[valid])
            r_v = radii[valid]
            hit = dist < r_v
            coll[np.where(valid)[0][hit]] = True
        return coll

    def self_collide(self, x_sph: np.ndarray) -> np.ndarray:
        """Per-sphere self-collision flags using pairwise sphere overlap +
        link-level self_collision_ignore filtering.
        """
        n = x_sph.shape[0]
        flags = np.zeros(n, dtype=bool)
        radii = x_sph[:, 3]
        valid_mask = radii > 0
        valid = np.where(valid_mask)[0]
        if valid.size == 0:
            return flags
        c = x_sph[valid, :3]
        r = x_sph[valid, 3]
        d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
        thresh = (r[:, None] + r[None, :])
        overlap = d < thresh  # includes diagonal — mask it out below
        np.fill_diagonal(overlap, False)
        # Filter link pairs in ignore set.
        link_of = self.link_sphere_idx_map[valid]
        for ii in range(valid.size):
            row = overlap[ii]
            if not row.any():
                continue
            la = self._link_idx_to_name.get(int(link_of[ii]))
            for jj in np.where(row)[0]:
                lb = self._link_idx_to_name.get(int(link_of[jj]))
                if la == lb:
                    continue
                if (la, lb) in self._self_ignore_pairs:
                    continue
                flags[valid[ii]] = True
                break
        return flags

    def _rebuild_world(self):
        wc = WorldConfig(mesh=list(self._world_meshes.values()))
        with self._lock:
            self.rw.update_world(wc)
            self.ik_solver.update_world(wc)

    def solve_ik(self, target_T: np.ndarray, retract_q: np.ndarray | None = None):
        """Run cuRobo IK to a 4x4 SE(3) target pose for the EE link.

        Returns (success: bool, q: np.ndarray of curobo joint order).
        On failure q is whatever cuRobo returned (often best-effort).
        """
        rot = target_T[:3, :3]
        qxyzw = R.from_matrix(rot).as_quat()
        # cuRobo Pose: position xyz, quaternion wxyz
        pos_t = torch.tensor([target_T[0, 3], target_T[1, 3], target_T[2, 3]],
                             device=self.tensor_args.device, dtype=self.tensor_args.dtype).view(1, 3)
        quat_t = torch.tensor([qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]],
                              device=self.tensor_args.device, dtype=self.tensor_args.dtype).view(1, 4)
        goal = CuroboPose(position=pos_t, quaternion=quat_t)

        retract_t = None
        if retract_q is not None:
            retract_t = torch.tensor(retract_q.reshape(1, -1),
                                     device=self.tensor_args.device,
                                     dtype=self.tensor_args.dtype)

        with self._lock:
            result = self.ik_solver.solve_single(goal, retract_config=retract_t)
        succ = bool(result.success.view(-1)[0].item())
        q = result.solution.view(-1).detach().cpu().numpy()
        return succ, q

    def fk_ee(self, q: np.ndarray) -> np.ndarray:
        """Return 4x4 SE(3) of EE link for the given (curobo-ordered) joint config."""
        q_t = torch.tensor(q.reshape(1, -1), dtype=self.tensor_args.dtype,
                           device=self.tensor_args.device)
        with self._lock:
            state = self.rw.get_kinematics(q_t)
            pos = state.ee_pose.position.view(-1).detach().cpu().numpy()
            wxyz = state.ee_pose.quaternion.view(-1).detach().cpu().numpy()
        T = np.eye(4)
        T[:3, 3] = pos
        T[:3, :3] = R.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]]).as_matrix()
        return T

    def query(self, q: np.ndarray):
        """Return sphere world positions+radii (N,4) for the given joint config."""
        q_t = torch.tensor(q.reshape(1, -1), dtype=self.tensor_args.dtype,
                           device=self.tensor_args.device)
        with self._lock:
            state = self.rw.get_kinematics(q_t)
            x_sph = state.link_spheres_tensor.view(-1, 4).detach().cpu().numpy()
        return x_sph


# ----- URDF -> viser -----
class ViserURDF:
    def __init__(self, server: viser.ViserServer, urdf_path: Path, root: str = "/robot"):
        self.urdf = yourdfpy.URDF.load(
            str(urdf_path), mesh_dir=str(urdf_path.parent),
            build_collision_scene_graph=False, load_collision_meshes=False,
        )
        self.server = server
        self.root = root
        self._joint_frames = {}
        # link_name -> {mesh_local: trimesh.Trimesh in link frame,
        #               base_color: (r,g,b), handle: viser mesh handle}
        self.links = {}
        server.scene.add_frame(root, show_axes=False)

        scene = self.urdf.scene
        for joint in self.urdf.joint_map.values():
            self._joint_frames[joint.child] = server.scene.add_frame(
                self._frame_path(joint.child), show_axes=False,
            )

        for link_name, mesh in scene.geometry.items():
            assert isinstance(mesh, trimesh.Trimesh)
            # Visual: viser mesh is placed under the link's parent frame, with
            # the link's local transform pre-baked into the vertices.
            T_link = self.urdf.get_transform(link_name, scene.graph.transforms.parents[link_name])
            m_visual = mesh.copy()
            m_visual.apply_transform(T_link)
            color = np.array([200, 200, 200], dtype=np.uint8)
            vc = getattr(m_visual.visual, "vertex_colors", None)
            if vc is not None:
                vc_arr = np.asarray(vc)
                if vc_arr.ndim == 2 and vc_arr.shape[0] > 0:
                    color = vc_arr[:, :3].mean(axis=0).astype(np.uint8)
            handle = server.scene.add_mesh_simple(
                name=self._frame_path(link_name) + f"/{link_name}_mesh",
                vertices=m_visual.vertices, faces=m_visual.faces,
                color=tuple(int(c) for c in color),
            )
            # Collision mesh in scene-link frame (i.e. yourdfpy's per-link geometry).
            # Pair with self.get_link_world_transform(link_name) to place in world.
            self.links[link_name] = {
                "mesh_collision": mesh.copy(),
                "base_color": tuple(int(c) for c in color),
                "handle": handle,
            }

    def _frame_path(self, link_name: str) -> str:
        scene = self.urdf.scene
        parts = []
        cur = link_name
        while cur != scene.graph.base_frame:
            parts.append(cur)
            cur = scene.graph.transforms.parents[cur]
        return self.root + "/" + "/".join(reversed(parts))

    def update_cfg(self, q_actuated: np.ndarray):
        self.urdf.update_cfg(q_actuated)
        for joint in self.urdf.joint_map.values():
            T = self.urdf.get_transform(joint.child, joint.parent)
            pos, wxyz = mat_to_pos_wxyz(T)
            handle = self._joint_frames[joint.child]
            handle.position = pos
            handle.wxyz = wxyz

    def get_link_world_transform(self, link_name: str) -> np.ndarray:
        return self.urdf.get_transform(link_name, self.urdf.scene.graph.base_frame)

    def set_link_color(self, link_name: str, color):
        h = self.links[link_name]["handle"]
        h.color = tuple(int(c) for c in color)


# ----- App -----
class App:
    def __init__(self, port: int):
        self.server = viser.ViserServer(port=port)
        self.server.gui.configure_theme(dark_mode=True)
        self.server.scene.set_up_direction((0, 0, 1))
        self.server.scene.add_grid("/floor", width=2.0, height=2.0, plane="xy",
                                   position=(0, 0, 0), cell_size=0.1)

        self.viser_urdf = ViserURDF(self.server, URDF_PATH)
        self.actuated = list(self.viser_urdf.urdf.actuated_joint_names)
        self.q_actuated = np.zeros(len(self.actuated))
        for jn, val in zip(ARM_JOINTS, ARM_HOME):
            self.q_actuated[self.actuated.index(jn)] = val

        self.joint_limits = {}
        for jn in ARM_JOINTS + HAND_ACTUATED:
            j = self.viser_urdf.urdf.joint_map[jn]
            lim = j.limit
            lo = lim.lower if lim and lim.lower is not None else -3.14
            hi = lim.upper if lim and lim.upper is not None else 3.14
            self.joint_limits[jn] = (float(lo), float(hi))

        self._add_scene_mesh()
        self._add_handeye_frame()

        # cuRobo collision checker (loads scene mesh as a static world mesh).
        # CRITICAL: cuRobo allocates CUDA state on whichever thread first calls
        # it. Viser slider callbacks fire on a worker thread, so calling cuRobo
        # from there segfaults. We post all robot updates to a queue and let
        # the main thread (which constructed everything) drain it.
        print("[curobo] initializing RobotWorld...")
        self.curobo = CuroboChecker(ROBOT_CFG_NAME, SCENE_MESH_PATH)
        self._dirty = True  # set by callbacks; consumed by main loop
        self._dirty_lock = threading.Lock()
        self._pending_ik_target = None  # 4x4 SE(3) when EEF mode requests IK
        self._world_jobs: list = []  # (action, *args) for cuRobo world updates
        self._ik_failed = False  # last IK attempt failed → tint all spheres red
        # actuated -> curobo joint index map (curobo order may differ from URDF actuated order)
        self.act_to_curobo_idx = [
            self.curobo.curobo_joint_names.index(jn) for jn in self.actuated
        ]
        # Read sphere radii once (constant per robot). Use them to size viser
        # icospheres at creation; positions/colors are updated each frame.
        x_sph_init = self.curobo.query(self._q_curobo())
        self.sphere_radii = x_sph_init[:, 3].copy()
        self._sphere_default_color = (90, 170, 255)
        self._sphere_collision_color = (230, 40, 40)
        self.sphere_handles = []
        for i in range(self.curobo.total_spheres):
            r = float(self.sphere_radii[i])
            if r <= 0.0:
                self.sphere_handles.append(None)
                continue
            h = self.server.scene.add_icosphere(
                f"/curobo_spheres/{i}", radius=r,
                color=self._sphere_default_color,
                position=(float(x_sph_init[i, 0]), float(x_sph_init[i, 1]), float(x_sph_init[i, 2])),
            )
            self.sphere_handles.append(h)
        print(f"[curobo] ready: {self.curobo.total_spheres} spheres "
              f"({sum(1 for h in self.sphere_handles if h is not None)} valid), "
              f"dof={len(self.curobo.curobo_joint_names)}")

        # Object catalog: union of robothome + paradex object dirs.
        self.objects = {}  # instance -> dict(T, frame, mesh_handle, sliders, folder, mesh_path)
        names = []
        for root in OBJECT_DIRS:
            if not root.exists():
                continue
            for p in sorted(root.iterdir()):
                if (p / "raw_mesh").is_dir():
                    names.append(p.name)
        # de-dup, keep order
        seen = set()
        self.object_names = []
        for n in names:
            if n not in seen:
                seen.add(n)
                self.object_names.append(n)

        # Tracking JSON files in this dir
        self.tracking_files = sorted([p.name for p in HERE.glob("tracking_*.json")])

        self._build_gui()
        self._refresh_robot()

    # actuated <-> curobo joint order
    def _q_curobo(self) -> np.ndarray:
        q = np.zeros(len(self.curobo.curobo_joint_names))
        for i, _jn in enumerate(self.actuated):
            q[self.act_to_curobo_idx[i]] = float(self.q_actuated[i])
        return q

    def _apply_curobo_q_to_actuated(self, q_curobo: np.ndarray):
        for i, _jn in enumerate(self.actuated):
            self.q_actuated[i] = float(q_curobo[self.act_to_curobo_idx[i]])

    def _ee_fk(self) -> np.ndarray:
        return self.curobo.fk_ee(self._q_curobo())

    # scene
    def _add_scene_mesh(self):
        if not SCENE_MESH_PATH.exists():
            return
        m = trimesh.load(str(SCENE_MESH_PATH), force="mesh", process=False)
        color = np.array([150, 150, 150], dtype=np.uint8)
        vc = getattr(m.visual, "vertex_colors", None)
        if vc is not None:
            vc_arr = np.asarray(vc)
            if vc_arr.ndim == 2 and vc_arr.shape[0] > 0:
                color = vc_arr[:, :3].mean(axis=0).astype(np.uint8)
        self.server.scene.add_mesh_simple(
            "/scene/collision_mesh", vertices=m.vertices, faces=m.faces,
            color=tuple(int(c) for c in color),
        )

    def _add_handeye_frame(self):
        if not HANDEYE_PATH.exists():
            return
        try:
            data = joblib.load(HANDEYE_PATH)
        except Exception:
            return
        if "Z" not in data:
            return
        Z = np.asarray(data["Z"])
        pos, wxyz = mat_to_pos_wxyz(Z)
        self.server.scene.add_frame(
            "/handeye/Z_camera", position=pos, wxyz=wxyz,
            show_axes=True, axes_length=0.1, axes_radius=0.005,
        )

    # GUI
    def _build_gui(self):
        srv = self.server.gui

        with srv.add_folder("Mode"):
            self.mode = srv.add_dropdown("Control", ("Joint", "EEF"), initial_value="Joint")

        with srv.add_folder("Arm joints"):
            self.arm_sliders = {}
            for jn, init in zip(ARM_JOINTS, ARM_HOME):
                lo, hi = self.joint_limits[jn]
                s = srv.add_slider(jn, min=lo, max=hi, step=0.005, initial_value=float(init))
                s.on_update(self._make_joint_cb(jn, s))
                self.arm_sliders[jn] = s

        with srv.add_folder("Hand joints"):
            self.hand_sliders = {}
            for jn in HAND_ACTUATED:
                lo, hi = self.joint_limits[jn]
                s = srv.add_slider(jn, min=lo, max=hi, step=0.005, initial_value=0.0)
                s.on_update(self._make_joint_cb(jn, s))
                self.hand_sliders[jn] = s

        with srv.add_folder("EEF target"):
            T0 = self._ee_fk()
            x, y, z = T0[:3, 3]
            rx, ry, rz = R.from_matrix(T0[:3, :3]).as_euler("xyz")
            self.eef_x = srv.add_slider("x", min=-1.0, max=1.5, step=0.005, initial_value=float(x))
            self.eef_y = srv.add_slider("y", min=-1.0, max=1.0, step=0.005, initial_value=float(y))
            self.eef_z = srv.add_slider("z", min=-0.2, max=1.5, step=0.005, initial_value=float(z))
            self.eef_rx = srv.add_slider("roll",  min=-3.14, max=3.14, step=0.01, initial_value=float(rx))
            self.eef_ry = srv.add_slider("pitch", min=-3.14, max=3.14, step=0.01, initial_value=float(ry))
            self.eef_rz = srv.add_slider("yaw",   min=-3.14, max=3.14, step=0.01, initial_value=float(rz))
            for s in (self.eef_x, self.eef_y, self.eef_z, self.eef_rx, self.eef_ry, self.eef_rz):
                s.on_update(lambda _: self._on_eef())
            srv.add_button("Sync EEF ← FK").on_click(lambda _: self._sync_eef_sliders())
            self.ik_status = srv.add_text("IK status", "—", disabled=True)

        with srv.add_folder("Objects"):
            self.obj_dropdown = srv.add_dropdown(
                "mesh", tuple(self.object_names) if self.object_names else ("(none)",),
                initial_value=self.object_names[0] if self.object_names else "(none)",
            )
            srv.add_button("Add object").on_click(lambda _: self._add_object(self.obj_dropdown.value))
            self.objects_folder_parent = srv.add_folder("Active objects", expand_by_default=True)

        with srv.add_folder("Tracking JSON"):
            tracking_choices = tuple(self.tracking_files) if self.tracking_files else ("(none)",)
            self.tracking_dropdown = srv.add_dropdown(
                "file", tracking_choices,
                initial_value=tracking_choices[0],
            )
            srv.add_button("Load (add objects)").on_click(
                lambda _: self._on_load_tracking()
            )
            srv.add_button("Clear all objects").on_click(lambda _: self._clear_objects())

        with srv.add_folder("Collision (cuRobo)"):
            self.show_spheres = srv.add_checkbox("Show spheres", True)
            self.show_spheres.on_update(lambda _: self._update_sphere_visibility())
            self.coll_status = srv.add_text("Status", "—", disabled=True)

        with srv.add_folder("Utilities"):
            srv.add_button("Arm → home").on_click(lambda _: self._reset_arm_home())
            srv.add_button("Hand → zero").on_click(lambda _: self._reset_hand_zero())

    def _update_sphere_visibility(self):
        v = bool(self.show_spheres.value)
        for h in self.sphere_handles:
            h.visible = v

    def _on_load_tracking(self):
        v = self.tracking_dropdown.value
        if v == "(none)":
            return
        self._load_tracking_json(v)

    def _clear_objects(self):
        for instance in list(self.objects.keys()):
            self._make_object_remove_cb(instance)(None)

    def _make_joint_cb(self, joint_name, slider):
        def cb(_):
            self._on_joint(joint_name, slider.value)
        return cb

    # callbacks (run on viser worker thread — must NOT call cuRobo directly)
    def _on_joint(self, jn, val):
        if self.mode.value == "EEF" and jn in ARM_JOINTS:
            return
        self.q_actuated[self.actuated.index(jn)] = float(val)
        with self._dirty_lock:
            self._dirty = True

    def _on_eef(self):
        if self.mode.value != "EEF":
            return
        T = np.eye(4)
        T[:3, 3] = [self.eef_x.value, self.eef_y.value, self.eef_z.value]
        T[:3, :3] = euler_xyz_to_mat(self.eef_rx.value, self.eef_ry.value, self.eef_rz.value)
        with self._dirty_lock:
            self._pending_ik_target = T
            self._dirty = True

    def _sync_eef_sliders(self):
        # Called on main thread after FK. cuRobo FK is invoked here.
        T = self._ee_fk()
        self.eef_x.value = float(T[0, 3])
        self.eef_y.value = float(T[1, 3])
        self.eef_z.value = float(T[2, 3])
        rx, ry, rz = R.from_matrix(T[:3, :3]).as_euler("xyz")
        self.eef_rx.value = float(rx)
        self.eef_ry.value = float(ry)
        self.eef_rz.value = float(rz)

    def _reset_arm_home(self):
        for jn, val in zip(ARM_JOINTS, ARM_HOME):
            self.q_actuated[self.actuated.index(jn)] = float(val)
            self.arm_sliders[jn].value = float(val)
        with self._dirty_lock:
            self._dirty = True

    def _reset_hand_zero(self):
        for jn in HAND_ACTUATED:
            self.q_actuated[self.actuated.index(jn)] = 0.0
            self.hand_sliders[jn].value = 0.0
        with self._dirty_lock:
            self._dirty = True

    # objects
    def _add_object(self, obj_name: str):
        if obj_name == "(none)":
            return
        mesh_file = self._resolve_mesh_file(obj_name)
        if mesh_file is None:
            print(f"[skip] no mesh dir for '{obj_name}'")
            return

        # Generate unique instance id
        instance = obj_name
        idx = 1
        while instance in self.objects:
            idx += 1
            instance = f"{obj_name}_{idx}"

        m = trimesh.load(str(mesh_file), force="mesh", process=False)
        T = np.eye(4)
        T[:3, 3] = [0.5, 0.0, 0.05]

        srv = self.server.gui
        with self.objects_folder_parent:
            with srv.add_folder(instance, expand_by_default=False) as folder:
                sx = srv.add_slider("x", min=-1.0, max=1.5, step=0.005, initial_value=float(T[0, 3]))
                sy = srv.add_slider("y", min=-1.0, max=1.0, step=0.005, initial_value=float(T[1, 3]))
                sz = srv.add_slider("z", min=-0.2, max=1.5, step=0.005, initial_value=float(T[2, 3]))
                srx = srv.add_slider("roll",  min=-3.14, max=3.14, step=0.01, initial_value=0.0)
                sry = srv.add_slider("pitch", min=-3.14, max=3.14, step=0.01, initial_value=0.0)
                srz = srv.add_slider("yaw",   min=-3.14, max=3.14, step=0.01, initial_value=0.0)
                btn_remove = srv.add_button("Remove")

        frame = self.server.scene.add_frame(
            f"/objects/{instance}/frame", position=tuple(T[:3, 3]),
            show_axes=True, axes_length=0.05, axes_radius=0.002,
        )
        color = np.array([200, 180, 120], dtype=np.uint8)
        vc = getattr(m.visual, "vertex_colors", None)
        if vc is not None:
            vc_arr = np.asarray(vc)
            if vc_arr.ndim == 2 and vc_arr.shape[0] > 0:
                color = vc_arr[:, :3].mean(axis=0).astype(np.uint8)
        mesh_handle = self.server.scene.add_mesh_simple(
            f"/objects/{instance}/frame/mesh",
            vertices=m.vertices, faces=m.faces,
            color=tuple(int(c) for c in color),
        )

        entry = {
            "T": T, "frame": frame, "mesh_handle": mesh_handle,
            "sliders": (sx, sy, sz, srx, sry, srz),
            "folder": folder, "btn_remove": btn_remove,
            "mesh_path": mesh_file,
        }
        self.objects[instance] = entry
        # Defer cuRobo world update + collision query to main thread.
        with self._dirty_lock:
            self._world_jobs.append(("add", instance, mesh_file, T))
            self._dirty = True

        for s in entry["sliders"]:
            s.on_update(self._make_object_cb(instance))
        btn_remove.on_click(self._make_object_remove_cb(instance))
        return instance

    def _make_object_cb(self, instance):
        def cb(_):
            entry = self.objects.get(instance)
            if entry is None:
                return
            sx, sy, sz, srx, sry, srz = entry["sliders"]
            T = np.eye(4)
            T[:3, 3] = [sx.value, sy.value, sz.value]
            T[:3, :3] = euler_xyz_to_mat(srx.value, sry.value, srz.value)
            entry["T"] = T
            pos, wxyz = mat_to_pos_wxyz(T)
            entry["frame"].position = pos
            entry["frame"].wxyz = wxyz
            with self._dirty_lock:
                self._world_jobs.append(("update", instance, T))
                self._dirty = True
        return cb

    def _make_object_remove_cb(self, instance):
        def cb(_):
            entry = self.objects.pop(instance, None)
            if entry is None:
                return
            entry["mesh_handle"].remove()
            entry["frame"].remove()
            for s in entry["sliders"]:
                s.remove()
            entry["btn_remove"].remove()
            entry["folder"].remove()
            with self._dirty_lock:
                self._world_jobs.append(("remove", instance))
                self._dirty = True
        return cb

    def _resolve_mesh_file(self, obj_name: str):
        """Look for raw_mesh/{obj_name}.obj under any OBJECT_DIRS root."""
        for root in OBJECT_DIRS:
            mesh_dir = root / obj_name / "raw_mesh"
            if not mesh_dir.is_dir():
                continue
            preferred = mesh_dir / f"{obj_name}.obj"
            if preferred.exists():
                return preferred
            cands = list(mesh_dir.glob("*.obj"))
            if cands:
                return cands[0]
        return None

    # tracking JSON
    def _load_tracking_json(self, filename: str):
        path = HERE / filename
        if not path.exists():
            print(f"[tracking] file not found: {path}")
            return
        data = json.loads(path.read_text())
        poses = data.get("object", {}).get("poses", {})
        added = 0
        for key, mat in poses.items():
            base, _ = parse_tracking_key(key)
            mesh_dir = find_mesh_dir(base)
            if mesh_dir is None:
                print(f"[tracking] no mesh for '{key}' (base='{base}')")
                continue
            instance = self._add_object(mesh_dir.name)
            if instance is None:
                continue
            T = np.asarray(mat, dtype=float).reshape(4, 4)
            self._set_object_pose(instance, T)
            added += 1
        print(f"[tracking] loaded {added}/{len(poses)} objects from {filename}")

    def _set_object_pose(self, instance: str, T: np.ndarray):
        entry = self.objects.get(instance)
        if entry is None:
            return
        entry["T"] = T
        sx, sy, sz, srx, sry, srz = entry["sliders"]
        sx.value = float(T[0, 3])
        sy.value = float(T[1, 3])
        sz.value = float(T[2, 3])
        rx, ry, rz = R.from_matrix(T[:3, :3]).as_euler("xyz")
        srx.value = float(rx)
        sry.value = float(ry)
        srz.value = float(rz)
        # Slider on_update will sync the frame, but ensure frame matches now too.
        pos, wxyz = mat_to_pos_wxyz(T)
        entry["frame"].position = pos
        entry["frame"].wxyz = wxyz

    # viz
    def _refresh_robot(self):
        self.viser_urdf.update_cfg(self.q_actuated)
        self._update_collision_spheres()

    def _update_collision_spheres(self):
        q_curobo = self._q_curobo()
        try:
            x_sph = self.curobo.query(q_curobo)
            world_flags = self.curobo.world_collide(x_sph)
            self_flags = self.curobo.self_collide(x_sph)
        except Exception as e:
            print(f"[curobo] query failed: {e}")
            return
        coll = world_flags | self_flags
        if self._ik_failed:
            # IK didn't find a solution; force-tint everything red so the user
            # immediately sees that the target is unreachable.
            coll = np.ones_like(coll, dtype=bool)
            self.coll_status.value = "IK FAIL"
        else:
            self.coll_status.value = (
                f"world={int(world_flags.sum())}  self={int(self_flags.sum())}"
            )

        show = bool(self.show_spheres.value)
        for i, h in enumerate(self.sphere_handles):
            if h is None:
                continue
            x, y, z, _ = x_sph[i]
            h.position = (float(x), float(y), float(z))
            h.color = self._sphere_collision_color if coll[i] else self._sphere_default_color
            h.visible = show

    # main-thread tick: drain dirty flag / world jobs / IK target
    def _tick(self):
        with self._dirty_lock:
            if not self._dirty:
                return
            jobs = self._world_jobs
            self._world_jobs = []
            ik_target = self._pending_ik_target
            self._pending_ik_target = None
            self._dirty = False

        # 1) world updates (each rebuild reloads RobotWorld + IKSolver)
        for job in jobs:
            try:
                if job[0] == "add":
                    _, name, mesh_file, T = job
                    self.curobo.add_object_mesh(f"obj::{name}", mesh_file, T)
                elif job[0] == "update":
                    _, name, T = job
                    self.curobo.update_object_pose(f"obj::{name}", T)
                elif job[0] == "remove":
                    _, name = job
                    self.curobo.remove_object_mesh(f"obj::{name}")
            except Exception as e:
                print(f"[world] job {job[0]} failed: {e}")

        # 2) IK if requested
        if ik_target is not None:
            retract = self._q_curobo()
            try:
                succ, q_sol = self.curobo.solve_ik(ik_target, retract_q=retract)
            except Exception as e:
                print(f"[ik] solve failed: {e}")
                self.ik_status.value = "ERROR"
                succ = False
                q_sol = None
            if not succ:
                self.ik_status.value = "IK FAIL — target unreachable"
                self._ik_failed = True
                # Don't move arm sliders; user sees previous valid pose.
            else:
                self.ik_status.value = "OK"
                self._ik_failed = False
                for jn in ARM_JOINTS:
                    ci = self.curobo.curobo_joint_names.index(jn)
                    self.q_actuated[self.actuated.index(jn)] = float(q_sol[ci])
                    self.arm_sliders[jn].value = float(q_sol[ci])
        else:
            # Non-IK update path (joint mode, world change) — clear stale IK
            # failure state so spheres aren't stuck red.
            self._ik_failed = False

        # 3) viser URDF + sphere/collision viz
        self.viser_urdf.update_cfg(self.q_actuated)
        try:
            self._update_collision_spheres()
        except Exception as e:
            print(f"[curobo] viz update failed: {e}")

        # 4) keep EEF sliders consistent in Joint mode (FK-derived)
        if self.mode.value == "Joint":
            try:
                self._sync_eef_sliders()
            except Exception:
                pass

    def run(self):
        while True:
            self._tick()
            time.sleep(0.03)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()
    App(args.port).run()


if __name__ == "__main__":
    main()
