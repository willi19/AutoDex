"""Viser viewer for robothome FR3 + Inspire(left) planning.

GUI:
  - Mode toggle: Joint sliders | EEF xyz/rpy (IK via pinocchio)
  - Hand: 6 actuated sliders (mimic resolved by yourdfpy)
  - Scene mesh (table/collision) loaded as fixed background
  - Object: dropdown of robothome objects → Add → per-object pose sliders + Remove
"""
import argparse
import json
import math
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
from tqdm import tqdm

from curobo.geom.sdf.world import CollisionQueryBuffer
from curobo.geom.types import Mesh as CuroboMesh
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose as CuroboPose
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.types.state import JointState
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig


HERE = Path(__file__).resolve().parent
URDF_PATH = HERE / "fr3_inspire_left.urdf"
SCENE_MESH_PATH = HERE / "scene_mesh.obj"
TRAJ_DIR = HERE / "traj"

# Fixed waypoint→waypoint trajectories to pre-compute and cache. These are
# scene-only (no manipulated object) and reusable across pick-and-trash runs.
# mode: "plan" → cuRobo plan_js; "sequential" → joint-by-joint interpolation.
FIXED_TRAJ_PAIRS = [
    ("home", "pick_start", "plan"),
    ("pick_start", "place_start", "sequential"),
    ("place_start", "drop_left", "sequential"),
    ("place_start", "drop_right", "sequential"),
]

# Objects that are roughly y-axis cylindrical-symmetric in their AutoDex mesh
# frame. For these we sweep grasp candidates 6× around object-Y so perception
# can't tell us the spin angle. Anything else (crushed cans, irregular
# shapes) gets a single orientation — Y-rotated copies would be physically
# wrong grasps.
Y_SYMMETRIC_OBJECTS = {"paperCup", "Jp_Water"}
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
HAND_NAME = "inspire_left"
CANDIDATE_VERSION = "selected_100"
CANDIDATE_ROOT = Path(f"/home/mingi/AutoDex/candidates/{HAND_NAME}/{CANDIDATE_VERSION}")
FLOATING_HAND_URDF = (Path("/home/mingi/AutoDex/autodex/planner/src/curobo/content/assets")
                      / "robot/inspire_description/inspire_left_floating.urdf")


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


def load_candidates_for_obj(obj_name: str):
    """Load all BODex grasp candidates for the given object across all scenes.

    Returns list of dicts: {scene, scene_id, grasp_id, wrist_obj (4,4),
    pregrasp (6,), grasp (6,)}. wrist_obj is in object-local frame.
    """
    obj_dir = CANDIDATE_ROOT / obj_name
    out = []
    if not obj_dir.is_dir():
        return out
    for scene in sorted(obj_dir.iterdir()):
        if not scene.is_dir():
            continue
        for scene_id in sorted(scene.iterdir()):
            if not scene_id.is_dir():
                continue
            for grasp in sorted(scene_id.iterdir()):
                if not grasp.is_dir():
                    continue
                wp = grasp / "wrist_se3.npy"
                pp = grasp / "pregrasp_pose.npy"
                gp = grasp / "grasp_pose.npy"
                if not (wp.exists() and pp.exists()):
                    continue
                wrist = np.load(wp)
                pre = np.load(pp)
                g = np.load(gp) if gp.exists() else pre
                out.append({
                    "scene": scene.name,
                    "scene_id": scene_id.name,
                    "grasp_id": grasp.name,
                    "wrist_obj": np.asarray(wrist, dtype=float),
                    "pregrasp": np.asarray(pre, dtype=float),
                    "grasp": np.asarray(g, dtype=float),
                })
    return out


def find_mesh_dir(base_name: str):
    """Resolve `'paper cup'` / `'jp_water'` -> Path to mesh dir under OBJECT_DIRS.

    Tries (in order, across both roots): camelCase, underscore-joined, and
    no-space forms — each compared **case-insensitively** against actual
    directory names. Falls back to a case-insensitive whole-name match. We do
    *not* substring-match because that's what was incorrectly picking
    `Jp_WaterCrush2` for `jp_water`.
    """
    forms = [
        _camel(base_name),
        base_name.replace(" ", "_"),
        base_name.replace(" ", ""),
    ]
    forms_lower = {f.lower() for f in forms if f}
    for root in OBJECT_DIRS:
        if not root.exists():
            continue
        # Build a single case-insensitive index of subdir names.
        for sub in root.iterdir():
            if not sub.is_dir():
                continue
            if sub.name.lower() in forms_lower:
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
DROP_RELEASE_QPOS_RIGHT = np.array([-1.53, 0.0, 0.0, -2.82, -0.01, 3.2, -0.91])
DROP_RELEASE_QPOS_LEFT  = np.array([+1.53, 0.0, 0.0, -2.82, -0.01, 3.2, -0.91])


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
        # Re-entrant: query() holds the lock and then calls world_collide()
        # which also wants it; a plain Lock would deadlock there.
        self._lock = threading.RLock()

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
        # Reusable buffer + tensors for per-sphere world collision queries.
        self._sphere_buf = None
        self._sphere_buf_shape = None
        self._sphere_weight = torch.tensor([1.0], device=self.tensor_args.device,
                                            dtype=self.tensor_args.dtype)
        self._sphere_act = torch.tensor([0.0], device=self.tensor_args.device,
                                         dtype=self.tensor_args.dtype)
        # MotionGen is heavy (~7s warmup); only built on first plan_js() call.
        self._motion_gen = None
        self._plan_cfg = None
        self._motion_gen_world = wc

    def _ensure_motion_gen(self):
        if self._motion_gen is not None:
            return
        print("[curobo] initializing MotionGen (one-time, ~7s)...")
        cfg = MotionGenConfig.load_from_robot_config(
            self.cfg_dict, self._motion_gen_world, self.tensor_args,
            num_trajopt_seeds=32,
            num_graph_seeds=1,
            num_ik_seeds=32,
            use_cuda_graph=False,  # same threading reason as IKSolver
            interpolation_dt=0.01,
            ik_opt_iters=200,
            grad_trajopt_iters=200,
            trajopt_tsteps=64,
            collision_activation_distance=0.005,
            # Pre-allocate the world collision cache. Without this, cuRobo's
            # WorldMeshCollision.cache stays None, and clear_world_cache()
            # crashes on `self.cache["mesh"]` when objects are added/removed
            # later. planner.py ships the same fix.
            collision_cache={"obb": 30, "mesh": 10},
        )
        self._motion_gen = MotionGen(cfg)
        self._motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
        self._plan_cfg = MotionGenPlanConfig(
            enable_graph=True,
            enable_opt=True,
            enable_graph_attempt=4,
            max_attempts=20,
            enable_finetune_trajopt=True,
            num_trajopt_seeds=32,
            num_ik_seeds=32,
            timeout=60.0,
            parallel_finetune=True,
        )
        print("[curobo] MotionGen ready")

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

    def world_collide_from_q(self, q: np.ndarray) -> np.ndarray:
        """Per-sphere world collision using cuRobo's WorldCollision directly.

        Returns (N,) bool array. Uses `get_sphere_distance(sum_collisions=False)`
        which is ~1300× faster than the previous trimesh closest-point path.
        """
        q_t = torch.tensor(q.reshape(1, -1), dtype=self.tensor_args.dtype,
                           device=self.tensor_args.device)
        with self._lock:
            state = self.rw.get_kinematics(q_t)
            x_sph = state.link_spheres_tensor.unsqueeze(1)  # (B=1, H=1, N, 4)
            wc = self.rw.collision_cost.world_coll_checker
            if (self._sphere_buf is None
                    or tuple(self._sphere_buf_shape) != tuple(x_sph.shape)):
                self._sphere_buf = CollisionQueryBuffer.initialize_from_shape(
                    x_sph.shape, self.tensor_args, wc.collision_types,
                )
                self._sphere_buf_shape = x_sph.shape
            d = wc.get_sphere_distance(
                x_sph, self._sphere_buf,
                self._sphere_weight, self._sphere_act,
                sum_collisions=False,
            )
        # d > 0 means colliding (cuRobo convention)
        return (d.view(-1).detach().cpu().numpy() > 0)

    def world_collide(self, x_sph: np.ndarray) -> np.ndarray:
        """Compatibility wrapper: callers that already have x_sph but only need
        a per-sphere collision flag still work, but we recompute from q for
        speed. Prefer `world_collide_from_q`.
        """
        # If only x_sph is given we don't have q, but x_sph[:, :3] are world
        # positions and we can reuse cuRobo if we treat x_sph as the kinematic
        # output directly.
        x_sph_t = torch.tensor(x_sph.reshape(1, 1, -1, 4),
                               dtype=self.tensor_args.dtype,
                               device=self.tensor_args.device)
        with self._lock:
            wc = self.rw.collision_cost.world_coll_checker
            if (self._sphere_buf is None
                    or tuple(self._sphere_buf_shape) != tuple(x_sph_t.shape)):
                self._sphere_buf = CollisionQueryBuffer.initialize_from_shape(
                    x_sph_t.shape, self.tensor_args, wc.collision_types,
                )
                self._sphere_buf_shape = x_sph_t.shape
            d = wc.get_sphere_distance(
                x_sph_t, self._sphere_buf,
                self._sphere_weight, self._sphere_act,
                sum_collisions=False,
            )
        return (d.view(-1).detach().cpu().numpy() > 0)

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
        self._motion_gen_world = wc
        with self._lock:
            self.rw.update_world(wc)
            self.ik_solver.update_world(wc)
            if self._motion_gen is not None:
                self._motion_gen.clear_world_cache()
                self._motion_gen.update_world(wc)
            # New world → buffer may need to grow (different collision types).
            self._sphere_buf = None
            self._sphere_buf_shape = None

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

    GRASP_IK_BATCH = 50

    def _ensure_grasp_ik_solver(self):
        """Lazy-build a separate IK solver for grasp reachability checks.

        Uses scene mesh ONLY (no dynamic object meshes), matching the working
        pattern in autodex/planner/planner.py: when checking if a wrist pose
        is reachable, the hand is supposed to be ON the object, so including
        the object in the IK world makes every grasp fail. Collision against
        the full world (with object) is handled separately after IK.

        Arm null_space_weight is overridden to non-zero values so the
        retract_config (= pick_start qpos) actually pulls solutions toward
        the close-range start pose. The shipped fr3_inspire_left.yml has
        null_space_weight = all zeros which makes retract_config a no-op
        and lets IK pick joint configs on the far side of joint limits
        (especially fr3_joint5, range ±2.876, can't 2π-wrap).

        cuda_graph is enabled because this solver is only ever called from
        the main `_tick` thread.
        """
        if getattr(self, "_grasp_ik_solver", None) is not None:
            return
        scene_only = {SCENE_MESH_KEY: self._world_meshes[SCENE_MESH_KEY]} \
            if SCENE_MESH_KEY in self._world_meshes else {}
        wc = WorldConfig(mesh=list(scene_only.values()))
        # Deep-copy cfg_dict so we don't mutate the shared config (the main
        # ik_solver / motion_gen still use the original null_space=0 defaults).
        import copy
        grasp_cfg = copy.deepcopy(self.cfg_dict)
        cspace = grasp_cfg["robot_cfg"]["kinematics"]["cspace"]
        # Arm: 7 joints — strong pull toward pick_start.
        # Hand: 6 joints — leave at 0 (we overwrite hand q from pregrasp anyway).
        cspace["null_space_weight"] = [1.0] * 7 + [0.0] * 6
        ik_cfg = IKSolverConfig.load_from_robot_config(
            grasp_cfg, wc, tensor_args=self.tensor_args,
            num_seeds=32,
            collision_activation_distance=0.005,
            use_cuda_graph=True,
        )
        self._grasp_ik_solver = IKSolver(ik_cfg)

    def solve_ik_batch(self, target_Ts: np.ndarray,
                       retract_q: np.ndarray | None = None,
                       batch_size: int | None = None):
        """Batched IK for B target poses, chunked + padded.

        - separate IK solver without dynamic object meshes
        - retract_config + non-zero arm null_space_weight (set in
          _ensure_grasp_ik_solver) bias solutions toward retract_q so plans
          don't wrap around through joint limits (esp. fr3_joint5 ±2.876)
        - chunks padded to fixed batch_size for consistent solver shape
        target_Ts: (B, 4, 4). Returns (success (B,) bool, q (B, n_dof)).
        """
        self._ensure_grasp_ik_solver()
        if batch_size is None:
            batch_size = self.GRASP_IK_BATCH
        retract_t = None
        if retract_q is not None:
            retract_t = torch.tensor(
                np.broadcast_to(np.asarray(retract_q, dtype=np.float32),
                                (batch_size, len(retract_q))).copy(),
                device=self.tensor_args.device, dtype=self.tensor_args.dtype,
            ).contiguous()
        target_Ts = np.asarray(target_Ts, dtype=np.float32)
        N = target_Ts.shape[0]
        n_dof = len(self.curobo_joint_names)
        succ_all = np.zeros(N, dtype=bool)
        q_all = np.zeros((N, n_dof), dtype=np.float32)
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            chunk = target_Ts[s:e]
            B = chunk.shape[0]
            if B < batch_size:
                pad = np.tile(chunk[:1], (batch_size - B, 1, 1))
                chunk = np.concatenate([chunk, pad], axis=0)
            pos = torch.tensor(chunk[:, :3, 3],
                               device=self.tensor_args.device,
                               dtype=self.tensor_args.dtype).contiguous()
            qxyzw = R.from_matrix(chunk[:, :3, :3]).as_quat()
            quat = torch.tensor(qxyzw[:, [3, 0, 1, 2]],
                                device=self.tensor_args.device,
                                dtype=self.tensor_args.dtype).contiguous()
            goal = CuroboPose(position=pos, quaternion=quat)
            with self._lock:
                result = self._grasp_ik_solver.solve_batch(
                    goal, retract_config=retract_t,
                )
            # Match planner.py exactly: success is (B_padded,) or (B_padded, 1);
            # solution is (B_padded, return_seeds, dof) or (B_padded, dof).
            s_chunk = result.success.cpu().numpy()[:B]
            q_chunk = result.solution.cpu().numpy()[:B]
            if q_chunk.ndim == 3:
                q_chunk = q_chunk[:, 0, :]
            succ_all[s:e] = np.asarray(s_chunk).astype(bool).reshape(-1)[:B]
            q_all[s:e] = q_chunk
        return succ_all, q_all

    def plan_js(self, q_start: np.ndarray, q_goal: np.ndarray):
        """Joint-space motion plan from q_start → q_goal (curobo joint order).

        Returns interpolated trajectory (T, n_dof) in curobo joint order, or
        None on failure.
        """
        self._ensure_motion_gen()
        start_t = torch.tensor(q_start.reshape(1, -1), dtype=self.tensor_args.dtype,
                               device=self.tensor_args.device)
        goal_t = torch.tensor(q_goal.reshape(1, -1), dtype=self.tensor_args.dtype,
                              device=self.tensor_args.device)
        start = JointState.from_position(start_t)
        goal = JointState.from_position(goal_t)
        with self._lock:
            result = self._motion_gen.plan_single_js(
                start_state=start, goal_state=goal, plan_config=self._plan_cfg,
            )
        if not bool(result.success.item()):
            return None
        return result.get_interpolated_plan().position.cpu().numpy()

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


# ----- Floating hand viewer (single instance, recolored per candidate) -----
class FloatingHand:
    """Renders inspire_left floating hand (6-DOF base + 12 joints) at a given
    wrist 4x4 SE(3) and 6-actuated finger qpos. Used to visualize candidate grasps.
    """
    # cspace order in floating yml expects 6 actuated finger joints in this order.
    HAND_ACTUATED = [
        "left_thumb_1_joint", "left_thumb_2_joint",
        "left_index_1_joint", "left_middle_1_joint",
        "left_ring_1_joint", "left_little_1_joint",
    ]
    BASE_JOINTS = ["x_joint", "y_joint", "z_joint",
                   "x_rotation_joint", "y_rotation_joint", "z_rotation_joint"]

    def __init__(self, server: viser.ViserServer, urdf_path: Path,
                 root: str = "/floating_hand"):
        self.server = server
        self.root = root
        self.urdf = yourdfpy.URDF.load(
            str(urdf_path), mesh_dir=str(urdf_path.parent),
            build_collision_scene_graph=False, load_collision_meshes=False,
        )
        self._joint_frames = {}
        self._mesh_handles = []
        scene = self.urdf.scene
        server.scene.add_frame(root, show_axes=False, visible=False)
        for joint in self.urdf.joint_map.values():
            self._joint_frames[joint.child] = server.scene.add_frame(
                self._frame_path(joint.child), show_axes=False,
            )
        for link_name, mesh in scene.geometry.items():
            T_link = self.urdf.get_transform(
                link_name, scene.graph.transforms.parents[link_name])
            m = mesh.copy()
            m.apply_transform(T_link)
            handle = server.scene.add_mesh_simple(
                name=self._frame_path(link_name) + f"/{link_name}_mesh",
                vertices=m.vertices, faces=m.faces,
                color=(120, 200, 120),
            )
            self._mesh_handles.append(handle)
        self._root_handle = server.scene.add_frame(root, show_axes=False, visible=False)

    def _frame_path(self, link_name: str) -> str:
        scene = self.urdf.scene
        parts = []
        cur = link_name
        while cur != scene.graph.base_frame:
            parts.append(cur)
            cur = scene.graph.transforms.parents[cur]
        return self.root + "/" + "/".join(reversed(parts))

    def set_visible(self, visible: bool):
        # Root frame at /floating_hand is created with visible=False; viser
        # cascades visibility, so toggling only the mesh handles isn't
        # enough — the root must also be flipped.
        self._root_handle.visible = visible
        for h in self._mesh_handles:
            h.visible = visible

    def set_color(self, color):
        for h in self._mesh_handles:
            h.color = tuple(int(c) for c in color)

    def set_pose_and_finger(self, T_world: np.ndarray, finger_qpos: np.ndarray):
        # Build full URDF cfg: 6-DOF base + 12 hand joints (with mimics).
        # actuated joint order from yourdfpy:
        actuated = list(self.urdf.actuated_joint_names)
        q = np.zeros(len(actuated))
        # Decompose T into xyz + rpy (xyz / xyz_rotation joints sequentially apply)
        x, y, z = T_world[:3, 3]
        # Intrinsic XYZ: matches URDF base chain x_rot → y_rot → z_rot which
        # composes as Rx(α)·Ry(β)·Rz(γ) in the moving frame. Lowercase "xyz"
        # is extrinsic and gives angles for Rz·Ry·Rx — wrong for this chain.
        rx, ry, rz = R.from_matrix(T_world[:3, :3]).as_euler("XYZ")
        base = {"x_joint": x, "y_joint": y, "z_joint": z,
                "x_rotation_joint": rx, "y_rotation_joint": ry, "z_rotation_joint": rz}
        for i, jn in enumerate(actuated):
            if jn in base:
                q[i] = float(base[jn])
            elif jn in self.HAND_ACTUATED:
                k = self.HAND_ACTUATED.index(jn)
                q[i] = float(finger_qpos[k])
        self.urdf.update_cfg(q)
        for joint in self.urdf.joint_map.values():
            T = self.urdf.get_transform(joint.child, joint.parent)
            pos, wxyz = mat_to_pos_wxyz(T)
            handle = self._joint_frames[joint.child]
            handle.position = pos
            handle.wxyz = wxyz


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
        self._pending_seq_traj = None  # dict for sequential trajectory job
        self._pending_grasp_check = False  # set True to queue a grasp check
        self._pending_precompute_traj = False  # queue fixed-pair traj precompute
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

        # Latest tracking JSON in this dir (we only keep one).
        tracking_jsons = sorted(HERE.glob("tracking_*.json"))
        self.tracking_files = [tracking_jsons[-1].name] if tracking_jsons else []

        # Floating hand for grasp candidate visualization (hidden by default).
        self.floating_hand = FloatingHand(self.server, FLOATING_HAND_URDF)
        self.floating_hand.set_visible(False)

        # Candidate state: list of dicts (after Check), filled by _check_grasps.
        # Each dict adds: 'wrist_world' (4,4), 'ik_success' (bool), 'ik_q' (curobo-order or None).
        self._candidates = []
        self._traj = None  # (T, n_dof) or None
        self._traj_q_curobo_init = None

        self._build_gui()
        self._refresh_grasp_objects()
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
        self.handeye_Z = None  # 4x4 robot_T_camera
        if not HANDEYE_PATH.exists():
            return
        try:
            data = joblib.load(HANDEYE_PATH)
        except Exception:
            return
        if "Z" not in data:
            return
        Z = np.asarray(data["Z"])
        self.handeye_Z = Z.astype(float)
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
            srv.add_button("Save object poses").on_click(lambda _: self._save_object_poses())
            self.object_poses_status = srv.add_text("Object poses", "—", disabled=True)
            self.objects_folder_parent = srv.add_folder("Active objects", expand_by_default=True)

        with srv.add_folder("Tracking JSON"):
            tracking_choices = tuple(self.tracking_files) if self.tracking_files else ("(none)",)
            self.tracking_dropdown = srv.add_dropdown(
                "file", tracking_choices,
                initial_value=tracking_choices[0],
            )
            self.tracking_pose_field = srv.add_dropdown(
                "pose field", ("mesh_poses", "poses"), initial_value="mesh_poses",
            )
            self.tracking_apply_zinv = srv.add_checkbox("Apply handeye Z^-1", True)
            srv.add_button("Load (add objects)").on_click(
                lambda _: self._on_load_tracking()
            )
            srv.add_button("Clear all objects").on_click(lambda _: self._clear_objects())

        with srv.add_folder("Collision (cuRobo)"):
            self.show_spheres = srv.add_checkbox("Show spheres", True)
            self.show_spheres.on_update(lambda _: self._update_sphere_visibility())
            self.coll_status = srv.add_text("Status", "—", disabled=True)

        with srv.add_folder("Grasp"):
            self.grasp_obj_dropdown = srv.add_dropdown(
                "object", ("(none)",), initial_value="(none)",
            )
            srv.add_button("Refresh objects").on_click(lambda _: self._refresh_grasp_objects())
            srv.add_button("Check grasps").on_click(lambda _: self._queue_grasp_check())
            self.grasp_stats = srv.add_text("Stats", "—", disabled=True)
            # max=1 (not 0) to avoid degenerate min==max sliders, which make
            # the client compute a NaN position and send NaN back to the int
            # handle (viser then raises ValueError on int(NaN)).
            self.grasp_slider = srv.add_slider(
                "Candidate #", min=0, max=1, step=1, initial_value=0,
            )
            self.grasp_slider.on_update(lambda _: self._on_grasp_slider())
            self.grasp_status = srv.add_text("This candidate", "—", disabled=True)
            srv.add_button("Plan to this grasp").on_click(lambda _: self._plan_to_grasp())
            self.plan_status = srv.add_text("Plan status", "—", disabled=True)
            self.traj_slider = srv.add_slider(
                "Trajectory frame", min=0, max=1, step=1, initial_value=0,
            )
            self.traj_slider.on_update(lambda _: self._on_traj_slider())

        with srv.add_folder("Utilities"):
            srv.add_button("Arm → home").on_click(lambda _: self._reset_arm_home())
            srv.add_button("Hand → zero").on_click(lambda _: self._reset_hand_zero())
            srv.add_button("Drop release (right)").on_click(
                lambda _: self._set_arm_qpos(DROP_RELEASE_QPOS_RIGHT))
            srv.add_button("Drop release (left)").on_click(
                lambda _: self._set_arm_qpos(DROP_RELEASE_QPOS_LEFT))

        with srv.add_folder("Waypoint"):
            self.waypoint_name = srv.add_text("Name", initial_value="wp")
            srv.add_button("Save current qpos").on_click(lambda _: self._save_waypoint())
            self.waypoint_status = srv.add_text("Last save", "—", disabled=True)

            wp_choices = self._list_waypoints() or ["(none)"]
            self.wp_start_dropdown = srv.add_dropdown(
                "Start waypoint", tuple(wp_choices),
                initial_value=wp_choices[0],
            )
            self.wp_goal_dropdown = srv.add_dropdown(
                "Goal waypoint", tuple(wp_choices),
                initial_value=wp_choices[-1] if len(wp_choices) > 1 else wp_choices[0],
            )
            srv.add_button("Refresh waypoints").on_click(lambda _: self._refresh_waypoints())
            srv.add_button("Move to start").on_click(
                lambda _: self._jump_to_waypoint(self.wp_start_dropdown.value))
            srv.add_button("Move to goal").on_click(
                lambda _: self._jump_to_waypoint(self.wp_goal_dropdown.value))
            srv.add_button("Sequential start→goal").on_click(
                lambda _: self._build_sequential_trajectory_wp())
            self.seq_status = srv.add_text("Sequential plan", "—", disabled=True)
            self.seq_traj_slider = srv.add_slider(
                "Seq frame", min=0, max=1, step=1, initial_value=0,
            )
            self.seq_traj_slider.on_update(lambda _: self._on_traj_slider())

            srv.add_button("Precompute fixed traj").on_click(
                lambda _: self._queue_precompute_traj())
            self.precompute_status = srv.add_text("Precompute", "—", disabled=True)

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
        self._set_arm_qpos(ARM_HOME)

    def _set_arm_qpos(self, qpos):
        for jn, val in zip(ARM_JOINTS, qpos):
            self.q_actuated[self.actuated.index(jn)] = float(val)
            self.arm_sliders[jn].value = float(val)
        with self._dirty_lock:
            self._dirty = True

    def _list_waypoints(self):
        wp_dir = HERE / "waypoints"
        if not wp_dir.is_dir():
            return []
        return sorted([p.stem for p in wp_dir.glob("*.npz")])

    def _refresh_waypoints(self):
        names = self._list_waypoints()
        choices = tuple(names) if names else ("(none)",)
        for dd in (self.wp_start_dropdown, self.wp_goal_dropdown):
            try:
                cur = dd.value
                dd.options = choices
                dd.value = cur if cur in choices else choices[0]
            except Exception:
                pass
        print(f"[waypoint] available: {names}")

    def _jump_to_waypoint(self, name: str):
        if name == "(none)":
            return
        wp = self._load_waypoint(name)
        if wp is None:
            print(f"[waypoint] missing: {name}")
            return
        for jn, v in zip(wp["actuated"], wp["q_actuated"]):
            if jn in self.actuated:
                self.q_actuated[self.actuated.index(jn)] = float(v)
        for jn in ARM_JOINTS:
            self.arm_sliders[jn].value = float(self.q_actuated[self.actuated.index(jn)])
        for jn in HAND_ACTUATED:
            self.hand_sliders[jn].value = float(self.q_actuated[self.actuated.index(jn)])
        with self._dirty_lock:
            self._dirty = True

    def _build_sequential_trajectory_wp(self):
        """Queue a sequential trajectory job; the main tick consumes it."""
        start_name = self.wp_start_dropdown.value
        goal_name = self.wp_goal_dropdown.value
        if start_name == "(none)" or goal_name == "(none)":
            self.seq_status.value = "select start + goal waypoints"
            return
        wp_g = self._load_waypoint(goal_name)
        if wp_g is None:
            self.seq_status.value = f"can't load {goal_name}"
            return
        arm_goal = np.zeros(len(ARM_JOINTS))
        for jn, v in zip(wp_g["actuated"], wp_g["q_actuated"]):
            if jn in ARM_JOINTS:
                arm_goal[ARM_JOINTS.index(jn)] = float(v)
        with self._dirty_lock:
            self._pending_seq_traj = {
                "arm_goal": arm_goal,
                "start_wp_name": start_name,
                "goal_label": goal_name,
            }
            self._dirty = True
        self.seq_status.value = f"queued {start_name} → {goal_name}"

    def _load_waypoint(self, name: str):
        path = HERE / "waypoints" / f"{name}.npz"
        if not path.exists():
            return None
        d = np.load(path, allow_pickle=True)
        return {
            "actuated": list(d["actuated_joint_names"]),
            "q_actuated": d["q_actuated"],
            "curobo": list(d["curobo_joint_names"]),
            "q_curobo": d["q_curobo"],
        }

    def _build_sequential_trajectory(self, arm_goal: np.ndarray,
                                     start_wp_name: str | None = None,
                                     goal_label: str = "drop_right",
                                     steps_per_joint: int = 30):
        """From `start_wp_name` (or the start dropdown), move FR3 joints 0→6
        one at a time to `arm_goal` (length-7). Hand joints stay at start
        values. Each frame is collision-checked.
        """
        wp_name = start_wp_name or self.wp_start_dropdown.value
        if wp_name == "(none)":
            self.seq_status.value = "no start waypoint"
            return
        wp = self._load_waypoint(wp_name)
        if wp is None:
            self.seq_status.value = f"can't load {wp_name}"
            return
        # Map waypoint actuated -> our actuated order
        q_start_act = np.zeros(len(self.actuated))
        for jn, val in zip(wp["actuated"], wp["q_actuated"]):
            if jn in self.actuated:
                q_start_act[self.actuated.index(jn)] = float(val)

        # Goal: same as start except arm joints 0..6 set to arm_goal.
        q_goal_act = q_start_act.copy()
        for i, jn in enumerate(ARM_JOINTS):
            q_goal_act[self.actuated.index(jn)] = float(arm_goal[i])

        # Build sequential trajectory in CURobo joint order.
        n_curobo = len(self.curobo.curobo_joint_names)
        # actuated -> curobo conversion fn
        def to_curobo(q_act):
            qc = np.zeros(n_curobo)
            for i, jn in enumerate(self.actuated):
                qc[self.act_to_curobo_idx[i]] = float(q_act[i])
            return qc

        q_start_curobo = to_curobo(q_start_act)
        traj = [q_start_curobo.copy()]
        q_cur_act = q_start_act.copy()
        for ji, jn in enumerate(ARM_JOINTS):
            ai = self.actuated.index(jn)
            v0 = float(q_cur_act[ai])
            v1 = float(q_goal_act[ai])
            if abs(v1 - v0) < 1e-6:
                continue
            for s in range(1, steps_per_joint + 1):
                alpha = s / steps_per_joint
                q_cur_act[ai] = v0 + alpha * (v1 - v0)
                traj.append(to_curobo(q_cur_act))
        traj = np.asarray(traj)

        # Batch collision-check all frames at once via cuRobo (very fast).
        n_frames = len(traj)
        try:
            coll_flags = self.curobo.batch_world_collide_any(traj)
        except Exception as e:
            print(f"[seq] batch collide failed, fallback per-frame: {e}")
            coll_flags = np.zeros(n_frames, dtype=bool)
            for i, q in enumerate(traj):
                x_sph = self.curobo.query(q)
                w = self.curobo.world_collide(x_sph)
                s = self.curobo.self_collide(x_sph)
                coll_flags[i] = bool(w.any() or s.any())
        n_coll = int(coll_flags.sum())
        self._traj = traj
        self._traj_coll = coll_flags
        new_max = max(1, len(traj) - 1)
        self.traj_slider.max = new_max
        self.traj_slider.value = 0
        if hasattr(self, "seq_traj_slider"):
            self.seq_traj_slider.max = new_max
            self.seq_traj_slider.value = 0
        self.seq_status.value = f"frames={n_frames}  collides={n_coll}"
        print(f"[seq] {n_frames} frames, {n_coll} colliding")
        # Save trajectory next to waypoints for re-use.
        out = HERE / "waypoints" / f"{wp_name}__to_{goal_label}.npz"
        np.savez(out, traj_curobo=traj, coll_flags=coll_flags,
                 curobo_joint_names=np.array(self.curobo.curobo_joint_names, dtype=object))
        print(f"[seq] saved {out}")
        self._on_traj_slider()

    def _save_object_poses(self):
        """Dump every active object's instance name, base name, mesh path,
        and 4×4 pose to traj/object_poses.npz so the replay script can
        restore the scene before the trajectory starts.
        """
        if not self.objects:
            self.object_poses_status.value = "no active objects"
            return
        instances, bases, mesh_paths, Ts = [], [], [], []
        for inst, entry in self.objects.items():
            base = inst
            while base not in self.object_names and "_" in base:
                base = base.rsplit("_", 1)[0]
            instances.append(inst)
            bases.append(base)
            mesh_paths.append(str(entry["mesh_path"]))
            Ts.append(np.asarray(entry["T"], dtype=np.float64))
        TRAJ_DIR.mkdir(exist_ok=True)
        out = TRAJ_DIR / "object_poses.npz"
        np.savez(
            out,
            instances=np.array(instances, dtype=object),
            base_names=np.array(bases, dtype=object),
            mesh_paths=np.array(mesh_paths, dtype=object),
            T=np.stack(Ts, axis=0),
        )
        self.object_poses_status.value = f"saved {len(instances)} → {out.name}"
        print(f"[objects] saved {len(instances)} poses → {out}")

    def _save_waypoint(self):
        name = (self.waypoint_name.value or "wp").strip()
        # sanitize file name
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name) or "wp"
        out_dir = HERE / "waypoints"
        out_dir.mkdir(exist_ok=True)
        path = out_dir / f"{safe}.npz"
        # Save both actuated (URDF order) and curobo-ordered for convenience.
        np.savez(
            path,
            actuated_joint_names=np.array(self.actuated, dtype=object),
            q_actuated=self.q_actuated.copy(),
            curobo_joint_names=np.array(self.curobo.curobo_joint_names, dtype=object),
            q_curobo=self._q_curobo(),
        )
        self.waypoint_status.value = f"saved → {path.name}"
        print(f"[waypoint] saved {path}")

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
        """Find an obj for `obj_name`, preferring `visual_mesh/` over
        `raw_mesh/` (raw meshes are huge — >1M faces each)."""
        for root in OBJECT_DIRS:
            for sub in ("visual_mesh", "raw_mesh"):
                mesh_dir = root / obj_name / sub
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
        # User-selectable: poses (raw tracker) vs mesh_poses (silhouette-refined)
        field = self.tracking_pose_field.value
        apply_zinv = bool(self.tracking_apply_zinv.value)
        poses = data.get("object", {}).get(field, {})
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
            T_raw = np.asarray(mat, dtype=float).reshape(4, 4)
            # Match the reference grasp_planner_gui.py: just `Z^-1 @ T_world`,
            # no orig_to_autodex remap. The tracking JSON's `poses` /
            # `mesh_poses` are already in the AutoDex mesh frame.
            if apply_zinv and self.handeye_Z is not None:
                T_obj = np.linalg.inv(self.handeye_Z) @ T_raw
            else:
                T_obj = T_raw
            self._set_object_pose(instance, T_obj)
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

    # ---------------- Grasp candidate check + plan ----------------
    def _refresh_grasp_objects(self):
        names = sorted(self.objects.keys())
        choices = tuple(names) if names else ("(none)",)
        # Re-create dropdown isn't supported; just update the options list when
        # viser allows. Fallback: keep a stale list (user can re-add to refresh).
        try:
            self.grasp_obj_dropdown.options = choices
            self.grasp_obj_dropdown.value = choices[0]
        except Exception:
            print(f"[grasp] active objects: {names}")

    def _queue_precompute_traj(self):
        """Worker-thread callback: only flips the precompute flag. The
        actual cuRobo plan / sequential build runs from the main tick."""
        with self._dirty_lock:
            self._pending_precompute_traj = True
            self._dirty = True
        self.precompute_status.value = "queued — running on main thread..."

    def _build_traj_between_waypoints(self, start_name: str, goal_name: str,
                                      mode: str):
        """Build a trajectory from waypoint `start_name` to `goal_name` and
        save it to traj/{start}__to_{goal}.npz.

        mode = "plan"       → cuRobo plan_js (collision-aware planning)
        mode = "sequential" → joint-by-joint interpolation (FR3 joints 0→6),
                              hand joints kept from start. May contain
                              collisions; flagged in coll_flags.
        """
        TRAJ_DIR.mkdir(exist_ok=True)
        wp_s = self._load_waypoint(start_name)
        wp_g = self._load_waypoint(goal_name)
        if wp_s is None or wp_g is None:
            print(f"[traj] missing waypoint(s): {start_name} / {goal_name}")
            return None
        # actuated -> curobo
        n_curobo = len(self.curobo.curobo_joint_names)

        def to_curobo(wp):
            qc = np.zeros(n_curobo)
            for jn, v in zip(wp["actuated"], wp["q_actuated"]):
                if jn in self.curobo.curobo_joint_names:
                    qc[self.curobo.curobo_joint_names.index(jn)] = float(v)
            return qc

        q_start_curobo = to_curobo(wp_s)
        q_goal_curobo = to_curobo(wp_g)

        if mode == "plan":
            traj = self.curobo.plan_js(q_start_curobo, q_goal_curobo)
            if traj is None:
                print(f"[traj] plan FAIL: {start_name} → {goal_name}")
                return None
        elif mode == "sequential":
            traj = [q_start_curobo.copy()]
            steps_per_joint = 30
            q_cur = q_start_curobo.copy()
            for jn in ARM_JOINTS:
                ci = self.curobo.curobo_joint_names.index(jn)
                v0, v1 = float(q_cur[ci]), float(q_goal_curobo[ci])
                if abs(v1 - v0) < 1e-6:
                    continue
                for s in range(1, steps_per_joint + 1):
                    a = s / steps_per_joint
                    q_cur[ci] = v0 + a * (v1 - v0)
                    traj.append(q_cur.copy())
            traj = np.asarray(traj)
        else:
            raise ValueError(f"unknown mode: {mode}")

        # Per-frame collision flags (informational; sequential may include
        # colliding frames — the user said that's OK for some segments).
        try:
            coll_flags = self.curobo.batch_world_collide_any(traj)
        except Exception:
            coll_flags = np.zeros(len(traj), dtype=bool)
            for i, q in enumerate(traj):
                x = self.curobo.query(q)
                w = self.curobo.world_collide(x)
                s_ = self.curobo.self_collide(x)
                coll_flags[i] = bool(w.any() or s_.any())

        out = TRAJ_DIR / f"{start_name}__to_{goal_name}.npz"
        np.savez(
            out,
            traj_curobo=traj,
            coll_flags=coll_flags,
            curobo_joint_names=np.array(self.curobo.curobo_joint_names, dtype=object),
            mode=np.array(mode, dtype=object),
            start_waypoint=np.array(start_name, dtype=object),
            goal_waypoint=np.array(goal_name, dtype=object),
        )
        n_coll = int(coll_flags.sum())
        print(f"[traj] {mode}: {start_name} → {goal_name}  "
              f"frames={len(traj)}  collides={n_coll}  → {out.name}")
        return {"path": out, "frames": len(traj), "collides": n_coll, "mode": mode}

    def _precompute_fixed_trajectories(self):
        """Walk FIXED_TRAJ_PAIRS, build + save each. Called from main tick."""
        results = []
        for start, goal, mode in FIXED_TRAJ_PAIRS:
            r = self._build_traj_between_waypoints(start, goal, mode)
            if r is not None:
                results.append((start, goal, r))
        ok = len(results)
        total = len(FIXED_TRAJ_PAIRS)
        self.precompute_status.value = (
            f"done {ok}/{total}: " +
            ", ".join(f"{s}→{g}({r['frames']}f,{r['collides']}c)"
                      for s, g, r in results)
        )

    def _queue_grasp_check(self):
        """Worker-thread callback: only flips a flag. The actual cuRobo work
        runs from the main tick (per the threading model in HANDOFF.md)."""
        with self._dirty_lock:
            self._pending_grasp_check = True
            self._dirty = True
        self.grasp_stats.value = "queued — running on main thread..."

    def _check_grasps(self):
        instance = self.grasp_obj_dropdown.value
        entry = self.objects.get(instance)
        if entry is None:
            self.grasp_stats.value = "no object selected"
            return
        # Resolve base object name (instance may be "name_2" for duplicates)
        base = instance
        while base not in self.object_names and "_" in base:
            base = base.rsplit("_", 1)[0]
        cands = load_candidates_for_obj(base)
        if not cands:
            self.grasp_stats.value = f"no candidates for '{base}'"
            return

        # Y-axis sweep: cylindrical-symmetric objects (paperCup, Jp_Water,
        # ...) hide the spin angle from perception, so generate 6 copies of
        # every grasp at 0/60/120/180/240/300° around the object's local Y.
        # Asymmetric objects (crushed cans, etc.) keep one orientation —
        # Y-rotated copies would be physically wrong grasps for them.
        if base in Y_SYMMETRIC_OBJECTS:
            expanded = []
            for c in cands:
                for k in range(6):
                    ang = math.radians(60.0 * k)
                    cy, sy = math.cos(ang), math.sin(ang)
                    Ry = np.array([
                        [ cy, 0.0,  sy, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [-sy, 0.0,  cy, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ])
                    cc = dict(c)
                    cc["wrist_obj"] = Ry @ c["wrist_obj"]
                    cc["y_rot_deg"] = 60 * k
                    cc["grasp_id"] = f"{c['grasp_id']}_y{60 * k:03d}"
                    expanded.append(cc)
            cands = expanded

        T_obj = entry["T"]

        # IK target is the wrist pose directly (cuRobo's ee_link is "wrist").
        target_Ts = np.empty((len(cands), 4, 4), dtype=np.float64)
        for i, c in enumerate(cands):
            T_wrist_world = T_obj @ c["wrist_obj"]
            c["wrist_world"] = T_wrist_world
            target_Ts[i] = T_wrist_world

        # Bias IK toward pick_start qpos so solutions don't end up on the
        # far side of joint limits (esp. fr3_joint5 which has range ±2.876
        # and can't wrap by 2π). pick_start is the close-range pre-pick
        # waypoint so its qpos is the right regularization target.
        retract_q = None
        wp_ps = self._load_waypoint("pick_start")
        if wp_ps is not None:
            retract_q = np.zeros(len(self.curobo.curobo_joint_names),
                                 dtype=np.float32)
            for jn, v in zip(wp_ps["actuated"], wp_ps["q_actuated"]):
                if jn in self.curobo.curobo_joint_names:
                    ci = self.curobo.curobo_joint_names.index(jn)
                    retract_q[ci] = float(v)

        # Batched IK against scene-only world (matches planner.py reference).
        t0 = time.perf_counter()
        try:
            succ_arr, q_arr = self.curobo.solve_ik_batch(
                target_Ts, retract_q=retract_q,
            )
        except Exception as e:
            print(f"[grasp] batch IK error: {e}")
            succ_arr = np.zeros(len(cands), dtype=bool)
            q_arr = np.zeros((len(cands), len(self.curobo.curobo_joint_names)),
                             dtype=np.float32)
        print(f"[grasp] batch IK ({len(cands)} poses) "
              f"ok={int(succ_arr.sum())} in {time.perf_counter() - t0:.2f}s")

        # Per-candidate self-collision check on the IK solution. Object
        # meshes are intentionally NOT checked — the wrist target is on the
        # object so any sphere overlap with it is expected; scene-mesh
        # collisions are already ruled out inside the IK solver.
        for i, c in enumerate(tqdm(cands, desc=f"[grasp] coll {base}", unit="grasp")):
            ok = bool(succ_arr[i])
            c["ik_success"] = ok
            c["ik_q"] = np.asarray(q_arr[i]) if ok else None
            in_coll = False
            if ok:
                try:
                    x_sph = self.curobo.query(c["ik_q"])
                    s_coll = self.curobo.self_collide(x_sph)
                    in_coll = bool(s_coll.any())
                except Exception as e:
                    print(f"[grasp] collision error: {e}")
            c["in_collision"] = in_coll
            if not ok:
                c["status"] = "ik_fail"
            elif in_coll:
                c["status"] = "collision"
            else:
                c["status"] = "success"

        # Sort: success first, then collision, then ik_fail
        order = {"success": 0, "collision": 1, "ik_fail": 2}
        cands.sort(key=lambda c: order[c["status"]])
        self._candidates = cands
        n_succ = sum(1 for c in cands if c["status"] == "success")
        n_coll = sum(1 for c in cands if c["status"] == "collision")
        n_fail = sum(1 for c in cands if c["status"] == "ik_fail")
        self.grasp_stats.value = (
            f"object={base}  total={len(cands)}  "
            f"success={n_succ}  collision={n_coll}  ik_fail={n_fail}"
        )
        self.grasp_slider.max = max(1, len(cands) - 1)
        self.grasp_slider.value = 0
        self._show_candidate(0)

    def _on_grasp_slider(self):
        i = int(self.grasp_slider.value)
        self._show_candidate(i)

    def _show_candidate(self, idx: int):
        if not self._candidates:
            self.floating_hand.set_visible(False)
            self.grasp_status.value = "—"
            return
        idx = max(0, min(idx, len(self._candidates) - 1))
        c = self._candidates[idx]
        self.floating_hand.set_visible(True)
        # Use pregrasp finger qpos for visualization (open hand pose).
        self.floating_hand.set_pose_and_finger(c["wrist_world"], c["pregrasp"])
        # green = reachable + collision-free, yellow = reachable but collides,
        # red = IK failed.
        status = c.get("status", "ik_fail")
        color = {
            "success":   (40, 200, 90),
            "collision": (230, 170, 40),
            "ik_fail":   (220, 40, 40),
        }[status]
        self.floating_hand.set_color(color)
        self.grasp_status.value = (
            f"#{idx} {c['scene']}/{c['scene_id']}/{c['grasp_id']} {status.upper()}"
        )

    def _plan_to_grasp(self):
        if not self._candidates:
            self.plan_status.value = "no candidates"
            return
        idx = int(self.grasp_slider.value)
        c = self._candidates[idx]
        if not c["ik_success"]:
            self.plan_status.value = "selected candidate has no IK solution"
            return
        # Goal qpos (curobo joint order)
        q_goal_curobo = c["ik_q"].copy()
        # Override hand joints with pregrasp (so we keep arm IK + open hand)
        for k, jn in enumerate(FloatingHand.HAND_ACTUATED):
            ci = self.curobo.curobo_joint_names.index(jn)
            q_goal_curobo[ci] = float(c["pregrasp"][k])

        # Start from `pick_start` waypoint — fixed pre-pick pose so the plan
        # is always close-range (planning from far away tends to fail).
        wp = self._load_waypoint("pick_start")
        if wp is None:
            self.plan_status.value = "missing pick_start waypoint"
            return
        q_start_curobo = np.zeros(len(self.curobo.curobo_joint_names))
        for jn, v in zip(wp["actuated"], wp["q_actuated"]):
            if jn in self.curobo.curobo_joint_names:
                q_start_curobo[self.curobo.curobo_joint_names.index(jn)] = float(v)

        try:
            traj = self.curobo.plan_js(q_start_curobo, q_goal_curobo)
        except Exception as e:
            self.plan_status.value = f"plan error: {e}"
            print(f"[plan] {e}")
            return
        if traj is None:
            self.plan_status.value = "plan FAIL"
            return
        self._traj = traj  # (T, n_dof) curobo order
        self.plan_status.value = f"plan OK: {len(traj)} frames"
        self.traj_slider.max = max(1, len(traj) - 1)
        self.traj_slider.value = 0
        self._on_traj_slider()

        # Multi-object trash flow: each "Plan to this grasp" click is one
        # pick segment in the sequence. The npz carries `instance` so we
        # don't lose paperCup_1 vs paperCup_2 (the filename only has the
        # base name, which is shared across same-class instances).
        instance = self.grasp_obj_dropdown.value
        base = instance
        while base not in self.object_names and "_" in base:
            base = base.rsplit("_", 1)[0]
        ts = time.strftime("%Y%m%d_%H%M%S")
        TRAJ_DIR.mkdir(exist_ok=True)
        out = TRAJ_DIR / (
            f"{ts}_pick_start__to_grasp_{base}_"
            f"{c['scene']}_{c['scene_id']}_{c['grasp_id']}.npz"
        )
        np.savez(
            out,
            traj_curobo=traj,
            curobo_joint_names=np.array(self.curobo.curobo_joint_names, dtype=object),
            instance=np.array(instance, dtype=object),
            base_name=np.array(base, dtype=object),
            scene=np.array(c["scene"], dtype=object),
            scene_id=np.array(c["scene_id"], dtype=object),
            grasp_id=np.array(c["grasp_id"], dtype=object),
        )
        print(f"[plan] saved {out.name}  instance={instance}  ({len(traj)} frames)")

    def _on_traj_slider(self):
        if self._traj is None:
            return
        # Either slider can drive playback; pick whichever just changed by
        # taking the larger raw value (sliders share state).
        idx = int(self.traj_slider.value)
        if hasattr(self, "seq_traj_slider"):
            idx = max(idx, int(self.seq_traj_slider.value))
        idx = max(0, min(idx, len(self._traj) - 1))
        q_curobo = self._traj[idx]
        # Sync to actuated array and arm/hand sliders, then refresh robot.
        for i, jn in enumerate(self.actuated):
            self.q_actuated[i] = float(q_curobo[self.act_to_curobo_idx[i]])
        for jn in ARM_JOINTS:
            self.arm_sliders[jn].value = float(self.q_actuated[self.actuated.index(jn)])
        for jn in HAND_ACTUATED:
            self.hand_sliders[jn].value = float(self.q_actuated[self.actuated.index(jn)])
        with self._dirty_lock:
            self._dirty = True

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
            seq_job = self._pending_seq_traj
            self._pending_seq_traj = None
            grasp_check = self._pending_grasp_check
            self._pending_grasp_check = False
            precompute_traj = self._pending_precompute_traj
            self._pending_precompute_traj = False
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
                import traceback
                print(f"[world] job {job[0]} failed: {e}")
                traceback.print_exc()

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

        # 2.3) Pre-compute fixed waypoint trajectories (cuRobo plan or
        # sequential, saved under traj/). Long-running; main thread.
        if precompute_traj:
            try:
                self._precompute_fixed_trajectories()
            except Exception as e:
                print(f"[traj] precompute failed: {e}")
                self.precompute_status.value = f"err: {e}"

        # 2.4) Grasp check (IK + collision over all candidates × Y rotations).
        # Long-running; runs here on the main thread per the threading model.
        if grasp_check:
            try:
                self._check_grasps()
            except Exception as e:
                print(f"[grasp] check failed: {e}")
                self.grasp_stats.value = f"err: {e}"

        # 2.5) Sequential trajectory build (collision-checks every frame).
        if seq_job is not None:
            try:
                self._build_sequential_trajectory(
                    seq_job["arm_goal"],
                    start_wp_name=seq_job["start_wp_name"],
                    goal_label=seq_job["goal_label"],
                )
            except Exception as e:
                print(f"[seq] build failed: {e}")
                self.seq_status.value = f"err: {e}"
            # _build_sequential_trajectory ends with _on_traj_slider(), which
            # sets q_actuated + arm_sliders. Mark dirty so the joint-mode
            # post-step (FK eef sync) doesn't fight the new pose, and so the
            # next tick re-publishes the robot.
            with self._dirty_lock:
                self._dirty = True

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
