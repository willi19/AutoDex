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
import sys
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

# cuRobo JIT-compiles CUDA kernels on first import in a fresh env, which
# can take 1–5 minutes with no console output. Print before/after each
# heavy import so the user sees something is happening.
def _t(msg):
    print(msg, flush=True, file=sys.stderr)

_t0 = time.perf_counter()
_t(f"[import 0.00s] cuRobo... (first run JIT-compiles CUDA kernels, can take minutes)")
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
_t(f"[import {time.perf_counter() - _t0:.2f}s] cuRobo done")


HERE = Path(__file__).resolve().parent
URDF_PATH = HERE / "fr3_inspire_left.urdf"
SCENE_MESH_PATH = HERE / "scene_mesh.obj"
TRAJ_DIR = HERE / "traj"
GRASP_CACHE_DIR = HERE / "grasp_cache"
# How close the object pose has to be (translation/rotation matrix entries) for
# a cached IK+collision result to be reused. ~5mm / a few degrees.
GRASP_CACHE_ATOL = 5e-3

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
Y_SYMMETRIC_OBJECTS = {"paperCup", "Jp_Water", "big_Oi_Ocha"}

# Post-grasp lift sequence: list of Cartesian world-frame offsets (meters)
# applied one-after-another. Each entry becomes a separate IK + plan_js stage.
# Default: just lift 10cm up.
# big_Oi_Ocha / paperCup / Jp_Water: lift 15cm up, THEN move 20cm in +y so
# the carried object clears the shelf before the lift→place plan runs.
DEFAULT_LIFT_OFFSETS = [(0.0, 0.0, 0.10)]
LIFT_OFFSETS = {
    "big_Oi_Ocha": [(0.0, 0.0, 0.15), (0.0, 0.20, 0.0)],
    "paperCup":    [(0.0, 0.0, 0.15), (0.0, 0.20, 0.0)],
    "Jp_Water":    [(0.0, 0.0, 0.15), (0.0, 0.20, 0.0)],
}
SCENE_VOXEL_PATH = Path("/home/mingi/Downloads/scene_voxel.npz")
HANDEYE_PATH = HERE / "hand_eye_result.pkl"
# Robot region (in world frame) to mask out from the scene SDF — same region
# we cut from the mesh to remove the robot's own embedding in the scan.
ROBOT_MASK_X_MAX = 0.2
ROBOT_MASK_Z_MIN = 0.0
OBJECT_DIRS = [
    Path("/home/mingi/shared_data/AutoDex/object/paradex"),
    Path("/home/mingi/shared_data/AutoDex/object/robothome"),
]
ROBOT_CFG_NAME = "fr3_inspire_left.yml"
SCENE_MESH_KEY = "scene_collision"
HAND_NAME = "inspire_left"
CANDIDATE_VERSION = "selected_100"
CANDIDATE_ROOT = Path(f"/home/mingi/AutoDex/candidates/{HAND_NAME}/{CANDIDATE_VERSION}")
FLOATING_HAND_URDF = (Path("/home/mingi/AutoDex/autodex/planner/src/curobo/content/assets")
                      / "robot/inspire_description/inspire_left_floating.urdf")

# Two-robot setup: jangja (장자, FR3 + inspire_left) at world origin (=
# viewer origin = jangja base), seoja (서자, FR3 + inspire_f1 right hand) at
# +1.50m on jangja's +x axis, ~180° rotated. Base pose computed from handeye
# files at startup (see _compute_seoja_base_pose).
SEOJA_URDF_PATH = (Path("/home/mingi/AutoDex/autodex/planner/src/curobo/content/assets")
                   / "robot/fr3_inspire_f1_description/fr3_inspire_f1.urdf")
SEOJA_ROBOT_CFG_NAME = "fr3_inspire_f1.yml"
HANDEYE_PATH_JANGJA = HERE / "hand_eye_result.pkl"  # active = jangja
HANDEYE_PATH_SEOJA  = Path("/home/mingi/Downloads/hand_eye_result (6).pkl")
SEOJA_HAND_ACTUATED = [
    "right_thumb_1_joint", "right_thumb_2_joint",
    "right_index_1_joint", "right_middle_1_joint",
    "right_ring_1_joint", "right_little_1_joint",
]


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


DEMO_CANDIDATES = {
    # obj_name -> (subsampled_dir, list of npz indices). BODex can't generate
    # grasps for flat-on-table objects, so we use captured demo trajectories
    # as candidates instead. Each (file × Z-rotation) becomes one candidate.
    "chocoSong-i": (HERE / "subsampled" / "choco",
                    ["0", "2", "3", "4", "7", "8", "9"]),
}
DEMO_Z_ANGLES_DEG = [0, 60, 120, 180, 240, 300]

# Captured-demo hand_qpos columns are stored in raw controller order which is
# the REVERSE of URDF/HAND_ACTUATED order (raw[0]=little, raw[5]=thumb_1).
# Use raw[:, RAW_HAND_TO_URDF] to get HAND_ACTUATED-ordered columns. Mirrors
# test_demo_plan.py's _hand_qpos_urdf_order; without this swap thumb/little
# ended up swapped in viewer playback.
RAW_HAND_TO_URDF = [5, 4, 3, 2, 1, 0]


def _hand_qpos_urdf_order(d: np.lib.npyio.NpzFile) -> np.ndarray:
    """Return demo hand qpos in HAND_ACTUATED column order. Honors a stored
    `hand_qpos_order` field if present; otherwise assumes raw controller order
    and applies RAW_HAND_TO_URDF.
    """
    hand = np.asarray(d["hand_qpos"], dtype=np.float64)
    order = str(d["hand_qpos_order"].item()) if "hand_qpos_order" in d.files else "raw"
    if order == "urdf":
        return hand
    if order == "raw":
        return hand[:, RAW_HAND_TO_URDF]
    raise ValueError(f"unknown hand_qpos_order={order!r}")


def _load_demo_candidates(obj_name: str):
    """Build BODex-shape candidates from saved subsampled demo trajectories.

    Each subsampled npz has wrist_rel_se3 (N,4,4) in object-local frame
    (rotation = I, translation = wrist - occluded_centroid). We use the
    last waypoint as the grasp pose and sweep R_z about the object center.
    """
    cfg = DEMO_CANDIDATES.get(obj_name)
    if cfg is None:
        return []
    sub_dir, idxs = cfg
    out = []
    for i in idxs:
        f = sub_dir / f"{i}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        wrist_rel = d["wrist_rel_se3"]      # (N, 4, 4) in object frame
        hand = _hand_qpos_urdf_order(d)     # (N, 6) in HAND_ACTUATED order
        last = wrist_rel[-1]
        finger_open = hand[0]
        finger_grasp = hand[-1]
        source_start = int(d["start"]) if "start" in d.files else 0
        source_stride = int(d["stride"]) if "stride" in d.files else 1
        for ang_deg in DEMO_Z_ANGLES_DEG:
            ang = math.radians(ang_deg)
            cz, sz = math.cos(ang), math.sin(ang)
            Rz = np.array([
                [cz, -sz, 0.0, 0.0],
                [sz,  cz, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
            wrist_obj = Rz @ last
            out.append({
                "scene": "demo",
                "scene_id": f"choco_{i}",
                "grasp_id": f"z{ang_deg:03d}",
                "wrist_obj": wrist_obj.astype(float),
                "pregrasp": np.asarray(finger_open, dtype=float),
                "grasp": np.asarray(finger_grasp, dtype=float),
                # Full demo for execution: wrist sequence in object frame
                # (Z-rotated) + finger qpos sequence.
                "demo_wrist_rel_seq": (Rz @ wrist_rel).astype(float),
                "demo_hand_qpos_seq": np.asarray(hand, dtype=float),
                "demo_source_start": source_start,
                "demo_source_stride": source_stride,
            })
    return out


def load_candidates_for_obj(obj_name: str):
    """Load all BODex grasp candidates for the given object across all scenes.

    Returns list of dicts: {scene, scene_id, grasp_id, wrist_obj (4,4),
    pregrasp (6,), grasp (6,)}. wrist_obj is in object-local frame.
    """
    # Demo-trajectory exception: objects without BODex candidates fall back
    # to captured demos under subsampled/.
    if obj_name in DEMO_CANDIDATES:
        return _load_demo_candidates(obj_name)

    # Candidate dirs use mixed naming (camelCase like `paperCup`, underscore
    # like `pepper_tuna`). Tracking keys feed in the underscore form, and
    # `_camel` only collapses spaces — so `_camel("paper_cup") == "paper_cup"`.
    # Adding the no-separator form makes `paper_cup` match `paperCup` via lower.
    forms_lower = {f.lower() for f in [
        obj_name,
        obj_name.replace("_", ""),
        obj_name.replace(" ", ""),
    ] if f}
    obj_dir = None
    if CANDIDATE_ROOT.is_dir():
        for sub in CANDIDATE_ROOT.iterdir():
            if sub.is_dir() and sub.name.lower() in forms_lower:
                obj_dir = sub
                break
    out = []
    if obj_dir is None or not obj_dir.is_dir():
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


def _grasp_cache_path(base: str):
    return GRASP_CACHE_DIR / f"{base}.npz"


def save_grasp_cache(base: str, T_obj: np.ndarray, cands: list):
    """Persist IK + collision results for `base` at `T_obj`. wrist_world is
    re-derived on load (it's just T_obj @ wrist_obj), so we strip it here.
    """
    GRASP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = []
    for c in cands:
        d = dict(c)
        d.pop("wrist_world", None)
        payload.append(d)
    np.savez(_grasp_cache_path(base),
             T_obj=np.asarray(T_obj, dtype=np.float64),
             cands=np.array(payload, dtype=object))


def try_load_grasp_cache(base: str, T_obj: np.ndarray):
    """Return cached cands list if `base` has a cache file whose stored T_obj
    is within GRASP_CACHE_ATOL of the current pose. Else None.
    """
    p = _grasp_cache_path(base)
    if not p.exists():
        return None
    try:
        d = np.load(p, allow_pickle=True)
        T_cached = np.asarray(d["T_obj"])
        if not np.allclose(T_cached, T_obj, atol=GRASP_CACHE_ATOL, rtol=0):
            return None
        cands = list(d["cands"])
        for c in cands:
            c["wrist_world"] = T_obj @ c["wrist_obj"]
        return cands
    except Exception as e:
        print(f"[grasp cache] load failed: {e}")
        return None


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


# ----- Demo trajectory IK helpers (ported from test_demo_plan.py) -----
# Used only by the per-waypoint backward IK chain on captured demo trajectories
# (chocoSong-i, etc.). Sparse keyframe IK + linear interpolation + wrist/hand
# smoothing makes the chain robust against per-frame null-space branch flips.
DEMO_IK_KEYFRAME_RAW_STRIDE = 60  # IK every ~N original capture frames
DEMO_SMOOTH_WRIST_POS_WINDOW = 7
DEMO_SMOOTH_WRIST_ROT_WINDOW = 7
DEMO_SMOOTH_HAND_WINDOW = 7
DEMO_MAX_HAND_STEP = 0.05      # rad/frame clamp on hand qpos
DEMO_MAX_ARM_STEP = 0.0        # rad/frame clamp on arm; <=0 disables
DEMO_SMOOTH_JOINT_WINDOW = 1   # post-interp arm smoothing; 1 disables


def _odd_window(n: int, length: int) -> int:
    n = int(max(1, n))
    n = min(n, max(1, int(length)))
    if n % 2 == 0:
        n = max(1, n - 1)
    return n


def _moving_average_reflect(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x)
    window = _odd_window(window, len(x))
    if window <= 1 or len(x) <= 2:
        return x.copy()
    pad = window // 2
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(x, [(pad, pad)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    out = np.empty_like(x, dtype=np.float64)
    for i in range(len(x)):
        out[i] = np.tensordot(kernel, padded[i:i + window], axes=(0, 0))
    return out.astype(x.dtype, copy=False)


def _smooth_wrist_rel_sequence(wrist_rel: np.ndarray, pos_window: int,
                               rot_window: int) -> np.ndarray:
    """Low-pass filter demo wrist SE(3) before IK. Endpoints preserved.

    Quaternion signs are made continuous before averaging so q/-q does not
    inject a fake rotation jump.
    """
    out = np.asarray(wrist_rel, dtype=np.float64).copy()
    if len(out) <= 2:
        return out
    first = out[0].copy()
    last = out[-1].copy()
    if pos_window > 1:
        out[:, :3, 3] = _moving_average_reflect(out[:, :3, 3], pos_window)
    if rot_window > 1:
        quat = R.from_matrix(out[:, :3, :3]).as_quat()  # xyzw
        for i in range(1, len(quat)):
            if float(np.dot(quat[i - 1], quat[i])) < 0.0:
                quat[i] *= -1.0
        quat = _moving_average_reflect(quat, rot_window)
        quat /= np.linalg.norm(quat, axis=1, keepdims=True).clip(1e-12)
        out[:, :3, :3] = R.from_quat(quat).as_matrix()
    out[0] = first
    out[-1] = last
    return out


def _limit_joint_steps(q_seq: np.ndarray, joint_idx: list,
                       max_step: float) -> np.ndarray:
    out = np.asarray(q_seq, dtype=np.float32).copy()
    if max_step <= 0.0 or len(out) <= 1:
        return out
    idx = np.asarray(joint_idx, dtype=np.int64)
    q0 = out[0, idx].copy()
    qn = out[-1, idx].copy()
    for k in range(1, len(out)):
        dq = out[k, idx] - out[k - 1, idx]
        out[k, idx] = out[k - 1, idx] + np.clip(dq, -max_step, max_step)
    out[-1, idx] = qn
    for k in range(len(out) - 2, -1, -1):
        dq = out[k, idx] - out[k + 1, idx]
        out[k, idx] = out[k + 1, idx] + np.clip(dq, -max_step, max_step)
    out[0, idx] = q0
    return out


def _smooth_joint_sequence(q_seq: np.ndarray, joint_idx: list,
                           window: int) -> np.ndarray:
    out = np.asarray(q_seq, dtype=np.float32).copy()
    window = _odd_window(window, len(out))
    if window <= 1 or len(out) <= 2:
        return out
    idx = np.asarray(joint_idx, dtype=np.int64)
    smoothed = _moving_average_reflect(out[:, idx], window)
    out[:, idx] = smoothed
    out[0, idx] = q_seq[0, idx]
    out[-1, idx] = q_seq[-1, idx]
    return out


def _keyframe_indices(length: int, stride: int) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.int64)
    stride = max(1, int(stride))
    idx = list(range(0, length, stride))
    if idx[-1] != length - 1:
        idx.append(length - 1)
    return np.asarray(idx, dtype=np.int64)


def _linear_interpolate_q(key_idx: np.ndarray, key_q: np.ndarray,
                          length: int) -> np.ndarray:
    key_idx = np.asarray(key_idx, dtype=np.int64)
    key_q = np.asarray(key_q, dtype=np.float32)
    if len(key_idx) == 0:
        return np.zeros((length, key_q.shape[-1]), dtype=np.float32)
    if len(key_idx) == 1:
        return np.repeat(key_q[:1], length, axis=0).astype(np.float32)
    out = np.empty((length, key_q.shape[1]), dtype=np.float32)
    for a, b, qa, qb in zip(key_idx[:-1], key_idx[1:], key_q[:-1], key_q[1:]):
        span = max(1, int(b - a))
        alpha = (np.arange(a, b + 1, dtype=np.float32) - float(a)) / float(span)
        out[a:b + 1] = (1.0 - alpha[:, None]) * qa[None, :] + alpha[:, None] * qb[None, :]
    return out


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
        # Captured demo grasps legitimately have fingers touching each other
        # (thumb-index pinch, etc.). The hand also stays in the same closed
        # config throughout lift→place, so any cross-finger sphere overlap
        # at the start persists at the goal — there's no trajectory motion
        # that could resolve it. Extend self_collision_ignore so every pair
        # of inspire hand links (thumb / index / middle / ring / little, +
        # base_link) is ignored. fr3 self-collision and fr3↔hand collision
        # are unchanged.
        kin_root = self.cfg_dict["robot_cfg"]["kinematics"]
        sci = kin_root.setdefault("self_collision_ignore", {})
        _hand_links = [
            "base_link",
            "left_thumb_1", "left_thumb_2", "left_thumb_3", "left_thumb_4",
            "left_index_1", "left_index_2",
            "left_middle_1", "left_middle_2",
            "left_ring_1", "left_ring_2",
            "left_little_1", "left_little_2",
        ]
        for a in _hand_links:
            existing = list(sci.get(a, []))
            for b in _hand_links:
                if a == b:
                    continue
                if b not in existing:
                    existing.append(b)
            sci[a] = existing
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
        # Pre-warm MotionGen now (scene mesh is loaded; dynamic objects come
        # later but motion_gen's world is scene-only by design). Pays the ~7s
        # warmup at startup so the first plan-button click is instant.
        self._ensure_motion_gen()

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
        # Separate buffer for grasp_world_collide: queries the grasp IK solver's
        # scene-only world checker, whose collision_types may differ from main.
        self._grasp_sphere_buf = None
        self._grasp_sphere_buf_shape = None
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
            # 0.0 — penetration only. Anything > 0 is a buffer that flags
            # valid pre-grasps near the target surface as collision.
            collision_activation_distance=0.0,
            # Pre-allocate the world collision cache. Without this, cuRobo's
            # WorldMeshCollision.cache stays None, and clear_world_cache()
            # crashes on `self.cache["mesh"]` when objects are added/removed
            # later. planner.py ships the same fix.
            collision_cache={"obb": 30, "mesh": 10},
        )
        self._motion_gen = MotionGen(cfg)
        self._motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
        self._plan_cfg = MotionGenPlanConfig(
            # Graph search (PRM) does its own start/goal feasibility check
            # via rollout_constraint that disagrees with raw sphere distance.
            # We've verified start/goal collision-free with 4 sphere checks,
            # so skip graph entirely and let trajopt drive the plan.
            enable_graph=False,
            # cuRobo's default `enable_graph_attempt=3` flips graph back ON
            # after 3 failed trajopt attempts — None here disables that
            # fallback so trajopt-only stays trajopt-only.
            enable_graph_attempt=None,
            enable_opt=True,
            max_attempts=20,
            enable_finetune_trajopt=True,
            num_trajopt_seeds=32,
            num_ik_seeds=32,
            timeout=60.0,
            parallel_finetune=True,
            check_start_validity=False,
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
            # Invalidate the scene-only grasp checkers — they cache the scene
            # mesh from their first build, so without this they go stale and
            # silently pass IK solutions that clip the new scene. Symptom:
            # `world_collide_scene_from_q` returns False for actually-colliding
            # candidates → marked "success" → plan_js fails GRAPH_FAIL on the
            # fresh main world.
            self._grasp_rw = None
            self._grasp_ik_solver = None
            self._grasp_sphere_buf = None
            self._grasp_sphere_buf_shape = None

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

    def _ensure_grasp_rw(self):
        """Lazy RobotWorld for grasp-time collision checks.

        Full world (scene + dynamic target objects). At a valid grasp the wrist
        is at the grasp pose with PRE-GRASP (open) fingers — the hand should
        wrap around the target without touching it. If pre-grasp fingers do
        contact the target the candidate is bad and gets filtered.

        Using the same world as motion_gen guarantees that a candidate marked
        collision-free here is also accepted by plan_single_js's start/end
        validity check (no more "End state in collision" surprises).
        """
        if getattr(self, "_grasp_rw", None) is not None:
            return
        wc = WorldConfig(mesh=list(self._world_meshes.values()))
        # All three checkers (motion_gen, _grasp_ik_solver, _grasp_rw) use 0.0
        # so they answer the same question (penetration only) on the same q.
        rw_cfg = RobotWorldConfig.load_from_config(
            self.cfg_dict, wc, tensor_args=self.tensor_args,
            collision_activation_distance=0.0,
        )
        self._grasp_rw = RobotWorld(rw_cfg)
        self._grasp_sphere_buf = None
        self._grasp_sphere_weight = torch.tensor(
            [1.0], device=self.tensor_args.device, dtype=self.tensor_args.dtype,
        )
        self._grasp_sphere_act = torch.tensor(
            [0.0], device=self.tensor_args.device, dtype=self.tensor_args.dtype,
        )

    def world_collide_scene_from_q(self, q: np.ndarray) -> np.ndarray:
        """Per-sphere world collision against the FULL world (scene + target
        objects). Pre-grasp finger config means the hand should not touch the
        target — if it does, this returns True for the offending sphere(s).
        Name is kept for backwards-compat; semantics now match motion_gen.
        """
        self._ensure_grasp_rw()
        q_t = torch.tensor(q.reshape(1, -1), dtype=self.tensor_args.dtype,
                           device=self.tensor_args.device)
        with self._lock:
            state = self._grasp_rw.get_kinematics(q_t)
            x_sph = state.link_spheres_tensor.unsqueeze(1)  # (1, 1, N, 4)
            wc = self._grasp_rw.collision_cost.world_coll_checker
            if (self._grasp_sphere_buf is None
                    or tuple(self._grasp_sphere_buf.shape) != tuple(x_sph.shape)):
                self._grasp_sphere_buf = CollisionQueryBuffer.initialize_from_shape(
                    x_sph.shape, self.tensor_args, wc.collision_types,
                )
            d = wc.get_sphere_distance(
                x_sph, self._grasp_sphere_buf,
                self._grasp_sphere_weight, self._grasp_sphere_act,
                sum_collisions=False,
            )
        return (d.view(-1).detach().cpu().numpy() > 0)

    def _ensure_grasp_ik_solver(self):
        """Lazy-build a separate IK solver for grasp reachability checks.

        Full world (scene + dynamic objects), matching motion_gen and
        `_grasp_rw`. The IK target is the grasp wrist pose; with pre-grasp
        (open) fingers the hand should not penetrate the target object. If
        a candidate's pre-grasp fingers do clip the target IK will fail it.

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
        wc = WorldConfig(mesh=list(self._world_meshes.values()))
        # Deep-copy cfg_dict so we don't mutate the shared config (the main
        # ik_solver / motion_gen still use the original null_space=0 defaults).
        import copy
        grasp_cfg = copy.deepcopy(self.cfg_dict)
        cspace = grasp_cfg["robot_cfg"]["kinematics"]["cspace"]
        # Arm: 7 joints — strong pull toward pick_start.
        # Hand: 6 joints — leave at 0 (we overwrite hand q from pregrasp anyway).
        cspace["null_space_weight"] = [1.0] * 7 + [0.0] * 6
        # cuda_graph=True: solve_ik_batch already pads each chunk to fixed
        # GRASP_IK_BATCH (=50), so the captured graph reuses across chunks
        # and across re-clicks. With graph off the same call took tens of
        # seconds; with it, sub-second for ~600 padded candidates.
        ik_cfg = IKSolverConfig.load_from_robot_config(
            grasp_cfg, wc, tensor_args=self.tensor_args,
            num_seeds=32,
            collision_activation_distance=0.0,
            use_cuda_graph=True,
        )
        self._grasp_ik_solver = IKSolver(ik_cfg)

    def solve_ik_batch(self, target_Ts: np.ndarray,
                       retract_q: np.ndarray | None = None):
        """Batched IK for N target poses in a single solve_batch call.

        - separate IK solver without dynamic object meshes
        - retract_config + non-zero arm null_space_weight (set in
          _ensure_grasp_ik_solver) bias solutions toward retract_q so plans
          don't wrap around through joint limits (esp. fr3_joint5 ±2.876)
        target_Ts: (N, 4, 4). Returns (success (N,) bool, q (N, n_dof)).
        """
        self._ensure_grasp_ik_solver()
        target_Ts = np.asarray(target_Ts, dtype=np.float32)
        N = target_Ts.shape[0]
        retract_t = None
        if retract_q is not None and N > 0:
            retract_t = torch.tensor(
                np.broadcast_to(np.asarray(retract_q, dtype=np.float32),
                                (N, len(retract_q))).copy(),
                device=self.tensor_args.device, dtype=self.tensor_args.dtype,
            ).contiguous()
        pos = torch.tensor(target_Ts[:, :3, 3],
                           device=self.tensor_args.device,
                           dtype=self.tensor_args.dtype).contiguous()
        qxyzw = R.from_matrix(target_Ts[:, :3, :3]).as_quat()
        quat = torch.tensor(qxyzw[:, [3, 0, 1, 2]],
                            device=self.tensor_args.device,
                            dtype=self.tensor_args.dtype).contiguous()
        goal = CuroboPose(position=pos, quaternion=quat)
        with self._lock:
            result = self._grasp_ik_solver.solve_batch(
                goal, retract_config=retract_t,
            )
        # success: (N,) or (N, 1); solution: (N, return_seeds, dof) or (N, dof).
        succ_all = result.success.cpu().numpy().astype(bool).reshape(-1)[:N]
        q_all = result.solution.cpu().numpy()
        if q_all.ndim == 3:
            q_all = q_all[:, 0, :]
        return succ_all, q_all

    def _ensure_demo_chain_ik_solver(self):
        """Lazy IK solver dedicated to backward IK along a captured demo
        trajectory (chocoSong-i, etc.).

        Differences from the main / grasp IK solvers:
          - World contains the static scene mesh ONLY — dynamic target
            objects are excluded. Rationale: the captured demo wrist is on/
            near the target object so the target would always trigger a
            false-positive collision; the scene (table/walls) however still
            constrains the franka arm so the trajectory can't pass through
            the table. The approach plan (pick_start → first waypoint) is
            unchanged and still uses full-collision motion_gen, so inspire
            hand vs scene/object IS checked there exactly like a normal
            BODex grasp.
          - Inspire (hand) collision is disabled by stripping all `left_*`
            and `base_link` entries from `collision_link_names` (and the
            corresponding self_collision_buffer / ignore entries). Net
            effect: only fr3_link0..7 spheres participate in collision, so
            the captured hand trajectory which legitimately contacts the
            object/table is not flagged.
          - `null_space_weight = arm 1.0 / hand 0.0` makes `retract_config`
            (= the next waypoint's qpos in the backward chain) act as a real
            continuity prior on the arm. Hand is overwritten from captured
            demo qpos after IK, so its weight stays 0.
          - cuda_graph=False because demo objects are infrequent and the chain
            is sequential (next call's retract depends on previous result), so
            graph capture wouldn't amortize.
        """
        if getattr(self, "_demo_chain_ik_solver", None) is not None:
            return
        import copy
        chain_cfg = copy.deepcopy(self.cfg_dict)
        kin = chain_cfg["robot_cfg"]["kinematics"]
        cspace = kin["cspace"]
        n_total = len(cspace["joint_names"])
        cspace["null_space_weight"] = (
            [1.0] * len(ARM_JOINTS) + [0.0] * (n_total - len(ARM_JOINTS))
        )
        # Keep only fr3 links in collision; drop base_link + every left_*.
        def _is_fr3(link: str) -> bool:
            return link.startswith("fr3_link")
        kin["collision_link_names"] = [
            ln for ln in kin.get("collision_link_names", []) if _is_fr3(ln)
        ]
        if "self_collision_buffer" in kin:
            kin["self_collision_buffer"] = {
                ln: v for ln, v in kin["self_collision_buffer"].items() if _is_fr3(ln)
            }
        if "self_collision_ignore" in kin:
            kin["self_collision_ignore"] = {
                a: [b for b in bs if _is_fr3(b)]
                for a, bs in kin["self_collision_ignore"].items() if _is_fr3(a)
            }
        # Scene mesh only — exclude dynamic target objects so a wrist touching
        # the target is not flagged.
        scene_meshes = [m for k, m in self._world_meshes.items()
                        if k == SCENE_MESH_KEY]
        wc_scene = WorldConfig(mesh=scene_meshes)
        ik_cfg = IKSolverConfig.load_from_robot_config(
            chain_cfg, wc_scene, tensor_args=self.tensor_args,
            num_seeds=32,
            collision_activation_distance=0.005,
            use_cuda_graph=False,
        )
        self._demo_chain_ik_solver = IKSolver(ik_cfg)

    def _solve_ik_demo_chain(self, target_T: np.ndarray,
                             retract_q: np.ndarray | None):
        """Single-pose IK on the demo-chain solver. Returns (success, q)."""
        self._ensure_demo_chain_ik_solver()
        rot = target_T[:3, :3]
        qxyzw = R.from_matrix(rot).as_quat()
        pos_t = torch.tensor([target_T[0, 3], target_T[1, 3], target_T[2, 3]],
                             device=self.tensor_args.device,
                             dtype=self.tensor_args.dtype).view(1, 3)
        quat_t = torch.tensor([qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]],
                              device=self.tensor_args.device,
                              dtype=self.tensor_args.dtype).view(1, 4)
        goal = CuroboPose(position=pos_t, quaternion=quat_t)
        retract_t = None
        if retract_q is not None:
            retract_t = torch.tensor(
                np.asarray(retract_q, dtype=np.float32).reshape(1, -1),
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            )
        with self._lock:
            result = self._demo_chain_ik_solver.solve_single(
                goal, retract_config=retract_t,
            )
        succ = bool(result.success.view(-1)[0].item())
        q = result.solution.view(-1).detach().cpu().numpy()
        return succ, q

    def solve_demo_chain(self, T_obj: np.ndarray, wrist_rel_seq: np.ndarray,
                         hand_qpos_seq: np.ndarray, q_last: np.ndarray,
                         arm_idx: list, hand_idx: list,
                         source_stride: int = 1,
                         keyframe_raw_stride: int = DEMO_IK_KEYFRAME_RAW_STRIDE):
        """Backward keyframe IK along a captured demo trajectory.

        The demo trajectory is N captured wrist poses in object frame. Solving
        IK for every frame is both expensive and prone to per-frame null-space
        branch flips, so we instead:
          1. Smooth wrist SE(3) (continuous quaternions + moving average).
          2. Pick sparse keyframes (~every `keyframe_raw_stride` original
             capture frames) including the last frame.
          3. Backward-chain IK across keyframes, seeding each waypoint with
             the next waypoint's solved qpos for arm-joint continuity.
          4. Linear-interpolate dense arm qpos for the original demo frames.
          5. Smooth hand qpos and clamp per-frame deltas, then write hand
             joints back into the dense arm sequence.

        Args:
            T_obj: 4x4 object pose in world.
            wrist_rel_seq: (N, 4, 4) wrist SE(3) in object frame, with any
                Z-rotation already applied. Last waypoint is the grasp.
            hand_qpos_seq: (N, 6) demo hand qpos in HAND_ACTUATED order.
            q_last: cuRobo-ordered qpos for the FINAL waypoint, already solved
                via the main IK path. Used to seed the backward chain so the
                grasp pose is preserved exactly.
            arm_idx: cuRobo joint indices for the 7 arm joints.
            hand_idx: cuRobo joint indices for the 6 actuated hand joints.

        Returns dict with:
          - "arm_qpos_seq": (N, n_dof) cuRobo-ordered, hand overlaid; or None
            on chain failure.
          - "fail_frame": int or None.
        """
        wrist_rel = _smooth_wrist_rel_sequence(
            np.asarray(wrist_rel_seq, dtype=np.float64),
            DEMO_SMOOTH_WRIST_POS_WINDOW, DEMO_SMOOTH_WRIST_ROT_WINDOW,
        )
        hand_raw = np.asarray(hand_qpos_seq, dtype=np.float64)
        hand = _moving_average_reflect(hand_raw, DEMO_SMOOTH_HAND_WINDOW)
        # Lock endpoints so first/last frame match captured pre-grasp / grasp.
        hand[0] = hand_raw[0]
        hand[-1] = hand_raw[-1]
        hand = _limit_joint_steps(
            hand, list(range(hand.shape[1])), DEMO_MAX_HAND_STEP,
        )
        N = len(wrist_rel)
        key_stride = max(1, int(math.ceil(
            float(keyframe_raw_stride) / max(1, int(source_stride))
        )))
        key_idx = _keyframe_indices(N, key_stride)
        K = len(key_idx)
        arm_key_seq = [None] * K
        arm_key_seq[K - 1] = np.asarray(q_last, dtype=np.float32)
        fail_frame = None
        # Backward chain over keyframes only.
        for ki in range(K - 2, -1, -1):
            frame_i = int(key_idx[ki])
            target_T = T_obj @ wrist_rel[frame_i]
            ok_k, q_k = self._solve_ik_demo_chain(target_T, arm_key_seq[ki + 1])
            if not ok_k:
                fail_frame = frame_i
                return {"arm_qpos_seq": None, "fail_frame": fail_frame,
                        "n_key": K, "key_stride": key_stride}
            arm_key_seq[ki] = np.asarray(q_k, dtype=np.float32)
        arm_key_arr = np.stack(arm_key_seq, axis=0).astype(np.float32)
        arm_seq_arr = _linear_interpolate_q(key_idx, arm_key_arr, N)
        arm_seq_arr = _limit_joint_steps(arm_seq_arr, arm_idx, DEMO_MAX_ARM_STEP)
        arm_seq_arr = _smooth_joint_sequence(
            arm_seq_arr, arm_idx, DEMO_SMOOTH_JOINT_WINDOW,
        )
        # Overlay smoothed hand qpos (open → closed) onto the dense arm seq.
        for k in range(N):
            for h, ci in enumerate(hand_idx):
                arm_seq_arr[k, ci] = float(hand[k, h])
        return {"arm_qpos_seq": arm_seq_arr, "fail_frame": None,
                "n_key": K, "key_stride": key_stride}

    def plan_js(self, q_start: np.ndarray, q_goal: np.ndarray,
                exclude_obstacle: str | None = None):
        """Joint-space motion plan from q_start → q_goal (curobo joint order).

        Returns interpolated trajectory (T, n_dof) in curobo joint order, or
        None on failure.

        `exclude_obstacle`: name of a mesh in `_world_meshes` to temporarily
        remove from motion_gen for this call only (restored on return). Use
        when planning AFTER a grasp — the grasped target moves with the hand,
        so it must not be a fixed obstacle in the world during lift/place.
        """
        self._ensure_motion_gen()
        start_t = torch.tensor(q_start.reshape(1, -1), dtype=self.tensor_args.dtype,
                               device=self.tensor_args.device)
        goal_t = torch.tensor(q_goal.reshape(1, -1), dtype=self.tensor_args.dtype,
                              device=self.tensor_args.device)
        start = JointState.from_position(start_t)
        goal = JointState.from_position(goal_t)
        swap_back = (exclude_obstacle is not None
                     and exclude_obstacle in self._world_meshes)
        if swap_back:
            reduced = WorldConfig(mesh=[m for k, m in self._world_meshes.items()
                                        if k != exclude_obstacle])
            with self._lock:
                self._motion_gen.clear_world_cache()
                self._motion_gen.update_world(reduced)
        try:
            with self._lock:
                result = self._motion_gen.plan_single_js(
                    start_state=start, goal_state=goal, plan_config=self._plan_cfg,
                )
        finally:
            if swap_back:
                with self._lock:
                    self._motion_gen.clear_world_cache()
                    self._motion_gen.update_world(self._motion_gen_world)
        if not bool(result.success.item()):
            # Surface cuRobo's diagnostic so the user can tell whether it's
            # a collision-at-start/goal vs a trajopt convergence failure
            # vs IK seed problem etc.
            status = getattr(result, "status", None)
            valid = getattr(result, "valid_query", None)
            print(f"[plan_js] FAIL  status={status}  valid_query={valid}")
            self._last_plan_status = str(status)
            return None
        self._last_plan_status = "OK"
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

    def export_collision_debug(self, q: np.ndarray, label: str,
                               out_dir: str = str(HERE / "collision_debug")):
        """Export robot collision spheres (at q) and ALL world meshes/cuboids
        as OBJ files for MeshLab inspection. Three sources are dumped so the
        user can compare what each checker sees:

          {label}_spheres_main.obj      — main RobotWorld kinematics
          {label}_spheres_motiongen.obj — motion_gen's kinematics
          world_mesh_{name}.obj         — every mesh in motion_gen.world_model
          world_cube_{name}.obj         — every cuboid in motion_gen.world_model

        Open all together in MeshLab and look for spheres clipping any mesh.
        """
        import os
        from scipy.spatial.transform import Rotation as Rot
        os.makedirs(out_dir, exist_ok=True)
        q_t = torch.tensor(q.reshape(1, -1), dtype=self.tensor_args.dtype,
                           device=self.tensor_args.device)

        def _save_spheres(state, path):
            x = state.link_spheres_tensor.view(-1, 4).detach().cpu().numpy()
            meshes = []
            for pos, rad in zip(x[:, :3], x[:, 3]):
                if rad <= 0:
                    continue
                m = trimesh.creation.icosphere(radius=float(rad), subdivisions=1)
                m.apply_translation(pos)
                meshes.append(m)
            if meshes:
                trimesh.util.concatenate(meshes).export(path)
                print(f"[debug] {path} ({len(meshes)} spheres)")

        # 1) main rw kinematics spheres
        with self._lock:
            try:
                _save_spheres(self.rw.get_kinematics(q_t),
                              os.path.join(out_dir, f"{label}_spheres_main.obj"))
            except Exception as e:
                print(f"[debug] main spheres export failed: {e}")
        # 2) motion_gen kinematics spheres (the ones plan_single_js actually checks)
        try:
            self._ensure_motion_gen()
            mg_kin = self._motion_gen.kinematics
            with self._lock:
                _save_spheres(mg_kin.compute_kinematics(JointState.from_position(q_t)),
                              os.path.join(out_dir, f"{label}_spheres_motiongen.obj"))
        except Exception as e:
            print(f"[debug] motion_gen spheres export failed: {e}")
        # 3) Dump motion_gen's world (meshes + cuboids) — once per call to refresh
        try:
            wm = getattr(self._motion_gen, "world_model", None)
            if wm is not None:
                for m in (getattr(wm, "mesh", None) or []):
                    name = getattr(m, "name", "mesh")
                    pose = np.asarray(getattr(m, "pose",
                                              [0, 0, 0, 1, 0, 0, 0]) or [0, 0, 0, 1, 0, 0, 0])
                    fp = getattr(m, "file_path", None)
                    verts, faces = m.vertices, m.faces
                    if (verts is None or faces is None) and fp:
                        tm = trimesh.load(fp, force="mesh", process=False)
                    else:
                        if hasattr(verts, "cpu"): verts = verts.cpu().numpy()
                        if hasattr(faces, "cpu"): faces = faces.cpu().numpy()
                        tm = trimesh.Trimesh(vertices=np.asarray(verts),
                                             faces=np.asarray(faces))
                    T = np.eye(4)
                    T[:3, 3] = pose[:3]
                    T[:3, :3] = Rot.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
                    tm.apply_transform(T)
                    p = os.path.join(out_dir, f"world_mesh_{name}.obj")
                    tm.export(p)
                    print(f"[debug] {p}")
                for c in (getattr(wm, "cuboid", None) or []):
                    name = getattr(c, "name", "cube")
                    dims = np.asarray(c.dims)
                    pose = np.asarray(c.pose)
                    box = trimesh.creation.box(extents=dims)
                    T = np.eye(4)
                    T[:3, 3] = pose[:3]
                    T[:3, :3] = Rot.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
                    box.apply_transform(T)
                    p = os.path.join(out_dir, f"world_cube_{name}.obj")
                    box.export(p)
                    print(f"[debug] {p}")
        except Exception as e:
            print(f"[debug] world export failed: {e}")


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
    def __init__(self, server: viser.ViserServer, urdf_path: Path,
                 root: str = "/robot", base_pose: np.ndarray | None = None):
        """If `base_pose` (4x4) is provided, the root frame is positioned at
        that pose so the entire URDF tree renders offset from world origin.
        Used for the seoja robot which sits at +1.5m from jangja (which is at
        viewer origin)."""
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
        if base_pose is not None:
            pos, wxyz = mat_to_pos_wxyz(np.asarray(base_pose, dtype=np.float64))
            self.root_handle = server.scene.add_frame(
                root, show_axes=False, position=pos, wxyz=wxyz,
            )
        else:
            self.root_handle = server.scene.add_frame(root, show_axes=False)

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
        _t0 = time.perf_counter()
        def _step(label):
            nonlocal _t0
            now = time.perf_counter()
            print(f"[init {now - _t0:5.2f}s] {label}", flush=True)
            _t0 = now

        _step("start")
        self.server = viser.ViserServer(port=port)
        self.server.gui.configure_theme(dark_mode=True)
        self.server.scene.set_up_direction((0, 0, 1))
        self.server.scene.add_grid("/floor", width=2.0, height=2.0, plane="xy",
                                   position=(0, 0, 0), cell_size=0.1)
        _step("viser server up")

        self.viser_urdf = ViserURDF(self.server, URDF_PATH, root="/robot_jangja")
        _step("ViserURDF (jangja) loaded")
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

        # Stage A: render the seoja (FR3 + inspire_f1) URDF as a visual-only
        # second robot at its handeye-derived base pose in jangja frame. No
        # cuRobo for seoja yet — Stage B. Seoja keeps its own last-commanded
        # qpos (init at retract pose).
        T_seoja_in_jangja = self._compute_seoja_base_pose()
        self.seoja_base_pose = T_seoja_in_jangja
        if T_seoja_in_jangja is not None and SEOJA_URDF_PATH.exists():
            self.viser_urdf_seoja = ViserURDF(
                self.server, SEOJA_URDF_PATH, root="/robot_seoja",
                base_pose=T_seoja_in_jangja,
            )
            self.actuated_seoja = list(self.viser_urdf_seoja.urdf.actuated_joint_names)
            self.q_actuated_seoja = np.zeros(len(self.actuated_seoja))
            for jn, val in zip(ARM_JOINTS, ARM_HOME):
                if jn in self.actuated_seoja:
                    self.q_actuated_seoja[self.actuated_seoja.index(jn)] = val
            self.viser_urdf_seoja.update_cfg(self.q_actuated_seoja)
            _step("ViserURDF (seoja) loaded")
        else:
            self.viser_urdf_seoja = None
            self.actuated_seoja = []
            self.q_actuated_seoja = np.zeros(0)
            print(f"[init] seoja URDF not loaded (path exists: "
                  f"{SEOJA_URDF_PATH.exists()}, base pose: "
                  f"{T_seoja_in_jangja is not None})")

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
        # Auto-restore the scene on startup. Prefer the saved `Save object
        # poses` snapshot (traj/object_poses.npz) since it's the explicit
        # last-known-good scene; fall back to the latest tracking_*.json
        # only if no snapshot exists.
        snapshot = TRAJ_DIR / "object_poses.npz"
        if snapshot.exists():
            try:
                self._load_object_poses()
                self._refresh_grasp_objects()
            except Exception as e:
                print(f"[init] auto-load object_poses.npz failed: {e}")
        elif self.tracking_files:
            try:
                self._load_tracking_json(self.tracking_files[0])
                self._refresh_grasp_objects()
            except Exception as e:
                print(f"[init] auto-load tracking failed: {e}")

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
    def _compute_seoja_base_pose(self) -> np.ndarray | None:
        """Return the 4x4 SE(3) pose of seoja base in jangja (= world) frame.

        T_seoja_in_jangja = inv(Z_jangja) @ Z_seoja, where Z_* is the
        camera_T_robot transform from each robot's handeye result.
        """
        try:
            Zj = np.asarray(joblib.load(HANDEYE_PATH_JANGJA)["Z"], dtype=np.float64)
            Zs = np.asarray(joblib.load(HANDEYE_PATH_SEOJA)["Z"], dtype=np.float64)
        except Exception as e:
            print(f"[seoja] failed to load handeye files: {e}")
            return None
        T = np.linalg.inv(Zj) @ Zs
        print(f"[seoja] base pose in jangja frame: t={T[:3,3].round(4).tolist()}")
        return T

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
            srv.add_button("Load object poses").on_click(lambda _: self._load_object_poses())
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
            # Post-grasp lift mode. Two presets:
            #   pocket OFF (default): +z 10cm only
            #   pocket ON:            +z 15cm → +y 20cm (clears shelf/box)
            # The dict in LIFT_OFFSETS is no longer consulted; this
            # checkbox is the single source of truth at plan time.
            self.pocket_mode_chk = srv.add_checkbox(
                "Pocket mode (+z 15cm → +y 20cm)", False,
            )
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

    def _load_object_poses(self):
        """Restore the scene from `traj/object_poses.npz` (saved earlier
        via "Save object poses"). Re-creates each instance via _add_object
        and applies the stored 4×4 pose. Skips entries whose mesh dir is
        missing.
        """
        path = TRAJ_DIR / "object_poses.npz"
        if not path.exists():
            self.object_poses_status.value = f"missing {path.name}"
            return
        d = np.load(path, allow_pickle=True)
        instances = [str(x) for x in d["instances"]]
        bases = [str(x) for x in d["base_names"]]
        Ts = np.asarray(d["T"], dtype=np.float64)
        added = 0
        for inst, base, T in zip(instances, bases, Ts):
            if base not in self.object_names:
                print(f"[objects] skip {inst!r}: base {base!r} not in object_names")
                continue
            # _add_object generates a unique instance id; if the saved
            # name was e.g. "paperCup_2", we still want that exact label
            # so per-object planning data lines up. _add_object dedups by
            # appending "_N", so adding twice for the same base picks up
            # _2, _3, ... naturally if loaded in saved order.
            new_inst = self._add_object(base)
            if new_inst is None:
                continue
            self._set_object_pose(new_inst, T)
            added += 1
        self.object_poses_status.value = f"loaded {added}/{len(instances)} from {path.name}"
        print(f"[objects] loaded {added}/{len(instances)} from {path}")

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
                sz = srv.add_slider("z", min=-0.5, max=1.5, step=0.005, initial_value=float(T[2, 3]))
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
        self._refresh_grasp_objects()
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
            self._refresh_grasp_objects()
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
            # chocoSong-i: tracking rotation is unreliable (Z-axis comes
            # out flipped) — drop rotation and force identity. Z position
            # is also off (~10-20cm too high); user adjusts via the per-
            # object z slider (range widened to -0.5..1.5).
            if base == "chocoSong-i" or mesh_dir.name == "chocoSong-i":
                T_pos_only = np.eye(4)
                T_pos_only[:3, 3] = T_obj[:3, 3]
                T_obj = T_pos_only
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
        print(f"[grasp] instance='{instance}'  base='{base}'  "
              f"in_object_names={base in self.object_names}  "
              f"CANDIDATE_ROOT={CANDIDATE_ROOT}  exists={CANDIDATE_ROOT.is_dir()}")
        cands = load_candidates_for_obj(base)
        if not cands:
            self.grasp_stats.value = f"no candidates for '{base}'"
            return

        # Y-axis sweep: cylindrical-symmetric objects (paperCup, Jp_Water,
        # ...) hide the spin angle from perception, so generate 6 copies of
        # every grasp at 0/60/120/180/240/300° around the object's local Y.
        # Asymmetric objects (crushed cans, etc.) keep one orientation —
        # Y-rotated copies would be physically wrong grasps for them.
        base_norm = base.replace("_", "").lower()
        if any(base_norm == x.replace("_", "").lower() for x in Y_SYMMETRIC_OBJECTS):
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

        # Try disk cache first — IK + collision is the slow step (~18s/100
        # poses) and results only depend on (base, T_obj).
        cached = try_load_grasp_cache(base, T_obj)
        if cached is not None:
            cached.sort(key=lambda c: {"success": 0, "collision": 1,
                                        "ik_fail": 2}[c["status"]])
            self._candidates = cached
            n_succ = sum(1 for c in cached if c["status"] == "success")
            n_coll = sum(1 for c in cached if c["status"] == "collision")
            n_fail = sum(1 for c in cached if c["status"] == "ik_fail")
            self.grasp_stats.value = (
                f"object={base}  total={len(cached)}  "
                f"success={n_succ}  collision={n_coll}  ik_fail={n_fail}  "
                f"(from cache)"
            )
            self.grasp_slider.max = max(1, len(cached) - 1)
            self.grasp_slider.value = 0
            self._show_candidate(0)
            print(f"[grasp] cache HIT for {base} ({len(cached)} candidates)")
            return

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

        # Demo candidates have grasp orientations far from pick_start. The
        # cuda-graph-cached `_grasp_ik_solver` has strong null_space_weight
        # pulling toward retract_q, so for demo we instead use the
        # non-cuda-graph `self.ik_solver` (which has null_space_weight = 0
        # by default → no bias) one-at-a-time. BODex still uses the fast
        # batched path biased by pick_start.
        demo_mask = np.array([("demo_wrist_rel_seq" in c) for c in cands], dtype=bool)
        succ_arr = np.zeros(len(cands), dtype=bool)
        q_arr = np.zeros((len(cands), len(self.curobo.curobo_joint_names)),
                         dtype=np.float32)
        t0 = time.perf_counter()
        if (~demo_mask).any():
            try:
                ss, qq = self.curobo.solve_ik_batch(
                    target_Ts[~demo_mask], retract_q=retract_q,
                )
                succ_arr[~demo_mask] = ss
                q_arr[~demo_mask] = qq
            except Exception as e:
                print(f"[grasp] batch IK error (bodex): {e}")
        for i in np.where(demo_mask)[0]:
            try:
                ok_i, q_i = self.curobo.solve_ik(target_Ts[i], retract_q=None)
                succ_arr[i] = bool(ok_i)
                q_arr[i] = q_i
            except Exception as e:
                print(f"[grasp] single IK error (demo {i}): {e}")
        print(f"[grasp] IK ({len(cands)} poses) "
              f"ok={int(succ_arr.sum())} in {time.perf_counter() - t0:.2f}s")

        # Collision check against the scene-only world (table/floor) plus
        # self-collision. Object meshes are intentionally excluded — the
        # wrist target is on the object, so the dynamic object would
        # always trigger a false positive there.
        # Pre-compute curobo indices for the hand joints we override below.
        hand_idx_in_curobo = [
            self.curobo.curobo_joint_names.index(jn)
            for jn in FloatingHand.HAND_ACTUATED
        ]
        for i, c in enumerate(tqdm(cands, desc=f"[grasp] coll {base}", unit="grasp")):
            ok = bool(succ_arr[i])
            c["ik_success"] = ok
            c["ik_q"] = np.asarray(q_arr[i]) if ok else None
            in_coll = False
            # Demo (contact-based) grasps: skip scene + self collision —
            # the captured trajectory legitimately makes contact with the
            # table/object, so collision flagging would always reject.
            is_demo = "demo_wrist_rel_seq" in c
            if ok and not is_demo:
                try:
                    # Check the SAME q that plan_js will see — IK keeps the
                    # solver's hand seed (~retract qpos), but we override hand
                    # joints with `pregrasp` before planning. Different hand
                    # config → different sphere positions → different
                    # collision answer. Match motion_gen by overriding here too.
                    q_check = np.asarray(c["ik_q"], dtype=np.float64).copy()
                    for k, ci in enumerate(hand_idx_in_curobo):
                        q_check[ci] = float(c["pregrasp"][k])
                    w_coll = self.curobo.world_collide_scene_from_q(q_check)
                    x_sph = self.curobo.query(q_check)
                    s_coll = self.curobo.self_collide(x_sph)
                    in_coll = bool(w_coll.any() or s_coll.any())
                except Exception as e:
                    print(f"[grasp] collision error: {e}")
            c["in_collision"] = in_coll
            if not ok:
                c["status"] = "ik_fail"
            elif in_coll:
                c["status"] = "collision"
            else:
                c["status"] = "success"

            # Demo trajectory: keyframe-stride backward IK + linear
            # interpolation, smoothing, hand qpos overlay. The solver used
            # here is `_demo_chain_ik_solver` — scene-mesh-only world
            # (dynamic target excluded), inspire collision spheres stripped
            # so only fr3 links collide, arm-only null_space_weight for
            # continuity. The captured demo legitimately contacts the
            # target/table with the hand, so excluding the target object and
            # disabling hand collision here is intentional. Approach
            # planning (pick_start → first waypoint) still uses full
            # collision, so that motion is checked normally.
            if c.get("status") == "success" and "demo_wrist_rel_seq" in c:
                seq_rel = np.asarray(c["demo_wrist_rel_seq"])
                seq_hand = np.asarray(c["demo_hand_qpos_seq"])
                arm_idx = [self.curobo.curobo_joint_names.index(jn)
                           for jn in ARM_JOINTS]
                hand_idx = [self.curobo.curobo_joint_names.index(jn)
                            for jn in FloatingHand.HAND_ACTUATED]
                source_stride = int(c.get("demo_source_stride", 1))
                chain = self.curobo.solve_demo_chain(
                    T_obj=T_obj,
                    wrist_rel_seq=seq_rel,
                    hand_qpos_seq=seq_hand,
                    q_last=np.asarray(c["ik_q"], dtype=np.float32),
                    arm_idx=arm_idx,
                    hand_idx=hand_idx,
                    source_stride=source_stride,
                )
                if chain["arm_qpos_seq"] is not None:
                    c["demo_arm_qpos_seq"] = chain["arm_qpos_seq"]
                    c["demo_seq_in_collision"] = False
                    c["demo_n_key"] = chain["n_key"]
                    c["demo_key_stride"] = chain["key_stride"]
                else:
                    c["demo_arm_qpos_seq"] = None
                    c["demo_chain_fail_frame"] = chain["fail_frame"]
                    c["status"] = "ik_fail"

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
        try:
            save_grasp_cache(base, T_obj, cands)
        except Exception as e:
            print(f"[grasp cache] save failed: {e}")

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
        # Resolve target object name (instance + canonical base) up-front so
        # later stages can use them for LIFT_OFFSETS lookup and obstacle
        # exclusion without re-deriving.
        instance = self.grasp_obj_dropdown.value
        base = instance
        while base not in self.object_names and "_" in base:
            base = base.rsplit("_", 1)[0]
        # Goal qpos (curobo joint order)
        q_goal_curobo = c["ik_q"].copy()
        # Override hand joints with pregrasp (so we keep arm IK + open hand)
        for k, jn in enumerate(FloatingHand.HAND_ACTUATED):
            ci = self.curobo.curobo_joint_names.index(jn)
            q_goal_curobo[ci] = float(c["pregrasp"][k])

        # Demo candidates ship a full per-waypoint qpos sequence
        # (chocoSong-i etc.). Insert it between the approach plan and the
        # lift so playback shows pick_start → first waypoint → captured
        # demo (open→closed hand) → lift. Non-demo candidates plan straight
        # to the grasp pose as before.
        demo_seq = c.get("demo_arm_qpos_seq")
        if demo_seq is not None:
            demo_seq = np.asarray(demo_seq, dtype=np.float64)
            q_pg_goal = demo_seq[0].copy()
            # Last demo waypoint already has the closed grasp hand qpos
            # baked in by solve_demo_chain — use it directly as the post-
            # grasp / lift starting state (no extra hand override).
            q_grasp_curobo = demo_seq[-1].copy()
        else:
            q_pg_goal = q_goal_curobo
            q_grasp_curobo = q_goal_curobo

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

        # Stage 1: pick_start → grasp_qpos
        # Diagnostic: query the SAME q against four checkers and see which
        # one disagrees. The fourth uses motion_gen's OWN world_coll_checker
        # (the one plan_single_js actually uses for the validity check) so we
        # can see whether motion_gen is looking at a different world or
        # different spheres than our two RobotWorld instances.
        try:
            self.curobo._ensure_motion_gen()
            mg_wcc = self.curobo._motion_gen.world_coll_checker
        except Exception as e:
            mg_wcc = None
            print(f"[plan] motion_gen world checker access error: {e}")
        for label, q in [("start", q_start_curobo),
                         ("end", q_pg_goal)]:
            try:
                w_grasp = self.curobo.world_collide_scene_from_q(q)
                x_sph = self.curobo.query(q)
                w_main = self.curobo.world_collide(x_sph)
                s = self.curobo.self_collide(x_sph)
                w_mg_str = "n/a"
                if mg_wcc is not None:
                    x_sph_t = torch.tensor(x_sph.reshape(1, 1, -1, 4),
                                           dtype=self.curobo.tensor_args.dtype,
                                           device=self.curobo.tensor_args.device)
                    buf = CollisionQueryBuffer.initialize_from_shape(
                        x_sph_t.shape, self.curobo.tensor_args, mg_wcc.collision_types,
                    )
                    d = mg_wcc.get_sphere_distance(
                        x_sph_t, buf,
                        self.curobo._sphere_weight, self.curobo._sphere_act,
                        sum_collisions=False,
                    )
                    w_mg = (d.view(-1).detach().cpu().numpy() > 0)
                    w_mg_str = f"{int(w_mg.sum())}/{w_mg.size}"
                print(f"[plan] {label}: "
                      f"_grasp_rw={int(w_grasp.sum())}/{w_grasp.size}  "
                      f"main_rw={int(w_main.sum())}/{w_main.size}  "
                      f"motion_gen={w_mg_str}  "
                      f"self={int(s.sum())}/{s.size}")
            except Exception as e:
                print(f"[plan] {label} check error: {e}")
        try:
            traj_pg = self.curobo.plan_js(q_start_curobo, q_pg_goal)
        except Exception as e:
            self.plan_status.value = f"plan error: {e}"
            print(f"[plan] {e}")
            return
        if traj_pg is None:
            status = getattr(self.curobo, "_last_plan_status", None) or "?"
            self.plan_status.value = f"plan FAIL (pick→grasp): {status}"
            return

        # Stage 2: post-grasp lift sequence. Each entry is a Cartesian
        # world-frame offset applied on top of the previous wrist pose. We
        # IK each waypoint and JOINT-SPACE LINEAR INTERPOLATE between
        # consecutive qpos — these motions are short and the target is
        # already in the hand, so a straight joint interp is faster and
        # more robust than re-running trajopt for each stage.
        # Mode is driven by the "Pocket mode" checkbox (single source of
        # truth):
        #   OFF: [+z 10cm]                           (default)
        #   ON:  [+z 15cm, +y 20cm]                  (clears shelf/box)
        if bool(self.pocket_mode_chk.value):
            offsets = [(0.0, 0.0, 0.15), (0.0, 0.20, 0.0)]
        else:
            offsets = [(0.0, 0.0, 0.10)]
        T_cur = c["wrist_world"].copy()
        q_cur = q_grasp_curobo.copy()
        traj_lift_segments = []
        n_steps_per_stage = 30
        alphas = np.linspace(0.0, 1.0, n_steps_per_stage)
        for stage_i, (dx, dy, dz) in enumerate(offsets, start=1):
            T_cur = T_cur.copy()
            T_cur[0, 3] += dx
            T_cur[1, 3] += dy
            T_cur[2, 3] += dz
            try:
                ok_lift, q_next = self.curobo.solve_ik(T_cur, retract_q=q_cur)
            except Exception as e:
                print(f"[plan] lift stage {stage_i} IK error: {e}")
                ok_lift, q_next = False, None
            if not ok_lift or q_next is None:
                self.plan_status.value = f"lift IK FAIL (stage {stage_i})"
                return
            q_next = np.asarray(q_next, dtype=np.float64).copy()
            for jn in FloatingHand.HAND_ACTUATED:
                ci = self.curobo.curobo_joint_names.index(jn)
                q_next[ci] = float(q_grasp_curobo[ci])
            seg = (1 - alphas)[:, None] * q_cur[None, :] + alphas[:, None] * q_next[None, :]
            traj_lift_segments.append(seg)
            q_cur = q_next
        traj_lift = np.concatenate(traj_lift_segments, axis=0)
        q_lift = q_cur

        # Stage 3: lift_qpos → place_start_qpos (cuRobo plan_js).
        wp_ps = self._load_waypoint("place_start")
        if wp_ps is None:
            self.plan_status.value = "missing place_start waypoint"
            return
        q_place_curobo = np.zeros(len(self.curobo.curobo_joint_names))
        for jn, v in zip(wp_ps["actuated"], wp_ps["q_actuated"]):
            if jn in self.curobo.curobo_joint_names:
                q_place_curobo[self.curobo.curobo_joint_names.index(jn)] = float(v)
        # Carry the grasp hand pose into place_start (don't drop the object).
        for jn in FloatingHand.HAND_ACTUATED:
            ci = self.curobo.curobo_joint_names.index(jn)
            q_place_curobo[ci] = float(q_grasp_curobo[ci])
        # Target object is now in the hand — exclude it from the world so
        # motion_gen doesn't treat it as a fixed obstacle hand-overlapping
        # itself.
        instance = self.grasp_obj_dropdown.value
        # Diagnostics: q_lift and q_place_curobo against the SAME world
        # motion_gen will see (target object excluded). If trajopt fails we
        # need to know whether it's start/goal feasibility (collision /
        # self-collision) or the optimizer not finding a path.
        excl_name = f"obj::{instance}"
        excl_meshes = [m for k, m in self.curobo._world_meshes.items()
                       if k != excl_name]
        wc_lp = WorldConfig(mesh=excl_meshes)
        try:
            self.curobo._ensure_motion_gen()
            with self.curobo._lock:
                self.curobo._motion_gen.clear_world_cache()
                self.curobo._motion_gen.update_world(wc_lp)
            mg_wcc_lp = self.curobo._motion_gen.world_coll_checker
        except Exception as e:
            mg_wcc_lp = None
            print(f"[plan] lift→place world swap error: {e}")
        for label, q in [("lift_start", q_lift),
                         ("place_goal", q_place_curobo)]:
            try:
                x_sph = self.curobo.query(q)
                s = self.curobo.self_collide(x_sph)
                w_mg_str = "n/a"
                if mg_wcc_lp is not None:
                    x_sph_t = torch.tensor(x_sph.reshape(1, 1, -1, 4),
                                           dtype=self.curobo.tensor_args.dtype,
                                           device=self.curobo.tensor_args.device)
                    buf = CollisionQueryBuffer.initialize_from_shape(
                        x_sph_t.shape, self.curobo.tensor_args, mg_wcc_lp.collision_types,
                    )
                    d = mg_wcc_lp.get_sphere_distance(
                        x_sph_t, buf,
                        self.curobo._sphere_weight, self.curobo._sphere_act,
                        sum_collisions=False,
                    )
                    w_mg = (d.view(-1).detach().cpu().numpy() > 0)
                    w_mg_str = f"{int(w_mg.sum())}/{w_mg.size}"
                # Per-joint magnitude of motion to spot impossible plans.
                arm_idx_dbg = [self.curobo.curobo_joint_names.index(jn)
                               for jn in ARM_JOINTS]
                d_arm = np.degrees(q_place_curobo[arm_idx_dbg] - q_lift[arm_idx_dbg])
                d_arm_str = "  ".join(f"{v:+6.1f}°" for v in d_arm)
                print(f"[plan] lift→place {label}: "
                      f"motion_gen_world={w_mg_str}  "
                      f"self={int(s.sum())}/{s.size}  "
                      f"d_arm[deg]={d_arm_str}")
            except Exception as e:
                print(f"[plan] lift→place {label} check error: {e}")
        # Restore world cache (plan_js below will swap again, but it expects
        # the full world to start from).
        try:
            with self.curobo._lock:
                self.curobo._motion_gen.clear_world_cache()
                self.curobo._motion_gen.update_world(self.curobo._motion_gen_world)
        except Exception:
            pass
        try:
            traj_lp = self.curobo.plan_js(
                q_lift, q_place_curobo,
                exclude_obstacle=f"obj::{instance}",
            )
        except Exception as e:
            self.plan_status.value = f"plan error (lift→place): {e}"
            print(f"[plan] lift→place {e}")
            return
        if traj_lp is None:
            status = getattr(self.curobo, "_last_plan_status", None) or "?"
            self.plan_status.value = f"plan FAIL (lift→place): {status}"
            return

        # Concatenate. grasp_frame = index where the hand first reaches the
        # final grasp qpos (end of demo for demo candidates, end of approach
        # plan for non-demo). For demo we splice the captured trajectory
        # between traj_pg and traj_lift; we drop the duplicate first frame
        # since traj_pg already ends at demo_seq[0].
        if demo_seq is not None:
            traj_demo = demo_seq[1:].astype(traj_pg.dtype, copy=False)
            traj_pre_lift = np.concatenate([traj_pg, traj_demo], axis=0)
            grasp_frame = len(traj_pre_lift) - 1
            traj = np.concatenate([traj_pre_lift, traj_lift, traj_lp], axis=0)
            self._traj = traj
            self.plan_status.value = (
                f"plan OK (demo): pg={len(traj_pg)} demo={len(traj_demo)} "
                f"lift={len(traj_lift)} lp={len(traj_lp)} total={len(traj)}"
            )
        else:
            grasp_frame = len(traj_pg) - 1
            traj = np.concatenate([traj_pg, traj_lift, traj_lp], axis=0)
            self._traj = traj
            self.plan_status.value = (
                f"plan OK: pg={len(traj_pg)} lift={len(traj_lift)} "
                f"lp={len(traj_lp)} total={len(traj)}"
            )
        self.traj_slider.max = max(1, len(traj) - 1)
        self.traj_slider.value = 0
        self._on_traj_slider()

        # Multi-object trash flow save. Filename keeps the legacy
        # `_pick_start__to_grasp_` format (replay regex matches it), but the
        # npz now contains the concat trajectory through place_start; the
        # `grasp_frame` index tells replay where to attach the carried object.
        # `instance` / `base` already resolved at the top of _plan_to_grasp.
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
            grasp_frame=np.array(grasp_frame, dtype=np.int64),
            instance=np.array(instance, dtype=object),
            base_name=np.array(base, dtype=object),
            scene=np.array(c["scene"], dtype=object),
            scene_id=np.array(c["scene_id"], dtype=object),
            grasp_id=np.array(c["grasp_id"], dtype=object),
        )
        print(f"[plan] saved {out.name}  instance={instance}  "
              f"frames={len(traj)} (grasp_frame={grasp_frame})")

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
    # Register SIGINT / SIGTSTP at module top-level BEFORE viser or cuRobo
    # initialise — both spawn threads and cuRobo holds CUDA state, so if
    # we wait until inside App.run() to install the handler something else
    # may have already wrapped or replaced it. The handler calls
    # `os._exit(0)` directly, skipping Python exception unwinding so
    # bare-except clauses in third-party code can't swallow it.
    import os
    import signal
    def _on_sig(sig, frame):
        print(f"\n[viewer] signal {sig} — force exit", flush=True)
        os._exit(0)
    signal.signal(signal.SIGINT, _on_sig)
    signal.signal(signal.SIGTSTP, _on_sig)

    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()
    App(args.port).run()


if __name__ == "__main__":
    main()
