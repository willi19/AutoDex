"""Standalone planning test for chocoSong-i demo grasps + viser playback.

Strips everything from viewer.py except cuRobo IK + MotionGen so we can
debug demo trajectory feasibility in isolation:

- No scene mesh, no object collision (object placement is just a pose
  reference for the demo's relative wrist trajectory).
- Object fixed at OBJ_POS = (0.74, -0.215, -0.09).
- 7 subsampled choco trajectories × 6 Z-rotations (0/60/.../300°) = 42
  candidates. Per candidate:
      1. Backward-chain IK from grasp pose to first waypoint.
      2. cuRobo MotionGen plan from ARM_HOME → first waypoint qpos.
- Successful candidates' (plan + demo) trajectories are concatenated and
  shown in a viser scene with a dropdown + frame slider + play/pause.

Usage:
    /home/mingi/miniconda3/envs/mingi/bin/python \\
        src/validation/robothome/test_demo_plan.py [--port 8200]
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
import viser
import yourdfpy
from scipy.spatial.transform import Rotation as R
from viser.extras import ViserUrdf

PARADEX_ROOT = Path("/home/mingi/paradex")
if str(PARADEX_ROOT) not in sys.path:
    sys.path.insert(0, str(PARADEX_ROOT))

from curobo.geom.types import Cuboid, WorldConfig               # noqa: E402
from curobo.types.base import TensorDeviceType                  # noqa: E402
from curobo.types.math import Pose as CuroboPose                # noqa: E402
from curobo.util_file import (                                  # noqa: E402
    get_robot_configs_path, join_path, load_yaml,
)
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig  # noqa: E402
from curobo.wrap.reacher.motion_gen import (                    # noqa: E402
    MotionGen, MotionGenConfig, MotionGenPlanConfig,
)

HERE = Path(__file__).resolve().parent
SUB_DIR = HERE / "subsampled" / "choco"
URDF_PATH = HERE / "fr3_inspire_left.urdf"
ROBOT_CFG_NAME = "fr3_inspire_left.yml"
OBJ_POS = np.array([0.74, -0.215, -0.09], dtype=np.float64)
OBJ_NAME = "chocoSong-i"
OBJ_MESH = (Path.home() / "shared_data" / "AutoDex" / "object" / "robothome" /
            OBJ_NAME / "visual_mesh" / f"{OBJ_NAME}.obj")
Z_ANGLES_DEG = [0, 60, 120, 180, 240, 300]
ARM_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
                    dtype=np.float32)
ARM_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
HAND_DRIVERS = [
    "left_thumb_1_joint", "left_thumb_2_joint",
    "left_index_1_joint", "left_middle_1_joint",
    "left_ring_1_joint", "left_little_1_joint",
]
RAW_HAND_TO_URDF = [5, 4, 3, 2, 1, 0]


def _hand_qpos_urdf_order(d: np.lib.npyio.NpzFile) -> np.ndarray:
    hand = np.asarray(d["hand_qpos"], dtype=np.float64)
    order = str(d["hand_qpos_order"].item()) if "hand_qpos_order" in d.files else "raw"
    if order == "urdf":
        return hand
    if order == "raw":
        return hand[:, RAW_HAND_TO_URDF]
    raise ValueError(f"unknown hand_qpos_order={order!r}")


def _ik_pose(ik_solver: IKSolver, T: np.ndarray, retract_q: np.ndarray | None,
             tensor_args: TensorDeviceType):
    """Single-pose IK; returns (success, q in curobo joint order)."""
    qxyzw = R.from_matrix(T[:3, :3]).as_quat()
    pos_t = torch.tensor(T[:3, 3], device=tensor_args.device,
                         dtype=tensor_args.dtype).view(1, 3)
    quat_t = torch.tensor([qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]],
                          device=tensor_args.device,
                          dtype=tensor_args.dtype).view(1, 4)
    goal = CuroboPose(position=pos_t, quaternion=quat_t)
    retract_t = None
    if retract_q is not None:
        retract_t = torch.tensor(np.asarray(retract_q, dtype=np.float32).reshape(1, -1),
                                 device=tensor_args.device, dtype=tensor_args.dtype)
    res = ik_solver.solve_single(goal, retract_config=retract_t)
    succ = bool(res.success.view(-1)[0].item())
    q = res.solution.view(-1).detach().cpu().numpy()
    return succ, q


def _abs_target(T_obj: np.ndarray, Rz: np.ndarray, T_rel: np.ndarray) -> np.ndarray:
    """T_world = T_obj · Rz · T_rel  (object rotation = I, only translation)."""
    return T_obj @ Rz @ T_rel


def _odd_window(n: int, length: int) -> int:
    n = int(max(1, n))
    n = min(n, max(1, int(length)))
    if n % 2 == 0:
        n = max(1, n - 1)
    return n


def _moving_average_reflect(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge padding; preserves shape."""
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
    """Low-pass filter demo wrist SE(3) before IK.

    The captured wrist poses are already close, but millimeter-level pose noise
    can be amplified by a redundant arm near limits/singularities. Quaternion
    signs are made continuous before averaging so q/-q does not introduce a
    fake rotation jump.
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


def _limit_joint_steps(q_seq: np.ndarray, joint_idx: list[int],
                       max_step: float) -> np.ndarray:
    """Clamp per-frame arm joint jumps after IK branch selection.

    This is a continuity post-process for playback/debug trajectories. It can
    slightly move the wrist off the exact IK target, but avoids visually violent
    branch flips when single-pose IK changes null-space solution between
    adjacent waypoints.
    """
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


def _smooth_joint_sequence(q_seq: np.ndarray, joint_idx: list[int],
                           window: int) -> np.ndarray:
    out = np.asarray(q_seq, dtype=np.float32).copy()
    window = _odd_window(window, len(out))
    if window <= 1 or len(out) <= 2:
        return out
    idx = np.asarray(joint_idx, dtype=np.int64)
    smoothed = _moving_average_reflect(out[:, idx], window)
    out[:, idx] = smoothed
    # Keep endpoints exact so the MotionGen approach still joins the demo and
    # the final grasp pose remains the solved target.
    out[0, idx] = q_seq[0, idx]
    out[-1, idx] = q_seq[-1, idx]
    return out


def _joint_step_stats(q_seq: np.ndarray, joint_idx: list[int]) -> tuple[float, float]:
    if len(q_seq) <= 1:
        return 0.0, 0.0
    dq = np.abs(np.diff(q_seq[:, joint_idx], axis=0))
    per_frame = dq.max(axis=1)
    return float(np.degrees(per_frame.mean())), float(np.degrees(per_frame.max()))


def _keyframe_indices(length: int, stride: int) -> np.ndarray:
    """Sparse demo indices for IK; always include first and last frame."""
    if length <= 0:
        return np.zeros(0, dtype=np.int64)
    stride = max(1, int(stride))
    idx = list(range(0, length, stride))
    if idx[-1] != length - 1:
        idx.append(length - 1)
    return np.asarray(idx, dtype=np.int64)


def _linear_interpolate_q(key_idx: np.ndarray, key_q: np.ndarray, length: int) -> np.ndarray:
    """Fill dense q sequence by linear interpolation between sparse IK keyframes."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--smooth-wrist-window", type=int, default=7,
                    help="Odd moving-average window for demo wrist translation; 1 disables.")
    ap.add_argument("--smooth-wrist-rot-window", type=int, default=7,
                    help="Odd moving-average window for demo wrist rotation; 1 disables.")
    ap.add_argument("--ik-keyframe-raw-stride", type=int, default=30,
                    help="Run sparse IK about every N original capture frames.")
    ap.add_argument("--ik-keyframe-stride", type=int, default=None,
                    help="Override sparse IK stride in this subsampled npz frame index.")
    ap.add_argument("--smooth-joint-window", type=int, default=1,
                    help="Optional odd moving-average window after interpolation; 1 disables.")
    ap.add_argument("--max-joint-step", type=float, default=0.0,
                    help="Optional per-frame arm joint delta clamp in radians; <=0 disables.")
    ap.add_argument("--smooth-hand-window", type=int, default=7,
                    help="Odd moving-average window for demo hand qpos; 1 disables.")
    ap.add_argument("--max-hand-step", type=float, default=0.05,
                    help="Clamp per-frame hand joint delta in radians; <=0 disables.")
    args = ap.parse_args()

    print(f"[init] tensor args + cuRobo (this can take ~10s)")
    tensor_args = TensorDeviceType()
    cfg_dict = load_yaml(join_path(get_robot_configs_path(), ROBOT_CFG_NAME))
    # cuRobo requires at least one collision obstacle, so park a tiny cuboid
    # far from the workspace. Self-collision still applies as usual.
    dummy = Cuboid(name="dummy_far", pose=[10.0, 10.0, 10.0, 1, 0, 0, 0],
                   dims=[0.001, 0.001, 0.001])
    wc_empty = WorldConfig(cuboid=[dummy])

    # Make retract_config actually act as a continuity prior during the
    # backward IK chain. With all zeros, cuRobo may solve each nearby wrist
    # pose on a different null-space branch, which looks jittery in playback.
    # Keep hand weights at zero because hand joints are overwritten from the
    # captured demo qpos after IK.
    cspace = cfg_dict["robot_cfg"]["kinematics"]["cspace"]
    cspace["null_space_weight"] = [1.0] * len(ARM_JOINTS) + (
        [0.0] * (len(cspace["joint_names"]) - len(ARM_JOINTS))
    )

    ik_cfg = IKSolverConfig.load_from_robot_config(
        cfg_dict, wc_empty, tensor_args=tensor_args,
        num_seeds=32, use_cuda_graph=False,
        collision_activation_distance=0.005,
    )
    ik_solver = IKSolver(ik_cfg)
    curobo_joint_names = list(ik_solver.kinematics.joint_names)
    arm_idx = [curobo_joint_names.index(j) for j in ARM_JOINTS]

    print(f"[init] MotionGen warmup (~7s)")
    mg_cfg = MotionGenConfig.load_from_robot_config(
        cfg_dict, wc_empty, tensor_args=tensor_args,
        num_trajopt_seeds=32, num_graph_seeds=1, num_ik_seeds=32,
        use_cuda_graph=False, interpolation_dt=0.01,
        ik_opt_iters=200, grad_trajopt_iters=200, trajopt_tsteps=64,
        collision_activation_distance=0.0,
        collision_cache={"obb": 30, "mesh": 10},
    )
    motion_gen = MotionGen(mg_cfg)
    motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
    plan_cfg = MotionGenPlanConfig(
        enable_graph=False, enable_graph_attempt=None,
        enable_opt=True, max_attempts=20,
        enable_finetune_trajopt=True,
        num_trajopt_seeds=32, num_ik_seeds=32,
        timeout=60.0, parallel_finetune=True,
        check_start_validity=False,
    )
    print(f"[init] ready, dof={len(curobo_joint_names)}")

    # Build start state (ARM_HOME for arm joints, zeros for hand).
    n_dof = len(curobo_joint_names)
    q_home = np.zeros(n_dof, dtype=np.float32)
    for i, ji in enumerate(arm_idx):
        q_home[ji] = ARM_HOME[i]

    # Object at fixed pose (rotation = I).
    T_obj = np.eye(4)
    T_obj[:3, 3] = OBJ_POS

    files = sorted(SUB_DIR.glob("*.npz"))
    print(f"[run] {len(files)} subsampled files × {len(Z_ANGLES_DEG)} Z-angles "
          f"= {len(files) * len(Z_ANGLES_DEG)} candidates")
    print(f"[run] OBJ_POS = {OBJ_POS.tolist()}")

    n_cand = 0
    n_ik_chain_ok = 0
    n_plan_ok = 0
    fail_at = {"ik_last": 0, "ik_chain": 0, "plan": 0}
    summary = []
    successes = []  # list of {label, full_qpos (T_total, n_dof)} for viser
    t_total = time.perf_counter()

    # Pre-resolve URDF actuated indices to map our hand_qpos → cuRobo qpos.
    urdf = yourdfpy.URDF.load(
        str(URDF_PATH), mesh_dir=str(URDF_PATH.parent),
        build_collision_scene_graph=False, load_collision_meshes=False,
    )
    hand_idx_in_curobo = [curobo_joint_names.index(jn) for jn in HAND_DRIVERS]

    for f in files:
        d = np.load(f)
        wrist_rel_raw = d["wrist_rel_se3"]   # (N, 4, 4) in object frame
        wrist_rel = _smooth_wrist_rel_sequence(
            wrist_rel_raw, args.smooth_wrist_window, args.smooth_wrist_rot_window,
        )
        hand_raw = _hand_qpos_urdf_order(d)  # (N, 6) in HAND_DRIVERS order
        hand = _moving_average_reflect(hand_raw, args.smooth_hand_window)
        hand[0] = hand_raw[0]
        hand[-1] = hand_raw[-1]
        hand = _limit_joint_steps(hand, list(range(hand.shape[1])), args.max_hand_step)
        hand_step_before = _joint_step_stats(hand_raw, list(range(hand_raw.shape[1])))
        hand_step_after = _joint_step_stats(hand, list(range(hand.shape[1])))
        N = len(wrist_rel)
        source_start = int(d["start"]) if "start" in d.files else 0
        source_stride = int(d["stride"]) if "stride" in d.files else 1
        key_stride = (
            int(args.ik_keyframe_stride)
            if args.ik_keyframe_stride is not None
            else max(1, int(math.ceil(float(args.ik_keyframe_raw_stride) / source_stride)))
        )
        key_idx = _keyframe_indices(N, key_stride)
        K = len(key_idx)
        for ang_deg in Z_ANGLES_DEG:
            n_cand += 1
            ang = math.radians(ang_deg)
            cz, sz = math.cos(ang), math.sin(ang)
            Rz = np.eye(4)
            Rz[:3, :3] = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

            # 1) IK last waypoint (no retract bias).
            t_last = _abs_target(T_obj, Rz, wrist_rel[-1])
            t0 = time.perf_counter()
            ok_last, q_last = _ik_pose(ik_solver, t_last, None, tensor_args)
            dt_ik_last = time.perf_counter() - t0
            if not ok_last:
                fail_at["ik_last"] += 1
                line = f"  {f.stem} z={ang_deg:3d}°  IK_LAST_FAIL"
                summary.append(line); print(line, flush=True)
                continue

            # 2) Backward keyframe chain. Solve sparse IK targets only, then
            # linearly interpolate dense arm qpos for the original demo frames.
            arm_key_seq = [None] * K
            arm_key_seq[K - 1] = q_last.astype(np.float32)
            seq_ok = True
            t0 = time.perf_counter()
            fail_key = None
            for ki in range(K - 2, -1, -1):
                frame_i = int(key_idx[ki])
                t_k = _abs_target(T_obj, Rz, wrist_rel[frame_i])
                ok_k, q_k = _ik_pose(ik_solver, t_k, arm_key_seq[ki + 1], tensor_args)
                if not ok_k:
                    seq_ok = False
                    fail_key = frame_i
                    break
                arm_key_seq[ki] = q_k.astype(np.float32)
            dt_chain = time.perf_counter() - t0
            if not seq_ok:
                fail_at["ik_chain"] += 1
                line = f"  {f.stem} z={ang_deg:3d}°  IK_CHAIN_FAIL@frame{fail_key}"
                summary.append(line); print(line, flush=True)
                continue
            arm_key_arr = np.stack(arm_key_seq, axis=0).astype(np.float32)
            key_step_mean, key_step_max = _joint_step_stats(arm_key_arr, arm_idx)
            arm_seq_arr = _linear_interpolate_q(key_idx, arm_key_arr, N)
            step_mean_before, step_max_before = _joint_step_stats(arm_seq_arr, arm_idx)
            arm_seq_arr = _limit_joint_steps(arm_seq_arr, arm_idx, args.max_joint_step)
            arm_seq_arr = _smooth_joint_sequence(
                arm_seq_arr, arm_idx, args.smooth_joint_window,
            )
            step_mean_after, step_max_after = _joint_step_stats(arm_seq_arr, arm_idx)
            n_ik_chain_ok += 1

            # 3) Plan ARM_HOME → arm_seq[0].
            q_start_t = torch.tensor(q_home.reshape(1, -1),
                                     device=tensor_args.device,
                                     dtype=tensor_args.dtype)
            q_goal_t = torch.tensor(arm_seq_arr[0].astype(np.float32).reshape(1, -1),
                                    device=tensor_args.device,
                                    dtype=tensor_args.dtype)
            from curobo.types.state import JointState
            start_js = JointState.from_position(q_start_t, joint_names=curobo_joint_names)
            goal_js = JointState.from_position(q_goal_t, joint_names=curobo_joint_names)
            t0 = time.perf_counter()
            try:
                res = motion_gen.plan_single_js(start_js, goal_js, plan_cfg.clone())
            except Exception as e:
                res = None
                print(f"  plan err: {e}")
            dt_plan = time.perf_counter() - t0
            ok_plan = bool(res is not None and res.success.item())
            if ok_plan:
                n_plan_ok += 1
                line = (f"  {f.stem} z={ang_deg:3d}°  PLAN_OK   "
                        f"(ik_last={dt_ik_last*1e3:.0f}ms "
                        f"chain[{K}/{N}, stride={key_stride}]={dt_chain*1e3:.0f}ms "
                        f"plan={dt_plan*1e3:.0f}ms "
                        f"key_step={key_step_mean:.1f}/{key_step_max:.1f}° "
                        f"step={step_mean_before:.1f}/{step_max_before:.1f}°→"
                        f"{step_mean_after:.1f}/{step_max_after:.1f}° "
                        f"hand={hand_step_before[0]:.1f}/{hand_step_before[1]:.1f}°→"
                        f"{hand_step_after[0]:.1f}/{hand_step_after[1]:.1f}°)")
                summary.append(line); print(line, flush=True)
                # Build the full qpos sequence (plan trajectory + demo).
                # plan_single_js returns interpolated trajectory; concatenate
                # the demo waypoints (with per-frame hand_qpos overlaid) on
                # the end so playback shows approach + grasp.
                plan_q = res.get_interpolated_plan().position.detach().cpu().numpy()
                # Open hand during approach.
                for k_app in range(plan_q.shape[0]):
                    for h, ci in enumerate(hand_idx_in_curobo):
                        plan_q[k_app, ci] = float(hand[0, h])
                # Demo waypoints with per-frame hand qpos.
                demo_q = arm_seq_arr.copy()
                for k_dem in range(demo_q.shape[0]):
                    for h, ci in enumerate(hand_idx_in_curobo):
                        demo_q[k_dem, ci] = float(hand[k_dem, h])
                full_q = np.concatenate([plan_q, demo_q], axis=0)
                successes.append({
                    "label": f"{f.stem} z={ang_deg:03d}",
                    "full_q": full_q,
                    "hand_raw": hand_raw.astype(np.float32),
                    "hand": hand.astype(np.float32),
                    "source_start": int(source_start),
                    "source_stride": int(source_stride),
                    "n_plan": int(plan_q.shape[0]),
                    "n_demo": int(demo_q.shape[0]),
                    "n_key": int(K),
                    "key_stride": int(key_stride),
                })
            else:
                fail_at["plan"] += 1
                summary.append(f"  {f.stem} z={ang_deg:3d}°  PLAN_FAIL")

    dt_total = time.perf_counter() - t_total
    print(f"\n=== summary ({dt_total:.1f}s) ===")
    for line in summary:
        print(line)
    print(f"\ntotal {n_cand}  ik_chain_ok={n_ik_chain_ok}  plan_ok={n_plan_ok}  "
          f"fail: ik_last={fail_at['ik_last']} chain={fail_at['ik_chain']} plan={fail_at['plan']}")

    # ---------------- viser playback ----------------
    if not successes:
        print("[viser] no successful trajectories to play; exiting")
        return

    server = viser.ViserServer(port=args.port)
    print(f"[viser] http://localhost:{args.port}")

    viser_urdf = ViserUrdf(server, urdf, root_node_name="/robot")
    actuated = list(urdf.actuated_joint_names)
    # Map cuRobo qpos (T, n_dof) → URDF actuated cfg.
    cur_to_urdf = [curobo_joint_names.index(jn) if jn in curobo_joint_names else -1
                   for jn in actuated]

    # ChocoSong-i mesh at OBJ_POS (visualization only — no collision).
    if OBJ_MESH.exists():
        m = trimesh.load(str(OBJ_MESH), force="mesh", process=False)
        server.scene.add_mesh_simple(
            "/object",
            vertices=np.asarray(m.vertices, dtype=np.float32),
            faces=np.asarray(m.faces, dtype=np.int32),
            color=(220, 190, 100),
            position=tuple(OBJ_POS.astype(np.float32).tolist()),
        )

    labels = [s["label"] for s in successes]
    rec_dd = server.gui.add_dropdown("Candidate", labels, initial_value=labels[0])
    play_cb = server.gui.add_checkbox("Playing", initial_value=False)
    fps_sl = server.gui.add_slider("FPS", min=1, max=120, step=1, initial_value=30)
    frame_sl = server.gui.add_slider(
        "Frame", min=0, max=max(1, len(successes[0]["full_q"]) - 1),
        step=1, initial_value=0,
    )
    prev_btn = server.gui.add_button("Prev frame")
    next_btn = server.gui.add_button("Next frame")
    info_txt = server.gui.add_text("Phase", "—", disabled=True)
    arm_txt = server.gui.add_text("Arm q", "—", disabled=True)
    hand_txt = server.gui.add_text("Hand q", "—", disabled=True)
    source_hand_txt = server.gui.add_text("Source hand q", "—", disabled=True)

    state = {"data": successes[0], "idx": 0, "dirty": True}

    def set_frame(idx, update_slider=True):
        n = len(state["data"]["full_q"])
        state["idx"] = int(np.clip(idx, 0, n - 1))
        if update_slider:
            frame_sl.value = state["idx"]
        state["dirty"] = True

    def load(label):
        for s in successes:
            if s["label"] == label:
                state["data"] = s
                frame_sl.max = max(1, len(s["full_q"]) - 1)
                set_frame(0)
                return

    @rec_dd.on_update
    def _(_): load(rec_dd.value)

    @frame_sl.on_update
    def _(_):
        set_frame(frame_sl.value, update_slider=False)

    @prev_btn.on_click
    def _(_):
        play_cb.value = False
        set_frame(state["idx"] - 1)

    @next_btn.on_click
    def _(_):
        play_cb.value = False
        set_frame(state["idx"] + 1)

    def render(idx):
        s = state["data"]
        full_q = s["full_q"]
        idx = int(np.clip(idx, 0, len(full_q) - 1))
        cur = full_q[idx]
        # Build URDF actuated cfg from cuRobo qpos via cur_to_urdf mapping.
        cfg = np.zeros(len(actuated))
        for ai, ci in enumerate(cur_to_urdf):
            if ci >= 0:
                cfg[ai] = cur[ci]
        viser_urdf.update_cfg(cfg)
        if idx < s["n_plan"]:
            info_txt.value = f"approach[{idx}/{s['n_plan']}]"
            source_hand_txt.value = "approach uses first demo hand q"
        else:
            demo_i = idx - s["n_plan"]
            source_frame = s.get("source_start", 0) + demo_i * s.get("source_stride", 1)
            info_txt.value = (
                f"demo[{demo_i}/{s['n_demo']}]  "
                f"source_frame={source_frame}  "
                f"ik_keyframes={s.get('n_key', '?')} stride={s.get('key_stride', '?')}"
            )
            source_hand_txt.value = "  ".join(
                f"{jn}:{s['hand_raw'][demo_i, j]:+.3f}"
                for j, jn in enumerate(HAND_DRIVERS)
            )
        arm_txt.value = "  ".join(
            f"{jn}:{cur[curobo_joint_names.index(jn)]:+.3f}" for jn in ARM_JOINTS
        )
        hand_txt.value = "  ".join(
            f"{jn}:{cur[curobo_joint_names.index(jn)]:+.3f}" for jn in HAND_DRIVERS
        )

    last = time.time()
    frame_accum = 0.0
    while True:
        now = time.time(); dt = now - last; last = now
        if play_cb.value and len(state["data"]["full_q"]) > 1:
            frame_accum += dt * float(fps_sl.value)
            adv = int(frame_accum)
            if adv > 0:
                frame_accum -= adv
                state["idx"] = (state["idx"] + adv) % len(state["data"]["full_q"])
                frame_sl.value = state["idx"]
                state["dirty"] = True
        if state["dirty"]:
            render(state["idx"])
            state["dirty"] = False
        time.sleep(1.0 / 120.0)


if __name__ == "__main__":
    main()
