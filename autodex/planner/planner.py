import os
import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.spatial.transform import Rotation

import torch

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'

from curobo.util_file import load_yaml
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.geom.types import WorldConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.util.trajectory import InterpolateType

from autodex.utils.path import robot_configs_path, load_candidate
from autodex.utils.conversion import se32action, cart2se3
from autodex.utils.robot_config import (
    INIT_STATE, XARM_INIT, INSPIRE_INIT,
    ALLEGRO_LINK6_TO_WRIST, INSPIRE_LINK6_TO_WRIST,
)


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class PlanResult:
    success: bool
    traj: Optional[np.ndarray]        # (T, dof)
    wrist_se3: Optional[np.ndarray]   # (4, 4)
    pregrasp_pose: np.ndarray         # (16,) hand joints
    grasp_pose: np.ndarray            # (16,) hand joints
    scene_info: list
    timing: Optional[dict] = None     # per-stage timing breakdown


# ── cuRobo format conversion (private) ───────────────────────────────────────

def _se3_to_7vec(mat: np.ndarray) -> list:
    """4x4 SE3 -> [x, y, z, qx, qy, qz, qw]."""
    t = mat[:3, 3].tolist()
    q = Rotation.from_matrix(mat[:3, :3]).as_quat().tolist()
    return t + q


def _to_curobo_world(scene_cfg: dict) -> dict:
    """scene_cfg -> cuRobo WorldConfig dict. Poses are already 7D [x,y,z,qw,qx,qy,qz]."""
    cfg = {"cuboid": {}, "mesh": {}}
    for name, info in scene_cfg.get("cuboid", {}).items():
        cfg["cuboid"][name] = {
            "dims": info["dims"],
            "pose": info["pose"],
            "color": info.get("color", [0.5, 0.5, 0.5, 1.0]),
        }
    for name, info in scene_cfg.get("mesh", {}).items():
        cfg["mesh"][name] = {
            "pose": info["pose"],
            "file_path": info["file_path"],
        }
    return cfg


def _to_curobo_pose(poses_se3: np.ndarray, device) -> Pose:
    """(B, 4, 4) -> cuRobo Pose."""
    position = torch.tensor(poses_se3[:, :3, 3], dtype=torch.float32, device=device).contiguous()
    xyzw = Rotation.from_matrix(poses_se3[:, :3, :3]).as_quat()
    wxyz = torch.tensor(xyzw[:, [3, 0, 1, 2]], dtype=torch.float32, device=device).contiguous()
    return Pose(position=position, quaternion=wxyz)


# ── Planner ───────────────────────────────────────────────────────────────────

class GraspPlanner:
    """
    scene_cfg + grasp candidates -> collision-free trajectory.

    Usage:
        planner = GraspPlanner()
        result = planner.plan(scene_cfg, obj_name="bottle", grasp_version="v1")
        if result.success:
            execute(result.traj)
    """

    BATCH_SIZE = 50
    N_CUBOIDS = 30
    N_MESHES = 5

    HAND_CONFIGS = {
        "allegro": ("xarm_allegro.yml", "allegro_floating.yml", 0.01, 32, InterpolateType.CUBIC),
        "inspire": ("xarm_inspire.yml", "inspire_floating.yml", 0.002, 32, InterpolateType.LINEAR_CUDA),
    }

    def __init__(self, robot_cfg_path: Optional[str] = None, hand_cfg_path: Optional[str] = None,
                 hand: str = "allegro"):
        if robot_cfg_path is None:
            robot_file, hand_file, self._collision_act_dist, self._num_trajopt_seeds, self._interpolation_type = self.HAND_CONFIGS.get(hand, self.HAND_CONFIGS["allegro"])
            robot_cfg_path = os.path.join(robot_configs_path, robot_file)
            if hand_cfg_path is None:
                hand_cfg_path = os.path.join(robot_configs_path, hand_file)
        else:
            self._collision_act_dist = 0.01
            self._num_trajopt_seeds = 1024
            self._interpolation_type = InterpolateType.LINEAR_CUDA

        self._robot_cfg = load_yaml(robot_cfg_path)["robot_cfg"]
        self._hand_cfg = load_yaml(hand_cfg_path)["robot_cfg"]
        self._tensor_args = TensorDeviceType()
        self._motion_gen: Optional[MotionGen] = None
        self._plan_cfg: Optional[MotionGenPlanConfig] = None
        self._ik_solver: Optional[IKSolver] = None

        # Init state: same arm position for all hands, hand-specific finger init
        if hand == "inspire":
            self._init_state = np.concatenate([XARM_INIT, INSPIRE_INIT]).astype(np.float32)
            self._link6_to_wrist_rot = INSPIRE_LINK6_TO_WRIST[:3, :3]
        else:
            self._init_state = INIT_STATE.astype(np.float32)
            self._link6_to_wrist_rot = ALLEGRO_LINK6_TO_WRIST[:3, :3]

        # Precompute link6 y-axis in wrist frame for backward filter
        self._link6_y_in_wrist = np.linalg.inv(self._link6_to_wrist_rot) @ np.array([0, 1, 0])
        self._hand = hand

    # ── world setup ───────────────────────────────────────────────────────────

    def _init_motion_gen(self, world_cfg: dict):
        config = MotionGenConfig.load_from_robot_config(
            self._robot_cfg,
            WorldConfig.from_dict(world_cfg),
            self._tensor_args,
            num_trajopt_seeds=self._num_trajopt_seeds,
            num_graph_seeds=1,
            num_ik_seeds=32,
            use_cuda_graph=True,
            interpolation_dt=0.01,
            interpolation_type=self._interpolation_type,
            collision_cache={"obb": self.N_CUBOIDS, "mesh": self.N_MESHES},
            ik_opt_iters=200,
            grad_trajopt_iters=200,
            trajopt_tsteps=64,
            collision_activation_distance=self._collision_act_dist,
        )
        self._motion_gen = MotionGen(config)
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

    def _update_world(self, world_cfg: dict):
        self._motion_gen.clear_world_cache()
        self._motion_gen.update_world(WorldConfig.from_dict(world_cfg))

    # ── collision check ───────────────────────────────────────────────────────

    def _check_collision(self, world_cfg: dict, wrist_se3: np.ndarray, pregrasp: np.ndarray) -> np.ndarray:
        """bool array (N,): True = collides."""
        rw_config = RobotWorldConfig.load_from_config(
            self._hand_cfg,
            WorldConfig.from_dict(world_cfg),
            collision_activation_distance=0.0,
            tensor_args=self._tensor_args,
        )
        rw = RobotWorld(rw_config)
        n_dof = rw.kinematics.get_dof()

        # Build q vector matching robot's expected DOF
        if n_dof == len(pregrasp[0]) + 6:
            # Floating hand (e.g. allegro_floating): [x,y,z,roll,pitch,yaw] + joints
            q = np.array([se32action(w, g) for w, g in zip(wrist_se3, pregrasp)])
        else:
            # use_root_pose hand (e.g. inspire): [x,y,z,qw,qx,qy,qz] + joints
            from autodex.utils.conversion import se32cart
            q = np.array([np.concatenate([se32cart(w), g]) for w, g in zip(wrist_se3, pregrasp)])
            # If still doesn't match, just use joints
            if q.shape[1] != n_dof:
                q = pregrasp

        q_t = torch.tensor(q, dtype=torch.float32, device=self._tensor_args.device)
        d_world, d_self = rw.get_world_self_collision_distance_from_joints(q_t)
        world_coll = (d_world > 0).cpu().numpy()
        self_coll = (d_self > 0).cpu().numpy()
        return world_coll | self_coll

    # ── IK solver ─────────────────────────────────────────────────────────────

    def _init_ik_solver(self, world_cfg: dict):
        config = IKSolverConfig.load_from_robot_config(
            self._robot_cfg,
            WorldConfig.from_dict(world_cfg),
            self._tensor_args,
            num_seeds=32,
            collision_cache={"obb": self.N_CUBOIDS, "mesh": self.N_MESHES},
            collision_activation_distance=self._collision_act_dist,
        )
        self._ik_solver = IKSolver(config)

    def solve_ik(self, scene_cfg: dict, obj_name: str, grasp_version: str,
                 seed: Optional[int] = None, hand: str = "allegro"):
        """
        IK-only reachability check for all grasp candidates.

        Skips hand-object collision check (hand is supposed to be near the object).
        Only applies backward filter. IK solver handles arm-scene collision internally.

        Returns:
            dict with per-candidate success, qpos, and timing.
        """
        import time as _time

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        t0 = _time.time()
        obj_pose = cart2se3(scene_cfg["mesh"]["target"]["pose"])
        wrist_se3, pregrasp, grasp, scene_info = load_candidate(obj_name, obj_pose, grasp_version, hand=hand)
        t_load = _time.time() - t0

        t0 = _time.time()
        # IK solver uses table-only world (no target mesh — arm shouldn't collide with table)
        world_cfg_no_target = _to_curobo_world(scene_cfg)
        world_cfg_no_target["mesh"] = {}
        if self._ik_solver is None:
            self._init_ik_solver(world_cfg_no_target)
        else:
            self._ik_solver.update_world(WorldConfig.from_dict(world_cfg_no_target))
        t_world = _time.time() - t0

        # Filter: backward + hand-table collision (no object mesh — hand should be near object)
        t0 = _time.time()
        backward = np.zeros(len(wrist_se3), dtype=bool) if self._hand == "inspire" else (wrist_se3[:, :3, :3] @ self._link6_y_in_wrist)[:, 2] < 0.3
        collision = self._check_collision(world_cfg_no_target, wrist_se3, pregrasp)
        filtered = backward | collision
        valid = np.where(~filtered)[0]
        t_filter = _time.time() - t0

        N = len(wrist_se3)
        ik_success = np.zeros(N, dtype=bool)
        ik_qpos = np.full((N, len(self._init_state)), np.nan)  # 6 arm + 16 fingers

        t0 = _time.time()
        if len(valid) > 0:
            # Process in fixed-size chunks for consistent CUDA graph shape
            for chunk_start in range(0, len(valid), self.BATCH_SIZE):
                chunk_idx = valid[chunk_start : chunk_start + self.BATCH_SIZE]
                chunk_poses = wrist_se3[chunk_idx]
                B = len(chunk_poses)

                if B < self.BATCH_SIZE:
                    pad = self.BATCH_SIZE - B
                    chunk_poses = np.concatenate(
                        [chunk_poses, np.tile(chunk_poses[:1], (pad, 1, 1))], axis=0)

                goal = _to_curobo_pose(chunk_poses, self._tensor_args.device)
                result = self._ik_solver.solve_batch(goal)
                succ = result.success.cpu().numpy()[:B]
                q_sol = result.solution.cpu().numpy()[:B]

                if q_sol.ndim == 3:
                    q_sol = q_sol[:, 0, :]

                for i, idx in enumerate(chunk_idx):
                    if succ[i]:
                        ik_success[idx] = True
                        arm_q = q_sol[i, :6].copy()
                        # Snap joint 6 to nearest equivalent angle to init_state
                        # IK can return any angle in [-2π, 2π]; pick closest to start
                        diff = arm_q[5] - self._init_state[5]
                        arm_q[5] -= np.round(diff / (2 * np.pi)) * 2 * np.pi
                        ik_qpos[idx, :6] = arm_q
                        ik_qpos[idx, 6:] = pregrasp[idx]
        t_ik = _time.time() - t0

        timing = {
            "load_candidates_s": round(t_load, 3),
            "world_setup_s": round(t_world, 3),
            "filter_s": round(t_filter, 3),
            "ik_solve_s": round(t_ik, 3),
        }

        return {
            "n_total": N,
            "n_backward": int(backward.sum()),
            "n_table_collision": int(collision.sum()),
            "n_valid": int(len(valid)),
            "n_ik_success": int(ik_success.sum()),
            "ik_success": ik_success,
            "ik_qpos": ik_qpos,
            "wrist_se3": wrist_se3,
            "pregrasp": pregrasp,
            "grasp": grasp,
            "scene_info": scene_info,
            "timing": timing,
        }

    # ── motion planning ───────────────────────────────────────────────────────

    def _plan_goalset(self, goal_poses_se3: np.ndarray):
        """INIT_STATE -> best among N goals. Returns (local_idx, traj) or (None, None)."""
        init_js = JointState.from_position(
            torch.tensor(self._init_state, dtype=torch.float32, device=self._tensor_args.device).unsqueeze(0)
        )
        goal = _to_curobo_pose(goal_poses_se3, self._tensor_args.device)
        goal = Pose(position=goal.position.unsqueeze(0), quaternion=goal.quaternion.unsqueeze(0))

        result = self._motion_gen.plan_goalset(start_state=init_js, goal_pose=goal, plan_config=self._plan_cfg)
        if not result.success.item():
            return None, None
        return result.goalset_index.item(), result.get_interpolated_plan().position.cpu().numpy()

    def _plan_batch(self, init_states: np.ndarray, goal_poses_se3: np.ndarray):
        """(B, dof), (B, 4, 4) -> success (B,), trajs (B, T, dof)."""
        B = len(init_states)
        # Pad to BATCH_SIZE so cuRobo gets a consistent batch size
        if B < self.BATCH_SIZE:
            pad = self.BATCH_SIZE - B
            init_states = np.concatenate([init_states, np.tile(init_states[:1], (pad, 1))], axis=0)
            goal_poses_se3 = np.concatenate([goal_poses_se3, np.tile(goal_poses_se3[:1], (pad, 1, 1))], axis=0)

        init_js = JointState.from_position(
            torch.tensor(init_states, dtype=torch.float32, device=self._tensor_args.device)
        )
        try:
            result = self._motion_gen.plan_batch(
                start_state=init_js,
                goal_pose=_to_curobo_pose(goal_poses_se3, self._tensor_args.device),
                plan_config=self._plan_cfg,
            )
        except RuntimeError:
            # cuRobo crashes when IK finds 0 solutions (internal shape mismatch)
            return np.zeros(B, dtype=bool), None
        success = result.success.cpu().numpy()[:B]
        trajs = result.optimized_plan.position.cpu().numpy()[:B] if success.any() else None
        if trajs is not None and trajs.ndim == 2:
            trajs = trajs[np.newaxis]
        return success, trajs

    def _refine_fingers(self, init_state: np.ndarray, goal_joint: np.ndarray):
        """Joint-space trajopt for full DOF (arm + fingers). Returns (ok, traj)."""
        start = JointState.from_position(
            torch.tensor(init_state, dtype=torch.float32, device=self._tensor_args.device).unsqueeze(0)
        )
        goal = JointState.from_position(
            torch.tensor(goal_joint, dtype=torch.float32, device=self._tensor_args.device).unsqueeze(0)
        )
        result = self._motion_gen.plan_single_js(start_state=start, goal_state=goal, plan_config=self._plan_cfg)
        if not result.success.item():
            if hasattr(result, 'status') and result.status is not None:
                print(f"    [plan_single_js] status={result.status} (act_dist={self._collision_act_dist})")
            if hasattr(result, 'valid_query') and result.valid_query is not None:
                print(f"    [plan_single_js] valid_query={result.valid_query}")
            # Export collision debug meshes on failure
            self._export_collision_debug(goal_joint)
        if result.success.item():
            return True, result.get_interpolated_plan().position.cpu().numpy()
        return False, None

    def _export_collision_debug(self, goal_joint: np.ndarray):
        """Export hand collision spheres + world meshes at goal state for debugging."""
        try:
            import trimesh
            debug_dir = "/tmp/collision_debug"
            os.makedirs(debug_dir, exist_ok=True)

            # Get collision spheres at goal state
            q = torch.tensor(goal_joint, dtype=torch.float32, device=self._tensor_args.device).unsqueeze(0)
            kin = self._motion_gen.kinematics
            spheres = kin.get_robot_as_spheres(q)

            # Save spheres as mesh. spheres is List[List[Sphere]] (per-batch, per-link).
            sphere_meshes = []
            for sphere_batch in spheres:
                for s in sphere_batch:
                    pos = np.asarray(s.position)
                    rad = float(s.radius)
                    if rad > 0:
                        m = trimesh.creation.icosphere(radius=rad)
                        m.apply_translation(pos)
                        sphere_meshes.append(m)
            if sphere_meshes:
                combined = trimesh.util.concatenate(sphere_meshes)
                out = os.path.join(debug_dir, "hand_spheres.obj")
                combined.export(out)
                print(f"    [debug] Hand spheres -> {out}")

            # Save world meshes + cuboids
            if self._motion_gen.world_model is not None:
                wm = self._motion_gen.world_model
                # Mesh primitives. cuRobo Mesh may store file_path instead of verts/faces.
                meshes = getattr(wm, "mesh", None) or []
                for m in meshes:
                    name = getattr(m, "name", "mesh")
                    pose = np.asarray(getattr(m, "pose", [0, 0, 0, 1, 0, 0, 0]) or [0, 0, 0, 1, 0, 0, 0])
                    file_path = getattr(m, "file_path", None)
                    verts, faces = m.vertices, m.faces
                    if (verts is None or faces is None) and file_path:
                        tm = trimesh.load(file_path, force="mesh")
                    else:
                        if hasattr(verts, "cpu"): verts = verts.cpu().numpy()
                        if hasattr(faces, "cpu"): faces = faces.cpu().numpy()
                        tm = trimesh.Trimesh(vertices=np.asarray(verts), faces=np.asarray(faces))
                    # Apply mesh pose
                    T = np.eye(4)
                    T[:3, 3] = pose[:3]
                    from scipy.spatial.transform import Rotation as Rot
                    T[:3, :3] = Rot.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
                    tm.apply_transform(T)
                    out = os.path.join(debug_dir, f"world_mesh_{name}.obj")
                    tm.export(out)
                    print(f"    [debug] World mesh -> {out}")
                # Cuboid primitives (table, shelf walls)
                cubes = getattr(wm, "cuboid", None) or []
                for c in cubes:
                    name = getattr(c, "name", "cube")
                    dims = np.asarray(c.dims)
                    pose = np.asarray(c.pose)  # [x,y,z,qw,qx,qy,qz]
                    box = trimesh.creation.box(extents=dims)
                    T = np.eye(4)
                    T[:3, 3] = pose[:3]
                    from scipy.spatial.transform import Rotation as Rot
                    T[:3, :3] = Rot.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
                    box.apply_transform(T)
                    out = os.path.join(debug_dir, f"world_cube_{name}.obj")
                    box.export(out)
                    print(f"    [debug] World cube -> {out}")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"    [debug] Export failed: {e}")

    # ── internal pipeline ─────────────────────────────────────────────────────

    def _find_trajectory(self, world_cfg: dict, wrist_se3: np.ndarray, pregrasp: np.ndarray, mode: str):
        """Filter candidates -> motion plan -> finger refinement. Returns (idx, traj, timing)."""
        import time as _time
        timing = {}

        t0 = _time.time()
        collision = self._check_collision(world_cfg, wrist_se3, pregrasp)
        backward = np.zeros(len(wrist_se3), dtype=bool) if self._hand == "inspire" else (wrist_se3[:, :3, :3] @ self._link6_y_in_wrist)[:, 2] < 0.3
        valid = np.where(~(collision | backward))[0]
        timing["collision_check_s"] = round(_time.time() - t0, 3)

        print(f"[planner] total={len(wrist_se3)}  collision={collision.sum()}  backward={backward.sum()}  valid={len(valid)}")

        if len(valid) == 0:
            return None, None, timing

        if mode == "goalset":
            t0 = _time.time()
            local_idx, traj = self._plan_goalset(wrist_se3[valid])
            timing["arm_plan_s"] = round(_time.time() - t0, 3)
            if local_idx is None:
                return None, None, timing
            idx = valid[local_idx]
            goal = traj[-1].copy()
            goal[6:] = pregrasp[idx]
            t0 = _time.time()
            ok, traj = self._refine_fingers(self._init_state, goal)
            timing["finger_refine_s"] = round(_time.time() - t0, 3)
            return (idx, traj, timing) if ok else (None, None, timing)

        # batch mode
        timing["arm_plan_s"] = 0.0
        timing["finger_refine_s"] = 0.0
        timing["n_batches"] = 0
        timing["n_refine_attempts"] = 0
        inits = np.tile(self._init_state, (len(valid), 1))
        for start in range(0, len(valid), self.BATCH_SIZE):
            batch = valid[start : start + self.BATCH_SIZE]

            t0 = _time.time()
            success, trajs = self._plan_batch(inits[start : start + len(batch)], wrist_se3[batch])
            timing["arm_plan_s"] += _time.time() - t0
            timing["n_batches"] += 1

            for i, idx in enumerate(batch):
                if not success[i]:
                    continue
                goal = trajs[i, -1].copy()
                goal[6:] = pregrasp[idx]
                t0 = _time.time()
                ok, traj = self._refine_fingers(inits[start + i], goal)
                timing["finger_refine_s"] += _time.time() - t0
                timing["n_refine_attempts"] += 1
                if ok:
                    timing["arm_plan_s"] = round(timing["arm_plan_s"], 3)
                    timing["finger_refine_s"] = round(timing["finger_refine_s"], 3)
                    return idx, traj, timing

        timing["arm_plan_s"] = round(timing["arm_plan_s"], 3)
        timing["finger_refine_s"] = round(timing["finger_refine_s"], 3)
        return None, None, timing

    # ── public API ────────────────────────────────────────────────────────────

    def get_candidates(self, scene_cfg: dict, obj_name: str, grasp_version: str,
                        success_only: bool = False, skip_done: bool = False, hand: str = "allegro"):
        """
        Return all grasp candidates with collision filter applied (no motion planning).

        Returns:
            wrist_se3  (N, 4, 4)
            pregrasp   (N, 16)
            grasp_pose (N, 16)
            collision  (N,) bool  — True if filtered out (collision OR backward)
        """
        obj_pose = cart2se3(scene_cfg["mesh"]["target"]["pose"])
        wrist_se3, pregrasp, grasp, _ = load_candidate(obj_name, obj_pose, grasp_version,
                                                         skip_done=skip_done, success_only=success_only, hand=hand)

        world_cfg = _to_curobo_world(scene_cfg)
        if self._motion_gen is None:
            self._init_motion_gen(world_cfg)
        else:
            self._update_world(world_cfg)

        collision = self._check_collision(world_cfg, wrist_se3, pregrasp)
        backward  = np.zeros(len(wrist_se3), dtype=bool)
        filtered  = collision | backward
        print(f"[planner] total={len(wrist_se3)}  collision={collision.sum()}  backward={backward.sum()}  valid={(~filtered).sum()}")
        return wrist_se3, pregrasp, grasp, filtered

    def plan_all(self, scene_cfg: dict, obj_name: str, grasp_version: str,
                 stop_on_first: bool = True, hand: str = "allegro"):
        """
        Plan trajectories for all candidates (for visualization / debugging).

        Args:
            stop_on_first: If True (default), stop after first successful grasp.
                           If False, attempt planning for ALL valid candidates.

        Returns:
            wrist_se3    (N, 4, 4)
            grasp_pose   (N, 16)
            succ_mask    (N,) bool — trajectory planning success
            collision    (N,) bool — collision or backward filtered
            traj_list    list[N] of (T, dof) arrays or None
        """
        import time as _time
        t_total = _time.time()

        t0 = _time.time()
        obj_pose = cart2se3(scene_cfg["mesh"]["target"]["pose"])
        wrist_se3, pregrasp, grasp, scene_info = load_candidate(obj_name, obj_pose, grasp_version, hand=hand)
        print(f"[planner] load candidates: {_time.time() - t0:.2f}s ({len(wrist_se3)} candidates)")

        t0 = _time.time()
        world_cfg = _to_curobo_world(scene_cfg)
        if self._motion_gen is None:
            self._init_motion_gen(world_cfg)
        else:
            self._update_world(world_cfg)
        print(f"[planner] init/update motion gen: {_time.time() - t0:.2f}s")

        t0 = _time.time()
        N = len(wrist_se3)
        collision = self._check_collision(world_cfg, wrist_se3, pregrasp)
        backward = np.zeros(len(wrist_se3), dtype=bool) if self._hand == "inspire" else (wrist_se3[:, :3, :3] @ self._link6_y_in_wrist)[:, 2] < 0.3
        filtered = collision | backward
        valid = np.where(~filtered)[0]
        print(f"[planner] collision check: {_time.time() - t0:.2f}s")

        print(f"[planner] total={N}  collision={collision.sum()}  backward={backward.sum()}  valid={len(valid)}")

        succ_mask = np.zeros(N, dtype=bool)
        traj_list = [None] * N

        if len(valid) == 0:
            return wrist_se3, pregrasp, grasp, succ_mask, filtered, traj_list

        inits = np.tile(self._init_state, (len(valid), 1))
        has_succ = False
        t_batch_total = 0.0
        t_refine_total = 0.0
        n_batches = 0
        n_refines = 0

        for start in range(0, len(valid), self.BATCH_SIZE):
            batch = valid[start : start + self.BATCH_SIZE]
            if has_succ:
                break

            t0 = _time.time()
            success, trajs = self._plan_batch(
                inits[start : start + len(batch)], wrist_se3[batch]
            )
            t_batch_total += _time.time() - t0
            n_batches += 1
            print(f"[planner] batch {n_batches}: {success.sum()}/{len(batch)} arm plan success ({_time.time() - t0:.2f}s)")

            if trajs is not None and trajs.ndim == 2:
                trajs = trajs[np.newaxis]

            for i, idx in enumerate(batch):
                if has_succ:
                    break
                if not success[i]:
                    continue
                goal = trajs[i, -1].copy()
                goal[6:] = pregrasp[idx]
                t1 = _time.time()
                ok, traj = self._refine_fingers(self._init_state, goal)
                t_refine_total += _time.time() - t1
                n_refines += 1
                print(f"[planner] plan_single #{n_refines} (idx={idx}): {'ok' if ok else 'fail'} ({_time.time() - t1:.2f}s)")
                if ok:
                    succ_mask[idx] = True
                    traj_list[idx] = traj
                    has_succ = True

        print(f"[planner] timing: plan_batch={t_batch_total:.2f}s ({n_batches} calls)  plan_single={t_refine_total:.2f}s ({n_refines} calls)")
        print(f"[planner] total plan_all: {_time.time() - t_total:.2f}s")

        return wrist_se3, pregrasp, grasp, succ_mask, filtered, traj_list

    def plan(self, scene_cfg: dict, obj_name: str, grasp_version: str,
             mode: str = "batch", seed: Optional[int] = None,
             skip_done: bool = True, success_only: bool = False,
             hand: str = "allegro") -> PlanResult:
        import time as _time

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # 1. Load candidates
        t0 = _time.time()
        obj_pose = cart2se3(scene_cfg["mesh"]["target"]["pose"])
        wrist_se3, pregrasp, grasp, scene_info = load_candidate(obj_name, obj_pose, grasp_version, skip_done=skip_done, success_only=success_only, hand=hand)
        t_load = _time.time() - t0

        if len(wrist_se3) == 0:
            print(f"[planner] No candidates available (all done or no success)")
            return PlanResult(
                success=False, traj=None, wrist_se3=None,
                pregrasp_pose=None, grasp_pose=None, scene_info=[],
                timing={"load_candidates_s": round(t_load, 3), "n_total": 0},
            )

        # 2. World setup (motion_gen for trajectory, ik_solver for IK)
        t0 = _time.time()
        world_cfg = _to_curobo_world(scene_cfg)
        if self._motion_gen is None:
            self._init_motion_gen(world_cfg)
        else:
            self._update_world(world_cfg)
        world_cfg_no_target = dict(world_cfg)
        world_cfg_no_target["mesh"] = {}
        if self._ik_solver is None:
            self._init_ik_solver(world_cfg_no_target)
        else:
            self._ik_solver.update_world(WorldConfig.from_dict(world_cfg_no_target))
        t_world = _time.time() - t0

        # 3. Filter: backward + hand-table collision
        t0 = _time.time()
        backward = np.zeros(len(wrist_se3), dtype=bool) if self._hand == "inspire" else (wrist_se3[:, :3, :3] @ self._link6_y_in_wrist)[:, 2] < 0.3
        print(f"[backward] wrist x-axis z: {wrist_se3[:, 0, 2]}")
        collision = self._check_collision(world_cfg_no_target, wrist_se3, pregrasp)
        valid = np.where(~(backward | collision))[0]
        t_filter = _time.time() - t0

        N = len(wrist_se3)
        print(f"[planner] total={N}  backward={backward.sum()}  collision={collision.sum()}  valid={len(valid)}")

        def _fail_result(timing):
            return PlanResult(
                success=False, traj=None, wrist_se3=None,
                pregrasp_pose=pregrasp[0], grasp_pose=grasp[0], scene_info=[],
                timing=timing,
            )

        base_timing = {
            "load_candidates_s": round(t_load, 3),
            "world_setup_s": round(t_world, 3),
            "filter_s": round(t_filter, 3),
            "n_total": N,
            "n_backward": int(backward.sum()),
            "n_collision": int(collision.sum()),
            "n_valid": int(len(valid)),
        }

        if len(valid) == 0:
            return _fail_result({**base_timing, "ik_s": 0.0, "plan_single_js_s": 0.0})

        # 4. IK solve on valid candidates
        t0 = _time.time()
        ik_success = np.zeros(N, dtype=bool)
        ik_qpos = np.full((N, len(self._init_state)), np.nan)
        for chunk_start in range(0, len(valid), self.BATCH_SIZE):
            chunk_idx = valid[chunk_start : chunk_start + self.BATCH_SIZE]
            chunk_poses = wrist_se3[chunk_idx]
            B = len(chunk_poses)
            if B < self.BATCH_SIZE:
                pad = self.BATCH_SIZE - B
                chunk_poses = np.concatenate(
                    [chunk_poses, np.tile(chunk_poses[:1], (pad, 1, 1))], axis=0)
            goal = _to_curobo_pose(chunk_poses, self._tensor_args.device)
            # Retract toward init_state so IK solutions stay near start config
            B_padded = chunk_poses.shape[0]
            retract = torch.tensor(
                self._init_state, dtype=torch.float32, device=self._tensor_args.device
            ).unsqueeze(0).repeat(B_padded, 1)
            result = self._ik_solver.solve_batch(goal, retract_config=retract)
            succ = result.success.cpu().numpy()[:B]
            q_sol = result.solution.cpu().numpy()[:B]
            if q_sol.ndim == 3:
                q_sol = q_sol[:, 0, :]
            for i, idx in enumerate(chunk_idx):
                if succ[i]:
                    ik_success[idx] = True
                    arm_q = q_sol[i, :6].copy()
                    # Snap joint 6 to nearest equivalent angle to init_state
                    # IK can return any angle in [-2π, 2π]; pick closest to start
                    diff = arm_q[5] - self._init_state[5]
                    arm_q[5] -= np.round(diff / (2 * np.pi)) * 2 * np.pi
                    ik_qpos[idx, :6] = arm_q
                    ik_qpos[idx, 6:] = pregrasp[idx]
        t_ik = _time.time() - t0

        # Lift IK check: verify z+12cm pose is reachable (avoids joint limit errors during lift)
        LIFT_HEIGHT = 0.10
        ik_valid_pre = np.where(ik_success)[0]
        if len(ik_valid_pre) > 0:
            lift_poses = wrist_se3[ik_valid_pre].copy()
            lift_poses[:, 2, 3] += LIFT_HEIGHT
            for chunk_start in range(0, len(ik_valid_pre), self.BATCH_SIZE):
                chunk = ik_valid_pre[chunk_start : chunk_start + self.BATCH_SIZE]
                chunk_poses = lift_poses[chunk_start : chunk_start + len(chunk)]
                B = len(chunk_poses)
                if B < self.BATCH_SIZE:
                    pad = self.BATCH_SIZE - B
                    chunk_poses = np.concatenate(
                        [chunk_poses, np.tile(chunk_poses[:1], (pad, 1, 1))], axis=0)
                goal = _to_curobo_pose(chunk_poses, self._tensor_args.device)
                result = self._ik_solver.solve_batch(goal)
                lift_succ = result.success.cpu().numpy()[:B]
                for i, idx in enumerate(chunk):
                    if not lift_succ[i]:
                        ik_success[idx] = False
            n_lift_fail = len(ik_valid_pre) - int(ik_success.sum())
            if n_lift_fail > 0:
                print(f"[planner] Lift IK check: {n_lift_fail} candidates failed (z+{LIFT_HEIGHT}m unreachable)")

        ik_valid = np.where(ik_success)[0]
        n_ik_success = len(ik_valid)
        print(f"[planner] IK: {n_ik_success}/{len(valid)} success (after lift check)")
        base_timing["ik_s"] = round(t_ik, 3)
        base_timing["n_ik_success"] = n_ik_success
        base_timing["n_valid"] = int(len(valid))

        if n_ik_success == 0:
            return _fail_result({**base_timing, "plan_single_js_s": 0.0})

        # 5. plan_single_js for each IK-reachable candidate until success
        t0 = _time.time()
        n_attempts = 0
        for idx in ik_valid:
            t1 = _time.time()
            ok, traj = self._refine_fingers(self._init_state, ik_qpos[idx])
            n_attempts += 1
            print(f"[planner] plan_single_js #{n_attempts} (idx={idx}): "
                  f"{'ok' if ok else 'fail'} ({_time.time() - t1:.2f}s)")
            if ok:
                t_plan = _time.time() - t0
                print(f"[planner] Selected candidate #{idx}/{N}")
                return PlanResult(
                    success=True, traj=traj, wrist_se3=wrist_se3[idx],
                    pregrasp_pose=pregrasp[idx], grasp_pose=grasp[idx],
                    scene_info=scene_info[idx],
                    timing={**base_timing, "plan_single_js_s": round(t_plan, 3),
                            "n_plan_attempts": n_attempts,
                            "candidate_idx": int(idx)},
                )

        t_plan = _time.time() - t0
        return _fail_result({**base_timing, "plan_single_js_s": round(t_plan, 3),
                             "n_plan_attempts": n_attempts})