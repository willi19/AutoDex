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
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

from rsslib.path import robot_configs_path, load_candidate
from rsslib.conversion import se32action
from rsslib.robot_config import INIT_STATE


@dataclass
class PlanResult:
    success: bool
    traj: Optional[np.ndarray]        # (T, dof)
    wrist_se3: Optional[np.ndarray]   # (4, 4)
    pregrasp_pose: np.ndarray         # (16,) hand joints
    grasp_pose: np.ndarray            # (16,) hand joints
    scene_info: list


# ── scene_cfg conversion ──────────────────────────────────────────────────────

def _se3_to_7vec(mat: np.ndarray) -> list:
    """4x4 SE3 → [x,y,z,qx,qy,qz,qw]"""
    t = mat[:3, 3].tolist()
    q = Rotation.from_matrix(mat[:3, :3]).as_quat().tolist()  # xyzw
    return t + q


def _to_curobo_cfg(scene_cfg: dict) -> dict:
    """
    SE3-based scene_cfg → curobo WorldConfig dict

    Input format:
        {
            "cuboid": {name: {"dims": [...], "pose": np.ndarray (4,4)}},
            "mesh":   {name: {"pose": np.ndarray (4,4), "file_path": str}},
        }
    """
    cfg = {"cuboid": {}, "mesh": {}}
    for name, info in scene_cfg.get("cuboid", {}).items():
        cfg["cuboid"][name] = {
            "dims": info["dims"],
            "pose": _se3_to_7vec(info["pose"]),
            "color": info.get("color", [0.5, 0.5, 0.5, 1.0]),
        }
    for name, info in scene_cfg.get("mesh", {}).items():
        cfg["mesh"][name] = {
            "pose": _se3_to_7vec(info["pose"]),
            "file_path": info["file_path"],
        }
    return cfg


def _to_curobo_pose(poses_se3: np.ndarray, device) -> Pose:
    """(B, 4, 4) → curobo Pose"""
    position = torch.tensor(poses_se3[:, :3, 3], dtype=torch.float32, device=device)
    xyzw = Rotation.from_matrix(poses_se3[:, :3, :3]).as_quat()
    wxyz = torch.tensor(xyzw[:, [3, 0, 1, 2]], dtype=torch.float32, device=device)
    return Pose(position=position, quaternion=wxyz)


# ── Planner ───────────────────────────────────────────────────────────────────

class Planner:
    """
    scene_cfg (SE3 poses) + grasp candidates → trajectory

    Usage:
        planner = Planner()
        result = planner.plan(scene_cfg, obj_name="bottle", grasp_version="v1")
    """

    BATCH_SIZE = 150
    N_CUBOIDS = 30
    N_MESHES = 100

    def __init__(self, robot_cfg_path: Optional[str] = None, hand_cfg_path: Optional[str] = None):
        if robot_cfg_path is None:
            robot_cfg_path = os.path.join(robot_configs_path, "xarm_allegro.yml")
        if hand_cfg_path is None:
            hand_cfg_path = os.path.join(robot_configs_path, "allegro_floating.yml")

        self.robot_cfg = load_yaml(robot_cfg_path)["robot_cfg"]
        self.hand_cfg = load_yaml(hand_cfg_path)["robot_cfg"]
        self.tensor_args = TensorDeviceType()
        self._motion_gen: Optional[MotionGen] = None
        self._plan_cfg: Optional[MotionGenPlanConfig] = None

    # ── setup ─────────────────────────────────────────────────────────────────

    def _init_motion_gen(self, world_cfg: dict):
        config = MotionGenConfig.load_from_robot_config(
            self.robot_cfg,
            WorldConfig.from_dict(world_cfg),
            self.tensor_args,
            num_trajopt_seeds=1024,
            num_graph_seeds=1,
            num_ik_seeds=32,
            use_cuda_graph=True,
            interpolation_dt=0.01,
            collision_cache={"obb": self.N_CUBOIDS, "mesh": self.N_MESHES},
            ik_opt_iters=200,
            grad_trajopt_iters=200,
            trajopt_tsteps=64,
            collision_activation_distance=0.01,
        )
        self._motion_gen = MotionGen(config)
        self._motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
        self._plan_cfg = MotionGenPlanConfig(
            enable_graph=True,
            enable_opt=True,
            enable_graph_attempt=2,
            max_attempts=10,
            enable_finetune_trajopt=True,
            num_trajopt_seeds=32,
            num_ik_seeds=32,
            timeout=60.0,
            parallel_finetune=True,
        )

    def _update_world(self, world_cfg: dict):
        self._motion_gen.clear_world_cache()
        self._motion_gen.update_world(WorldConfig.from_dict(world_cfg))

    # ── collision filter ──────────────────────────────────────────────────────

    def _check_collision(self, world_cfg: dict, wrist_se3: np.ndarray, pregrasp_pose: np.ndarray) -> np.ndarray:
        """Returns bool array (N,): True = collides"""
        rw_config = RobotWorldConfig.load_from_config(
            self.hand_cfg,
            WorldConfig.from_dict(world_cfg),
            collision_activation_distance=0.0,
            tensor_args=self.tensor_args,
        )
        q = np.array([se32action(w, g) for w, g in zip(wrist_se3, pregrasp_pose)])
        q_t = torch.tensor(q, dtype=torch.float32, device=self.tensor_args.device)
        rw = RobotWorld(rw_config)
        d_world, d_self = rw.get_world_self_collision_distance_from_joints(q_t)
        return torch.logical_or(d_world > 0, d_self > 0).cpu().numpy()

    # ── motion planning ───────────────────────────────────────────────────────

    def _plan_goalset(self, goal_poses_se3: np.ndarray):
        """1 start (INIT_STATE) → best among N goals. Returns (found_local_idx, traj) or (None, None)"""
        init_js = JointState.from_position(
            torch.tensor(INIT_STATE, dtype=torch.float32, device=self.tensor_args.device).unsqueeze(0)
        )
        goal_pose = _to_curobo_pose(goal_poses_se3, self.tensor_args.device)
        # goalset expects (1, N, 3) and (1, N, 4)
        goal_pose = Pose(
            position=goal_pose.position.unsqueeze(0),
            quaternion=goal_pose.quaternion.unsqueeze(0),
        )
        result = self._motion_gen.plan_goalset(
            start_state=init_js,
            goal_pose=goal_pose,
            plan_config=self._plan_cfg,
        )
        if not result.success.item():
            return None, None
        idx = result.goalset_index.item()
        traj = result.get_interpolated_plan().position.cpu().numpy()
        return idx, traj

    def _plan_batch(self, init_states: np.ndarray, goal_poses_se3: np.ndarray):
        """(B, dof), (B, 4, 4) → success (B,), trajs (B, T, dof) or None"""
        init_js = JointState.from_position(
            torch.tensor(init_states, dtype=torch.float32, device=self.tensor_args.device)
        )
        result = self._motion_gen.plan_batch(
            start_state=init_js,
            goal_pose=_to_curobo_pose(goal_poses_se3, self.tensor_args.device),
            plan_config=self._plan_cfg,
        )
        success = result.success.cpu().numpy()
        trajs = result.optimized_plan.position.cpu().numpy() if success.any() else None
        if trajs is not None and trajs.ndim == 2:
            trajs = trajs[np.newaxis]
        return success, trajs

    def _refine_with_fingers(self, init_state: np.ndarray, goal_joint: np.ndarray):
        """Single-pose trajopt with full DOF (arm + fingers). Returns (success, traj)"""
        start = JointState.from_position(
            torch.tensor(init_state, dtype=torch.float32, device=self.tensor_args.device).unsqueeze(0)
        )
        goal = JointState.from_position(
            torch.tensor(goal_joint, dtype=torch.float32, device=self.tensor_args.device).unsqueeze(0)
        )
        result = self._motion_gen.plan_single_js(
            start_state=start, goal_state=goal, plan_config=self._plan_cfg
        )
        if result.success.item():
            return True, result.optimized_plan.position.cpu().numpy()
        return False, None

    def _find_trajectory(self, world_cfg: dict, wrist_se3: np.ndarray, pregrasp_pose: np.ndarray, mode: str = "batch"):
        """Collision filter → plan → finger refinement. Returns (found_idx, traj)"""
        collision = self._check_collision(world_cfg, wrist_se3, pregrasp_pose)
        backward = wrist_se3[:, 0, 2] < 0.3  # wrist pointing backward
        valid_idx = np.where(~np.logical_or(collision, backward))[0]

        if len(valid_idx) == 0:
            return None, None

        if mode == "goalset":
            local_idx, traj = self._plan_goalset(wrist_se3[valid_idx])
            if local_idx is None:
                return None, None
            global_idx = valid_idx[local_idx]
            goal_joint = traj[-1].copy()
            goal_joint[6:] = pregrasp_pose[global_idx]
            ok, traj = self._refine_with_fingers(INIT_STATE, goal_joint)
            return (global_idx, traj) if ok else (None, None)

        # mode == "batch"
        init_states = np.tile(INIT_STATE, (len(valid_idx), 1))
        for start in range(0, len(valid_idx), self.BATCH_SIZE):
            batch_idx = valid_idx[start:start + self.BATCH_SIZE]
            success, trajs = self._plan_batch(init_states[start:start + len(batch_idx)], wrist_se3[batch_idx])
            for i, global_idx in enumerate(batch_idx):
                if not success[i]:
                    continue
                goal_joint = trajs[i, -1].copy()
                goal_joint[6:] = pregrasp_pose[global_idx]
                ok, traj = self._refine_with_fingers(init_states[start + i], goal_joint)
                if ok:
                    return global_idx, traj

        return None, None

    # ── public API ────────────────────────────────────────────────────────────

    def plan(self, scene_cfg: dict, obj_name: str, grasp_version: str, mode: str = "batch") -> PlanResult:
        obj_pose = scene_cfg["mesh"]["target"]["pose"]  # (4,4) SE3
        wrist_se3, pregrasp_pose, grasp_pose, scene_info = load_candidate(obj_name, obj_pose, grasp_version)

        world_cfg = _to_curobo_cfg(scene_cfg)
        if self._motion_gen is None:
            self._init_motion_gen(world_cfg)
        else:
            self._update_world(world_cfg)

        found_idx, traj = self._find_trajectory(world_cfg, wrist_se3.copy(), pregrasp_pose, mode=mode)

        if found_idx is None:
            return PlanResult(
                success=False, traj=None, wrist_se3=None,
                pregrasp_pose=pregrasp_pose[0], grasp_pose=grasp_pose[0], scene_info=[],
            )
        return PlanResult(
            success=True,
            traj=traj,
            wrist_se3=wrist_se3[found_idx],
            pregrasp_pose=pregrasp_pose[found_idx],
            grasp_pose=grasp_pose[found_idx],
            scene_info=scene_info[found_idx],
        )