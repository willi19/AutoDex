"""
Real-world grasp executor for xArm + Allegro hand.

Two modes:
    - "auto" (default): autonomous trajectory execution (no GUI)
    - "gui":  interactive Tkinter GUI with manual control

Usage:
    executor = RealExecutor(mode="auto")
    executor.execute(plan_result)
    executor.release(plan_result)
    executor.shutdown()
"""
import time
import numpy as np
from scipy.spatial.transform import Rotation

from autodex.planner import PlanResult
from autodex.utils.robot_config import XARM_INIT, ALLEGRO_INIT, LINK6_TO_WRIST


def _convert_hand_pose(hand_pose: np.ndarray) -> np.ndarray:
    """Reorder Allegro joints: move last 4 (thumb) to front."""
    if hand_pose.ndim == 1:
        out = hand_pose.copy()
        out[:4] = hand_pose[12:]
        out[4:] = hand_pose[:12]
    else:
        out = hand_pose.copy()
        out[:, :4] = hand_pose[:, 12:]
        out[:, 4:] = hand_pose[:, :12]
    return out


class RealExecutor:
    """
    Execute a planned grasp on real hardware.

    Args:
        mode: "auto" (default) or "gui"
        arm_name: robot arm type (default "xarm")
        hand_name: robot hand type (default "allegro")
        dt: control loop period in seconds
        squeeze_level: how many squeeze increments (default 10)
    """

    def __init__(
        self,
        mode: str = "auto",
        arm_name: str = "xarm",
        hand_name: str = "allegro",
        dt: float = 0.01,
        squeeze_level: int = 10,
    ):
        if mode not in ("auto", "gui"):
            raise ValueError(f"mode must be 'auto' or 'gui', got '{mode}'")
        self.mode = mode
        self.dt = dt
        self.squeeze_level = squeeze_level

        from paradex.io.robot_controller import get_arm, get_hand
        self.arm = get_arm(arm_name)
        self.hand = get_hand(hand_name)

        # Safety velocity limits (auto mode)
        self.joint_vel_limit = 0.05
        self.cart_vel_limit = 0.002
        self.rot_vel_limit = 0.01
        self.hand_vel_limit = 0.03

    # ── low-level motion primitives (auto mode) ──────────────────────────

    def _safe_joint_step(self, current, target, vel_limit=None):
        delta = target - current
        limit = vel_limit if vel_limit is not None else self.joint_vel_limit
        norm = np.linalg.norm(delta)
        if norm > limit:
            delta = delta / norm * limit
        return current + delta

    def _move_joints(self, arm_traj, hand_traj=None, threshold=0.02):
        """Follow a joint-space trajectory with velocity limiting."""
        for i in range(len(arm_traj)):
            target_arm = arm_traj[i]
            target_hand = hand_traj[i] if hand_traj is not None else None
            if target_hand is not None:
                self.hand.move(target_hand)
            for _ in range(500):
                cur = self.arm.get_data()["qpos"]
                nxt = self._safe_joint_step(cur, target_arm)
                self.arm.move(nxt, is_servo=True)
                time.sleep(self.dt)
                if np.linalg.norm(self.arm.get_data()["qpos"] - target_arm) < threshold:
                    break

    def _move_hand(self, target):
        self.hand.move(target)
        time.sleep(self.dt)

    def _move_cartesian(self, target_pose, threshold_t=0.002, threshold_r=0.02):
        """Cartesian move with velocity limiting."""
        target_rot = Rotation.from_matrix(target_pose[:3, :3])
        for _ in range(500):
            cur = self.arm.get_data()["position"].copy()
            # translation
            t_delta = target_pose[:3, 3] - cur[:3, 3]
            t_dist = np.linalg.norm(t_delta)
            if t_dist > self.cart_vel_limit:
                t_delta = t_delta / t_dist * self.cart_vel_limit
            cur[:3, 3] += t_delta
            # rotation
            cur_rot = Rotation.from_matrix(cur[:3, :3])
            r_delta = (target_rot * cur_rot.inv()).as_rotvec()
            r_dist = np.linalg.norm(r_delta)
            if r_dist > self.rot_vel_limit:
                r_delta = r_delta / r_dist * self.rot_vel_limit
            if r_dist > 0.001:
                cur[:3, :3] = (Rotation.from_rotvec(r_delta) * cur_rot).as_matrix()
            self.arm.move(cur, is_servo=True)
            time.sleep(self.dt)
            # convergence check
            actual = self.arm.get_data()["position"]
            if (np.linalg.norm(actual[:3, 3] - target_pose[:3, 3]) < threshold_t
                    and np.linalg.norm((target_rot * Rotation.from_matrix(actual[:3, :3]).inv()).as_rotvec()) < threshold_r):
                break

    def _move_joint_sequential(self, target_qpos, joint_order, threshold=0.01):
        """Move joints one-by-one in the given order."""
        current_target = self.arm.get_data()["qpos"].copy()
        for j in joint_order:
            current_target[j] = target_qpos[j]
            while True:
                cur = self.arm.get_data()["qpos"]
                nxt = self._safe_joint_step(cur, current_target, vel_limit=0.06)
                self.arm.move(nxt, is_servo=True)
                time.sleep(self.dt)
                if np.abs(self.arm.get_data()["qpos"][j] - target_qpos[j]) < threshold:
                    break

    # ── public API ────────────────────────────────────────────────────────

    def start_recording(self, save_dir: str):
        """Start recording arm and hand trajectories."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        self.hand.start(os.path.join(save_dir, "hand"))
        self.arm.start(os.path.join(save_dir, "arm"))

    def stop_recording(self):
        self.arm.stop()
        self.hand.stop()

    def execute(self, plan_result: PlanResult, lift_height: float = 0.12):
        """
        Execute the grasp: approach -> pre-grasp -> grasp -> squeeze -> lift.

        Returns the squeezed hand pose (for logging).
        """
        if not plan_result.success:
            print("Planning failed — nothing to execute.")
            return None

        traj = plan_result.traj                          # (T, 22) arm6 + hand16
        pg_hand = _convert_hand_pose(plan_result.pregrasp_pose)
        g_hand = _convert_hand_pose(plan_result.grasp_pose)
        wrist_se3 = plan_result.wrist_se3 @ np.linalg.inv(LINK6_TO_WRIST)

        if self.mode == "gui":
            return self._execute_gui(traj, pg_hand, g_hand, wrist_se3, lift_height)
        return self._execute_auto(traj, pg_hand, g_hand, wrist_se3, lift_height)

    def _execute_auto(self, traj, pg_hand, g_hand, wrist_se3, lift_height):
        sl = self.squeeze_level

        # Move to init
        self._move_joint_sequential(XARM_INIT[:6], [0], threshold=0.06)

        # Approach trajectory
        hand_traj = np.array([_convert_hand_pose(traj[i, 6:]) for i in range(len(traj))])
        self._move_joints(traj[:, :6], hand_traj)

        # Grasp sequence
        self._move_hand(pg_hand)
        self._move_hand(g_hand)
        for i in range(sl * 5):
            s_hand = g_hand * (1 + i / 5) - pg_hand * (i / 5)
            self._move_hand(s_hand)
            time.sleep(0.01)

        # Lift
        lift_pose = wrist_se3.copy()
        lift_pose[2, 3] += lift_height
        self._move_cartesian(lift_pose)

        return s_hand

    def _execute_gui(self, traj, pg_hand, g_hand, wrist_se3, lift_height):
        from paradex.io.robot_controller.gui_controller import RobotGUIController

        sl = self.squeeze_level
        s_hand = g_hand * sl - pg_hand * (sl - 1)

        lift_pose = wrist_se3.copy()
        lift_pose[2, 3] += lift_height

        rgc = RobotGUIController(
            self.arm,
            self.hand,
            grasp_pose={
                "start": _convert_hand_pose(ALLEGRO_INIT),
                "pregrasp": pg_hand,
                "grasp": g_hand,
                "squeezed": s_hand,
            },
            approach_traj=np.column_stack([
                traj[:, :6],
                np.array([_convert_hand_pose(traj[i, 6:]) for i in range(len(traj))])
            ]),
            lift_distance=lift_height * 1000,  # m -> mm
        )
        rgc.run()  # blocks until GUI is closed
        return s_hand

    def release(self, plan_result: PlanResult):
        """Release object and return arm to init pose."""
        if not plan_result.success:
            return

        pg_hand = _convert_hand_pose(plan_result.pregrasp_pose)
        g_hand = _convert_hand_pose(plan_result.grasp_pose)
        sl = self.squeeze_level

        if self.mode == "gui":
            self._release_gui(pg_hand, g_hand)
        else:
            self._release_auto(pg_hand, g_hand)

    def _release_auto(self, pg_hand, g_hand):
        sl = self.squeeze_level

        # Reverse squeeze
        for i in range(sl * 5):
            s_hand = g_hand * (sl - i / 5) - pg_hand * (sl - 1 - i / 5)
            self._move_hand(s_hand)
            time.sleep(0.01)

        self._move_hand(g_hand)
        time.sleep(0.01)
        self._move_hand(pg_hand)
        time.sleep(0.01)
        self._move_hand(_convert_hand_pose(ALLEGRO_INIT))

        # Return arm to init
        execute_order = [1, 2, 5, 0, 3, 4]
        if self.arm.get_data()["qpos"][1] < XARM_INIT[1]:
            execute_order = [2, 1, 5, 0, 3, 4]

        clear_view = XARM_INIT.copy()
        clear_view[0] -= 60 * np.pi / 180
        self._move_joint_sequential(clear_view[:6], execute_order, threshold=0.06)

    def _release_gui(self, pg_hand, g_hand):
        from paradex.io.robot_controller.gui_controller import RobotGUIController

        sl = self.squeeze_level
        s_hand = g_hand * sl - pg_hand * (sl - 1)

        rgc = RobotGUIController(
            self.arm,
            self.hand,
            grasp_pose={
                "start": s_hand,
                "pregrasp": g_hand,
                "grasp": pg_hand,
                "squeezed": _convert_hand_pose(ALLEGRO_INIT),
            },
        )
        rgc.run()

    def shutdown(self):
        """Shut down arm and hand controllers."""
        self.arm.end()
        self.hand.end()