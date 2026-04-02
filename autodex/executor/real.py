"""
Real-world grasp executor for xArm + Allegro hand.

Two modes:
    - "auto" (default): autonomous trajectory execution (no GUI)
    - "gui":  interactive Tkinter GUI with manual control

Execution sequence (matches RSS2026 reference: planner/inference/train/run_auto_v2.py):
    execute:  init(joint0) -> approach(traj) -> pregrasp -> grasp -> squeeze -> lift
    release:  reverse_squeeze -> grasp -> pregrasp -> allegro_init -> arm_return

Usage:
    executor = RealExecutor(mode="auto")
    executor.execute(plan_result)
    executor.release(plan_result)
    executor.shutdown()
"""
import datetime
import time
import numpy as np
from scipy.spatial.transform import Rotation

from autodex.planner import PlanResult
from autodex.utils.robot_config import (
    XARM_INIT, XARM_INSPIRE_INIT,
    ALLEGRO_INIT, ALLEGRO_LINK6_TO_WRIST,
    INSPIRE_INIT, INSPIRE_LINK6_TO_WRIST,
)

# Per-hand config: (init_joints, link6_to_wrist, convert_fn)
def _convert_allegro(hand_pose: np.ndarray) -> np.ndarray:
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

def _convert_inspire(hand_pose: np.ndarray) -> np.ndarray:
    """Inspire hand: no reordering needed."""
    return hand_pose

HAND_CONFIG = {
    "allegro": {
        "init": ALLEGRO_INIT,
        "link6_to_wrist": ALLEGRO_LINK6_TO_WRIST,
        "convert": _convert_allegro,
        "xarm_init": XARM_INIT,
    },
    "inspire": {
        "init": INSPIRE_INIT,
        "link6_to_wrist": INSPIRE_LINK6_TO_WRIST,
        "convert": _convert_inspire,
        "xarm_init": XARM_INSPIRE_INIT,
    },
}


class RealExecutor:
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
        if hand_name not in HAND_CONFIG:
            raise ValueError(f"Unknown hand: {hand_name}. Choose from {list(HAND_CONFIG)}")
        self.mode = mode
        self.dt = dt
        self.squeeze_level = squeeze_level
        self.hand_name = hand_name

        hcfg = HAND_CONFIG[hand_name]
        self._convert = hcfg["convert"]
        self._hand_init = hcfg["init"]
        self._link6_to_wrist = hcfg["link6_to_wrist"]
        self._xarm_init = hcfg["xarm_init"]

        from paradex.io.robot_controller import get_arm, get_hand
        self.arm = get_arm(arm_name)
        self.hand = get_hand(hand_name)

        # Safety velocity limits
        self.joint_vel_limit = 0.05
        self.cart_vel_limit = 0.002
        self.rot_vel_limit = 0.01
        self.hand_vel_limit = 0.03

    # ── low-level motion primitives ──────────────────────────────────────

    def _safe_joint_step(self, current, target, vel_limit=None):
        delta = target - current
        limit = vel_limit if vel_limit is not None else self.joint_vel_limit
        norm = np.linalg.norm(delta)
        if norm > limit:
            delta = delta / norm * limit
        return current + delta

    def _move_joints(self, arm_traj, hand_traj=None, threshold=0.02):
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

    def _move_cartesian(self, target_pose, threshold_t=0.002, threshold_r=0.02, vel_scale=1.0):
        target_rot = Rotation.from_matrix(target_pose[:3, :3])
        for _ in range(500):
            cur = self.arm.get_data()["position"].copy()
            t_delta = target_pose[:3, 3] - cur[:3, 3]
            t_dist = np.linalg.norm(t_delta)
            vel = self.cart_vel_limit * vel_scale
            if t_dist > vel:
                t_delta = t_delta / t_dist * vel
            cur[:3, 3] += t_delta
            cur_rot = Rotation.from_matrix(cur[:3, :3])
            r_delta = (target_rot * cur_rot.inv()).as_rotvec()
            r_dist = np.linalg.norm(r_delta)
            if r_dist > self.rot_vel_limit:
                r_delta = r_delta / r_dist * self.rot_vel_limit
            if r_dist > 0.001:
                cur[:3, :3] = (Rotation.from_rotvec(r_delta) * cur_rot).as_matrix()
            self.arm.move(cur, is_servo=True)
            time.sleep(self.dt)
            actual = self.arm.get_data()["position"]
            if (np.linalg.norm(actual[:3, 3] - target_pose[:3, 3]) < threshold_t
                    and np.linalg.norm((target_rot * Rotation.from_matrix(actual[:3, :3]).inv()).as_rotvec()) < threshold_r):
                break

    def _move_joint_sequential(self, target_qpos, joint_order, threshold=0.01):
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
        import os
        os.makedirs(save_dir, exist_ok=True)
        self.hand.start(os.path.join(save_dir, "hand"))
        self.arm.start(os.path.join(save_dir, "arm"))

    def stop_recording(self):
        self.arm.stop()
        self.hand.stop()

    def _log_state(self, state):
        ts = datetime.datetime.now().isoformat()
        self.state_timestamps.append({"state": state, "time": ts})

    def execute(self, plan_result: PlanResult, lift_height: float = 0.12):
        """
        Execute: init -> approach -> pregrasp -> grasp -> squeeze -> lift.
        State timestamps stored in self.state_timestamps.
        Returns the squeezed hand pose.
        """
        if not plan_result.success:
            print("Planning failed — nothing to execute.")
            return None

        self.state_timestamps = []
        traj = plan_result.traj
        pg_hand = self._convert(plan_result.pregrasp_pose)
        g_hand = self._convert(plan_result.grasp_pose)
        wrist_ee = plan_result.wrist_se3 @ np.linalg.inv(self._link6_to_wrist)

        if self.mode == "gui":
            return self._execute_gui(traj, pg_hand, g_hand, wrist_ee, lift_height)
        return self._execute_auto(traj, pg_hand, g_hand, wrist_ee, lift_height)

    def _execute_auto(self, traj, pg_hand, g_hand, wrist_ee, lift_height):
        """Reference: run_auto_v2.py lines 318-335"""
        sl = self.squeeze_level

        # 1. Return to init pose (joint 0 first)
        self._log_state("init")
        self._move_joint_sequential(self._xarm_init[:6], [0], threshold=0.06)

        # 2. Approach trajectory
        self._log_state("approach")
        hand_traj = np.array([self._convert(traj[i, 6:]) for i in range(len(traj))])
        self._move_joints(traj[:, :6], hand_traj)

        # 3. Pregrasp
        self._log_state("pregrasp")
        self._move_hand(pg_hand)

        # 4. Grasp
        self._log_state("grasp")
        self._move_hand(g_hand)

        # 5. Squeeze
        self._log_state("squeeze")
        for i in range(sl * 5):
            s_hand = g_hand * (1 + i / 5) - pg_hand * (i / 5)
            self._move_hand(s_hand)
            time.sleep(0.01)

        # 6. Lift
        self._log_state("lift")
        lift_pose = wrist_ee.copy()
        lift_pose[2, 3] += lift_height
        self._move_cartesian(lift_pose, vel_scale=1/1.5)

        self._log_state("done")
        return s_hand

    def _execute_gui(self, traj, pg_hand, g_hand, wrist_ee, lift_height):
        """Same sequence as _execute_auto, but via GUI waypoints."""
        from paradex.io.robot_controller.gui_controller import RobotGUIController

        sl = self.squeeze_level
        s_hand = g_hand * sl - pg_hand * (sl - 1)
        last_arm = traj[-1, :6]

        lift_pose = wrist_ee.copy()
        lift_pose[2, 3] += lift_height

        rgc = RobotGUIController(self.arm, self.hand)

        # 1. Init (joint 0 only — same as _execute_auto)
        self._log_state("init")
        rgc.add_waypoint("init", "joint", target=self._xarm_init[:6])

        # 2. Approach (full trajectory)
        self._log_state("approach")
        hand_traj = np.array([self._convert(traj[i, 6:]) for i in range(len(traj))])
        for i in range(len(traj)):
            rgc.add_waypoint(f"approach_{i}", "joint", target=traj[i, :6], hand_qpos=hand_traj[i])

        # 3. Pregrasp
        self._log_state("pregrasp")
        rgc.add_waypoint("pregrasp", "joint", target=last_arm, hand_qpos=pg_hand)

        # 4. Grasp
        self._log_state("grasp")
        rgc.add_waypoint("grasp", "joint", target=last_arm, hand_qpos=g_hand)

        # 5. Squeeze
        self._log_state("squeeze")
        rgc.add_waypoint("squeeze", "joint", target=last_arm, hand_qpos=s_hand)

        # 6. Lift
        self._log_state("lift")
        rgc.add_waypoint("lift", "cartesian", target=lift_pose)

        self._log_state("done")
        rgc.run()
        return s_hand

    def release(self, plan_result: PlanResult):
        """Release object and return arm to init pose."""
        if not plan_result.success:
            return

        pg_hand = self._convert(plan_result.pregrasp_pose)
        g_hand = self._convert(plan_result.grasp_pose)

        if self.mode == "gui":
            self._release_gui(pg_hand, g_hand)
        else:
            self._release_auto(pg_hand, g_hand)

    def _release_auto(self, pg_hand, g_hand):
        """Reference: run_auto_v2.py lines 357-375"""
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
        self._move_hand(self._convert(self._hand_init))

        # Return arm to init
        execute_order = [1, 2, 5, 0, 3, 4]
        if self.arm.get_data()["qpos"][1] < self._xarm_init[1]:
            execute_order = [2, 1, 5, 0, 3, 4]

        clear_view = self._xarm_init.copy()
        clear_view[0] -= 60 * np.pi / 180
        self._move_joint_sequential(clear_view[:6], execute_order, threshold=0.06)

    def _release_gui(self, pg_hand, g_hand):
        """Same release sequence via GUI waypoints."""
        from paradex.io.robot_controller.gui_controller import RobotGUIController

        sl = self.squeeze_level
        s_hand = g_hand * sl - pg_hand * (sl - 1)

        clear_view = self._xarm_init.copy()
        clear_view[0] -= 60 * np.pi / 180

        rgc = RobotGUIController(self.arm, self.hand)

        # Reverse: squeeze -> grasp -> pregrasp -> open -> arm return
        rgc.add_waypoint("release_grasp", "joint",
                         target=self.arm.get_data()["qpos"], hand_qpos=g_hand)
        rgc.add_waypoint("release_pregrasp", "joint",
                         target=self.arm.get_data()["qpos"], hand_qpos=pg_hand)
        rgc.add_waypoint("release_open", "joint",
                         target=self.arm.get_data()["qpos"], hand_qpos=self._convert(self._hand_init))
        rgc.add_waypoint("return_init", "joint", target=clear_view[:6])

        rgc.run()

    def shutdown(self):
        self.arm.end()
        self.hand.end()
