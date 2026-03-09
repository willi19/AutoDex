import os
import numpy as np

from autodex.visualizer.scene_viewer import SceneViewer
from autodex.utils.path import urdf_path


class GraspPlanningVisualizer(SceneViewer):
    """Interactive viewer for motion planning results.

    Two modes:
    - Overview: All candidates color-coded showing pregrasp pose
    - Trajectory: Play back a successful trajectory

    Args:
        scene_cfg: Scene dict with 'mesh' and 'cuboid' keys.
        wrist_se3: (N, 4, 4) wrist poses.
        pregrasp: (N, 16) hand joint configs at pregrasp.
        grasp_pose: (N, 16) hand joint configs at grasp.
        collision: (N,) bool array.
        succ: (N,) bool array (planning success).
        traj_list: List of (T_i, 22) arrays or None.
    """

    def __init__(self, scene_cfg, wrist_se3, pregrasp, grasp_pose, collision, succ, traj_list):
        super().__init__()

        self.wrist_se3 = wrist_se3
        self.pregrasp = pregrasp
        self.grasp_pose = grasp_pose
        self.collision = collision
        self.succ = succ
        self._traj_list = traj_list

        self.n_grasps = len(wrist_se3)
        self.success_indices = np.where(succ)[0]

        self.current_mode = "overview"

        # Load scene
        self.load_scene_cfg(scene_cfg)

        # Add hand robots for each grasp candidate (showing pregrasp pose)
        urdf_hand = os.path.join(urdf_path, "allegro_hand_description_right.urdf")
        for i in range(self.n_grasps):
            name = f"grasp_{i}"
            self.add_robot(name, urdf_hand, pose=self.wrist_se3[i])
            self.robot_dict[name].update_cfg(self.pregrasp[i])

            if self.collision[i]:
                color = [1, 0, 0, 0.6]
            elif not self.succ[i]:
                color = [1, 1, 0, 0.6]
            else:
                color = [0, 1, 0, 0.6]
            self.change_color(name, color)

        # Full arm+hand robot for trajectory playback (hidden initially)
        urdf_full = os.path.join(urdf_path, "xarm_allegro.urdf")
        self.add_robot("traj_robot", urdf_full)
        self.robot_dict["traj_robot"].set_visibility(False)

        # GUI
        with self.server.gui.add_folder("Grasp Planning"):
            self.mode_selector = self.server.gui.add_dropdown(
                "Mode", options=["Overview", "Trajectory"], initial_value="Overview"
            )

            with self.server.gui.add_folder("Filter"):
                self.show_success = self.server.gui.add_checkbox("Success", initial_value=True)
                self.show_planning_failed = self.server.gui.add_checkbox("Planning Failed", initial_value=True)
                self.show_collision = self.server.gui.add_checkbox("Collision", initial_value=True)

            success_options = [f"Grasp {i}" for i in self.success_indices]
            if not success_options:
                success_options = ["None"]

            self.grasp_selector = self.server.gui.add_dropdown(
                "Select Grasp", options=success_options, initial_value=success_options[0],
                disabled=(len(self.success_indices) == 0)
            )

            self.stats_text = self.server.gui.add_text(
                "Statistics", initial_value=self._get_stats_text(), disabled=True
            )

        # Event handlers
        @self.mode_selector.on_update
        def _(event):
            self._on_mode_change()

        @self.grasp_selector.on_update
        def _(event):
            if self.current_mode == "trajectory":
                self._show_trajectory()

        for cb in [self.show_success, self.show_planning_failed, self.show_collision]:
            @cb.on_update
            def _(event):
                if self.current_mode == "overview":
                    self._update_visibility()

        self._show_overview()

    def _get_stats_text(self):
        n_coll = self.collision.sum()
        n_fail = (~self.collision & ~self.succ).sum()
        n_succ = self.succ.sum()
        return f"Total: {self.n_grasps} | Collision: {n_coll} | Plan Failed: {n_fail} | Success: {n_succ}"

    def _update_visibility(self):
        for i in range(self.n_grasps):
            is_coll = self.collision[i]
            is_succ = self.succ[i]
            is_fail = not is_coll and not is_succ

            show = False
            if is_succ and self.show_success.value:
                show = True
            elif is_fail and self.show_planning_failed.value:
                show = True
            elif is_coll and self.show_collision.value:
                show = True

            self.robot_dict[f"grasp_{i}"].set_visibility(show)

    def _show_overview(self):
        self.current_mode = "overview"
        self.robot_dict["traj_robot"].set_visibility(False)
        self.gui_playing.value = False
        self.clear_traj()
        # Restore all hands to pregrasp pose
        for i in range(self.n_grasps):
            self.robot_dict[f"grasp_{i}"].update_cfg(self.pregrasp[i])
        self._update_visibility()

    def _show_trajectory(self):
        if len(self.success_indices) == 0:
            return

        grasp_name = self.grasp_selector.value
        if grasp_name == "None":
            return
        grasp_idx = int(grasp_name.split()[-1])

        traj = self._traj_list[grasp_idx]
        if traj is None:
            return

        self.gui_playing.value = False

        # Show only the selected grasp hand with grasp_pose, hide others
        for i in range(self.n_grasps):
            if i == grasp_idx:
                self.robot_dict[f"grasp_{i}"].update_cfg(self.grasp_pose[i])
                self.robot_dict[f"grasp_{i}"].set_visibility(True)
            else:
                self.robot_dict[f"grasp_{i}"].set_visibility(False)

        # Temporarily remove grasp_* from robot_dict so add_traj doesn't tile them
        grasp_robots = {}
        for i in range(self.n_grasps):
            key = f"grasp_{i}"
            grasp_robots[key] = self.robot_dict.pop(key)

        self.clear_traj()
        self.robot_dict["traj_robot"].set_visibility(True)
        self.add_traj(
            f"traj_{grasp_idx}",
            robot_traj={"traj_robot": traj},
        )

        # Restore grasp_* robots
        self.robot_dict.update(grasp_robots)

        self.gui_playing.value = True

    def _on_mode_change(self):
        mode = self.mode_selector.value.lower()
        if mode == "overview":
            self._show_overview()
        else:
            self.current_mode = "trajectory"
            self._show_trajectory()