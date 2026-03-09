import os
import numpy as np

from autodex.visualizer.scene_viewer import SceneViewer
from autodex.utils.path import urdf_path

# Color constants
COLOR_SUCCESS = [0, 1, 0, 0.6]
COLOR_FAIL = [1, 0, 0, 0.6]
COLOR_PLANNING_FAIL = [1, 1, 0, 0.6]
COLOR_CONTACT_OBJ = (0, 1, 0)
COLOR_CONTACT_ROBOT = (1, 0, 0)


class GraspViewer(SceneViewer):
    """Interactive viewer for grasp candidates on an object.

    Displays Allegro hand at grasp poses, color-coded by result category.
    Supports:
    - Success/fail/collision filtering via checkboxes
    - Contact point visualization
    - Trajectory animation (squeeze or sim playback)

    Args:
        scene_cfg: Scene dict with 'mesh' and 'cuboid' keys.
        wrist_se3: (N, 4, 4) wrist poses in world frame.
        hand_joint: (N, 16) hand joint configurations to display.
        labels: (N,) int/bool array. 1=success, 0=fail. Used for color coding.
        collision: (N,) bool array. True=collision detected (optional).
        hand_urdf: Path to hand URDF. Defaults to allegro_hand_description_right.
        contact_points: (N, K, 3, 2) contact points per grasp (optional).
            [..., 0] = object contact, [..., 1] = robot contact.
        traj_joints: List of (T_i, 16) arrays, per-grasp hand joint trajectory (optional).
    """

    def __init__(
        self,
        scene_cfg,
        wrist_se3,
        hand_joint,
        labels,
        collision=None,
        hand_urdf=None,
        contact_points=None,
        traj_joints=None,
    ):
        super().__init__()

        self.wrist_se3 = wrist_se3
        self.hand_joint = hand_joint
        self.labels = np.asarray(labels, dtype=bool)
        self.collision = np.asarray(collision, dtype=bool) if collision is not None else np.zeros(len(labels), dtype=bool)
        self.contact_points = contact_points
        self.traj_joints = traj_joints
        self.n_grasps = len(wrist_se3)

        if hand_urdf is None:
            hand_urdf = os.path.join(urdf_path, "allegro_hand_description_right.urdf")
        self.hand_urdf = hand_urdf

        # Load scene
        self.load_scene_cfg(scene_cfg)

        # Add all grasp robots
        for i in range(self.n_grasps):
            name = f"grasp_{i}"
            self.add_robot(name, self.hand_urdf, pose=self.wrist_se3[i])
            self.robot_dict[name].update_cfg(self.hand_joint[i])
            self.change_color(name, self._get_color(i))
            self.robot_dict[name].set_visibility(False)

        # GUI
        with self.server.gui.add_folder("Grasp Viewer"):
            with self.server.gui.add_folder("Filter"):
                self.show_success = self.server.gui.add_checkbox("Success", initial_value=True)
                self.show_fail = self.server.gui.add_checkbox("Fail", initial_value=True)
                self.show_collision = self.server.gui.add_checkbox("Collision", initial_value=True)

            self.grasp_slider = self.server.gui.add_slider(
                "Grasp Index", min=0, max=max(self.n_grasps - 1, 0), step=1, initial_value=0
            )

            self.show_contact = self.server.gui.add_checkbox("Show Contacts", initial_value=False)

            self.stats_text = self.server.gui.add_text(
                "Stats", initial_value=self._get_stats_text(), disabled=True
            )

            if self.traj_joints is not None:
                self.show_traj = self.server.gui.add_checkbox("Show Trajectory", initial_value=False)
                self.traj_slider = self.server.gui.add_slider(
                    "Frame", min=0, max=1, step=1, initial_value=0, disabled=True
                )

        # Event handlers
        for cb in [self.show_success, self.show_fail, self.show_collision]:
            @cb.on_update
            def _(event):
                self._update_visibility()

        @self.grasp_slider.on_update
        def _(event):
            self._on_grasp_select()

        @self.show_contact.on_update
        def _(event):
            if self.show_contact.value:
                self._show_contacts(int(self.grasp_slider.value))
            else:
                self._clear_contacts()

        if self.traj_joints is not None:
            @self.show_traj.on_update
            def _(event):
                self._on_traj_toggle()

            @self.traj_slider.on_update
            def _(event):
                self._on_traj_frame()

        # Show all
        self._update_visibility()

    def _get_color(self, idx):
        if self.collision[idx]:
            return COLOR_FAIL
        elif self.labels[idx]:
            return COLOR_SUCCESS
        else:
            return COLOR_PLANNING_FAIL

    def _get_stats_text(self):
        n_coll = self.collision.sum()
        n_succ = (self.labels & ~self.collision).sum()
        n_fail = (~self.labels & ~self.collision).sum()
        return f"Total: {self.n_grasps} | Success: {n_succ} | Fail: {n_fail} | Collision: {n_coll}"

    def _update_visibility(self):
        for i in range(self.n_grasps):
            is_coll = self.collision[i]
            is_succ = self.labels[i] and not is_coll
            is_fail = not self.labels[i] and not is_coll

            show = False
            if is_succ and self.show_success.value:
                show = True
            elif is_fail and self.show_fail.value:
                show = True
            elif is_coll and self.show_collision.value:
                show = True

            self.robot_dict[f"grasp_{i}"].set_visibility(show)

    def _on_grasp_select(self):
        idx = int(self.grasp_slider.value)
        if self.show_contact.value:
            self._show_contacts(idx)
        if self.traj_joints is not None and hasattr(self, 'show_traj') and self.show_traj.value:
            traj = self.traj_joints[idx]
            self.traj_slider.max = len(traj) - 1
            self.traj_slider.value = 0
            self.robot_dict[f"grasp_{idx}"].update_cfg(traj[0])

    def _show_contacts(self, grasp_idx):
        self._clear_contacts()
        if self.contact_points is None:
            return
        cp = self.contact_points[grasp_idx]
        if cp is None:
            return
        for k in range(len(cp)):
            obj_pt = cp[k, :, 0]
            rob_pt = cp[k, :, 1]
            self.server.scene.add_icosphere(
                name=f"/contacts/cp_obj_{k}", radius=0.002, color=COLOR_CONTACT_OBJ, position=obj_pt
            )
            self.server.scene.add_icosphere(
                name=f"/contacts/cp_rob_{k}", radius=0.002, color=COLOR_CONTACT_ROBOT, position=rob_pt
            )

    def _clear_contacts(self):
        try:
            self.server.scene.remove("/contacts")
        except Exception:
            pass

    def _on_traj_toggle(self):
        idx = int(self.grasp_slider.value)
        if self.show_traj.value and self.traj_joints is not None:
            traj = self.traj_joints[idx]
            self.traj_slider.max = len(traj) - 1
            self.traj_slider.disabled = False
        else:
            self.traj_slider.disabled = True

    def _on_traj_frame(self):
        idx = int(self.grasp_slider.value)
        frame = int(self.traj_slider.value)
        if self.traj_joints is not None:
            traj = self.traj_joints[idx]
            if frame < len(traj):
                self.robot_dict[f"grasp_{idx}"].update_cfg(traj[frame])
