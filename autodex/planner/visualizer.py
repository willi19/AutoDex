"""Scene + trajectory visualizer using paradex ViserViewer.

Shows scene_cfg (cuboids, meshes), the planned trajectory, and grasp hand poses.
No dependency on rsslib — uses paradex.visualization directly.
"""
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from paradex.visualization.visualizer.viser import ViserViewer

from autodex.utils.path import urdf_path
from autodex.utils.conversion import cart2se3
import os


class ScenePlanVisualizer(ViserViewer):
    """Visualize scene_cfg + PlanResult in viser.

    Usage:
        vis = ScenePlanVisualizer(scene_cfg, plan_result)
        vis.start_viewer(use_thread=True)
        # ... later
        vis.start_viewer(use_thread=False)  # blocks
    """

    def __init__(self, scene_cfg, plan_result=None, port=8080):
        super().__init__(port_number=port)
        self.scene_cfg = scene_cfg
        self.plan_result = plan_result

        self._add_scene()

        if plan_result is not None and plan_result.success:
            self._add_trajectory(plan_result)

    def _pose7d_to_se3(self, pose_list):
        se3 = np.eye(4)
        se3[:3, 3] = pose_list[:3]
        wxyz = pose_list[3:7]
        se3[:3, :3] = R.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]]).as_matrix()
        return se3

    def _add_scene(self):
        # Cuboids
        for name, data in self.scene_cfg.get("cuboid", {}).items():
            dims = data["dims"]
            pose_se3 = self._pose7d_to_se3(data["pose"])
            box = trimesh.creation.box(extents=dims)
            self.add_object(f"cuboid_{name}", box, pose_se3)
            self.change_color(f"cuboid_{name}", [0.5, 0.5, 0.5, 0.4])

        # Meshes
        for name, data in self.scene_cfg.get("mesh", {}).items():
            pose_se3 = self._pose7d_to_se3(data["pose"])
            mesh = trimesh.load(data["file_path"])
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            self.add_object(f"mesh_{name}", mesh, pose_se3)
            if name == "target":
                self.change_color(f"mesh_{name}", [0.2, 0.8, 0.2, 0.5])
            else:
                self.change_color(f"mesh_{name}", [0.6, 0.4, 0.2, 0.5])

    def _add_trajectory(self, result):
        """Add trajectory robot + grasp hand."""
        # Full robot trajectory
        urdf_full = os.path.join(urdf_path, "xarm_allegro.urdf")
        self.add_robot("traj_robot", urdf_full)

        # Show final pose
        traj = result.traj
        self.robot_dict["traj_robot"].update_cfg(traj[-1])

        # Grasp hand at wrist
        urdf_hand = os.path.join(urdf_path, "allegro_hand_description_right.urdf")
        self.add_robot("grasp_hand", urdf_hand, pose=result.wrist_se3)
        self.robot_dict["grasp_hand"].update_cfg(result.grasp_pose)
        self.change_color("grasp_hand", [0.0, 1.0, 0.0, 0.6])

        # Trajectory slider
        n_frames = len(traj)
        self.traj_slider = self.server.gui.add_slider(
            "Trajectory Frame",
            min=0, max=n_frames - 1, step=1, initial_value=n_frames - 1,
        )

        @self.traj_slider.on_update
        def _on_frame(_):
            idx = int(self.traj_slider.value)
            self.robot_dict["traj_robot"].update_cfg(traj[idx])

    def start_viewer(self, use_thread=False):
        self.add_frame("base_frame", np.eye(4))
        if use_thread:
            import threading
            t = threading.Thread(target=self._block, daemon=True)
            t.start()
        else:
            self._block()

    def _block(self):
        print(f"Visualizer running at http://localhost:{self.port_number}")
        while True:
            import time
            time.sleep(1)

    def add_frame(self, name, pose):
        self.server.scene.add_frame(
            f"/frames/{name}",
            position=pose[:3, 3],
            wxyz=R.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]],
            axes_length=0.1,
            axes_radius=0.003,
        )