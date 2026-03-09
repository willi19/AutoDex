"""
Planning Visualization
- Shows trajectory from initial pose to successful grasp with ghost trail effect
- Multiple keyframes are displayed with decreasing opacity (trail effect)
"""

import os
import json
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from paradex.visualization.visualizer.viser import ViserViewer

from rsslib.path import project_dir, urdf_path, robot_configs_path, obj_path, load_candidate
from rsslib.conversion import cart2se3, se32cart, se32action
from rsslib.curobo_util import filter_collision, CuroboPlanner, get_traj, xarm_init_pose, allegro_init_pose

from curobo.types.base import TensorDeviceType
from curobo.util_file import load_yaml




def get_tabletop_scene(obj_name, obj_pose):
    """Generate simple tabletop scene with object"""
    mesh_path = os.path.join(obj_path, obj_name, "processed_data", "mesh", "simplified.obj")

    # Convert SE3 to pose format [x, y, z, qw, qx, qy, qz]
    pos = obj_pose[:3, 3]
    quat_xyzw = R.from_matrix(obj_pose[:3, :3]).as_quat()
    quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]

    scene = {
        "mesh": {
            "target": {
                "pose": [pos[0], pos[1], pos[2], quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]],
                "file_path": mesh_path,
            }
        },
        "cuboid": {
            "table": {
                "dims": [10.0, 20.0, 0.02],
                "pose": [2.0, 0.0, -0.01, 1, 0, 0, 0],
            }
        }
    }
    return scene


class PlanningVisualizer(ViserViewer):
    def __init__(self, obj_name, version="baseline", num_trail_frames=8):
        super().__init__()

        self.obj_name = obj_name
        self.version = version
        self.num_trail_frames = num_trail_frames

        # Get object pose from tabletop pose (same as grasp_select.py)
        tabletop_pose_path = os.path.join(obj_path, obj_name, "processed_data", "info", "tabletop")
        tabletop_pose_filename = os.listdir(tabletop_pose_path)[1]
        self.obj_pose = np.load(os.path.join(tabletop_pose_path, tabletop_pose_filename))
        self.obj_pose[0, 3] += 0.4

        # Generate tabletop scene
        self.scene_cfg = get_tabletop_scene(self.obj_name, self.obj_pose)

        # Load grasps and run planning
        self.wrist_se3, self.hand_pose, self.scene_info = self._load_grasps()
        self.collision_mask, self.succ_mask, self.plan_trajectories = self._run_planning()

        # Find first successful trajectory
        self.selected_idx = self._find_first_success()

        # Set num_frames for ViserViewer (prevents division by zero)
        if self.selected_idx is not None and self.plan_trajectories[self.selected_idx] is not None:
            self.num_frames = len(self.plan_trajectories[self.selected_idx])
        else:
            self.num_frames = 1

        # Add scene to viewer
        self._add_scene()

        # Add trajectory robot for animation
        if self.selected_idx is not None:
            self._add_trajectory_robot()

        # Add GUI
        self._add_gui()
        self.add_video_capture_gui()
        self.add_view_save_gui()

    def _load_grasps(self):
        """Load grasp candidates from candidate folder"""
        wrist_se3_all, pregrasp_pose_all, grasp_pose_all, scene_info_all = load_candidate(
            self.obj_name, self.obj_pose, self.version, shuffle=False
        )

        total = len(wrist_se3_all)
        if total == 0:
            raise ValueError(f"No candidates found for {self.obj_name} with version {self.version}")

        # Same indices as grasp_select.py (starting from 25), but only first 5
        indices = list(range(25, min(5 + 25, total)))

        wrist_se3 = wrist_se3_all[indices]
        pregrasp_pose = pregrasp_pose_all[indices]
        scene_info = [scene_info_all[i] for i in indices]

        return wrist_se3, pregrasp_pose, scene_info

    def _run_planning(self):
        """Run collision check and motion planning using get_traj"""
        # Setup planner
        robot_cfg = load_yaml(os.path.join(robot_configs_path, "xarm_allegro.yml"))["robot_cfg"]
        tensor_args = TensorDeviceType()
        planner = CuroboPlanner(self.scene_cfg, robot_cfg, tensor_args)

        # Use get_traj to get trajectories with proper finger poses
        succ_mask, traj_list, collision = get_traj(
            self.scene_cfg, self.wrist_se3, self.hand_pose, planner
        )

        return collision, succ_mask, traj_list

    def _find_first_success(self):
        """Find index of first successful grasp (non-collision and planning success)"""
        valid_mask = (~self.collision_mask) & self.succ_mask
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) > 0:
            return valid_indices[0]
        return None

    def _add_scene(self):
        """Add scene objects to viewer"""
        # Add cuboids
        if 'cuboid' in self.scene_cfg:
            for name, data in self.scene_cfg['cuboid'].items():
                dims = data['dims']
                pose_list = data['pose']

                pose_se3 = np.eye(4)
                pose_se3[:3, 3] = pose_list[:3]
                wxyz = pose_list[3:7]
                pose_se3[:3, :3] = R.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]]).as_matrix()

                box = trimesh.creation.box(extents=dims)
                self.add_object(f"cuboid_{name}", box, pose_se3)

                # Table: light cool white (240, 240, 245) with opacity 0.5
                if name == "table":
                    self.change_color(f"cuboid_{name}", [240/255, 240/255, 245/255, 0.5])
                else:
                    self.change_color(f"cuboid_{name}", [0.5, 0.5, 0.5, 0.5])

        # Add meshes
        if 'mesh' in self.scene_cfg:
            for name, data in self.scene_cfg['mesh'].items():
                pose_list = data['pose']
                # Use raw_mesh instead of simplified
                file_path = os.path.join(obj_path, self.obj_name, "raw_mesh", f"{self.obj_name}.obj")

                pose_se3 = np.eye(4)
                pose_se3[:3, 3] = pose_list[:3]
                wxyz = pose_list[3:7]
                pose_se3[:3, :3] = R.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]]).as_matrix()

                mesh = trimesh.load(file_path)
                self.add_trimesh(f"mesh_{name}", mesh, pose_se3)

                if name == "target":
                    self.change_color(f"mesh_{name}", [0.8, 0.8, 0.8, 0.5])
                else:
                    self.change_color(f"mesh_{name}", [0.6, 0.4, 0.2, 0.5])

        # Add xarm_allegro at init pose (to show robot reach)
        urdf_full = os.path.join(urdf_path, "xarm_allegro.urdf")
        self.add_robot("xarm_init", urdf_full)

        # Init pose for xarm + allegro (from curobo_util)
        init_cfg = np.concatenate([xarm_init_pose, allegro_init_pose])
        self.robot_dict["xarm_init"].update_cfg(init_cfg)
        self.change_color("xarm_init", [0.7, 0.7, 0.7, 0.5])  # Gray, semi-transparent

        # Add floor
        self.add_grid(size=12.0, cell_size=0.1, height=0.0)

    def _add_trajectory_robot(self):
        """Add robots for trajectory animation with ghost trail"""
        if self.selected_idx is None:
            return

        traj = self.plan_trajectories[self.selected_idx]
        if traj is None:
            return

        urdf_full = os.path.join(urdf_path, "xarm_allegro.urdf")
        urdf_hand = os.path.join(urdf_path, "allegro_hand_description_right.urdf")

        # Add destination hand in blue (same as grasp_select valid color)
        self.add_robot("dest_hand", urdf_hand, pose=self.wrist_se3[self.selected_idx])
        self.robot_dict["dest_hand"].update_cfg(self.hand_pose[self.selected_idx])
        self.change_color("dest_hand", [0.3, 0.5, 1.0, 0.7])  # Blue like COLOR_VALID

        # Add main robot for current position
        self.add_robot("traj_robot", urdf_full)
        self.robot_dict["traj_robot"].update_cfg(traj[0])
        self.change_color("traj_robot", [0.7, 0.7, 0.7, 0.8])

        # Add ghost trail robots at fixed positions (every 6 timesteps)
        traj_len = len(traj)
        self.ghost_spacing = 6
        self.ghost_positions = list(range(0, traj_len, self.ghost_spacing))  # [0, 6, 12, 18, ...]
        self.num_ghosts = len(self.ghost_positions)

        for i, pos in enumerate(self.ghost_positions):
            ghost_name = f"ghost_{i}"
            self.add_robot(ghost_name, urdf_full)
            self.robot_dict[ghost_name].update_cfg(traj[pos])
            self.robot_dict[ghost_name].set_visibility(False)

    def _update_frame(self, frame):
        """Update robot pose based on current frame with ghost trail"""
        if self.selected_idx is None:
            return

        traj = self.plan_trajectories[self.selected_idx]
        if traj is None:
            return

        traj_len = len(traj)
        frame = min(frame, traj_len - 1)

        # Update main robot config
        self.robot_dict["traj_robot"].update_cfg(traj[frame])
        self.change_color("traj_robot", [0.7, 0.7, 0.7, 0.8])

        # Update ghost trail - show ghosts at positions 0, 6, 12, ... that are <= current frame
        for i, ghost_pos in enumerate(self.ghost_positions):
            ghost_name = f"ghost_{i}"

            # Show ghost if current frame has passed this position
            if frame >= ghost_pos:
                self.robot_dict[ghost_name].set_visibility(True)
                self.change_color(ghost_name, [0.7, 0.7, 0.7, 0.4])
            else:
                self.robot_dict[ghost_name].set_visibility(False)

    def _add_gui(self):
        """Add GUI controls with timeline"""
        with self.server.gui.add_folder("Planning Visualization"):
            # Statistics
            n_total = len(self.wrist_se3)
            n_collision = self.collision_mask.sum()
            n_planning_fail = ((~self.collision_mask) & (~self.succ_mask)).sum()
            n_success = ((~self.collision_mask) & self.succ_mask).sum()

            self.server.gui.add_text(
                "Statistics",
                initial_value=f"Total: {n_total} | Collision: {n_collision} | Plan Fail: {n_planning_fail} | Success: {n_success}",
                disabled=True
            )

            if self.selected_idx is not None:
                traj = self.plan_trajectories[self.selected_idx]
                traj_len = len(traj) if traj is not None else 0
                self.server.gui.add_text(
                    "Selected Grasp",
                    initial_value=f"Index: {self.selected_idx} | Trajectory Length: {traj_len}",
                    disabled=True
                )

                # Timeline slider
                self.timeline = self.server.gui.add_slider(
                    "Timeline",
                    min=0,
                    max=traj_len - 1,
                    step=1,
                    initial_value=0
                )

                @self.timeline.on_update
                def _(_):
                    self._update_frame(int(self.timeline.value))

                # Grasp selector dropdown
                valid_mask = (~self.collision_mask) & self.succ_mask
                valid_indices = np.where(valid_mask)[0]

                if len(valid_indices) > 1:
                    options = [f"Grasp {idx}" for idx in valid_indices]
                    self.grasp_selector = self.server.gui.add_dropdown(
                        "Select Grasp",
                        options=options,
                        initial_value=f"Grasp {self.selected_idx}"
                    )

                    @self.grasp_selector.on_update
                    def _(_):
                        self._on_grasp_change()
            else:
                self.server.gui.add_text(
                    "Status",
                    initial_value="No successful trajectory found!",
                    disabled=True
                )

    def _on_grasp_change(self):
        """Handle grasp selection change"""
        grasp_name = self.grasp_selector.value
        new_idx = int(grasp_name.split()[-1])

        if new_idx == self.selected_idx:
            return

        # Update selected index
        self.selected_idx = new_idx

        # Update timeline max and reset
        traj = self.plan_trajectories[self.selected_idx]
        if traj is not None:
            traj_len = len(traj)
            self.timeline.max = traj_len - 1
            self.timeline.value = 0

            # Update ghost positions for new trajectory length
            new_ghost_positions = list(range(0, traj_len, self.ghost_spacing))

            # Update existing ghosts or hide extra ones
            for i in range(self.num_ghosts):
                ghost_name = f"ghost_{i}"
                if i < len(new_ghost_positions):
                    self.robot_dict[ghost_name].update_cfg(traj[new_ghost_positions[i]])
                self.robot_dict[ghost_name].set_visibility(False)

            self.ghost_positions = new_ghost_positions
            self._update_frame(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Planning Visualizer with Trail Effect")
    parser.add_argument("--obj", type=str, default="attached_container", help="Object name")
    parser.add_argument("--version", type=str, default="tselected_100", help="Candidate version")
    parser.add_argument("--num_trail", type=int, default=8, help="Number of trail frames")

    args = parser.parse_args()

    vis = PlanningVisualizer(
        obj_name=args.obj,
        version=args.version,
        num_trail_frames=args.num_trail
    )
    vis.start_viewer()
