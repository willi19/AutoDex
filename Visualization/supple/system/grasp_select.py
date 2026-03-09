"""
Grasp Selection Visualization
- Red: Collision detected
- Orange: No valid IK solution
- Green: Valid grasp (no collision + IK success)
- Failed grasps fade out over time (timeline-based)
"""

import os
import json
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from paradex.visualization.visualizer.viser import ViserViewer

from rsslib.path import project_dir, urdf_path, robot_configs_path, obj_path, load_candidate, load_scene
from rsslib.conversion import cart2se3, se32cart, se32action
from rsslib.curobo_util import filter_collision, CuroboIkSolver, xarm_init_pose, allegro_init_pose

from curobo.types.base import TensorDeviceType
from curobo.util_file import load_yaml


# Colors for different states (RGBA)
COLOR_NEUTRAL = [0.6, 0.4, 0.8, 0.7]     # Purple: neutral/retrieving
COLOR_COLLISION = [1.0, 0.0, 0.0, 0.7]   # Red: collision
COLOR_NO_IK = [1.0, 0.5, 0.0, 0.7]       # Orange: no IK solution
COLOR_VALID = [0.3, 0.5, 1.0, 0.7]       # Blue: valid grasp

# Animation settings (at 30fps) - per grasp
NEUTRAL_FRAMES = 30    # 1 sec: grasp appears in neutral color
REVEAL_FRAMES = 15     # 0.5 sec: transition to result color
SHOW_FRAMES = 60       # 2 sec: show result
FADE_FRAMES = 45       # 1.5 sec: fade out if failed
GRASP_CYCLE = NEUTRAL_FRAMES + REVEAL_FRAMES + SHOW_FRAMES + FADE_FRAMES  # frames per grasp


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


class GraspSelectVisualizer(ViserViewer):
    def __init__(self, obj_name, version="baseline", num_grasps=7):
        super().__init__()

        self.obj_name = obj_name
        self.version = version
        self.num_grasps = num_grasps

        # Get object pose from tabletop pose
        tabletop_pose_path = os.path.join(obj_path, obj_name, "processed_data", "info", "tabletop")
        tabletop_pose_filename = os.listdir(tabletop_pose_path)[1]
        self.obj_pose = np.load(os.path.join(tabletop_pose_path, tabletop_pose_filename))
        self.obj_pose[0, 3] += 0.4
        
        # Load grasps from candidates
        self.wrist_se3, self.hand_pose, self.scene_info = self._load_grasps()

        # Generate tabletop scene
        self.scene_cfg = get_tabletop_scene(self.obj_name, self.obj_pose)

        # Check collision and IK
        self.collision_mask = self._check_collision()
        self.ik_success = self._check_ik()

        # Precompute opacity for each frame
        self._precompute_animation()

        # Add scene to viewer
        self._add_scene()

        # Add grasp visualizations
        self._add_grasps()

        # Add GUI with timeline
        self._add_gui()
        self.add_video_capture_gui()
        self.add_grid(size=10.0, cell_size=0.1, height=0.0)
        self.add_view_save_gui()

    def _load_grasps(self):
        """Load grasp candidates from candidate folder"""

        wrist_se3_all, pregrasp_pose_all, grasp_pose_all, scene_info_all = load_candidate(
            self.obj_name, self.obj_pose, self.version, shuffle=False
        )

        total = len(wrist_se3_all)
        if total == 0:
            raise ValueError(f"No candidates found for {self.obj_name} with version {self.version}")

        indices = list(range(25, min(self.num_grasps+25, total)))

        wrist_se3 = wrist_se3_all[indices]
        pregrasp_pose = pregrasp_pose_all[indices]
        scene_info = [scene_info_all[i] for i in indices]

        return wrist_se3, pregrasp_pose, scene_info

    def _check_collision(self):
        """Check collision for all grasps"""
        collision = filter_collision(self.scene_cfg, self.wrist_se3, self.hand_pose)
        return collision

    def _check_ik(self):
        """Check IK feasibility for non-colliding grasps"""
        robot_cfg = load_yaml(os.path.join(robot_configs_path, "xarm_allegro.yml"))["robot_cfg"]
        tensor_args = TensorDeviceType()
        ik_solver = CuroboIkSolver(self.scene_cfg, robot_cfg, tensor_args)
        result = ik_solver.solve_ik_batch(self.wrist_se3)

        # Store IK solutions for visualization
        self.ik_solutions = result["solution"]  # [n_grasps, n_seeds, dof]

        return result["success"]

    def _precompute_animation(self):
        """Precompute color and opacity for each grasp at each frame"""
        n_grasps = len(self.wrist_se3)

        # Total frames = each grasp gets its cycle sequentially
        self.total_frames = n_grasps * GRASP_CYCLE

        # Store color (RGBA) for each frame and grasp
        self.color_timeline = np.zeros((self.total_frames, n_grasps, 4))

        for i in range(n_grasps):
            # Determine final color based on status
            if self.collision_mask[i]:
                final_color = COLOR_COLLISION
            elif not self.ik_success[i]:
                final_color = COLOR_NO_IK
            else:
                final_color = COLOR_VALID

            is_valid = (not self.collision_mask[i]) and self.ik_success[i]

            # Start frame for this grasp
            start_frame = i * GRASP_CYCLE

            for frame in range(self.total_frames):
                local_frame = frame - start_frame

                if local_frame < 0:
                    # Before this grasp's turn - invisible
                    self.color_timeline[frame, i] = [0, 0, 0, 0]

                elif local_frame < NEUTRAL_FRAMES:
                    # Phase 1: Neutral (purple)
                    self.color_timeline[frame, i] = COLOR_NEUTRAL

                elif local_frame < NEUTRAL_FRAMES + REVEAL_FRAMES:
                    # Phase 2: Transition from neutral to result color
                    t = (local_frame - NEUTRAL_FRAMES) / REVEAL_FRAMES
                    color = [
                        COLOR_NEUTRAL[0] * (1-t) + final_color[0] * t,
                        COLOR_NEUTRAL[1] * (1-t) + final_color[1] * t,
                        COLOR_NEUTRAL[2] * (1-t) + final_color[2] * t,
                        0.7
                    ]
                    self.color_timeline[frame, i] = color

                elif local_frame < NEUTRAL_FRAMES + REVEAL_FRAMES + SHOW_FRAMES:
                    # Phase 3: Show result color
                    self.color_timeline[frame, i] = final_color

                else:
                    # Phase 4: Fade out (only for failed grasps)
                    fade_local = local_frame - (NEUTRAL_FRAMES + REVEAL_FRAMES + SHOW_FRAMES)
                    if is_valid:
                        # Valid grasps stay visible forever
                        self.color_timeline[frame, i] = final_color
                    else:
                        # Failed grasps fade out
                        fade_progress = min(fade_local / FADE_FRAMES, 1.0)
                        opacity = 0.7 * (1.0 - fade_progress)
                        self.color_timeline[frame, i] = [
                            final_color[0], final_color[1], final_color[2], opacity
                        ]

    def _add_scene(self):
        """Add scene objects to viewer"""
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
                print(name)
                if name == "table":
                    self.change_color(f"cuboid_{name}", [240/255, 240/255, 245/255, 0.7])
                else:
                    self.change_color(f"cuboid_{name}", [0.5, 0.5, 0.5, 0.5])

        if 'mesh' in self.scene_cfg:
            for name, data in self.scene_cfg['mesh'].items():
                pose_list = data['pose']
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

    def _add_grasps(self):
        """Add grasp visualizations with color coding"""
        urdf_hand = os.path.join(urdf_path, "allegro_hand_description_right.urdf")

        for i in range(len(self.wrist_se3)):
            robot_name = f"grasp_{i}"
            self.add_robot(robot_name, urdf_hand, pose=self.wrist_se3[i])
            self.robot_dict[robot_name].update_cfg(self.hand_pose[i])

            # Start invisible (will be shown when their turn comes)
            self.robot_dict[robot_name].set_visibility(False)

    def _update_frame(self, frame):
        """Update grasp colors based on current frame"""
        frame = min(frame, self.total_frames - 1)

        for i in range(len(self.wrist_se3)):
            robot_name = f"grasp_{i}"
            color = self.color_timeline[frame, i]

            if color[3] <= 0.01:
                self.robot_dict[robot_name].set_visibility(False)
            else:
                self.robot_dict[robot_name].set_visibility(True)
                self.change_color(robot_name, color.tolist())

    def _add_gui(self):
        """Add GUI controls with timeline"""
        with self.server.gui.add_folder("Grasp Selection"):
            # Statistics
            n_total = len(self.wrist_se3)
            n_collision = self.collision_mask.sum()
            n_no_ik = ((~self.collision_mask) & (~self.ik_success)).sum()
            n_valid = ((~self.collision_mask) & self.ik_success).sum()

            self.server.gui.add_text(
                "Statistics",
                initial_value=f"Total: {n_total} | Collision: {n_collision} | No IK: {n_no_ik} | Valid: {n_valid}",
                disabled=True
            )

            # Timeline slider
            self.timeline = self.server.gui.add_slider(
                "Timeline",
                min=0,
                max=self.total_frames - 1,
                step=1,
                initial_value=0
            )

            @self.timeline.on_update
            def _(_):
                self._update_frame(int(self.timeline.value))

            # Reset button
            reset_btn = self.server.gui.add_button("Reset")

            @reset_btn.on_click
            def _(_):
                self.timeline.value = 0
                self._update_frame(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Grasp Selection Visualizer")
    parser.add_argument("--obj", type=str, default="attached_container", help="Object name")
    parser.add_argument("--version", type=str, default="tselected_100", help="Candidate version")
    parser.add_argument("--num_grasps", type=int, default=50, help="Number of grasps to visualize")

    args = parser.parse_args()

    vis = GraspSelectVisualizer(
        obj_name=args.obj,
        version=args.version,
        num_grasps=args.num_grasps
    )
    vis.start_viewer()
