"""
Grasp Location Visualizer

Show where an object can be grasped by displaying all grasp candidate wrist poses
on/around the object mesh, color-coded by feasibility.

Colors:
    🟢 Green  = valid (no collision, no backward)
    🔴 Red    = collision with scene/self
    🟡 Yellow = backward-pointing wrist (filtered)

Each grasp is shown as the Allegro hand at the pregrasp pose, positioned at the
candidate wrist SE3. The object mesh is shown semi-transparent so you can see
grasps from all angles.

Usage:
    python src/validation/planning/grasp_locations.py --obj attached_container --version selected_100
    python src/validation/planning/grasp_locations.py --obj attached_container --version selected_100 --show_axes
    python src/validation/planning/grasp_locations.py --obj attached_container --version selected_100 --no_collision_check
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "paradex"))

from autodex.planner import GraspPlanner
from autodex.utils.path import obj_path, urdf_path, load_candidate
from autodex.utils.conversion import se32cart, cart2se3
from autodex.visualizer import SceneViewer


COLOR_VALID = [0, 1, 0, 0.6]       # green
COLOR_COLLISION = [1, 0, 0, 0.6]   # red
COLOR_BACKWARD = [1, 1, 0, 0.6]    # yellow


def load_tabletop_scene(obj_name, pose_idx="000"):
    """Load object pose and build scene_cfg."""
    pose_dir = os.path.join(obj_path, obj_name, "processed_data", "info", "tabletop")
    pose_file = os.path.join(pose_dir, f"{pose_idx}.npy")
    if not os.path.exists(pose_file):
        available = sorted(os.listdir(pose_dir))
        raise FileNotFoundError(f"Pose {pose_idx} not found. Available: {available}")

    obj_pose = np.load(pose_file)
    obj_pose[0, 3] += 0.4

    mesh_path = os.path.join(obj_path, obj_name, "processed_data", "mesh", "simplified.obj")

    scene_cfg = {
        "mesh": {
            "target": {
                "pose": se32cart(obj_pose).tolist(),
                "file_path": mesh_path,
            },
        },
        "cuboid": {
            "table": {
                "dims": [2, 3, 0.2],
                "pose": [1.1, 0, -0.1 + 0.037, 1, 0, 0, 0],
            },
        },
    }
    return scene_cfg, obj_pose


class GraspLocationViewer(SceneViewer):
    """Viewer showing all grasp locations on an object."""

    def __init__(self, scene_cfg, wrist_se3, pregrasp, collision, backward,
                 show_axes=False, max_display=200):
        super().__init__()

        self.wrist_se3 = wrist_se3
        self.pregrasp = pregrasp
        self.collision = collision
        self.backward = backward
        self.valid = ~(collision | backward)
        self.n_grasps = len(wrist_se3)
        self.show_axes = show_axes

        # Load scene
        self.load_scene_cfg(scene_cfg, target_color=[0.8, 0.8, 0.8, 0.3])

        # Subsample if too many grasps for display performance
        if self.n_grasps > max_display:
            print(f"[viewer] Subsampling {self.n_grasps} grasps to {max_display} for display")
            # Keep proportional representation of each category
            indices = np.random.choice(self.n_grasps, max_display, replace=False)
            indices = np.sort(indices)
        else:
            indices = np.arange(self.n_grasps)
        self.display_indices = indices

        # Add hand robots for displayed grasps
        urdf_hand = os.path.join(urdf_path, "allegro_hand_description_right.urdf")
        for i in indices:
            name = f"grasp_{i}"
            self.add_robot(name, urdf_hand, pose=self.wrist_se3[i])
            self.robot_dict[name].update_cfg(self.pregrasp[i])
            self.change_color(name, self._get_color(i))

            if show_axes:
                self.add_frame(f"axis_{i}", self.wrist_se3[i], axes_length=0.03)

        # GUI
        self._build_gui()
        self._update_visibility()

    def _get_color(self, idx):
        if self.collision[idx]:
            return COLOR_COLLISION
        elif self.backward[idx]:
            return COLOR_BACKWARD
        else:
            return COLOR_VALID

    def _build_gui(self):
        n_valid = self.valid.sum()
        n_coll = self.collision.sum()
        n_back = self.backward.sum()

        with self.server.gui.add_folder("Grasp Locations"):
            self.server.gui.add_text(
                "Stats",
                initial_value=(
                    f"Total: {self.n_grasps} | "
                    f"Valid: {n_valid} ({100*n_valid/max(self.n_grasps,1):.0f}%) | "
                    f"Collision: {n_coll} | Backward: {n_back}"
                ),
                disabled=True,
            )

            with self.server.gui.add_folder("Filter"):
                self.show_valid_cb = self.server.gui.add_checkbox(
                    "Valid (green)", initial_value=True
                )
                self.show_collision_cb = self.server.gui.add_checkbox(
                    "Collision (red)", initial_value=True
                )
                self.show_backward_cb = self.server.gui.add_checkbox(
                    "Backward (yellow)", initial_value=True
                )

            self.opacity_slider = self.server.gui.add_slider(
                "Hand opacity", min=0.1, max=1.0, step=0.1, initial_value=0.6
            )

        for cb in [self.show_valid_cb, self.show_collision_cb, self.show_backward_cb]:
            @cb.on_update
            def _(event):
                self._update_visibility()

        @self.opacity_slider.on_update
        def _(event):
            self._update_opacity()

    def _update_visibility(self):
        for i in self.display_indices:
            is_coll = self.collision[i]
            is_back = self.backward[i] and not is_coll
            is_valid = self.valid[i]

            show = False
            if is_valid and self.show_valid_cb.value:
                show = True
            elif is_coll and self.show_collision_cb.value:
                show = True
            elif is_back and self.show_backward_cb.value:
                show = True

            self.robot_dict[f"grasp_{i}"].set_visibility(show)
            if self.show_axes and f"axis_{i}" in self.frame_nodes:
                self.frame_nodes[f"axis_{i}"].visible = show

    def _update_opacity(self):
        alpha = self.opacity_slider.value
        for i in self.display_indices:
            base = self._get_color(i)
            self.change_color(f"grasp_{i}", [base[0], base[1], base[2], alpha])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize where an object can be grasped")
    parser.add_argument("--obj", type=str, required=True, help="Object name")
    parser.add_argument("--version", type=str, required=True, help="Grasp candidate version")
    parser.add_argument("--pose_idx", type=str, default="000", help="Tabletop pose index")
    parser.add_argument("--show_axes", action="store_true", help="Show coordinate axes at each grasp")
    parser.add_argument("--max_display", type=int, default=200,
                        help="Max grasps to display (subsampled if more)")
    parser.add_argument("--no_collision_check", action="store_true",
                        help="Skip collision check (show all grasps as valid)")
    args = parser.parse_args()

    scene_cfg, obj_pose = load_tabletop_scene(args.obj, args.pose_idx)

    if args.no_collision_check:
        # Load candidates without collision check
        wrist_se3, pregrasp, grasp, _ = load_candidate(
            args.obj, obj_pose, args.version, shuffle=False
        )
        collision = np.zeros(len(wrist_se3), dtype=bool)
        backward = wrist_se3[:, 0, 2] < 0.3
        print(f"[grasp_locations] Loaded {len(wrist_se3)} candidates (no collision check)")
        print(f"  backward={backward.sum()}, valid={(~backward).sum()}")
    else:
        # Use planner for collision filtering
        planner = GraspPlanner()
        wrist_se3, pregrasp, grasp, filtered = planner.get_candidates(
            scene_cfg, obj_name=args.obj, grasp_version=args.version
        )
        # Separate collision from backward for color coding
        backward = wrist_se3[:, 0, 2] < 0.3
        collision = filtered & ~backward  # collision-only (not backward)

    vis = GraspLocationViewer(
        scene_cfg=scene_cfg,
        wrist_se3=wrist_se3,
        pregrasp=pregrasp,
        collision=collision,
        backward=backward,
        show_axes=args.show_axes,
        max_display=args.max_display,
    )
    vis.start_viewer()
