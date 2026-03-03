"""
Planning Visualization

Two modes (--mode):
  trajectory  : show the planned trajectory with ghost trail (default)
  candidates  : show all grasp candidates color-coded by filter result
                🔴 filtered (collision/backward)  🟡 not yet planned  🟢 success
"""

import os
import sys
import argparse
import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "paradex"))
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "RSS_2026"))

from paradex.visualization.visualizer.viser import ViserViewer
from rsslib.visualizer import GraspPlanningVisualizer
from rsslib.path import urdf_path, obj_path
from rsslib.robot_config import INIT_STATE
from autodex.planner import GraspPlanner


def load_tabletop_scene(obj_name):
    """Load object pose and build scene_cfg (SE3 format)."""
    pose_dir = os.path.join(obj_path, obj_name, "processed_data", "info", "tabletop")
    pose_file = sorted(os.listdir(pose_dir))[0]
    obj_pose = np.load(os.path.join(pose_dir, pose_file))
    obj_pose[0, 3] += 0.4

    mesh_path = os.path.join(obj_path, obj_name, "processed_data", "mesh", "simplified.obj")

    table_pose = np.eye(4)
    table_pose[2, 3] = -0.01

    scene_cfg = {
        "mesh": {
            "target": {"pose": obj_pose, "file_path": mesh_path},
        },
        "cuboid": {
            "table": {"dims": [10.0, 20.0, 0.02], "pose": table_pose},
        },
    }
    return scene_cfg, obj_pose


class PlanningVisualizer(ViserViewer):
    def __init__(self, obj_name, version="baseline"):
        super().__init__()

        self.obj_name = obj_name

        # plan
        self.scene_cfg, self.obj_pose = load_tabletop_scene(obj_name)
        planner = GraspPlanner()
        self.result = planner.plan(self.scene_cfg, obj_name=obj_name, grasp_version=version)

        if not self.result.success:
            print("Planning failed — no collision-free trajectory found.")
            self.num_frames = 1
            self._add_scene()
            return

        self.traj = self.result.traj
        self.num_frames = len(self.traj)

        self._add_scene()
        self._add_trajectory_robot()
        self._add_gui()

    def _add_scene(self):
        # table
        for name, data in self.scene_cfg.get("cuboid", {}).items():
            box = trimesh.creation.box(extents=data["dims"])
            self.add_object(f"cuboid_{name}", box, data["pose"])
            if name == "table":
                self.change_color(f"cuboid_{name}", [240/255, 240/255, 245/255, 0.5])

        # object mesh
        for name, data in self.scene_cfg.get("mesh", {}).items():
            file_path = os.path.join(obj_path, self.obj_name, "raw_mesh", f"{self.obj_name}.obj")
            mesh = trimesh.load(file_path)
            self.add_trimesh(f"mesh_{name}", mesh, data["pose"])
            if name == "target":
                self.change_color(f"mesh_{name}", [0.8, 0.8, 0.8, 0.5])

        # robot at init pose
        urdf_full = os.path.join(urdf_path, "xarm_allegro.urdf")
        self.add_robot("xarm_init", urdf_full)
        self.robot_dict["xarm_init"].update_cfg(INIT_STATE)
        self.change_color("xarm_init", [0.7, 0.7, 0.7, 0.5])

        self.add_grid(size=12.0, cell_size=0.1, height=0.0)

    def _add_trajectory_robot(self):
        urdf_full = os.path.join(urdf_path, "xarm_allegro.urdf")
        urdf_hand = os.path.join(urdf_path, "allegro_hand_description_right.urdf")

        # destination hand (pregrasp at wrist pose)
        self.add_robot("dest_hand", urdf_hand, pose=self.result.wrist_se3)
        self.robot_dict["dest_hand"].update_cfg(self.result.pregrasp_pose)
        self.change_color("dest_hand", [0.3, 0.5, 1.0, 0.7])

        # main robot for current frame
        self.add_robot("traj_robot", urdf_full)
        self.robot_dict["traj_robot"].update_cfg(self.traj[0])
        self.change_color("traj_robot", [0.7, 0.7, 0.7, 0.8])

        # ghost trail
        traj_len = len(self.traj)
        self.ghost_spacing = 6
        self.ghost_positions = list(range(0, traj_len, self.ghost_spacing))

        for i, pos in enumerate(self.ghost_positions):
            ghost_name = f"ghost_{i}"
            self.add_robot(ghost_name, urdf_full)
            self.robot_dict[ghost_name].update_cfg(self.traj[pos])
            self.robot_dict[ghost_name].set_visibility(False)

    def _update_frame(self, frame):
        frame = min(frame, len(self.traj) - 1)
        self.robot_dict["traj_robot"].update_cfg(self.traj[frame])

        for i, ghost_pos in enumerate(self.ghost_positions):
            ghost_name = f"ghost_{i}"
            if frame >= ghost_pos:
                self.robot_dict[ghost_name].set_visibility(True)
                self.change_color(ghost_name, [0.7, 0.7, 0.7, 0.4])
            else:
                self.robot_dict[ghost_name].set_visibility(False)

    def _add_gui(self):
        with self.server.gui.add_folder("Planning"):
            traj_len = len(self.traj)
            self.server.gui.add_text(
                "Info",
                initial_value=f"Trajectory: {traj_len} frames",
                disabled=True,
            )
            self.timeline = self.server.gui.add_slider(
                "Timeline", min=0, max=traj_len - 1, step=1, initial_value=0,
            )

            @self.timeline.on_update
            def _(_):
                self._update_frame(int(self.timeline.value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj",     type=str, default="attached_container")
    parser.add_argument("--version", type=str, default="tselected_100")
    parser.add_argument("--mode",    type=str, default="trajectory",
                        choices=["trajectory", "candidates"])
    args = parser.parse_args()

    if args.mode == "candidates":
        scene_cfg, obj_pose = load_tabletop_scene(args.obj)
        planner = GraspPlanner()
        wrist_se3, grasp_pose, filtered = planner.get_candidates(
            scene_cfg, obj_name=args.obj, grasp_version=args.version
        )
        succ     = np.zeros(len(wrist_se3), dtype=bool)
        traj_list = [None] * len(wrist_se3)
        vis = GraspPlanningVisualizer(scene_cfg, wrist_se3, grasp_pose, filtered, succ, traj_list)
    else:
        vis = PlanningVisualizer(obj_name=args.obj, version=args.version)

    vis.start_viewer()