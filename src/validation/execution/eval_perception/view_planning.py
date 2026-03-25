#!/usr/bin/env python3
"""View planning trajectory in viser.

Two modes:
  trajectory  : show planned trajectory with ghost trail + timeline slider (default)
  candidates  : show all grasp candidates color-coded

Usage:
    python src/validation/execution/eval_perception/view_planning.py \
        --data_root ~/shared_data/mingi_object_test \
        --obj attached_container --episode 20260317_172712 --port 8080
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import trimesh

AUTODEX_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(AUTODEX_ROOT))
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "paradex"))

from paradex.visualization.visualizer.viser import ViserViewer
from autodex.utils.path import urdf_path, obj_path
from autodex.utils.robot_config import INIT_STATE
from autodex.utils.conversion import se32cart

MESH_ROOT = Path.home() / "shared_data/object_6d/data/mesh"
TABLE_POSE = [1.1, 0, -0.1 + 0.037, 1, 0, 0, 0]
TABLE_DIMS = [2, 3, 0.2]


def find_mesh(obj_name):
    for name in [f"{obj_name}.obj", "simplified.obj", "coacd.obj"]:
        p = MESH_ROOT / obj_name / name
        if p.exists():
            return str(p)
    # Try raw_mesh
    raw = Path(obj_path) / obj_name / "raw_mesh" / f"{obj_name}.obj"
    if raw.exists():
        return str(raw)
    objs = list((MESH_ROOT / obj_name).glob("*.obj"))
    if objs:
        return str(objs[0])
    raise FileNotFoundError(f"No mesh for {obj_name}")


def pose7_to_se3(pose_list):
    from scipy.spatial.transform import Rotation
    se3 = np.eye(4)
    se3[:3, 3] = pose_list[:3]
    wxyz = pose_list[3:7]
    se3[:3, :3] = Rotation.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]]).as_matrix()
    return se3


class PlanningResultViewer(ViserViewer):
    """Viewer for saved planning results (traj.npy, wrist_se3.npy)."""

    def __init__(self, capture_dir, obj_name, port=8080):
        super().__init__(port_number=port)

        self.obj_name = obj_name
        capture_dir = Path(capture_dir)

        # Load results
        self.traj = np.load(str(capture_dir / "traj.npy"))
        self.wrist_se3 = np.load(str(capture_dir / "wrist_se3.npy"))
        self.pregrasp_pose = np.load(str(capture_dir / "pregrasp_pose.npy"))
        self.pose_world = np.load(str(capture_dir / "pose_world.npy"))

        # C2R
        c2r_path = capture_dir / "cam_param" / "C2R.npy"
        if not c2r_path.exists():
            c2r_path = capture_dir / "C2R.npy"
        c2r = np.load(str(c2r_path)) if c2r_path.exists() else np.eye(4)
        self.pose_robot = np.linalg.inv(c2r) @ self.pose_world

        print(f"Trajectory: {len(self.traj)} frames")
        print(f"Object pose (robot): {self.pose_robot[:3, 3]}")

        self._add_scene()
        self._add_trajectory_robot()
        self._add_gui()

    def _add_scene(self):
        # Table
        table_se3 = pose7_to_se3(TABLE_POSE)
        table_mesh = trimesh.creation.box(extents=TABLE_DIMS)
        self.add_object("table", table_mesh, table_se3)
        self.change_color("table", [240/255, 240/255, 245/255, 0.5])

        # Object mesh
        mesh_path = find_mesh(self.obj_name)
        obj_mesh = trimesh.load(mesh_path, process=False)
        if isinstance(obj_mesh, trimesh.Scene):
            obj_mesh = trimesh.util.concatenate([g for g in obj_mesh.geometry.values()])
        self.add_trimesh("object", obj_mesh, self.pose_robot)

        # Robot at init
        urdf_full = os.path.join(urdf_path, "xarm_allegro.urdf")
        self.add_robot("xarm_init", urdf_full)
        self.robot_dict["xarm_init"].update_cfg(INIT_STATE)
        self.change_color("xarm_init", [0.7, 0.7, 0.7, 0.3])

        self.add_grid(size=12.0, cell_size=0.1, height=0.0)

    def _add_trajectory_robot(self):
        urdf_full = os.path.join(urdf_path, "xarm_allegro.urdf")
        urdf_hand = os.path.join(urdf_path, "allegro_hand_description_right.urdf")

        # Destination hand
        self.add_robot("dest_hand", urdf_hand, pose=self.wrist_se3)
        self.robot_dict["dest_hand"].update_cfg(self.pregrasp_pose)
        self.change_color("dest_hand", [0.3, 0.5, 1.0, 0.7])

        # Main trajectory robot
        self.add_robot("traj_robot", urdf_full)
        self.robot_dict["traj_robot"].update_cfg(self.traj[0])
        self.change_color("traj_robot", [0.7, 0.7, 0.7, 0.8])

        # Ghost trail (initially hidden)
        self.ghost_spacing = max(1, len(self.traj) // 10)
        self.ghost_positions = list(range(0, len(self.traj), self.ghost_spacing))

        for i, pos in enumerate(self.ghost_positions):
            ghost_name = f"ghost_{i}"
            self.add_robot(ghost_name, urdf_full)
            self.robot_dict[ghost_name].update_cfg(self.traj[pos])
            self.change_color(ghost_name, [0.7, 0.7, 0.7, 0.4])
            self.robot_dict[ghost_name].set_visibility(False)

    def _update_frame(self, frame):
        frame = min(frame, len(self.traj) - 1)
        self.robot_dict["traj_robot"].update_cfg(self.traj[frame])

        for i, ghost_pos in enumerate(self.ghost_positions):
            ghost_name = f"ghost_{i}"
            if frame >= ghost_pos:
                self.robot_dict[ghost_name].set_visibility(True)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--episode", type=str, required=True)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    capture_dir = Path(args.data_root) / args.obj / args.episode

    if not (capture_dir / "traj.npy").exists():
        print(f"No trajectory at {capture_dir / 'traj.npy'}")
        return

    mesh_path = find_mesh(args.obj)
    print(f"Mesh: {mesh_path}")

    vis = PlanningResultViewer(str(capture_dir), args.obj, port=args.port)

    print(f"Viewer running at http://localhost:{args.port}")
    while True:
        import time
        time.sleep(1)


if __name__ == "__main__":
    main()
