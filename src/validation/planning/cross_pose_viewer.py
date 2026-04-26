"""
Cross-Pose Grasp Viewer

For each grasp, visualize which tabletop poses are valid (no hand-table collision).
Shows object + hand at each tabletop pose, colored green (valid) or red (collision).
Supports switching between objects via GUI dropdown.

Usage:
    python src/validation/planning/cross_pose_viewer.py --obj attached_container
    python src/validation/planning/cross_pose_viewer.py --hand inspire
    python src/validation/planning/cross_pose_viewer.py  # all objects with selected_100
"""

import os
import sys
import argparse
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "paradex"))

from paradex.visualization.visualizer.viser import ViserViewer
from autodex.planner.planner import GraspPlanner
from autodex.utils.path import obj_path, load_candidate, get_candidate_path

_ASSET_ROOT = os.path.join(os.path.expanduser("~"), "shared_data", "AutoDex", "content", "assets", "robot")
HAND_URDFS = {
    "allegro": os.path.join(_ASSET_ROOT, "allegro_description", "allegro_hand_description_right.urdf"),
    "inspire": os.path.join(_ASSET_ROOT, "inspire_description", "inspire_hand_right.urdf"),
}

COLOR_VALID = (0, 200, 0)
COLOR_COLLISION = (200, 0, 0)


def _color_robot(viewer, robot_name, rgb_255):
    robot = viewer.robot_dict[robot_name]
    for handle in robot._meshes.values():
        handle.color = rgb_255


def get_tabletop_poses(obj_name):
    pose_dir = os.path.join(obj_path, obj_name, "processed_data", "info", "tabletop")
    if not os.path.isdir(pose_dir):
        return []
    return sorted([f.replace(".npy", "") for f in os.listdir(pose_dir) if f.endswith(".npy")])


def load_tabletop_pose(obj_name, pose_idx):
    pose_dir = os.path.join(obj_path, obj_name, "processed_data", "info", "tabletop")
    return np.load(os.path.join(pose_dir, f"{pose_idx}.npy"))


def build_table_world_cfg():
    return {
        "cuboid": {
            "table": {"dims": [2, 2, 0.2], "pose": [0.0, 0.0, -0.1, 1, 0, 0, 0], "color": [0.5, 0.5, 0.5, 1.0]},
        },
        "mesh": {},
    }


def get_available_objects(hand, version):
    """Find objects that have both candidates and tabletop poses."""
    cand_dir = os.path.join(get_candidate_path(hand), version)
    if not os.path.isdir(cand_dir):
        return []
    objects = []
    for obj_name in sorted(os.listdir(cand_dir)):
        if get_tabletop_poses(obj_name):
            objects.append(obj_name)
    return objects


def compute_collision(planner, obj_name, version, hand):
    """Compute per-pose collision for all candidates of an object."""
    pose_indices = get_tabletop_poses(obj_name)
    identity = np.eye(4)
    wrist_obj, pregrasp, grasp, scene_info = load_candidate(
        obj_name, identity, version, shuffle=False, skip_done=False, hand=hand
    )
    table_world_cfg = build_table_world_cfg()

    valid_per_pose = {}
    for pose_idx in pose_indices:
        obj_pose = load_tabletop_pose(obj_name, pose_idx)
        wrist_world = obj_pose @ wrist_obj
        collision = planner._check_collision(table_world_cfg, wrist_world, pregrasp)
        valid_per_pose[pose_idx] = ~collision
        print(f"  Pose {pose_idx}: {(~collision).sum()}/{len(wrist_obj)} valid")

    return {
        "wrist_obj": wrist_obj,
        "pregrasp": pregrasp,
        "scene_info": scene_info,
        "valid_per_pose": valid_per_pose,
        "pose_indices": pose_indices,
    }


class CrossPoseViewer(ViserViewer):
    def __init__(self, obj_data, hand="allegro", port=8080):
        """
        obj_data: dict of obj_name -> compute_collision() result
        """
        super().__init__(port_number=port)
        self.obj_data = obj_data
        self.obj_names = sorted(obj_data.keys())
        self.hand = hand
        self.urdf_hand = HAND_URDFS.get(hand, HAND_URDFS["allegro"])
        self.spacing = 0.3

        # Scene elements tracking
        self._current_obj = None
        self._obj_handles = []  # object mesh names
        self._hand_names = []   # hand robot names
        self._table_name = None

        self._build_gui()
        self._load_object(self.obj_names[0])

    def _build_gui(self):
        with self.server.gui.add_folder("Cross-Pose Grasp"):
            self.obj_dropdown = self.server.gui.add_dropdown(
                "Object", options=self.obj_names, initial_value=self.obj_names[0],
            )
            self.grasp_slider = self.server.gui.add_slider(
                "Grasp Index", min=0, max=1, step=1, initial_value=0,
            )
            self.info_text = self.server.gui.add_text("Info", initial_value="", disabled=True)
            self.pose_text = self.server.gui.add_text("Poses", initial_value="", disabled=True)

        @self.obj_dropdown.on_update
        def _(event):
            self._load_object(self.obj_dropdown.value)

        @self.grasp_slider.on_update
        def _(event):
            self._update_grasp(int(self.grasp_slider.value))

    def _clear_scene(self):
        """Remove current object meshes and hands."""
        for name in self._obj_handles:
            if name in self.obj_dict:
                self.obj_dict[name]['handle'].remove()
                del self.obj_dict[name]
        for name in self._hand_names:
            if name in self.robot_dict:
                self.robot_dict[name].remove()
                del self.robot_dict[name]
        if self._table_name and self._table_name in self.obj_dict:
            self.obj_dict[self._table_name]['handle'].remove()
            del self.obj_dict[self._table_name]
        self._obj_handles = []
        self._hand_names = []
        self._table_name = None

    def _load_object(self, obj_name):
        """Load a new object into the scene."""
        self._clear_scene()
        self._current_obj = obj_name

        data = self.obj_data[obj_name]
        pose_indices = data["pose_indices"]
        N = len(data["wrist_obj"])

        # Load object mesh
        mesh_path = os.path.join(obj_path, obj_name, "raw_mesh", f"{obj_name}.obj")
        obj_mesh = trimesh.load(mesh_path)
        if isinstance(obj_mesh, trimesh.Scene):
            obj_mesh = obj_mesh.dump(concatenate=True)

        # Load tabletop poses
        tabletop_poses = {p: load_tabletop_pose(obj_name, p) for p in pose_indices}
        self._tabletop_poses = tabletop_poses

        # Add table
        table_y = (len(pose_indices) - 1) * self.spacing
        table = trimesh.creation.box(extents=[2, table_y + 1.0, 0.2])
        table_pose = np.eye(4)
        table_pose[1, 3] = table_y / 2
        table_pose[2, 3] = -0.1
        self._table_name = f"table_{obj_name}"
        self.add_object(self._table_name, table, table_pose)
        self.change_color(self._table_name, [0.5, 0.5, 0.5, 0.3])

        # Add objects + hands for each pose
        for i, pose_idx in enumerate(pose_indices):
            offset = np.eye(4)
            offset[1, 3] = i * self.spacing
            op = offset @ tabletop_poses[pose_idx]

            obj_vis_name = f"obj_{obj_name}_{pose_idx}"
            self.add_object(obj_vis_name, obj_mesh, op)
            self._obj_handles.append(obj_vis_name)

            hand_name = f"hand_{obj_name}_{pose_idx}"
            self.add_robot(hand_name, self.urdf_hand, pose=np.eye(4))
            self._hand_names.append(hand_name)

        # Compute per-grasp pose count (before slider update triggers callback)
        reach_matrix = np.array([data["valid_per_pose"][p] for p in pose_indices])
        self._grasp_pose_count = reach_matrix.sum(axis=0)

        # Update slider range
        self.grasp_slider.max = N - 1
        self.grasp_slider.value = 0

        self._update_grasp(0)

    def _update_grasp(self, grasp_idx):
        data = self.obj_data[self._current_obj]
        si = data["scene_info"][grasp_idx]
        pose_indices = data["pose_indices"]
        n_valid = int(self._grasp_pose_count[grasp_idx])
        self.info_text.value = (
            f"[{grasp_idx}] {si[0]}/{si[1]}/{si[2]} — "
            f"valid in {n_valid}/{len(pose_indices)} poses"
        )

        pose_strs = []
        for i, pose_idx in enumerate(pose_indices):
            is_valid = bool(data["valid_per_pose"][pose_idx][grasp_idx])

            offset = np.eye(4)
            offset[1, 3] = i * self.spacing

            wrist_world = self._tabletop_poses[pose_idx] @ data["wrist_obj"][grasp_idx]
            wrist_offset = offset @ wrist_world

            hand_name = f"hand_{self._current_obj}_{pose_idx}"
            robot = self.robot_dict[hand_name]
            robot._visual_root_frame.position = wrist_offset[:3, 3]
            robot._visual_root_frame.wxyz = R.from_matrix(wrist_offset[:3, :3]).as_quat()[[3, 0, 1, 2]]
            robot.update_cfg(data["pregrasp"][grasp_idx])

            color = COLOR_VALID if is_valid else COLOR_COLLISION
            _color_robot(self, hand_name, color)

            pose_strs.append(f"{pose_idx}:{'O' if is_valid else 'X'}")

        self.pose_text.value = " | ".join(pose_strs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, default=None, help="Object name (omit for all)")
    parser.add_argument("--version", type=str, default="selected_100")
    parser.add_argument("--hand", type=str, default="allegro")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    if args.obj:
        obj_names = [args.obj]
    else:
        obj_names = get_available_objects(args.hand, args.version)
        print(f"Found {len(obj_names)} objects")

    planner = GraspPlanner(hand=args.hand)

    obj_data = {}
    for obj_name in obj_names:
        print(f"\n{obj_name}:")
        obj_data[obj_name] = compute_collision(planner, obj_name, args.version, args.hand)

    print(f"\nStarting viewer on port {args.port}...")
    vis = CrossPoseViewer(
        obj_data=obj_data,
        hand=args.hand,
        port=args.port,
    )
    vis.start_viewer()


if __name__ == "__main__":
    main()
