import os
import json
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from paradex.visualization.visualizer.viser import ViserViewer


def pose7_to_se3(pose_list):
    """Convert [x, y, z, w, qx, qy, qz] to 4x4 SE3 matrix."""
    se3 = np.eye(4)
    se3[:3, 3] = pose_list[:3]
    wxyz = pose_list[3:7]
    se3[:3, :3] = R.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]]).as_matrix()
    return se3


class SceneViewer(ViserViewer):
    """Base viewer for displaying scenes (meshes + cuboids) from scene_cfg or scene JSON.

    Provides:
    - load_scene_cfg(scene_cfg): Load from dict with 'mesh' and 'cuboid' keys
    - load_scene_json(json_path): Load from scene JSON file
    - clear_scene(): Remove all objects
    - obj_pose: SE3 of the target object (set after loading)
    """

    def __init__(self):
        super().__init__()
        self.obj_pose = np.eye(4)

    def load_scene_cfg(self, scene_cfg, target_color=None, obstacle_color=None):
        """Load scene from a scene_cfg dict.

        scene_cfg format:
            {
                "cuboid": {"name": {"dims": [x,y,z], "pose": [x,y,z,w,qx,qy,qz]}},
                "mesh":   {"name": {"file_path": "...", "pose": [x,y,z,w,qx,qy,qz]}}
            }
        """
        if target_color is None:
            target_color = [0.8, 0.8, 0.8, 0.3]
        if obstacle_color is None:
            obstacle_color = [0.5, 0.5, 0.5, 0.5]

        if 'cuboid' in scene_cfg:
            for name, data in scene_cfg['cuboid'].items():
                pose_se3 = pose7_to_se3(data['pose'])
                box = trimesh.creation.box(extents=data['dims'])
                self.add_object(f"cuboid_{name}", box, pose_se3)
                self.change_color(f"cuboid_{name}", obstacle_color)

        if 'mesh' in scene_cfg:
            for name, data in scene_cfg['mesh'].items():
                pose_se3 = pose7_to_se3(data['pose'])
                mesh = trimesh.load(data['file_path'])
                self.add_object(f"mesh_{name}", mesh, pose_se3)

                if name == "target":
                    self.obj_pose = pose_se3
                    self.change_color(f"mesh_{name}", target_color)
                else:
                    self.change_color(f"mesh_{name}", obstacle_color)

    def load_scene_json(self, json_path, target_color=None, obstacle_color=None):
        """Load scene from a JSON file (scene field contains mesh/cuboid)."""
        with open(json_path, 'r') as f:
            cfg = json.load(f)
        scene_cfg = cfg.get("scene", cfg)
        self.load_scene_cfg(scene_cfg, target_color, obstacle_color)

    def clear_scene(self):
        """Remove all objects and trajectories."""
        for name in list(self.obj_dict.keys()):
            try:
                self.obj_dict[name]['frame'].remove()
            except Exception:
                pass
        self.obj_dict.clear()
        self.frame_nodes.clear()
        self.clear_traj()
