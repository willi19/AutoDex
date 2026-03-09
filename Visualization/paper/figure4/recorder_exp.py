import os
import glob
import json
import numpy as np
import trimesh
import transforms3d
from scipy.spatial.transform import Rotation as R
import shapely.geometry as geom

from paradex.utils.path import shared_dir
from paradex.visualization.visualizer.viser import ViserViewer
from paradex.robot.robot_wrapper import RobotWrapper
from paradex.calibration.utils import load_current_C2R, load_camparam
from rsslib.conversion import se32cart, cart2se3
from rsslib.path import candidate_path, obj_path, urdf_path, project_dir

def parse_allegro(allegro_traj):
    ret_allegro_traj = np.zeros((allegro_traj.shape[0], 16))
    ret_allegro_traj[:, :4] = allegro_traj[:, 4:8] # index
    ret_allegro_traj[:, 4:8] = allegro_traj[:, :4] # middle
    ret_allegro_traj[:, 8:12] = allegro_traj[:, 12:16] # last
    ret_allegro_traj[:, 12:16] = allegro_traj[:, 8:12] # thumb

    # ret_allegro_traj[:, :12] = allegro_traj[:, :12]
    return ret_allegro_traj

class Renderer(ViserViewer):
    def __init__(self, version, obj_name, exp_idx, obj_T):
        super().__init__()
        exp_dir = os.path.join(
            project_dir, "experiment", version, obj_name, exp_idx
        )
        xarm_traj = np.load(os.path.join(exp_dir, "raw","arm",  "position.npy"))[:1500]
        hand_traj = np.load(os.path.join(exp_dir, "raw", "hand", "position.npy"))[:1500]
        hand_traj = parse_allegro(hand_traj)
        print(xarm_traj.shape, hand_traj.shape)
        self.add_robot("asdf", os.path.join(urdf_path, "xarm_allegro.urdf"))

        obj_mesh = trimesh.load(os.path.join(obj_path, obj_name, "raw_mesh", f"{obj_name}.obj"))
        c2r = load_current_C2R()
        pose_path = os.path.join(
            exp_dir, "outputs",
            f"{obj_name}_pose", "optimized_pose_world.txt"
        )
        
        obj_init_T = np.loadtxt(pose_path).reshape(4, 4)
        obj_init_T = np.linalg.inv(c2r) @ obj_init_T

        xarm = RobotWrapper(os.path.join(urdf_path, "xarm_allegro.urdf"))
        print(xarm.link_names)
        wrist_se3 = xarm.compute_forward_kinematics(
            np.concatenate([xarm_traj[650], hand_traj[650]]), ["base_link"]
        )["base_link"]

        wrist_se32 = xarm.compute_forward_kinematics(
            np.concatenate([xarm_traj[-1], hand_traj[-1]]), ["base_link"]
        )["base_link"]

        offset = np.linalg.inv(wrist_se3) @ obj_init_T
        obj_final_T = wrist_se32 @ offset
        obj_final_T[2, 3] += 0.02
        obj_final_T[0, 3] -= 0.02
        obj_final_T[1, 3] += 0.02
        self.add_trimesh("object", obj_mesh, obj_final_T)
        
        self.add_traj("robot_traj", {"asdf": np.concatenate([xarm_traj, hand_traj], axis=1)})
        
        self.add_floor(0.0)

        intrinsic, extrinsic = load_camparam(exp_dir)
        cam_list = list(intrinsic.keys())
        for cam_name in cam_list:
            extmat = extrinsic[cam_name]
            extmat_4x4 = np.eye(4)
            extmat_4x4[:3, :4] = extmat 
            
            extmat_4x4 = np.linalg.inv(extmat_4x4 @ c2r)

            self.add_camera(
                cam_name,
                extmat_4x4,
                intrinsic[cam_name]["intrinsics_undistort"],
                size=0.1,
                color=(255, 179, 140)
            )
if __name__ == "__main__":
    version = "fourcam"
    obj_name = "soaptray"
    exp_idx = "20260130_151255"

    scene_type = "wall"
    scene_idx = "24"
    grasp_idx = "105"

    scene = json.load(open(os.path.join(obj_path, obj_name, "scene", f"{scene_type}", f"{scene_idx}.json")))["scene"]
    obj_T = cart2se3(np.array(scene["mesh"]["target"]["pose"]))
    
    vis = Renderer(version, obj_name, exp_idx, obj_T)
    vis.start_viewer()