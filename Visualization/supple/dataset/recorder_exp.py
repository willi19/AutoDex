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
from paradex.calibration.utils import load_current_C2R, load_c2r
from rsslib.conversion import se32cart, cart2se3
from rsslib.path import candidate_path, obj_path, urdf_path, project_dir

def parse_allegro(allegro_traj):
    for i in range(allegro_traj.shape[1]):
        if allegro_traj[325, i] > 0.7:
            print(allegro_traj[325, i], i)
    ret_allegro_traj = np.zeros((allegro_traj.shape[0], 16))

    ret_allegro_traj[:, 0] = allegro_traj[:, 4] # index 5 11
    ret_allegro_traj[:, 1] = allegro_traj[:, 2] # index
    ret_allegro_traj[:, 2] = allegro_traj[:, 0] # index
    ret_allegro_traj[:, 3] = allegro_traj[:, 1] # index

    ret_allegro_traj[:, 4] = allegro_traj[:, 7] # middle
    ret_allegro_traj[:, 5] = allegro_traj[:, 15] # middle
    ret_allegro_traj[:, 6] = allegro_traj[:, 14] # middle
    ret_allegro_traj[:, 7] = allegro_traj[:, 12] # middle
    
    ret_allegro_traj[:, 8] = allegro_traj[:, 11] # last
    ret_allegro_traj[:, 9] = allegro_traj[:, 13] # last
    ret_allegro_traj[:, 10] = allegro_traj[:, 4] # last
    ret_allegro_traj[:, 11] = allegro_traj[:, 9] # last

    ret_allegro_traj[:, 12] = allegro_traj[:, 8] # thumb 15 2
    ret_allegro_traj[:, 13] = allegro_traj[:, 6] # thumb
    ret_allegro_traj[:, 14] = allegro_traj[:, 10] # thumb
    ret_allegro_traj[:, 15] = allegro_traj[:, 3] # thumb

    return ret_allegro_traj

class Renderer(ViserViewer):
    def __init__(self, version, obj_name, exp_idx, obj_T):
        super().__init__()
        exp_dir = os.path.join(
            project_dir, "experiment", version, obj_name, exp_idx
        )
        xarm_traj = np.load(os.path.join(exp_dir, "arm",  "position.npy"))
        hand_traj = np.load(os.path.join(exp_dir, "hand", "position.npy"))
        hand_traj = parse_allegro(hand_traj)
        print(xarm_traj.shape, hand_traj.shape)
        self.add_robot("asdf", os.path.join(urdf_path, "xarm_allegro.urdf"))

        obj_mesh = trimesh.load(os.path.join(obj_path, obj_name, "raw_mesh", f"{obj_name}.obj"))
        c2r = load_c2r(exp_dir)
        pose_path = os.path.join(
            exp_dir, "outputs",
            f"{obj_name}_pose", "optimized_pose_world.txt"
        )

        obj_init_T = np.loadtxt(pose_path).reshape(4, 4)
        obj_init_T = np.linalg.inv(c2r) @ obj_init_T

        # xarm = RobotWrapper(os.path.join(urdf_path, "xarm_allegro.urdf"))
        # print(xarm.link_names)
        # wrist_se3 = xarm.compute_forward_kinematics(
        #     np.concatenate([xarm_traj[809], hand_traj[809]]), ["base_link"]
        # )["base_link"]

        # wrist_se32 = xarm.compute_forward_kinematics(
        #     np.concatenate([xarm_traj[-1], hand_traj[-1]]), ["base_link"]
        # )["base_link"]

        # offset = np.linalg.inv(wrist_se3) @ obj_init_T
        # obj_final_T = wrist_se32 @ offset
        # obj_final_T[2, 3] += 0.02
        # obj_final_T[0, 3] += 0.01
        self.add_trimesh("object", obj_mesh, obj_init_T)

        self.add_traj("robot_traj", {"asdf": np.concatenate([xarm_traj, hand_traj], axis=1)})

        self.add_floor(0.0)

if __name__ == "__main__":
    version = "selected_100"
    obj_name = "attached_container"
    exp_idx = "20260121_163413"

    scene_type = "wall"
    scene_idx = "20"
    grasp_idx = "34"

    scene = json.load(open(os.path.join(obj_path, obj_name, "scene", f"{scene_type}", f"{scene_idx}.json")))["scene"]
    obj_T = cart2se3(np.array(scene["mesh"]["target"]["pose"]))

    vis = Renderer(version, obj_name, exp_idx, obj_T)
    vis.start_viewer()