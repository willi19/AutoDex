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

    ret_allegro_traj[:, 0] = allegro_traj[:, 7] # index 5 11
    ret_allegro_traj[:, 1] = allegro_traj[:, 2] # index
    ret_allegro_traj[:, 2] = allegro_traj[:, 0] # index
    ret_allegro_traj[:, 3] = allegro_traj[:, 1] # index

    ret_allegro_traj[:, 4] = allegro_traj[:, 5] # middle
    ret_allegro_traj[:, 5] = allegro_traj[:, 8] # middle
    ret_allegro_traj[:, 6] = allegro_traj[:, 14] # middle
    ret_allegro_traj[:, 7] = allegro_traj[:, 12] # middle
    
    ret_allegro_traj[:, 8] = allegro_traj[:, 11] # last
    ret_allegro_traj[:, 9] = allegro_traj[:, 13] # last
    ret_allegro_traj[:, 10] = allegro_traj[:, 4] # last
    ret_allegro_traj[:, 11] = allegro_traj[:, 9] # last

    ret_allegro_traj[:, 12] = allegro_traj[:, 0] # thumb 15 2
    ret_allegro_traj[:, 13] = allegro_traj[:, 6] # thumb
    ret_allegro_traj[:, 14] = allegro_traj[:, 10] # thumb
    ret_allegro_traj[:, 15] = allegro_traj[:, 3] # thumb

    return ret_allegro_traj

class Renderer(ViserViewer):
    def __init__(self):
        super().__init__()
        exp_dir = os.path.join(
            shared_dir, "allegro_order_debug"
        )
        hand_traj = np.load(os.path.join(exp_dir, "position.npy"))
        hand_traj = parse_allegro(hand_traj)
        self.add_robot("asdf", os.path.join(urdf_path, "allegro_hand_description_right.urdf"))
        self.add_traj("robot_traj", {"asdf": hand_traj})

if __name__ == "__main__":
    vis = Renderer()
    vis.start_viewer()