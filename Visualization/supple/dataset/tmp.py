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
    for i in range(allegro_traj.shape[0]):
        if abs(allegro_traj[i, 5]-allegro_traj[i, 11]) > 1.0:# -allegro_traj[i, 2]) > 0.8:
            print(allegro_traj[i, 5], allegro_traj[i, 11], i) #, allegro_traj[i, 2], i)
            return True
    return False

if __name__ == "__main__":
    version = "selected_100"
    # obj_name = "attached_container"
    obj_list = os.listdir(os.path.join(
        project_dir, "experiment", version
    ))
    for obj_name in obj_list:
        exp_idx_list = os.listdir(os.path.join(
            project_dir, "experiment", version, obj_name
        ))
        for exp_idx in exp_idx_list:
            if not os.path.exists(os.path.join(
                project_dir, "experiment", version, obj_name, exp_idx, "hand", "position.npy"
            )):
                continue

            allegro_traj = np.load(os.path.join(
                project_dir, "experiment", version, obj_name, exp_idx, "hand", "position.npy"
            ))
            if parse_allegro(allegro_traj):
                print("Found!", exp_idx, obj_name)
