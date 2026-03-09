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

def generate_sequential_trajectory(num_joints=22, steps_per_joint=100):
    """
    각 joint를 순차적으로 -1.57 ~ 1.57 사이를 왕복하는 trajectory 생성

    Args:
        num_joints: 전체 관절 개수 (xArm 6개 + Allegro Hand 16개 = 22개)
        steps_per_joint: 각 관절당 왕복에 사용할 타임스텝 수

    Returns:
        trajectory: (total_steps, num_joints) 배열
    """
    total_steps = num_joints * steps_per_joint
    trajectory = np.zeros((total_steps, num_joints))

    for joint_idx in range(num_joints):
        start_step = joint_idx * steps_per_joint
        end_step = (joint_idx + 1) * steps_per_joint

        # 절반은 -1.57 → 1.57, 나머지 절반은 1.57 → -1.57 (왕복)
        half_steps = steps_per_joint // 2

        # Forward: -1.57 → 1.57
        trajectory[start_step:start_step + half_steps, joint_idx] = np.linspace(-1.57, 1.57, half_steps)

        # Backward: 1.57 → -1.57
        trajectory[start_step + half_steps:end_step, joint_idx] = np.linspace(1.57, -1.57, steps_per_joint - half_steps)

    return trajectory

class Renderer(ViserViewer):
    def __init__(self):
        super().__init__()
        self.add_robot("asdf", os.path.join(urdf_path, "xarm_allegro.urdf"))

        # 전체 trajectory 생성 (xArm 6개 + Allegro Hand 16개 = 22개 관절)
        full_traj = generate_sequential_trajectory(num_joints=22, steps_per_joint=100)

        self.add_traj("robot_traj", {"asdf": full_traj})

        # self.add_floor(0.0)

if __name__ == "__main__":
    vis = Renderer()
    vis.start_viewer()
