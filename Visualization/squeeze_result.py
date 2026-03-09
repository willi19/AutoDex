import numpy as np
import os
import transforms3d
import trimesh
import torch
from scipy.spatial.transform import Rotation as R
import json

from paradex.utils.path import shared_dir
from paradex.visualization.visualizer.viser import ViserViewer

from curobo.types.base import TensorDeviceType
from curobo.util_file import load_yaml
from curobo.types.math import Pose
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

from paradex.utils.path import shared_dir, home_path
from paradex.utils.file_io import find_latest_directory
from paradex.visualization.visualizer.viser import ViserViewer
from paradex.robot.utils import get_robot_urdf_path
from paradex.robot.robot_wrapper import RobotWrapper

from rsslib.curobo_util import to_quat, load_world_config
from rsslib.path import robot_configs_path

mesh_path = os.path.join(shared_dir, "RSS2026_Mingi", "object", "paradex")
OBSTACLE = {
            'cuboid': 
                {
                 'table': {'dims': [2, 3, 0.2], 'pose': [1.1,0,-0.1+0.037,0,0,0,1]}, 
                }
            }

def se3_to_action(wrist_pose, hand_joints):
    action = np.zeros(6 + len(hand_joints))
    action[:3] = wrist_pose[:3, 3]
    action[3:6] = R.from_matrix(wrist_pose[:3, :3]).as_euler('zyx')
    action[6:] = hand_joints
    return action

def check_valid(q):

    robot_cfg = load_yaml(os.path.join(robot_configs_path, "allegro_floating.yml"))["robot_cfg"]
    world_config = load_world_config(OBSTACLE, {})
    tensor_args = TensorDeviceType()
    config = RobotWorldConfig.load_from_config(
        robot_cfg, world_config, collision_activation_distance=0.0, tensor_args=tensor_args
    )
    q = torch.tensor(q, dtype=torch.float32).to(tensor_args.device)
    rw = RobotWorld(config)
    
    d_world, d_self = rw.get_world_self_collision_distance_from_joints(q)
    collided = torch.logical_or(d_world > 0, d_self > 0)
    # collided = (d_world > 0)
    # print(d_world)
    return collided

class SqueezeResultVisualizer(ViserViewer):
    def __init__(self):
        super().__init__()

    def add_robot(self, name, urdf_path, pose=np.eye(4), succ_metric=[]):
        super().add_robot(name, urdf_path, pose)
        self.robot_dict[name]._visual_root_frame.visible = False
        self.robot_dict[name].succ_metric = succ_metric
    
    def update_scene(self, timestep):
        # 현재 timestep이 속한 trajectory 찾기
        cumulative_frames = 0
        current_traj = None
        local_timestep = timestep
        
        for traj_name, traj_data, traj_len in self.traj_list:
            if timestep < cumulative_frames + traj_len:
                # 이 trajectory에 속함
                current_traj = traj_data
                local_timestep = timestep - cumulative_frames
                # print(f"Updating scene to timestep {timestep} (trajectory '{traj_name}', local frame {local_timestep})")
                break
            
            cumulative_frames += traj_len
        
        if current_traj is None:
            print(f"Warning: timestep {timestep} out of range")
            return
        
        # 해당 trajectory의 local timestep으로 로봇 업데이트
        with self.server.atomic():
            for robot_name, robot in self.robot_dict.items():
                if robot_name in current_traj["robot"]:
                    robot.update_cfg(current_traj["robot"][robot_name][local_timestep])
                    if robot.succ_metric[timestep] == 1:
                        self.change_color(robot_name, [0,1,0,1])
                    elif robot.succ_metric[timestep] == 0:
                        self.change_color(robot_name, [1,0,0,1])
                    else:
                        self.change_color(robot_name, [1,1,0,1])

            for obj_name, obj in self.obj_dict.items():
                if obj_name in current_traj["object"]:
                    obj_transform = current_traj["object"][obj_name][local_timestep]
                    frame_handle = obj['frame']
                    
                    # Frame의 position과 rotation 업데이트
                    xyzw = R.from_matrix(obj_transform[:3, :3]).as_quat()
                    frame_handle.wxyz = xyzw[[3, 0, 1, 2]]
                    frame_handle.position = obj_transform[:3, 3]

        self.prev_timestep = timestep
        self.server.flush()
        
        if self.render_png.value:
            self.render_current_frame(timestep)
        
grasp_idx_list = os.listdir(os.path.join(shared_dir, "experiment", "squeeze_pose", "pringles"))
grasp_pose_path = os.path.join(shared_dir, "RSS2026_Mingi/candidate/after_sim/pringles")
obj_root_dir = os.path.join(shared_dir, "RSS2026_Mingi", "object", "paradex")

vis = SqueezeResultVisualizer()
mesh = trimesh.load(os.path.join(obj_root_dir, "pringles", "raw_mesh", "pringles.obj")) 
obj_pose = np.load(os.path.join(shared_dir, "inference", "bodex", "20251228_074306", "obj_T.npy"))
    
vis.add_object("object", mesh, obj_pose)
vis.add_floor(0.037)

traj_dict = {}
all_traj = []

squeeze_max = 10

for grasp_idx in os.listdir(grasp_pose_path):
    grasp_dir = os.path.join(shared_dir, "experiment", "squeeze_pose", "pringles", grasp_idx)
    robot_pose = np.load(os.path.join(grasp_pose_path, grasp_idx, "robot_pose.npy"))

    pregrasp_pose = robot_pose[0, :]
    grasp_pose = robot_pose[1, :]

    wrist_se3 = np.eye(4)
    wrist_se3[:3, :3] = transforms3d.quaternions.quat2mat(pregrasp_pose[3:7])
    wrist_se3[:3, 3] = pregrasp_pose[:3]

    wrist_se3 = obj_pose @ wrist_se3

    traj_dict[f"{grasp_idx}"] = np.zeros((squeeze_max+2, robot_pose.shape[1]-7))
    succ_result = [1, 1]
    for succ_idx in range(1, squeeze_max+1):
        result_path = f"/home/mingi/shared_data/experiment/squeeze_pose/pringles/{grasp_idx}/result_{succ_idx}.json"
        if not os.path.exists(result_path):
            succ_result.append(-1)
        else:
            succ_result.append(json.load(open(result_path, "r"))["success"])  

    collided = check_valid([se3_to_action(wrist_se3, grasp_pose[7:])])
    if collided[0]:
        succ_result = [0 for _ in succ_result]
    

    vis.add_robot(f"{grasp_idx}", "BODex/src/curobo/content/assets/robot/allegro_description/allegro_hand_description_right.urdf", pose=wrist_se3, succ_metric=succ_result)
    traj_dict[f"{grasp_idx}"][:2] = robot_pose[:2, 7:]
    for squeeze_idx in range(squeeze_max):
        traj_dict[f"{grasp_idx}"][squeeze_idx+2] = (robot_pose[1, 7:] * (squeeze_idx+2)) - robot_pose[0, 7:] * (squeeze_idx + 1)  # keep grasp pose during squeezing
    
    float_robot_pose = se3_to_action(wrist_se3, robot_pose[0, 7:])
    all_traj.append(float_robot_pose)


collided = check_valid(all_traj)
for i, grasp_idx in enumerate(os.listdir(grasp_pose_path)):
    if collided[i]:
        vis.change_color(f"{grasp_idx}", [1,0,0,1])
    else:
        vis.change_color(f"{grasp_idx}", [0,1,0,1])

vis.change_color("object", [0.8,0.8,0.8,0.3])
vis.add_traj("asdf",traj_dict)
vis.start_viewer()