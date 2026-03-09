import numpy as np
import transforms3d as t3d
from scipy.spatial.transform import Rotation as R


def cart2se3(cart):
    """7D Cartesian [x,y,z, qw,qx,qy,qz] -> 4x4 SE3 matrix."""
    ret = np.eye(4)
    ret[:3, 3] = cart[0:3]
    ret[:3, :3] = t3d.quaternions.quat2mat(cart[3:7])
    return ret


def se32cart(se3):
    """4x4 SE3 matrix -> 7D Cartesian [x,y,z, qw,qx,qy,qz]."""
    ret = np.zeros(7)
    ret[0:3] = se3[:3, 3]
    quat = t3d.quaternions.mat2quat(se3[:3, :3])
    ret[3:7] = quat
    return ret


def se32action(wrist_se3, hand_joints):
    """Wrist SE3 + hand joints -> action vector [x,y,z, roll,pitch,yaw, joints...]."""
    action = np.zeros(6 + len(hand_joints))
    action[:3] = wrist_se3[:3, 3]
    zyx = R.from_matrix(wrist_se3[:3, :3]).as_euler('zyx')
    action[3] = zyx[2]
    action[4] = zyx[1]
    action[5] = zyx[0]
    action[6:] = hand_joints
    return action
