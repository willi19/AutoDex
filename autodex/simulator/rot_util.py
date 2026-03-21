"""Rotation and interpolation utilities for MuJoCo simulation."""

import numpy as np
import transforms3d.quaternions as tq
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def interplote_pose(pose1, pose2, step):
    """Interpolate SE3 pose (pos + quat wxyz) with SLERP."""
    trans1, quat1 = pose1[:3], pose1[3:7]
    trans2, quat2 = pose2[:3], pose2[3:7]
    slerp = Slerp([0, 1], R.from_quat([quat1, quat2], scalar_first=True))
    trans_interp = np.linspace(trans1, trans2, step + 1)[1:]
    quat_interp = slerp(np.linspace(0, 1, step + 1))[1:].as_quat(scalar_first=True)
    return np.concatenate([trans_interp, quat_interp], axis=1)


def interplote_qpos(qpos1, qpos2, step):
    """Linear interpolation between joint configs."""
    return np.linspace(qpos1, qpos2, step + 1)[1:]


def np_get_delta_qpos(qpos1, qpos2):
    """Compute position delta (m) and angle delta (deg) between two [x,y,z,qw,qx,qy,qz]."""
    delta_pos = np.linalg.norm(qpos1[:3] - qpos2[:3])
    q1_inv = tq.qinverse(qpos1[3:])
    q_rel = tq.qmult(qpos2[3:], q1_inv)
    if np.abs(q_rel[0]) > 1:
        q_rel[0] = 1
    angle = 2 * np.arccos(q_rel[0])
    return delta_pos, np.degrees(angle)