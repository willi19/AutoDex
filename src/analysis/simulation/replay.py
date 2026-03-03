"""
Replay a recorded experiment in MuJoCo simulation and visualize the result.

Loads arm/hand trajectories from an experiment directory, runs them through
the MuJoCo simulator, and opens a ViserViewer to inspect the result.

Usage:
    python src/analysis/simulation/replay.py \
        --exp_dir /path/to/experiment \
        --obj_name organizer_beige
"""

import os
import sys
import argparse

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "paradex"))
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "RSS_2026"))

from paradex.visualization.visualizer.viser import ViserViewer
from rsslib.path import urdf_path, obj_path

from autodex.simulator.mujoco_sim import Simulator

XARM_ALLEGRO_URDF = os.path.join(urdf_path, "xarm_allegro.urdf")


# ---- Data loading ----

def load_trajectory(exp_dir):
    arm  = np.load(os.path.join(exp_dir, "arm",  "position.npy"))   # (T, 6)
    hand = np.load(os.path.join(exp_dir, "hand", "position.npy"))   # (T, 16)
    return arm, _parse_allegro(hand)


def load_obj_pose(exp_dir, obj_name):
    """Return object pose (4x4) in robot base frame."""
    ob_in_world = None
    for subdir in [f"{obj_name}_pose", "object_pose"]:
        p = os.path.join(exp_dir, "outputs", subdir, "optimized_pose_world.txt")
        if os.path.exists(p):
            ob_in_world = np.loadtxt(p)
            break
    if ob_in_world is None:
        for root, _, files in os.walk(os.path.join(exp_dir, "outputs")):
            if "optimized_pose_world.txt" in files:
                ob_in_world = np.loadtxt(os.path.join(root, "optimized_pose_world.txt"))
                break
    if ob_in_world is None:
        raise FileNotFoundError(
            f"optimized_pose_world.txt not found under {exp_dir}/outputs/"
        )
    C2R = np.load(os.path.join(exp_dir, "C2R.npy"))
    return np.linalg.inv(C2R) @ ob_in_world


def _parse_allegro(hand):
    """Reorder hand joints: position.npy order -> URDF joint order."""
    out = np.zeros_like(hand)
    out[...,  :4]  = hand[...,  4:8]
    out[...,  4:8] = hand[..., 12:16]
    out[..., 8:12] = hand[...,  :4]
    out[..., 12:16] = hand[...,  8:12]
    return out


def _unparse_allegro(hand):
    """Inverse: URDF joint order -> position.npy order (for ViserViewer)."""
    out = np.zeros_like(hand)
    out[...,  :4]  = hand[...,  8:12]
    out[...,  4:8] = hand[...,   :4]
    out[..., 8:12] = hand[..., 12:16]
    out[..., 12:16] = hand[...,  4:8]
    return out


# ---- Simulation ----

def run_sim(exp_dir, obj_name, obj_mass=0.1, table_z=0.037, steps_per_frame=5):
    """Run experiment in simulation. Returns (robot_traj, obj_traj, obj_init_pose, lift_m)."""
    arm_traj, hand_traj = load_trajectory(exp_dir)
    obj_pose = load_obj_pose(exp_dir, obj_name)

    sim = Simulator(headless=True, table_z=table_z, obj_mass=obj_mass)
    sim.load_robot_asset("xarm", "allegro")
    sim.load_object_asset(obj_name)
    sim.add_env("exp", {
        "robot":  {"xarm_allegro": ("xarm", "allegro")},
        "object": {obj_name: obj_name},
    }, obj_poses={obj_name: obj_pose})

    init_qpos = np.concatenate([arm_traj[0], hand_traj[0]])
    sim.reset("exp", {
        "robot":  {"xarm_allegro": init_qpos},
        "object": {obj_name: obj_pose},
    })
    init_obj_z = sim.get_state("exp")["object"][obj_name][2, 3]

    robot_states, obj_states = [], []
    for i in range(len(arm_traj)):
        qpos = np.concatenate([arm_traj[i], hand_traj[i]])
        sim.step("exp", {"robot": {"xarm_allegro": qpos}})
        sim.tick(n=steps_per_frame)
        state = sim.get_state("exp")
        robot_states.append(state["robot"]["xarm_allegro"]["qpos"])
        obj_states.append(state["object"][obj_name])

    robot_traj = np.stack(robot_states)   # (T, 22)
    obj_traj   = np.stack(obj_states)     # (T, 4, 4)
    lift       = float(obj_traj[:, 2, 3].max() - init_obj_z)
    sim.terminate()

    print(f"[replay] lift={lift*100:.1f} cm  "
          f"({'SUCCESS' if lift >= 0.05 else 'FAIL'})")

    return robot_traj, obj_traj, obj_pose, lift


# ---- Viewer ----

class ReplayViewer(ViserViewer):
    def __init__(self, exp_dir, obj_name, obj_mass=0.1,
                 table_z=0.037, steps_per_frame=5):
        super().__init__()

        robot_traj, obj_traj, obj_init_pose, _ = run_sim(
            exp_dir, obj_name, obj_mass, table_z, steps_per_frame
        )

        # ViserViewer expects hand joints in position.npy order
        arm_vis  = robot_traj[:, :6]
        hand_vis = _unparse_allegro(robot_traj[:, 6:])
        vis_traj = np.concatenate([arm_vis, hand_vis], axis=1)

        # Load object mesh
        mesh_file = os.path.join(obj_path, obj_name, "raw_mesh", f"{obj_name}.obj")
        obj_mesh  = trimesh.load(mesh_file)
        if isinstance(obj_mesh, trimesh.Scene):
            obj_mesh = obj_mesh.dump(concatenate=True)

        self.add_robot("xarm_allegro", XARM_ALLEGRO_URDF)
        self.add_trimesh("object", obj_mesh, obj_init_pose)
        self.add_traj("sim_replay",
                      robot_traj={"xarm_allegro": vis_traj},
                      obj_traj={"object": obj_traj})
        self.add_floor(table_z)

        print("[replay] Viewer ready at http://localhost:8080")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir",         required=True)
    parser.add_argument("--obj_name",        required=True)
    parser.add_argument("--obj_mass",        type=float, default=0.1)
    parser.add_argument("--table_z",         type=float, default=0.037)
    parser.add_argument("--steps_per_frame", type=int,   default=5)
    args = parser.parse_args()

    viewer = ReplayViewer(
        exp_dir=args.exp_dir,
        obj_name=args.obj_name,
        obj_mass=args.obj_mass,
        table_z=args.table_z,
        steps_per_frame=args.steps_per_frame,
    )
    viewer.start_viewer()