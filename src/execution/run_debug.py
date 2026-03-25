#!/usr/bin/env python3
"""Debug mode: Perception (distributed) -> Planning -> GUI Controller.

Uses distributed daemon pipeline for fast perception,
then launches the GUI controller for manual step-by-step execution.

Usage:
    python src/execution/run_debug.py --obj attached_container
"""
import argparse
import datetime
import json
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from paradex.io.robot_controller.gui_controller import RobotGUIController
from paradex.io.robot_controller import get_arm, get_hand
from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.calibration.utils import save_current_camparam, save_current_C2R, load_c2r

from autodex.utils.conversion import se32cart
from autodex.utils.path import project_dir, obj_path
from autodex.utils.robot_config import INIT_STATE, XARM_INIT, ALLEGRO_INIT, LINK6_TO_WRIST
from autodex.planner import GraspPlanner
from src.execution.daemon.perception_pipeline import PerceptionPipeline

logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')

MESH_ROOT = os.path.expanduser("~/shared_data/object_6d/data/mesh")

SAM3_HOSTS = [
    ("192.168.0.101", 5001),
    ("192.168.0.102", 5001),
    ("192.168.0.103", 5001),
]
FPOSE_HOSTS = [
    ("192.168.0.104", 5003),
    ("192.168.0.105", 5003),
    ("192.168.0.106", 5003),
]


def find_mesh(obj_name):
    base = os.path.join(MESH_ROOT, obj_name)
    for name in [f"{obj_name}.obj", "simplified.obj", "coacd.obj"]:
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    import glob
    objs = glob.glob(os.path.join(base, "*.obj"))
    if objs:
        return objs[0]
    raise FileNotFoundError(f"No mesh for {obj_name}")


def find_planning_mesh(obj_name):
    p = os.path.join(obj_path, obj_name, "processed_data", "mesh", "simplified.obj")
    if os.path.exists(p):
        return p
    p2 = os.path.join(obj_path, obj_name, "raw_mesh", f"{obj_name}.obj")
    if os.path.exists(p2):
        return p2
    raise FileNotFoundError(f"No planning mesh for {obj_name}")


def pose_world_to_scene_cfg(pose_world, c2r, obj_name):
    pose_robot = np.linalg.inv(c2r) @ pose_world
    return {
        "mesh": {
            "target": {
                "pose": se32cart(pose_robot).tolist(),
                "file_path": find_planning_mesh(obj_name),
            }
        },
        "cuboid": {
            "table": {
                "dims": [2, 3, 0.2],
                "pose": [1.1, 0, -0.1 + 0.037, 1, 0, 0, 0],
            }
        },
    }


def _convert_hand(hand_pose):
    """Reorder Allegro joints: move last 4 (thumb) to front."""
    out = hand_pose.copy()
    if hand_pose.ndim == 1:
        out[:4] = hand_pose[12:]
        out[4:] = hand_pose[:12]
    else:
        out[:, :4] = hand_pose[:, 12:]
        out[:, 4:] = hand_pose[:, :12]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--grasp_version", type=str, default="selected_100")
    parser.add_argument("--exp_name", type=str, default="debug")
    parser.add_argument("--depth", type=str, default="da3", choices=["da3", "stereo"])
    args = parser.parse_args()

    obj_name = args.obj
    exp_name = args.exp_name
    grasp_version = args.grasp_version

    rcc = remote_camera_controller("test_lookup_obstacle")

    # ── 1. Perception pipeline init ──────────────────────────────────────
    mesh_path = find_mesh(obj_name)
    print(f"Initializing perception pipeline (mesh={mesh_path}, depth={args.depth})...")
    pipeline = PerceptionPipeline(
        sam3_hosts=SAM3_HOSTS,
        fpose_hosts=FPOSE_HOSTS,
        mesh_path=mesh_path,
        depth_method=args.depth,
    )

    # ── 2. Capture images ────────────────────────────────────────────────
    dir_idx = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    img_dir = os.path.join(project_dir, "experiment", exp_name, obj_name, dir_idx)
    os.makedirs(img_dir, exist_ok=True)

    print(f"[1/4] Capturing images -> {dir_idx}")
    t0 = time.time()
    rcc.start("image", False, os.path.join("AutoDex", "experiment", exp_name, obj_name, dir_idx, "raw"))
    rcc.stop()
    save_current_C2R(img_dir)
    save_current_camparam(img_dir)
    print(f"    Capture: {time.time() - t0:.1f}s")

    # ── 3. Distributed perception ────────────────────────────────────────
    print(f"[2/4] Perception (distributed, depth={args.depth})...")
    t0 = time.time()
    pose_world, perc_timing = pipeline.run(capture_dir=img_dir)
    print(f"    Perception: {time.time() - t0:.1f}s")

    if pose_world is None:
        print("Perception failed. Exiting.")
        pipeline.close()
        rcc.end()
        return

    np.save(os.path.join(img_dir, "pose_world.npy"), pose_world)
    if perc_timing:
        with open(os.path.join(img_dir, "perception_timing.json"), "w") as f:
            json.dump(perc_timing, f, indent=2)

    # ── 4. Plan ──────────────────────────────────────────────────────────
    print(f"[3/4] Planning (version={grasp_version})...")
    t0 = time.time()
    c2r = load_c2r(img_dir)
    scene_cfg = pose_world_to_scene_cfg(pose_world, c2r, obj_name)
    planner = GraspPlanner()
    result = planner.plan(scene_cfg, obj_name, grasp_version)
    print(f"    Planning: {time.time() - t0:.1f}s  success={result.success}")

    if not result.success:
        print("No valid trajectory found. Exiting.")
        pipeline.close()
        rcc.end()
        return

    # Save plan
    plan_dir = os.path.join(img_dir, "plan")
    os.makedirs(plan_dir, exist_ok=True)
    np.save(os.path.join(plan_dir, "traj.npy"), result.traj)
    np.save(os.path.join(plan_dir, "wrist_se3.npy"), result.wrist_se3)
    np.save(os.path.join(plan_dir, "pregrasp_pose.npy"), result.pregrasp_pose)
    np.save(os.path.join(plan_dir, "grasp_pose.npy"), result.grasp_pose)
    if result.timing:
        with open(os.path.join(plan_dir, "timing.json"), "w") as f:
            json.dump(result.timing, f, indent=2)
    print(f"    Scene info: {result.scene_info}")

    # ── 5. GUI Controller ────────────────────────────────────────────────
    print(f"[4/4] Launching GUI controller...")
    traj = result.traj
    pg_hand = _convert_hand(result.pregrasp_pose)
    g_hand = _convert_hand(result.grasp_pose)
    wrist_ee = result.wrist_se3 @ np.linalg.inv(LINK6_TO_WRIST)

    squeeze_level = 10
    s_hand = g_hand * squeeze_level - pg_hand * (squeeze_level - 1)

    approach_traj = np.column_stack([
        traj[:, :6],
        np.array([_convert_hand(traj[i, 6:]) for i in range(len(traj))]),
    ])

    arm = get_arm("xarm")
    hand = get_hand("allegro")

    rgc = RobotGUIController(
        robot_controller=arm,
        hand_controller=hand,
        grasp_pose={
            "start": approach_traj[0, 6:],
            "pregrasp": pg_hand,
            "grasp": g_hand,
            "squeezed": s_hand,
        },
        approach_traj=approach_traj,
        lift_distance=120.0,
        place_distance=40.0,
    )
    rgc.run()

    # ── Label ────────────────────────────────────────────────────────────
    while True:
        label = input("Success? (y/n): ").strip().lower()
        if label in ("y", "n"):
            break

    json.dump(
        {"scene_info": result.scene_info, "success": label == "y"},
        open(os.path.join(img_dir, "result.json"), "w"),
    )
    print(f"Result saved to {img_dir}/result.json")

    arm.end()
    hand.end()
    pipeline.close()
    rcc.end()


if __name__ == "__main__":
    main()