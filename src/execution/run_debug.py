#!/usr/bin/env python3
"""Debug mode: Perception (distributed) -> Planning -> Visualize -> GUI Controller.

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

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.calibration.utils import save_current_camparam, save_current_C2R, load_c2r

from autodex.utils.conversion import se32cart
from autodex.utils.path import project_dir, obj_path
from autodex.utils.robot_config import INIT_STATE, XARM_INIT, ALLEGRO_INIT
from autodex.planner import GraspPlanner
from autodex.planner.obstacles import add_obstacles
from autodex.planner.visualizer import ScenePlanVisualizer
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


TABLE_SURFACE_Z = -0.1 + 0.039 + 0.1  # 0.039

CYLINDER_OBJECTS = [
    "pepper_tuna", "pepper_tuna_light", "pepsi", "pepsi_light",
]

def _snap_z_to_table(pose_robot, mesh_path):
    """Ensure mesh bottom doesn't go below table surface."""
    import trimesh

    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    verts = np.asarray(mesh.vertices)
    verts_h = np.hstack([verts, np.ones((len(verts), 1))])
    verts_robot = (pose_robot @ verts_h.T).T[:, :3]
    bottom_z = verts_robot[:, 2].min()

    if bottom_z < TABLE_SURFACE_Z:
        delta = TABLE_SURFACE_Z - bottom_z
        print(f"    [snap] Object bottom {bottom_z:.4f} < table {TABLE_SURFACE_Z:.4f}, raising by {delta:.4f}m")
        pose_robot = pose_robot.copy()
        pose_robot[2, 3] += delta

    return pose_robot


def _snap_cylinder_pose(pose_robot, obj_name):
    """For y-axis symmetric objects, snap rotation to nearest tabletop pose."""
    import glob

    tabletop_dir = os.path.join(obj_path, obj_name, "processed_data", "info", "tabletop")
    if not os.path.isdir(tabletop_dir):
        return pose_robot

    tabletop_files = sorted(glob.glob(os.path.join(tabletop_dir, "*.npy")))
    if not tabletop_files:
        return pose_robot

    R_est = pose_robot[:3, :3]
    y_est = R_est @ np.array([0, 1, 0])

    best_diff = float("inf")
    best_R_tab = R_est

    for tf in tabletop_files:
        R_tab = np.load(tf)[:3, :3]
        y_tab_z = R_tab[2, 1]
        diff = np.abs(np.abs(y_est[2]) - np.abs(y_tab_z))
        if diff < best_diff:
            best_diff = diff
            best_R_tab = R_tab.copy()
            if y_est[2] * y_tab_z < 0:
                best_R_tab = best_R_tab @ np.diag([1, -1, -1]).astype(float)

    y_tab = best_R_tab[:, 1]
    phi = np.arctan2(y_est[1], y_est[0]) - np.arctan2(y_tab[1], y_tab[0])
    c, s = np.cos(phi), np.sin(phi)
    R_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    best_R = R_z @ best_R_tab

    print(f"    [cylinder] Snapped (y-z diff={best_diff:.3f}, z-rot={np.degrees(phi):.1f}deg)")
    pose_robot = pose_robot.copy()
    pose_robot[:3, :3] = best_R

    return pose_robot


def pose_world_to_scene_cfg(pose_world, c2r, obj_name):
    pose_robot = np.linalg.inv(c2r) @ pose_world
    mesh_path = find_planning_mesh(obj_name)
    if obj_name in CYLINDER_OBJECTS:
        pose_robot = _snap_cylinder_pose(pose_robot, obj_name)
    pose_robot = _snap_z_to_table(pose_robot, mesh_path)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--grasp_version", type=str, default="selected_100")
    parser.add_argument("--exp_name", type=str, default="debug")
    parser.add_argument("--depth", type=str, default="da3", choices=["da3", "stereo"])
    parser.add_argument("--scene", type=str, default="table",
                        choices=["table", "wall", "shelf", "cluttered"])
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
        obj_name=obj_name,
        depth_method=args.depth,
    )

    timing = {}

    def _ts():
        return datetime.datetime.now().isoformat()

    # ── 2. Capture images ────────────────────────────────────────────────
    dir_idx = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    hand_type = "allegro"
    scene_prefix = args.scene if args.scene != "table" else ""
    img_dir = os.path.join(project_dir, "experiment", exp_name, scene_prefix, hand_type, obj_name, dir_idx) if scene_prefix else os.path.join(project_dir, "experiment", exp_name, hand_type, obj_name, dir_idx)
    os.makedirs(img_dir, exist_ok=True)

    print(f"[1/5] Capturing images -> {dir_idx}")
    timing["capture_start"] = _ts()
    t0 = time.time()
    rcc.start("image", False, os.path.join("shared_data", "AutoDex", "experiment", exp_name, scene_prefix, hand_type, obj_name, dir_idx, "raw"))
    rcc.stop()
    save_current_C2R(img_dir)
    save_current_camparam(img_dir)
    timing["capture_s"] = round(time.time() - t0, 1)
    print(f"    Capture: {timing['capture_s']}s")

    # ── 3. Distributed perception ────────────────────────────────────────
    print(f"[2/5] Perception (distributed, depth={args.depth})...")
    timing["perception_start"] = _ts()
    t0 = time.time()
    pose_world, perc_timing = pipeline.run(capture_dir=img_dir)
    timing["perception_s"] = round(time.time() - t0, 1)
    print(f"    Perception: {timing['perception_s']}s")

    if pose_world is None:
        print("Perception failed. Exiting.")
        timing["perception_failed"] = True
        with open(os.path.join(img_dir, "timing.json"), "w") as f:
            json.dump(timing, f, indent=2)
        pipeline.close()
        rcc.end()
        return

    np.save(os.path.join(img_dir, "pose_world.npy"), pose_world)
    if perc_timing:
        timing["perception_detail"] = perc_timing

    # ── 4. Plan ──────────────────────────────────────────────────────────
    print(f"[3/5] Planning (version={grasp_version})...")
    timing["planning_start"] = _ts()
    t0 = time.time()
    c2r = load_c2r(img_dir)
    scene_cfg = pose_world_to_scene_cfg(pose_world, c2r, obj_name)
    scene_cfg = add_obstacles(scene_cfg, args.scene)
    planner = GraspPlanner()
    result = planner.plan(scene_cfg, obj_name, grasp_version)
    timing["planning_s"] = round(time.time() - t0, 1)
    print(f"    Planning: {timing['planning_s']}s  success={result.success}")

    if not result.success:
        print("No valid trajectory found. Exiting.")
        timing["planning_success"] = False
        with open(os.path.join(img_dir, "timing.json"), "w") as f:
            json.dump(timing, f, indent=2)
        pipeline.close()
        rcc.end()
        return

    timing["planning_success"] = True
    if result.timing:
        timing["planning_detail"] = result.timing

    # Save plan
    plan_dir = os.path.join(img_dir, "plan")
    os.makedirs(plan_dir, exist_ok=True)
    np.save(os.path.join(plan_dir, "traj.npy"), result.traj)
    np.save(os.path.join(plan_dir, "wrist_se3.npy"), result.wrist_se3)
    np.save(os.path.join(plan_dir, "pregrasp_pose.npy"), result.pregrasp_pose)
    np.save(os.path.join(plan_dir, "grasp_pose.npy"), result.grasp_pose)
    print(f"    Scene info: {result.scene_info}")

    # ── 5. Visualize scene + trajectory ──────────────────────────────────
    print(f"[4/5] Launching scene visualizer (http://localhost:8080)...")
    vis = ScenePlanVisualizer(scene_cfg, result, port=8080)
    vis.start_viewer(use_thread=True)

    while True:
        cont = input("Press 'y' to execute on robot, 'q' to quit: ").strip().lower()
        if cont in ("y", "q"):
            break

    if cont == "q":
        print("Skipping execution.")
        with open(os.path.join(img_dir, "timing.json"), "w") as f:
            json.dump(timing, f, indent=2)
        pipeline.close()
        rcc.end()
        return

    # ── 6. GUI Controller ────────────────────────────────────────────────
    print(f"[5/5] Launching GUI controller...")
    timing["execution_start"] = _ts()

    from autodex.executor.real import RealExecutor
    executor = RealExecutor(mode="gui")
    s_hand = executor.execute(result)
    timing["execution_states"] = executor.state_timestamps

    # ── Label ────────────────────────────────────────────────────────────
    while True:
        label = input("Success? (y/n): ").strip().lower()
        if label in ("y", "n"):
            break

    timing["label_start"] = _ts()

    # Release & return to init
    executor.release(result)

    if s_hand is not None:
        np.save(os.path.join(img_dir, "squeeze_hand.npy"), s_hand)

    trial_result = {
        "scene_info": result.scene_info,
        "success": label == "y",
        "timing": timing,
    }
    with open(os.path.join(img_dir, "result.json"), "w") as f:
        json.dump(trial_result, f, indent=2)

    # Save result to candidate path (table only — other scenes are testing)
    if result.scene_info is not None and args.scene == "table":
        from autodex.utils.path import candidate_path
        sei = result.scene_info
        cand_result_path = os.path.join(candidate_path, grasp_version, obj_name, sei[0], sei[1], sei[2], "result.json")
        with open(cand_result_path, "w") as f:
            json.dump({"success": label == "y", "dir_idx": dir_idx}, f)

    print(f"Result saved to {img_dir}/result.json")

    executor.shutdown()
    pipeline.close()
    rcc.end()


if __name__ == "__main__":
    main()
