#!/usr/bin/env python3
"""Automated mode: Perception (distributed) -> Planning -> Execute -> Label.

Uses distributed daemon pipeline (SAM3 x3 + FPose x3 + DA3 local)
for fast perception, then plans and executes on the robot.

Usage:
    python src/execution/run_auto.py --obj attached_container --n_trials 10
"""
import argparse
import chime
import datetime
import json
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from paradex.io.robot_controller import get_arm, get_hand
from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.io.camera_system.signal_generator import UTGE900
from paradex.io.camera_system.timestamp_monitor import TimestampMonitor
from paradex.utils.system import network_info
from paradex.calibration.utils import save_current_camparam, save_current_C2R, load_c2r

from autodex.utils.conversion import se32cart
from autodex.utils.path import project_dir, obj_path
from autodex.planner import GraspPlanner
from autodex.planner.obstacles import add_obstacles
from autodex.planner.visualizer import ScenePlanVisualizer
from autodex.executor.real import RealExecutor
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
    """Find mesh for cuRobo planning (simplified.obj in object/paradex)."""
    p = os.path.join(obj_path, obj_name, "processed_data", "mesh", "simplified.obj")
    if os.path.exists(p):
        return p
    p2 = os.path.join(obj_path, obj_name, "raw_mesh", f"{obj_name}.obj")
    if os.path.exists(p2):
        return p2
    raise FileNotFoundError(f"No planning mesh for {obj_name}")


TABLE_SURFACE_Z = -0.1 + 0.039 + 0.1  # 0.039

# Objects with y-axis cylindrical symmetry — snap to nearest tabletop pose
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

    # Load all tabletop poses
    tabletop_files = sorted(glob.glob(os.path.join(tabletop_dir, "*.npy")))
    if not tabletop_files:
        return pose_robot

    R_est = pose_robot[:3, :3]
    y_est = R_est @ np.array([0, 1, 0])  # object y-axis in robot frame

    # 1. Pick tabletop pose with closest y-axis z-component (frame-independent)
    best_diff = float("inf")
    best_R_tab = R_est

    for tf in tabletop_files:
        R_tab = np.load(tf)[:3, :3]
        y_tab_z = R_tab[2, 1]  # z-component of object y-axis
        diff = np.abs(np.abs(y_est[2]) - np.abs(y_tab_z))
        if diff < best_diff:
            best_diff = diff
            best_R_tab = R_tab.copy()
            # If y-axis sign is flipped, flip tabletop pose around y
            if y_est[2] * y_tab_z < 0:
                best_R_tab = best_R_tab @ np.diag([1, -1, -1]).astype(float)

    # 2. Rotate R_tab around robot z-axis so y-axis xy-projection matches R_est
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
    """Convert world-frame 4x4 pose to scene_cfg dict for planner."""
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


def get_label():
    while True:
        chime.success()
        label = input("Press 'y' if the grasp was successful, 'n' if not: ").strip().lower()
        if label in ("y", "n"):
            return label == "y"


_active_vis = None

def run_single_trial(
    obj_name, exp_name, grasp_version, depth_method, scene_type, viz,
    wall_gap, wall_angle, clutter_seed, clutter_min_dist, clutter_max_dist, clutter_n, success_only,
    shelf_width, shelf_depth, shelf_height, shelf_gap, shelf_back, shelf_sides, shelf_top,
    planner, pipeline, executor, rcc, sync_generator, timestamp_monitor,
):
    global _active_vis
    if _active_vis is not None:
        try:
            _active_vis.server.stop()
        except Exception:
            pass
        _active_vis = None
    """Run one capture -> perceive -> plan -> execute -> label cycle."""
    hand_type = "allegro"
    dir_idx = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    scene_prefix = scene_type if scene_type != "table" else ""
    if success_only and scene_prefix:
        scene_prefix = f"{scene_prefix}_success_only"
    elif success_only:
        scene_prefix = "success_only"
    img_dir = os.path.join(project_dir, "experiment", exp_name, scene_prefix, hand_type, obj_name, dir_idx) if scene_prefix else os.path.join(project_dir, "experiment", exp_name, hand_type, obj_name, dir_idx)
    os.makedirs(img_dir, exist_ok=True)

    timing = {}

    def _ts():
        return datetime.datetime.now().isoformat()

    # ── 1. Capture images ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[1/6] Capturing -> {dir_idx}")
    timing["capture_start"] = _ts()
    t0 = time.time()

    rcc.start("image", False, os.path.join("shared_data", "AutoDex", "experiment", exp_name, scene_prefix, hand_type, obj_name, dir_idx, "raw"))
    rcc.stop()
    save_current_C2R(img_dir)
    save_current_camparam(img_dir)
    timing["capture_s"] = round(time.time() - t0, 1)
    print(f"    Capture: {timing['capture_s']}s")

    # ── 2. Distributed perception (SAM3 x3 + DA3 + FPose x3 + Sil) ──
    print(f"[2/6] Perception (distributed, depth={depth_method})...")
    timing["perception_start"] = _ts()
    t0 = time.time()
    pose_world, perc_timing = pipeline.run(capture_dir=img_dir)
    timing["perception_s"] = round(time.time() - t0, 1)

    if pose_world is None:
        print("    Perception failed.")
        fail_result = {"dir_idx": dir_idx, "scene_type": scene_type, "success": False, "reason": "perception_failed", "timing": timing}
        with open(os.path.join(img_dir, "result.json"), "w") as f:
            json.dump(fail_result, f, indent=2)
        return fail_result

    if perc_timing:
        timing["perception_detail"] = perc_timing
    print(f"    Perception: {timing['perception_s']}s")

    # Save pose
    np.save(os.path.join(img_dir, "pose_world.npy"), pose_world)

    # ── 3. Build scene_cfg & plan ────────────────────────────────────────
    print(f"[3/6] Planning (version={grasp_version}, scene={scene_type})...")
    timing["planning_start"] = _ts()
    t0 = time.time()
    c2r = load_c2r(img_dir)
    scene_cfg = pose_world_to_scene_cfg(pose_world, c2r, obj_name)
    scene_cfg = add_obstacles(scene_cfg, scene_type, wall_gap=wall_gap, wall_angle=wall_angle,
                              seed=clutter_seed, clutter_min_dist=clutter_min_dist, clutter_max_dist=clutter_max_dist, clutter_n=clutter_n,
                              shelf_width=shelf_width, shelf_depth=shelf_depth, shelf_height=shelf_height,
                              shelf_gap=shelf_gap, shelf_back=shelf_back, shelf_sides=shelf_sides, shelf_top=shelf_top)
    result = planner.plan(scene_cfg, obj_name, grasp_version,
                          skip_done=(scene_type == "table"),
                          success_only=success_only)
    timing["plan_s"] = round(time.time() - t0, 1)
    print(f"    Plan: {timing['plan_s']}s  success={result.success}")

    if not result.success:
        print("    Planning FAILED — launching visualizer to inspect...")
        # Show scene + all candidate wrist poses
        wrist_se3, _, grasp_pose, filtered = planner.get_candidates(scene_cfg, obj_name, grasp_version,
                                                                    success_only=success_only, skip_done=(scene_type == "table"))
        fail_vis = ScenePlanVisualizer(scene_cfg, None, port=8080)
        fail_vis.add_candidates(wrist_se3, grasp_pose, filtered)
        fail_vis.start_viewer(use_thread=True)
        _active_vis = fail_vis
        chime.error()
        fail_result = {"dir_idx": dir_idx, "scene_type": scene_type, "success": False, "reason": "planning_failed", "timing": timing}
        with open(os.path.join(img_dir, "result.json"), "w") as f:
            json.dump(fail_result, f, indent=2)
        return fail_result

    # Save plan
    plan_dir = os.path.join(img_dir, "plan")
    os.makedirs(plan_dir, exist_ok=True)
    np.save(os.path.join(plan_dir, "traj.npy"), result.traj)
    np.save(os.path.join(plan_dir, "wrist_se3.npy"), result.wrist_se3)
    if result.timing:
        with open(os.path.join(plan_dir, "timing.json"), "w") as f:
            json.dump(result.timing, f, indent=2)
    print(f"    Scene info: {result.scene_info}")

    # ── 3.5 Visualize (optional, stays open until next trial) ──────────
    if viz:
        print("    Launching visualizer (http://localhost:8080)...")
        scene_vis = ScenePlanVisualizer(scene_cfg, result, port=8080)
        scene_vis.start_viewer(use_thread=True)
        _active_vis = scene_vis

    # ── 4. Execute ───────────────────────────────────────────────────────
    print(f"[4/6] Executing on robot...")
    timing["execution_start"] = _ts()

    raw_dir = os.path.join(img_dir, "raw")
    rcc.start("video", True, os.path.join("AutoDex", "experiment", exp_name, scene_prefix, hand_type, obj_name, dir_idx, "raw"))
    timestamp_monitor.start(os.path.join(raw_dir, "timestamps"))
    executor.start_recording(raw_dir)
    sync_generator.start(fps=30)

    t0 = time.time()
    s_hand = executor.execute(result)
    timing["execute_s"] = round(time.time() - t0, 1)
    timing["execution_states"] = executor.state_timestamps

    rcc.stop()
    timestamp_monitor.stop()
    sync_generator.stop()

    # ── 5. Label ─────────────────────────────────────────────────────────
    timing["label_start"] = _ts()
    print(f"[5/6] Label the result")
    rcc.start("image", False, os.path.join("shared_data", "AutoDex", "experiment", exp_name, scene_prefix, hand_type, obj_name, dir_idx, "label", "raw"))
    rcc.stop()
    succ = get_label()

    # ── 6. Release & save ────────────────────────────────────────────────
    print(f"[6/6] Releasing...")
    executor.release(result)
    executor.stop_recording()

    if s_hand is not None:
        np.save(os.path.join(img_dir, "squeeze_hand.npy"), s_hand)

    trial_result = {
        "dir_idx": dir_idx,
        "scene_type": scene_type,
        "success": succ,
        "scene_info": result.scene_info,
        "candidate_idx": result.timing.get("candidate_idx") if result.timing else None,
        "timing": timing,
    }
    with open(os.path.join(img_dir, "result.json"), "w") as f:
        json.dump(trial_result, f, indent=2)

    # Save result to candidate path (table only — other scenes are testing)
    if result.scene_info is not None and scene_type == "table":
        from autodex.utils.path import candidate_path
        sei = result.scene_info
        cand_result_path = os.path.join(candidate_path, grasp_version, obj_name, sei[0], sei[1], sei[2], "result.json")
        with open(cand_result_path, "w") as f:
            json.dump({"success": succ, "dir_idx": dir_idx}, f)

    print(f"    Result: {'SUCCESS' if succ else 'FAIL'}  saved to {img_dir}/result.json")
    return trial_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--grasp_version", type=str, default="selected_100")
    parser.add_argument("--exp_name", type=str, default=None, help="Defaults to grasp_version")
    parser.add_argument("--depth", type=str, default="da3", choices=["da3", "stereo"])
    parser.add_argument("--scene", type=str, default="table",
                        choices=["table", "wall", "shelf", "cluttered"])
    parser.add_argument("--wall_gap", type=float, default=0.04, help="Wall distance from object (meters)")
    parser.add_argument("--wall_angle", type=float, default=0.0, help="Wall rotation around object (degrees, 0=+y)")
    parser.add_argument("--clutter_seed", type=int, default=42, help="Random seed for cluttered scene (same seed = same obstacles)")
    parser.add_argument("--clutter_min_dist", type=float, default=0.12, help="Min distance from object (meters)")
    parser.add_argument("--clutter_max_dist", type=float, default=0.20, help="Max distance from object (meters)")
    parser.add_argument("--clutter_n", type=int, default=4, help="Number of clutter obstacles")
    parser.add_argument("--shelf_width", type=float, default=0.30, help="Shelf inner width (meters)")
    parser.add_argument("--shelf_depth", type=float, default=0.30, help="Shelf inner depth (meters)")
    parser.add_argument("--shelf_height", type=float, default=0.30, help="Shelf inner height (meters)")
    parser.add_argument("--shelf_gap", type=float, default=0.02, help="Gap between object and shelf (meters)")
    parser.add_argument("--no_shelf_back", action="store_true", help="Remove shelf back wall")
    parser.add_argument("--no_shelf_sides", action="store_true", help="Remove shelf side walls")
    parser.add_argument("--no_shelf_top", action="store_true", help="Remove shelf top panel")
    parser.add_argument("--success_only", action="store_true", help="Only use previously successful grasps")
    parser.add_argument("--viz", action="store_true", help="Launch scene visualizer after planning")
    args = parser.parse_args()
    if args.exp_name is None:
        args.exp_name = args.grasp_version

    # Hardware init
    rcc = remote_camera_controller("test_lookup_obstacle")
    sync_generator = UTGE900(**network_info["signal_generator"]["param"])
    timestamp_monitor = TimestampMonitor(**network_info["timestamp"]["param"])

    # Perception pipeline init (distributed daemons)
    mesh_path = find_mesh(args.obj)
    print(f"Initializing perception pipeline (mesh={mesh_path}, depth={args.depth})...")
    pipeline = PerceptionPipeline(
        sam3_hosts=SAM3_HOSTS,
        fpose_hosts=FPOSE_HOSTS,
        mesh_path=mesh_path,
        obj_name=args.obj,
        depth_method=args.depth,
    )

    # Planner init (warmup once, reuse across trials)
    print("Initializing planner...")
    planner = GraspPlanner()

    # Executor init (reuse across trials — reference: arm/hand init once)
    print("Initializing executor...")
    executor = RealExecutor(mode="auto")

    results = []
    trial = 0
    while True:
        trial += 1
        print(f"\n{'#'*60}")
        print(f"# Trial {trial}")
        print(f"{'#'*60}")

        chime.info()
        cmd = input("Press Enter to start trial, 'q' to quit: ").strip().lower()
        if cmd == "q":
            break

        trial_result = run_single_trial(
            obj_name=args.obj,
            exp_name=args.exp_name,
            grasp_version=args.grasp_version,
            depth_method=args.depth,
            scene_type=args.scene,
            viz=args.viz,
            wall_gap=args.wall_gap,
            wall_angle=args.wall_angle,
            clutter_seed=args.clutter_seed,
            clutter_min_dist=args.clutter_min_dist,
            clutter_max_dist=args.clutter_max_dist,
            clutter_n=args.clutter_n,
            success_only=args.success_only,
            shelf_width=args.shelf_width,
            shelf_depth=args.shelf_depth,
            shelf_height=args.shelf_height,
            shelf_gap=args.shelf_gap,
            shelf_back=not args.no_shelf_back,
            shelf_sides=not args.no_shelf_sides,
            shelf_top=not args.no_shelf_top,
            planner=planner,
            pipeline=pipeline,
            executor=executor,
            rcc=rcc,
            sync_generator=sync_generator,
            timestamp_monitor=timestamp_monitor,
        )
        results.append(trial_result)

        n_succ = sum(1 for r in results if r.get("success"))
        print(f"\n    Running total: {n_succ}/{len(results)} success")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.obj} x {len(results)} trials")
    n_succ = sum(1 for r in results if r.get("success"))
    if results:
        print(f"  Success: {n_succ}/{len(results)} ({100*n_succ/len(results):.0f}%)")
    else:
        print(f"  No trials run.")
    for r in results:
        status = "OK" if r.get("success") else r.get("reason", "FAIL")
        print(f"  {r['dir_idx']}: {status}")

    scene_pfx = args.scene if args.scene != "table" else ""
    if args.success_only and scene_pfx:
        scene_pfx = f"{scene_pfx}_success_only"
    elif args.success_only:
        scene_pfx = "success_only"
    summary_path = os.path.join(project_dir, "experiment", args.exp_name, scene_pfx, "allegro", args.obj, "summary.json") if scene_pfx else os.path.join(project_dir, "experiment", args.exp_name, "allegro", args.obj, "summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    executor.shutdown()
    pipeline.close()
    timestamp_monitor.end()
    sync_generator.end()
    rcc.end()


if __name__ == "__main__":
    main()