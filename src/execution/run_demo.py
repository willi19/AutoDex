#!/usr/bin/env python3
"""Demo mode: Perception -> Planning -> Execute for multiple objects.

No labeling, no recording. Object name given interactively each round.

Usage:
    python src/execution/run_demo.py --hand allegro
    python src/execution/run_demo.py --hand inspire --viz
"""
import argparse
import datetime
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
from autodex.planner import GraspPlanner
from autodex.planner.visualizer import ScenePlanVisualizer
from autodex.executor.real import RealExecutor
from src.execution.daemon.perception_pipeline import PerceptionPipeline
from src.execution.run_auto import (
    SAM3_HOSTS, FPOSE_HOSTS,
    find_planning_mesh, pose_world_to_scene_cfg,
)

logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')
logging.getLogger("curobo").setLevel(logging.WARNING)

_active_vis = None


def run_demo_trial(obj_name, hand, planner, pipeline, executor, rcc, viz=False):
    global _active_vis
    if _active_vis is not None:
        try:
            _active_vis.server.stop()
        except Exception:
            pass
        _active_vis = None

    dir_idx = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    img_dir = os.path.join(project_dir, "experiment", "demo", hand, obj_name, dir_idx)
    os.makedirs(img_dir, exist_ok=True)

    # ── 1. Capture ──
    print(f"\n{'='*60}")
    print(f"[1/4] Capturing -> {dir_idx}")
    t0 = time.time()
    rcc.start("image", False, os.path.join("shared_data", "AutoDex", "experiment", "demo", hand, obj_name, dir_idx, "raw"))
    rcc.stop()
    save_current_C2R(img_dir)
    save_current_camparam(img_dir)
    print(f"    Capture: {time.time() - t0:.1f}s")

    # ── 2. Perception ──
    print(f"[2/4] Perception...")
    t0 = time.time()
    pose_world, perc_timing = pipeline.run(capture_dir=img_dir)
    print(f"    Perception: {time.time() - t0:.1f}s")

    if pose_world is None:
        print("    Perception FAILED")
        return False

    # ── 3. Plan ──
    print(f"[3/4] Planning...")
    t0 = time.time()
    c2r = load_c2r(img_dir)
    scene_cfg = pose_world_to_scene_cfg(pose_world, c2r, obj_name)
    result = planner.plan(scene_cfg, obj_name, "selected_100",
                          skip_done=False, success_only=True, hand=hand)
    print(f"    Plan: {time.time() - t0:.1f}s  success={result.success}")

    if not result.success:
        print("    Planning FAILED")
        return False

    if viz:
        scene_vis = ScenePlanVisualizer(scene_cfg, result, port=8080, hand=hand)
        scene_vis.start_viewer(use_thread=True)
        _active_vis = scene_vis

    # ── 4. Execute ──
    print(f"[4/4] Executing...")
    t0 = time.time()
    executor.execute(result, lift_height=0.07)
    time.sleep(3)
    executor.release(result)
    print(f"    Execute: {time.time() - t0:.1f}s")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", type=str, default="allegro", choices=["allegro", "inspire"])
    parser.add_argument("--viz", action="store_true")
    args = parser.parse_args()

    # Init hardware
    rcc = remote_camera_controller("test_lookup_obstacle")

    # Init pipeline (obj set per trial via change_object)
    print("Initializing perception pipeline...")
    pipeline = PerceptionPipeline(
        sam3_hosts=SAM3_HOSTS,
        fpose_hosts=FPOSE_HOSTS,
        obj_name=None,
        depth_method="da3",
    )

    # Init planner (warmup once)
    print("Initializing planner...")
    planner = GraspPlanner(hand=args.hand)

    # Init executor
    print("Initializing executor...")
    executor = RealExecutor(mode="auto", hand_name=args.hand)

    # Warmup planner with dummy plan
    print("Warming up planner...")
    planner._init_motion_gen({"cuboid": {"table": {"dims": [2, 3, 0.2], "pose": [1.1, 0, -0.063, 1, 0, 0, 0]}}})

    n_total = 0

    while True:
        try:
            obj_name = input(f"\nObject name (q=quit, trial #{n_total + 1}): ").strip()
        except KeyboardInterrupt:
            break
        if obj_name == "q" or obj_name == "":
            break

        # Update FPose mesh
        pipeline.change_object(obj_name)

        n_total += 1
        run_demo_trial(obj_name, args.hand, planner, pipeline, executor, rcc, viz=args.viz)

    print(f"\nDemo done: {n_total} trials")
    executor.shutdown()
    pipeline.close()
    rcc.end()


if __name__ == "__main__":
    main()
