#!/usr/bin/env python3
"""Sequential evaluation: perception + planning on all episodes.

Usage:
    python src/validation/execution/eval_perception/run_eval_pipeline.py \
        --data_root ~/shared_data/mingi_object_test --depth da3

    # Single object
    python src/validation/execution/eval_perception/run_eval_pipeline.py \
        --data_root ~/shared_data/mingi_object_test --obj attached_container --depth da3

    # Skip perception (reuse existing pose_world.npy), only planning
    python src/validation/execution/eval_perception/run_eval_pipeline.py \
        --data_root ~/shared_data/mingi_object_test --skip_perception --depth da3
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

AUTODEX_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(AUTODEX_ROOT))

logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')

MESH_ROOT = Path.home() / "shared_data/object_6d/data/mesh"
OBJ_PATH = Path.home() / "shared_data/object_6d/data/mesh"

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

TABLE_CFG = {"dims": [2, 3, 0.2], "pose": [1.1, 0, -0.1 + 0.037, 0, 0, 0, 1]}


def find_mesh(obj_name):
    for name in [f"{obj_name}.obj", "simplified.obj", "coacd.obj"]:
        p = MESH_ROOT / obj_name / name
        if p.exists():
            return str(p)
    objs = list((MESH_ROOT / obj_name).glob("*.obj"))
    if objs:
        return str(objs[0])
    raise FileNotFoundError(f"No mesh for {obj_name}")


def find_urdf(obj_name):
    candidates = [
        OBJ_PATH / obj_name / "processed_data" / "urdf" / "coacd.urdf",
        OBJ_PATH / obj_name / "coacd.urdf",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def pose_world_to_scene_cfg(pose_world, obj_name, c2r):
    """Convert colmap-frame pose to cuRobo scene_cfg in robot frame."""
    from autodex.utils.conversion import se32cart

    # colmap → robot frame
    pose_robot = np.linalg.inv(c2r) @ pose_world
    pose_7d = se32cart(pose_robot).tolist()

    mesh_path = find_mesh(obj_name)
    urdf_path = find_urdf(obj_name)

    scene_cfg = {
        "mesh": {
            "target": {
                "pose": pose_7d,
                "file_path": mesh_path,
            }
        },
        "cuboid": {
            "table": TABLE_CFG,
        }
    }
    if urdf_path:
        scene_cfg["mesh"]["target"]["urdf_path"] = urdf_path

    return scene_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--obj", type=str, default=None)
    parser.add_argument("--depth", type=str, default="da3", choices=["da3", "stereo"])
    parser.add_argument("--prompt", type=str, default="object on the checkerboard")
    parser.add_argument("--sil_iters", type=int, default=100)
    parser.add_argument("--sil_lr", type=float, default=0.002)
    parser.add_argument("--grasp_version", type=str, default="selected_100")
    parser.add_argument("--skip_perception", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)

    # Build episode list
    episodes = []
    if args.obj:
        obj_dir = data_root / args.obj
        for ep in sorted(obj_dir.iterdir()):
            if ep.is_dir():
                episodes.append((args.obj, str(ep)))
    else:
        for obj_dir in sorted(data_root.iterdir()):
            if not obj_dir.is_dir() or obj_dir.name in ("cam_param", "simulate"):
                continue
            for ep in sorted(obj_dir.iterdir()):
                if ep.is_dir():
                    episodes.append((obj_dir.name, str(ep)))

    print(f"Episodes: {len(episodes)}, Depth: {args.depth}, Skip perception: {args.skip_perception}")

    # Init perception pipeline
    perception = None
    if not args.skip_perception:
        from src.execution.daemon.perception_pipeline import PerceptionPipeline
        current_obj = episodes[0][0]
        mesh_path = find_mesh(current_obj)
        perception = PerceptionPipeline(
            sam3_hosts=SAM3_HOSTS,
            fpose_hosts=FPOSE_HOSTS,
            mesh_path=mesh_path,
            depth_method=args.depth,
        )

    # Init planner (lazy — first plan() call does warmup)
    from autodex.planner.planner import GraspPlanner
    planner = GraspPlanner()

    current_obj = None
    all_timings = []

    for i, (obj, capture_dir) in enumerate(episodes):
        capture_dir = Path(capture_dir)

        # Switch object
        if obj != current_obj:
            current_obj = obj
            mesh_path = find_mesh(current_obj)
            if perception:
                perception.change_object(mesh_path)
            print(f"\nObject: {current_obj}")

        print(f"\n[{i+1}/{len(episodes)}] {obj}/{capture_dir.name}")

        timing = {"obj": obj, "episode": capture_dir.name}

        # ── Perception ──
        if args.skip_perception:
            pose_path = capture_dir / "pose_world.npy"
            if not pose_path.exists():
                print(f"  SKIP: no pose_world.npy")
                continue
            pose_world = np.load(str(pose_path))
            # Load existing timing if available
            timing_path = capture_dir / "timing.json"
            if timing_path.exists():
                with open(timing_path) as f:
                    timing.update(json.load(f))
        else:
            pose_world, perc_timing = perception.run(
                capture_dir=str(capture_dir),
                prompt=args.prompt,
                sil_iters=args.sil_iters,
                sil_lr=args.sil_lr,
            )
            if pose_world is None:
                print(f"  Perception FAILED")
                continue
            timing.update(perc_timing)
            np.save(str(capture_dir / "pose_world.npy"), pose_world)
            with open(capture_dir / "timing.json", "w") as f:
                json.dump(timing, f, indent=2)

        # ── Planning ──
        c2r_path = capture_dir / "cam_param" / "C2R.npy"
        if not c2r_path.exists():
            c2r_path = capture_dir / "C2R.npy"
        if not c2r_path.exists():
            print(f"  SKIP planning: no C2R.npy")
            all_timings.append(timing)
            continue

        c2r = np.load(str(c2r_path))
        scene_cfg = pose_world_to_scene_cfg(pose_world, obj, c2r)

        t0 = time.perf_counter()
        try:
            result = planner.plan(scene_cfg, obj, args.grasp_version)
            t_plan = time.perf_counter() - t0
            timing["planning"] = t_plan
            timing["planning_success"] = result.success
            timing["planning_timing"] = result.timing

            if result.success:
                # Save trajectory
                np.save(str(capture_dir / "traj.npy"), result.traj)
                np.save(str(capture_dir / "wrist_se3.npy"), result.wrist_se3)
                np.save(str(capture_dir / "pregrasp_pose.npy"), result.pregrasp_pose)
                np.save(str(capture_dir / "grasp_pose.npy"), result.grasp_pose)
                print(f"  Planning: OK in {t_plan:.2f}s")
            else:
                print(f"  Planning: FAILED in {t_plan:.2f}s")
        except Exception as e:
            t_plan = time.perf_counter() - t0
            timing["planning"] = t_plan
            timing["planning_success"] = False
            timing["planning_error"] = str(e)
            print(f"  Planning ERROR: {e}")

        # Save timing
        with open(capture_dir / "timing.json", "w") as f:
            json.dump(timing, f, indent=2)
        all_timings.append(timing)

        perc_str = ""
        if "total" in timing:
            perc_str = f"Perc={timing['total']:.1f}s "
        print(f"  {perc_str}Plan={timing.get('planning', 0):.1f}s "
              f"Success={timing.get('planning_success', False)}")

    # Summary
    if all_timings:
        print(f"\n{'='*60}")
        print(f"Summary ({len(all_timings)}/{len(episodes)} episodes)")

        perc_keys = ["total", "sam3", "depth", "fpose", "select", "sil"]
        for key in perc_keys:
            vals = [t[key] for t in all_timings if key in t]
            if vals:
                print(f"  {key:>12}: mean={np.mean(vals):.2f}s  min={np.min(vals):.2f}s  max={np.max(vals):.2f}s")

        plan_vals = [t["planning"] for t in all_timings if "planning" in t]
        if plan_vals:
            print(f"  {'planning':>12}: mean={np.mean(plan_vals):.2f}s  min={np.min(plan_vals):.2f}s  max={np.max(plan_vals):.2f}s")

        n_plan_success = sum(1 for t in all_timings if t.get("planning_success"))
        n_plan_total = sum(1 for t in all_timings if "planning" in t)
        print(f"  Planning success: {n_plan_success}/{n_plan_total}")

        summary_path = data_root / "timing_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_timings, f, indent=2)
        print(f"\nSaved to {summary_path}")

    if perception:
        perception.close()


if __name__ == "__main__":
    main()
