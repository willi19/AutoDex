#!/usr/bin/env python3
"""Run perception pipeline — interactive loop or batch mode.

Usage:
    # Single episode
    python src/execution/run_perception.py \
        --capture_dir /path/to/episode --obj attached_container

    # Batch: all episodes under data_root for one object
    python src/execution/run_perception.py \
        --data_root /path/to/mingi_object_test --obj attached_container

    # Batch: all objects × all episodes
    python src/execution/run_perception.py \
        --data_root /path/to/mingi_object_test
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.execution.daemon.perception_pipeline import PerceptionPipeline

logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')

MESH_ROOT = Path.home() / "shared_data/object_6d/data/mesh"

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
    for name in [f"{obj_name}.obj", "simplified.obj", "coacd.obj"]:
        p = MESH_ROOT / obj_name / name
        if p.exists():
            return str(p)
    objs = list((MESH_ROOT / obj_name).glob("*.obj"))
    if objs:
        return str(objs[0])
    raise FileNotFoundError(f"No mesh for {obj_name}")


def run_episode(pipeline, capture_dir, prompt, sil_iters, sil_lr):
    capture_dir = Path(capture_dir)
    pose_world, timing = pipeline.run(
        capture_dir=str(capture_dir),
        prompt=prompt,
        sil_iters=sil_iters,
        sil_lr=sil_lr,
    )

    if pose_world is not None:
        np.save(str(capture_dir / "pose_world.npy"), pose_world)
        with open(capture_dir / "timing.json", "w") as f:
            json.dump(timing, f, indent=2)
        print(f"  Saved pose + timing to {capture_dir}")
    else:
        print(f"  FAILED: no pose estimated")

    return pose_world, timing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture_dir", type=str, default=None, help="Single episode")
    parser.add_argument("--data_root", type=str, default=None, help="Batch: root dir")
    parser.add_argument("--obj", type=str, default=None, help="Object name (required for single, optional for batch)")
    parser.add_argument("--depth", type=str, default="da3", choices=["da3", "stereo"])
    parser.add_argument("--prompt", type=str, default="object on the checkerboard")
    parser.add_argument("--sil_iters", type=int, default=100)
    parser.add_argument("--sil_lr", type=float, default=0.002)
    args = parser.parse_args()

    if args.capture_dir is None and args.data_root is None:
        parser.error("Either --capture_dir or --data_root required")

    # Build list of (obj, capture_dir) pairs
    episodes = []
    if args.capture_dir:
        if args.obj is None:
            parser.error("--obj required with --capture_dir")
        episodes.append((args.obj, args.capture_dir))
    else:
        data_root = Path(args.data_root)
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

    print(f"Episodes: {len(episodes)}, Depth: {args.depth}")

    # Init pipeline with first object
    current_obj = episodes[0][0]
    mesh_path = find_mesh(current_obj)
    print(f"Object: {current_obj}, Mesh: {mesh_path}")

    pipeline = PerceptionPipeline(
        sam3_hosts=SAM3_HOSTS,
        fpose_hosts=FPOSE_HOSTS,
        mesh_path=mesh_path,
        depth_method=args.depth,
    )

    all_timings = []
    for i, (obj, capture_dir) in enumerate(episodes):
        # Switch object if needed
        if obj != current_obj:
            current_obj = obj
            mesh_path = find_mesh(current_obj)
            print(f"\nSwitching to {current_obj}, Mesh: {mesh_path}")
            pipeline.change_object(mesh_path)

        print(f"\n[{i+1}/{len(episodes)}] {obj}/{Path(capture_dir).name}")
        pose, timing = run_episode(pipeline, capture_dir, args.prompt, args.sil_iters, args.sil_lr)
        if timing:
            timing["obj"] = obj
            timing["episode"] = Path(capture_dir).name
            all_timings.append(timing)

    # Summary
    if all_timings:
        print(f"\n{'='*60}")
        print(f"Summary ({len(all_timings)} episodes):")
        for key in ["total", "sam3", "depth", "fpose", "select", "sil"]:
            vals = [t[key] for t in all_timings]
            print(f"  {key:>8}: mean={np.mean(vals):.2f}s  min={np.min(vals):.2f}s  max={np.max(vals):.2f}s")

    pipeline.close()


if __name__ == "__main__":
    main()
