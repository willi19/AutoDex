#!/usr/bin/env python3
"""Run perception pipeline — real robot execution loop.

Usage:
    python src/execution/run_perception.py --obj attached_container

    # Then enter capture_dir paths, or press Enter to quit.
    # Type a new object name to switch objects.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--depth", type=str, default="da3", choices=["da3", "stereo"])
    parser.add_argument("--prompt", type=str, default="object on the checkerboard")
    parser.add_argument("--sil_iters", type=int, default=100)
    parser.add_argument("--sil_lr", type=float, default=0.002)
    args = parser.parse_args()

    current_obj = args.obj
    mesh_path = find_mesh(current_obj)
    print(f"Object: {current_obj}")
    print(f"Mesh: {mesh_path}")
    print(f"Depth: {args.depth}")

    pipeline = PerceptionPipeline(
        sam3_hosts=SAM3_HOSTS,
        fpose_hosts=FPOSE_HOSTS,
        mesh_path=mesh_path,
        depth_method=args.depth,
    )

    print("\nReady. Enter capture_dir path, object name to switch, or empty to quit.")

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            break

        # Check if it's an object name (no / in it)
        if "/" not in user_input and (MESH_ROOT / user_input).exists():
            current_obj = user_input
            mesh_path = find_mesh(current_obj)
            pipeline.change_object(mesh_path)
            print(f"Switched to {current_obj} ({mesh_path})")
            continue

        capture_dir = Path(user_input).expanduser()
        if not capture_dir.exists():
            print(f"Not found: {capture_dir}")
            continue

        pose_world, timing = pipeline.run(
            capture_dir=str(capture_dir),
            prompt=args.prompt,
            sil_iters=args.sil_iters,
            sil_lr=args.sil_lr,
        )

        if pose_world is not None:
            np.save(str(capture_dir / "pose_world.npy"), pose_world)
            with open(capture_dir / "timing.json", "w") as f:
                json.dump(timing, f, indent=2)
            print(f"\nPose saved to {capture_dir / 'pose_world.npy'}")
            print(f"Timing: total={timing['total']:.1f}s "
                  f"(SAM3={timing['sam3']:.1f} Depth={timing['depth']:.1f} "
                  f"FPose={timing['fpose']:.1f} Select={timing['select']:.1f} "
                  f"Sil={timing['sil']:.1f})")
        else:
            print("FAILED: no pose estimated")

    pipeline.close()
    print("Done.")


if __name__ == "__main__":
    main()
