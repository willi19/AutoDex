#!/usr/bin/env python3
"""Run perception pipeline on a capture directory.

Usage:
    python src/execution/run_perception.py \
        --capture_dir /path/to/episode \
        --obj attached_container \
        --depth da3
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.execution.daemon.perception_pipeline import PerceptionPipeline

logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')

MESH_ROOT = Path("/home/mingi/shared_data/object_6d/data/mesh")

# Default daemon hosts
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
    parser.add_argument("--capture_dir", type=str, required=True)
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--depth", type=str, default="da3", choices=["da3", "stereo"])
    parser.add_argument("--prompt", type=str, default="object on the checkerboard")
    parser.add_argument("--sil_iters", type=int, default=100)
    parser.add_argument("--sil_lr", type=float, default=0.002)
    args = parser.parse_args()

    mesh_path = find_mesh(args.obj)
    print(f"Mesh: {mesh_path}")
    print(f"Capture: {args.capture_dir}")
    print(f"Depth: {args.depth}")

    pipeline = PerceptionPipeline(
        sam3_hosts=SAM3_HOSTS,
        fpose_hosts=FPOSE_HOSTS,
        mesh_path=mesh_path,
        depth_method=args.depth,
    )

    pose_world = pipeline.run(
        capture_dir=args.capture_dir,
        prompt=args.prompt,
        sil_iters=args.sil_iters,
        sil_lr=args.sil_lr,
    )

    if pose_world is not None:
        import numpy as np
        print(f"\nObject pose (world frame):")
        print(pose_world)
        # Save
        out_path = Path(args.capture_dir) / "pose_world.npy"
        np.save(str(out_path), pose_world)
        print(f"Saved to {out_path}")
    else:
        print("FAILED: no pose estimated")

    pipeline.close()


if __name__ == "__main__":
    main()