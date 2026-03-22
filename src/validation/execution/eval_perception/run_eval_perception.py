#!/usr/bin/env python3
"""Sequential evaluation: perception + planning on all episodes.

Usage:
    python src/validation/execution/eval_perception/run_eval_perception.py \
        --data_root ~/shared_data/mingi_object_test --depth da3

    # Single object
    python src/validation/execution/eval_perception/run_eval_perception.py \
        --data_root ~/shared_data/mingi_object_test --obj attached_container --depth da3
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

AUTODEX_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(AUTODEX_ROOT))

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
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--obj", type=str, default=None)
    parser.add_argument("--depth", type=str, default="da3", choices=["da3", "stereo"])
    parser.add_argument("--prompt", type=str, default="object on the checkerboard")
    parser.add_argument("--sil_iters", type=int, default=100)
    parser.add_argument("--sil_lr", type=float, default=0.002)
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

    print(f"Episodes: {len(episodes)}, Depth: {args.depth}")

    # Init pipeline
    current_obj = episodes[0][0]
    mesh_path = find_mesh(current_obj)
    pipeline = PerceptionPipeline(
        sam3_hosts=SAM3_HOSTS,
        fpose_hosts=FPOSE_HOSTS,
        mesh_path=mesh_path,
        depth_method=args.depth,
    )

    all_timings = []
    for i, (obj, capture_dir) in enumerate(episodes):
        if obj != current_obj:
            current_obj = obj
            mesh_path = find_mesh(current_obj)
            pipeline.change_object(mesh_path)

        print(f"\n[{i+1}/{len(episodes)}] {obj}/{Path(capture_dir).name}")

        pose_world, timing = pipeline.run(
            capture_dir=capture_dir,
            prompt=args.prompt,
            sil_iters=args.sil_iters,
            sil_lr=args.sil_lr,
        )

        if pose_world is not None and timing is not None:
            np.save(str(Path(capture_dir) / "pose_world.npy"), pose_world)
            with open(Path(capture_dir) / "timing.json", "w") as f:
                json.dump(timing, f, indent=2)
            timing["obj"] = obj
            timing["episode"] = Path(capture_dir).name
            all_timings.append(timing)
            print(f"  total={timing['total']:.1f}s "
                  f"(SAM3={timing['sam3']:.1f} Depth={timing['depth']:.1f} "
                  f"FPose={timing['fpose']:.1f} Select={timing['select']:.1f} "
                  f"Sil={timing['sil']:.1f})")
        else:
            print(f"  FAILED")

    # Summary
    if all_timings:
        print(f"\n{'='*60}")
        print(f"Summary ({len(all_timings)}/{len(episodes)} episodes):")
        for key in ["total", "sam3", "depth", "fpose", "select", "sil"]:
            vals = [t[key] for t in all_timings]
            print(f"  {key:>8}: mean={np.mean(vals):.2f}s  min={np.min(vals):.2f}s  max={np.max(vals):.2f}s")

        # Save summary
        summary_path = data_root / "timing_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_timings, f, indent=2)
        print(f"\nTiming summary saved to {summary_path}")

    pipeline.close()


if __name__ == "__main__":
    main()
