"""Preprocess: compute per-frame synced qpos and write to experiment dirs.

For every experiment under ~/shared_data/AutoDex/experiment/selected_100/{hand}/{obj}/{ep}/,
reads raw/arm/, raw/hand/, raw/timestamps/timestamp.npy and writes:
  {exp}/arm/state.npy     (N, 6)   actual arm qpos (URDF order)
  {exp}/arm/action.npy    (N, 6)   commanded arm qpos (URDF order)
  {exp}/hand/state.npy    (N, H)   actual hand qpos (URDF order)
  {exp}/hand/action.npy   (N, H)   commanded hand qpos (URDF order)

Default timestamp offsets compensate for the video stream being ~0.09s slower
than the robot stream.

Usage:
    python src/process/precompute_synced_qpos.py --hand inspire
    python src/process/precompute_synced_qpos.py --hand allegro --obj banana
    python src/process/precompute_synced_qpos.py --hand inspire --overwrite
"""
import argparse
import sys
from pathlib import Path

PARADEX_ROOT = Path.home() / "paradex"
sys.path.insert(0, str(PARADEX_ROOT))

from autodex.utils.sync import precompute_synced_qpos
from tqdm import tqdm


EXP_BASE = Path.home() / "shared_data" / "AutoDex" / "experiment" / "selected_100"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hand", required=True, choices=["allegro", "inspire"])
    p.add_argument("--obj", nargs="+", default=None)
    p.add_argument("--ep", nargs="+", default=None)
    p.add_argument("--arm-time-offset", type=float, default=0.03)
    p.add_argument("--hand-time-offset", type=float, default=0.03)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    hand_dir = EXP_BASE / args.hand
    if not hand_dir.is_dir():
        print(f"No experiments at {hand_dir}")
        return

    objects = args.obj if args.obj else sorted(d.name for d in hand_dir.iterdir() if d.is_dir())

    eps = []
    for obj in objects:
        obj_dir = hand_dir / obj
        if not obj_dir.is_dir():
            continue
        for ep in sorted(obj_dir.iterdir()):
            if not ep.is_dir():
                continue
            if args.ep and ep.name not in args.ep:
                continue
            if not (ep / "raw" / "arm" / "time.npy").exists():
                continue
            eps.append((obj, ep))

    print(f"Processing {len(eps)} experiments (hand={args.hand})", flush=True)

    n_ok = 0
    n_skip = 0
    n_fail = 0
    for obj, ep in tqdm(eps, dynamic_ncols=True):
        try:
            # Check if already done
            req = [ep / "arm" / "state.npy", ep / "arm" / "action.npy",
                   ep / "hand" / "state.npy", ep / "hand" / "action.npy"]
            if not args.overwrite and all(p.exists() for p in req):
                n_skip += 1
                continue
            precompute_synced_qpos(
                ep, args.hand,
                arm_time_offset=args.arm_time_offset,
                hand_time_offset=args.hand_time_offset,
                overwrite=args.overwrite,
            )
            n_ok += 1
        except Exception as e:
            print(f"  [fail] {obj}/{ep.name}: {e}", flush=True)
            n_fail += 1

    print(f"Done: {n_ok} written, {n_skip} skipped, {n_fail} failed", flush=True)


if __name__ == "__main__":
    main()