"""Subsample a captured wrist trajectory and save it relative to the
occluded centroid for later use at test time.

Conventions:
- Object pose at capture and test are translation-only; rotation = I.
- "Relative SE3" of a wrist frame is simply
      (wrist_pos - occluded_centroid, wrist_rot)
  i.e. the wrist's translation re-expressed with the occluded centroid as
  origin. The rotation is unchanged.

At test time, given an object position ``p_obj`` and a yaw angle ``θ``
about the world z-axis through ``p_obj``::

      wrist_test_pos = p_obj + R_z(θ) @ rel_pos
      wrist_test_rot = R_z(θ) @ rel_rot

Output: ``src/validation/robothome/subsampled/<obj>/<idx>.npz`` with keys

- ``wrist_rel_se3`` (N, 4, 4) : relative SE3 sequence (rotation = original
                                wrist rotation, translation = wrist - occ)
- ``hand_qpos``    (N, 6)    : finger joints aligned to the same N samples
                               in URDF/RobotModule order
                               [thumb_1, thumb_2, index_1, middle_1,
                               ring_1, little_1]
- ``arm_t``        (N,)      : matching timestamps
- ``occ_centroid`` (3,)      : the capture-time object position used as
                                origin
- ``start``, ``end`` (int)   : selected slice on the extracted trajectory
- ``stride``       (int)     : sampling stride (default 3)

Usage:
    /home/mingi/miniconda3/envs/foundationpose/bin/python \
        src/validation/robothome/subsample_traj.py \
        --obj choco --idx 0 --start 1000 --end 1200 [--stride 3]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXTRACT_DIR = HERE / "extracted"
SUB_DIR = HERE / "subsampled"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True)
    ap.add_argument("--idx", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--stride", type=int, default=3)
    args = ap.parse_args()

    src = EXTRACT_DIR / args.obj / f"{args.idx}.npz"
    if not src.exists():
        raise SystemExit(f"missing extracted: {src}. Run extract_traj.py first.")
    d = np.load(src)
    wrist = d["wrist_se3_traj"]
    hand = d["hand_qpos"]
    arm_t = d["arm_t"]
    occ = d["occluded_centroid"].astype(np.float64)

    T = len(wrist)
    if not (0 <= args.start < args.end <= T):
        raise SystemExit(f"invalid slice [start={args.start}, end={args.end}] for T={T}")

    sl = slice(args.start, args.end, args.stride)
    wrist_sub = wrist[sl].copy()
    hand_sub = hand[sl].copy()
    arm_t_sub = arm_t[sl].copy()

    # Relative SE3: rotation unchanged, translation re-anchored to occluded
    # centroid (since the object pose is treated as identity rotation).
    wrist_rel = wrist_sub.copy()
    wrist_rel[:, :3, 3] = wrist_sub[:, :3, 3] - occ[None, :]

    out = SUB_DIR / args.obj / f"{args.idx}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        wrist_rel_se3=wrist_rel,
        hand_qpos=hand_sub,
        hand_qpos_order=(d["hand_qpos_order"] if "hand_qpos_order" in d.files else "raw"),
        arm_t=arm_t_sub,
        occ_centroid=occ,
        start=np.int32(args.start),
        end=np.int32(args.end),
        stride=np.int32(args.stride),
    )
    print(f"[ok] {args.obj}/{args.idx}: N={len(wrist_rel)} "
          f"slice=[{args.start},{args.end}) stride={args.stride}  -> {out}")


if __name__ == "__main__":
    main()
