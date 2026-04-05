#!/usr/bin/env python3
"""Move Allegro hand finger joints to a specific grasp pose.

Usage:
    python rebuttal/figure/1/pose_hand.py --obj soaptray --scene_type box --scene_id 0 --grasp 78
    python rebuttal/figure/1/pose_hand.py --obj soaptray --scene_type box --scene_id 0 --grasp 78 --version baseline_100
    python rebuttal/figure/1/pose_hand.py --obj soaptray --scene_type box --scene_id 0 --grasp 78 --pregrasp  # pregrasp pose instead
"""
import argparse
import os
import sys
import time

import numpy as np

from autodex.utils.path import candidate_path
from paradex.io.robot_controller import get_hand


def _convert_hand_pose(hand_pose):
    """Reorder: move last 4 (thumb) to front."""
    out = hand_pose.copy()
    out[:4] = hand_pose[12:]
    out[4:] = hand_pose[:12]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", required=True)
    parser.add_argument("--scene_type", required=True)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--grasp", required=True)
    parser.add_argument("--version", default="selected_100")
    parser.add_argument("--pregrasp", action="store_true", help="Use pregrasp pose instead of grasp")
    args = parser.parse_args()

    grasp_dir = os.path.join(candidate_path, args.version, args.obj,
                             args.scene_type, args.scene_id, args.grasp)
    if not os.path.isdir(grasp_dir):
        print(f"Not found: {grasp_dir}")
        sys.exit(1)

    prejoints = np.load(os.path.join(grasp_dir, "pregrasp_pose.npy")).flatten()
    joints = np.load(os.path.join(grasp_dir, "grasp_pose.npy")).flatten()

    target = _convert_hand_pose(joints*5-prejoints*4)
    print(f"Sending to hand (reordered): {target}")

    hand = get_hand("allegro")
    while True:
        hand.move(target)
        time.sleep(0.03)

    print("Done.")
    try:
        hand.end()
    except Exception:
        pass


if __name__ == "__main__":
    main()
