"""
Cross-Pose Grasp Compatibility

For each tabletop pose, check which grasp candidates are valid (no backward,
no hand-table collision). Then compute intersections across poses to find
grasps usable for pick-and-place between pose A and pose B.

Usage:
    python src/validation/planning/cross_pose_reachability.py --obj attached_container
    python src/validation/planning/cross_pose_reachability.py --obj attached_container --x_offset 0.35
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "paradex"))

from autodex.planner.planner import GraspPlanner
from autodex.utils.path import obj_path, load_candidate


def get_tabletop_poses(obj_name):
    pose_dir = os.path.join(obj_path, obj_name, "processed_data", "info", "tabletop")
    if not os.path.isdir(pose_dir):
        return []
    return sorted([f.replace(".npy", "") for f in os.listdir(pose_dir) if f.endswith(".npy")])


def load_tabletop_pose(obj_name, pose_idx, x_offset=0.4, z_rotation=0.0):
    pose_dir = os.path.join(obj_path, obj_name, "processed_data", "info", "tabletop")
    obj_pose = np.load(os.path.join(pose_dir, f"{pose_idx}.npy"))
    obj_pose[0, 3] += x_offset
    if z_rotation != 0.0:
        c, s = np.cos(z_rotation), np.sin(z_rotation)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        obj_pose[:3, :3] = Rz @ obj_pose[:3, :3]
    return obj_pose


def build_table_world_cfg():
    """Table-only world config matching BODex scene (surface at z=0)."""
    return {
        "cuboid": {
            "table": {
                "dims": [2, 2, 0.2],
                "pose": [0.0, 0.0, -0.1, 1, 0, 0, 0],
                "color": [0.5, 0.5, 0.5, 1.0],
            },
        },
        "mesh": {},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--version", type=str, default="selected_100")
    parser.add_argument("--hand", type=str, default="allegro")
    parser.add_argument("--x_offset", type=float, default=0.0)
    parser.add_argument("--z_rotation", type=float, default=0.0, help="degrees")
    args = parser.parse_args()

    obj_name = args.obj
    z_rad = np.radians(args.z_rotation)

    pose_indices = get_tabletop_poses(obj_name)
    if not pose_indices:
        print(f"No tabletop poses for {obj_name}")
        return

    print(f"Object: {obj_name}")
    print(f"Poses: {pose_indices}")
    print(f"Placement: x={args.x_offset}, z={args.z_rotation}°")
    print("=" * 70)

    # Load candidates in object frame (identity pose)
    identity = np.eye(4)
    wrist_obj, pregrasp, grasp, scene_info = load_candidate(
        obj_name, identity, args.version, shuffle=False, skip_done=False, hand=args.hand
    )
    N = len(wrist_obj)
    print(f"Candidates: {N}")

    # Collision checker (floating hand vs table)
    planner = GraspPlanner(hand=args.hand)
    table_world_cfg = build_table_world_cfg()

    # Per-pose validity
    valid_per_pose = {}  # pose_idx -> bool array (N,)

    for pose_idx in pose_indices:
        obj_pose = load_tabletop_pose(obj_name, pose_idx, args.x_offset, z_rad)
        wrist_world = obj_pose @ wrist_obj  # (N, 4, 4)

        # Hand-table collision only
        collision = planner._check_collision(table_world_cfg, wrist_world, pregrasp)

        valid = ~collision
        valid_per_pose[pose_idx] = valid

        print(f"  Pose {pose_idx}: {valid.sum()}/{N} valid (collision={collision.sum()})")

    # Intersection analysis
    print(f"\n{'=' * 70}")
    print("PAIRWISE INTERSECTION (grasps valid for BOTH poses)")
    print(f"{'=' * 70}")

    pair_results = []
    for pi, pj in combinations(pose_indices, 2):
        shared = valid_per_pose[pi] & valid_per_pose[pj]
        n_shared = int(shared.sum())
        pair_results.append({
            "pose_a": pi, "pose_b": pj,
            "shared": n_shared,
            "only_a": int((valid_per_pose[pi] & ~valid_per_pose[pj]).sum()),
            "only_b": int((~valid_per_pose[pi] & valid_per_pose[pj]).sum()),
            "shared_indices": np.where(shared)[0].tolist(),
        })

    pair_results.sort(key=lambda x: x["shared"], reverse=True)
    for pr in pair_results:
        print(f"  {pr['pose_a']} ↔ {pr['pose_b']}: "
              f"{pr['shared']} shared, {pr['only_a']} only-A, {pr['only_b']} only-B")

    # Union: valid in at least one pose
    any_valid = np.zeros(N, dtype=bool)
    for v in valid_per_pose.values():
        any_valid |= v
    never_valid = ~any_valid
    print(f"\nValid in at least 1 pose: {any_valid.sum()}/{N}")
    print(f"Never valid (collide in ALL poses): {never_valid.sum()}/{N}")

    # Universal grasps (valid for ALL poses)
    all_valid = np.ones(N, dtype=bool)
    for v in valid_per_pose.values():
        all_valid &= v
    universal_indices = np.where(all_valid)[0]

    print(f"\nUniversal grasps (all {len(pose_indices)} poses): {len(universal_indices)}")
    for idx in universal_indices[:20]:
        print(f"  [{idx}] {scene_info[idx][0]}/{scene_info[idx][1]}/{scene_info[idx][2]}")

    # Save
    save_dir = os.path.join("outputs", "cross_pose_reachability", obj_name)
    os.makedirs(save_dir, exist_ok=True)

    output = {
        "obj_name": obj_name,
        "version": args.version,
        "hand": args.hand,
        "x_offset": args.x_offset,
        "z_rotation_deg": args.z_rotation,
        "pose_indices": pose_indices,
        "n_candidates": N,
        "scene_info": scene_info,
        "per_pose_valid_count": {p: int(v.sum()) for p, v in valid_per_pose.items()},
        "per_pose_valid": {p: v.tolist() for p, v in valid_per_pose.items()},
        "pair_results": pair_results,
        "universal_grasps": universal_indices.tolist(),
    }

    out_path = os.path.join(save_dir, f"cross_pose_{args.version}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()