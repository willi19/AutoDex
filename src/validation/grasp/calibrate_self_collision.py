"""Find sphere pairs in self-collision at a given joint config and append
them to a cuRobo robot YAML's self_collision_ignore.

Usage:
    python src/validation/grasp/calibrate_self_collision.py \
        --urdf src/grasp_generation/BODex/src/curobo/content/assets/robot/inspire_f1_description/inspire_f1_hand_right.urdf \
        --robot_yml src/grasp_generation/BODex/src/curobo/content/configs/robot/inspire_f1.yml \
        --joints right_thumb_1_joint=1.3 right_thumb_2_joint=0 \
                 right_index_1_joint=0.8 right_middle_1_joint=0.8 \
                 right_ring_1_joint=0.8 right_little_1_joint=0.8 \
        --apply
"""
import argparse
import os

import numpy as np
import yaml
import yourdfpy


def parse_joints(items, actuated):
    overrides = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--joints expects name=value, got {item!r}")
        name, val = item.split("=", 1)
        overrides[name.strip()] = float(val)
    return np.array([overrides.get(n, 0.0) for n in actuated], dtype=np.float64)


def load_robot_cfg(robot_yml_path):
    with open(robot_yml_path) as f:
        cfg = yaml.safe_load(f)
    kin = cfg["robot_cfg"]["kinematics"]
    sphere_path = os.path.normpath(
        os.path.join(os.path.dirname(robot_yml_path), kin["collision_spheres"])
    )
    ignore = kin.get("self_collision_ignore") or {}
    pairs = set()
    for a, lst in ignore.items():
        for b in lst or []:
            pairs.add(frozenset([a, b]))
    coll_links = set(kin.get("collision_link_names") or [])
    buffer = kin.get("collision_sphere_buffer", 0.0) or 0.0
    return cfg, sphere_path, pairs, coll_links, buffer


def load_spheres(sphere_yml_path):
    with open(sphere_yml_path) as f:
        data = yaml.safe_load(f)
    spheres = data.get("collision_spheres", data) or {}
    out = {}
    for link, items in spheres.items():
        if not items:
            continue
        out[link] = [(np.asarray(it["center"], dtype=np.float64), float(it["radius"])) for it in items]
    return out


def find_collisions(world_spheres, ignore_pairs, coll_links):
    flat = []
    for link, items in world_spheres.items():
        if coll_links and link not in coll_links:
            continue
        for i, (c, r, _) in enumerate(items):
            flat.append((link, i, c, r))
    pair_overlap = {}
    for ai in range(len(flat)):
        la, ia, ca, ra = flat[ai]
        for bi in range(ai + 1, len(flat)):
            lb, ib, cb, rb = flat[bi]
            if la == lb:
                continue
            if frozenset([la, lb]) in ignore_pairs:
                continue
            d = float(np.linalg.norm(ca - cb))
            if d < ra + rb:
                key = tuple(sorted([la, lb]))
                pair_overlap.setdefault(key, []).append((ia, ib, ra + rb - d))
    return pair_overlap


def transform_spheres(robot, spheres, base_link):
    out = {}
    for link, items in spheres.items():
        T = robot.get_transform(frame_to=link, frame_from=base_link)
        R, t = T[:3, :3], T[:3, 3]
        out[link] = [(R @ c + t, r, link) for c, r in items]
    return out


def update_yml_in_place(robot_yml_path, new_pairs):
    """Append new pairs to self_collision_ignore (preserves existing entries)."""
    with open(robot_yml_path) as f:
        cfg = yaml.safe_load(f)
    ignore = cfg["robot_cfg"]["kinematics"].setdefault("self_collision_ignore", {})
    added = []
    for la, lb in new_pairs:
        existing = set(ignore.get(la) or [])
        rev = set(ignore.get(lb) or [])
        if lb in existing or la in rev:
            continue
        ignore.setdefault(la, [])
        ignore[la] = list(ignore[la]) + [lb]
        added.append((la, lb))
    with open(robot_yml_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--robot_yml", required=True)
    ap.add_argument("--joints", nargs="*", default=[])
    ap.add_argument("--apply", action="store_true",
                    help="Append detected pairs to self_collision_ignore (writes the yml)")
    args = ap.parse_args()

    robot = yourdfpy.URDF.load(args.urdf)
    actuated = list(robot.actuated_joint_names)
    cfg = parse_joints(args.joints, actuated)
    print("joint config:")
    for n, v in zip(actuated, cfg):
        print(f"  {n}: {v:.4f}")
    robot.update_cfg(cfg)

    _, sphere_path, ignore_pairs, coll_links, buffer = load_robot_cfg(args.robot_yml)
    spheres = load_spheres(sphere_path)
    if buffer:
        spheres = {l: [(c, r + buffer) for c, r in items] for l, items in spheres.items()}

    world = transform_spheres(robot, spheres, robot.base_link)
    pair_overlap = find_collisions(world, ignore_pairs, coll_links)

    if not pair_overlap:
        print("\nNo new self-collision pairs detected.")
        return

    print(f"\nDetected {len(pair_overlap)} colliding link pairs (not in ignore):")
    for (la, lb), hits in sorted(pair_overlap.items()):
        worst = max(h[2] for h in hits) * 1000
        print(f"  {la} <-> {lb}  ({len(hits)} sphere pairs, worst overlap {worst:.2f}mm)")

    if not args.apply:
        print("\n(Dry run — pass --apply to write to robot_yml.)")
        return

    added = update_yml_in_place(args.robot_yml, list(pair_overlap.keys()))
    print(f"\nAppended {len(added)} pairs to {args.robot_yml}")
    for la, lb in added:
        print(f"  + {la}: [{lb}]")


if __name__ == "__main__":
    main()
