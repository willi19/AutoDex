"""Quick visualizer to inspect robot URDF + collision spheres with FK.

Usage:
    python src/validation/planning/inspect_robot.py --hand inspire --port 8080
    python src/validation/planning/inspect_robot.py --hand allegro --port 8080
"""
import argparse
import os
import sys
import numpy as np
import yaml
import torch
import trimesh

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "paradex"))

from autodex.utils.robot_config import (
    XARM_INIT, XARM_INSPIRE_INIT,
    ALLEGRO_INIT, INSPIRE_INIT,
)
from autodex.utils.path import robot_configs_path

ASSET_ROOT = os.path.join(os.path.expanduser("~"), "shared_data", "AutoDex", "content", "assets", "robot")
CONFIG_ROOT = os.path.join(os.path.expanduser("~"), "shared_data", "AutoDex", "content", "configs", "robot")

HAND_CFG = {
    "allegro": {
        "urdf": os.path.join(ASSET_ROOT, "allegro_description", "xarm_allegro.urdf"),
        "robot_yml": "xarm_allegro.yml",
        "init": np.concatenate([XARM_INIT, ALLEGRO_INIT]),
    },
    "inspire": {
        "urdf": os.path.join(ASSET_ROOT, "inspire_description", "xarm_inspire.urdf"),
        "robot_yml": "xarm_inspire.yml",
        "init": np.concatenate([XARM_INSPIRE_INIT, INSPIRE_INIT]),
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", default="inspire", choices=["allegro", "inspire"])
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    cfg = HAND_CFG[args.hand]
    init = cfg["init"]
    print(f"URDF: {cfg['urdf']}")
    print(f"Init state ({len(init)} DOF): {init}")

    # === cuRobo collision check ===
    from curobo.util_file import load_yaml
    from curobo.types.base import TensorDeviceType
    from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
    from curobo.geom.types import WorldConfig

    tensor_args = TensorDeviceType()
    robot_cfg = load_yaml(os.path.join(robot_configs_path, cfg["robot_yml"]))["robot_cfg"]
    world_cfg = WorldConfig.from_dict({
        "cuboid": {"table": {"dims": [2, 3, 0.2], "pose": [1.1, 0, -0.1 + 0.037, 1, 0, 0, 0]}}
    })
    rw_config = RobotWorldConfig.load_from_config(
        robot_cfg, world_cfg,
        collision_activation_distance=0.01,
        tensor_args=tensor_args,
    )
    rw = RobotWorld(rw_config)

    q = torch.tensor(init, dtype=torch.float32, device=tensor_args.device).unsqueeze(0)
    d_world, d_self = rw.get_world_self_collision_distance_from_joints(q)

    print(f"\n=== cuRobo collision at init_state ===")
    print(f"  World collision cost: {d_world.item():.6f} (>0 means collision)")
    print(f"  Self collision cost:  {d_self.item():.6f} (>0 means collision)")

    # Get sphere positions (world frame) and link mapping
    state = rw.get_kinematics(q)
    spheres = state.link_spheres_tensor[0].cpu().numpy()  # (n_spheres, 4) x,y,z,r
    print(f"  Total spheres: {spheres.shape[0]}")

    # Build sphere_idx -> link_name from sphere yml (same order as cuRobo loads)
    with open(os.path.join(CONFIG_ROOT, "spheres", f"xarm_{args.hand}.yml")) as f:
        sphere_yml = yaml.safe_load(f)
    sphere_to_link = {}
    idx = 0
    for link_name, slist in sphere_yml.get("collision_spheres", {}).items():
        for _ in slist:
            sphere_to_link[idx] = link_name
            idx += 1
    print(f"  Sphere->link mapping: {idx} spheres across {len(sphere_yml['collision_spheres'])} links")

    # Load self_collision_ignore
    with open(os.path.join(CONFIG_ROOT, cfg["robot_yml"])) as f:
        yml = yaml.safe_load(f)
    ignore_map = yml["robot_cfg"]["kinematics"].get("self_collision_ignore", {})
    ignore_pairs = set()
    for a, bs in ignore_map.items():
        for b in bs:
            ignore_pairs.add((a, b))
            ignore_pairs.add((b, a))

    # Brute force: check all sphere pairs
    activation_dist = 0.01
    collisions = []
    n = spheres.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            li = sphere_to_link.get(i, "?")
            lj = sphere_to_link.get(j, "?")
            if li == lj:
                continue
            if (li, lj) in ignore_pairs:
                continue
            ri, rj = spheres[i, 3], spheres[j, 3]
            if ri < 0.001 or rj < 0.001:
                continue
            dist = np.linalg.norm(spheres[i, :3] - spheres[j, :3])
            gap = dist - ri - rj
            if gap < activation_dist:
                collisions.append((li, lj, i, j, gap, dist, ri, rj))

    print(f"\n=== Self-collision pairs (gap < {activation_dist}m) ===")
    if collisions:
        # Group by link pair
        pair_info = {}
        for li, lj, si, sj, gap, dist, ri, rj in collisions:
            key = tuple(sorted([li, lj]))
            if key not in pair_info:
                pair_info[key] = []
            pair_info[key].append((si, sj, gap, dist, ri, rj))

        for (a, b), infos in sorted(pair_info.items(), key=lambda x: min(i[2] for i in x[1])):
            worst = min(infos, key=lambda x: x[2])
            print(f"  {a} <-> {b}: {len(infos)} sphere pairs, worst gap={worst[2]:.4f}m")
            for si, sj, gap, dist, ri, rj in sorted(infos, key=lambda x: x[2])[:3]:
                print(f"    sph[{si}](r={ri:.4f}) <-> sph[{sj}](r={rj:.4f})  dist={dist:.4f}  gap={gap:.4f}")
    else:
        print("  No collisions found!")

    # === Visualize ===
    from paradex.visualization.visualizer.viser import ViserViewer

    vis = ViserViewer(port_number=args.port)
    vis.add_frame("base", np.eye(4))
    vis.add_robot("robot", cfg["urdf"])
    vis.robot_dict["robot"].update_cfg(init)

    # Table
    table = trimesh.creation.box([2, 3, 0.2])
    table_pose = np.eye(4)
    table_pose[:3, 3] = [1.1, 0, -0.1 + 0.037]
    vis.add_object("table", table, table_pose)
    vis.change_color("table", [0.6, 0.4, 0.2, 0.3])

    # Draw spheres — red for colliding, blue for ok
    colliding_spheres = set()
    for li, lj, si, sj, gap, dist, ri, rj in collisions:
        colliding_spheres.add(si)
        colliding_spheres.add(sj)

    for i in range(n):
        x, y, z, r = spheres[i]
        if r < 0.001:
            continue
        mesh = trimesh.creation.icosphere(radius=float(r))
        pose = np.eye(4)
        pose[:3, 3] = [x, y, z]
        vis.add_object(f"sph_{i}", mesh, pose)
        if i in colliding_spheres:
            vis.change_color(f"sph_{i}", [1.0, 0.0, 0.0, 0.5])
        else:
            vis.change_color(f"sph_{i}", [0.2, 0.6, 1.0, 0.15])

    print(f"\nVisualizer at http://localhost:{args.port}")
    print("Red = colliding spheres, Blue = ok")

    while True:
        import time
        time.sleep(1)


if __name__ == "__main__":
    main()
