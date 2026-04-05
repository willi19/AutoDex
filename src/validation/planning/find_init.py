"""Find init state that has zero self-collision for inspire.

Perturbs the current init_state slightly and checks self-collision cost.
"""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "paradex"))

import argparse
from autodex.utils.robot_config import XARM_INSPIRE_INIT, INSPIRE_INIT, XARM_INIT, ALLEGRO_INIT
from autodex.utils.path import robot_configs_path
from curobo.util_file import load_yaml
from curobo.types.base import TensorDeviceType
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.geom.types import WorldConfig

parser = argparse.ArgumentParser()
parser.add_argument("--hand", default="inspire", choices=["allegro", "inspire"])
args = parser.parse_args()

CONFIGS = {
    "inspire": ("xarm_inspire.yml", np.concatenate([XARM_INSPIRE_INIT, INSPIRE_INIT])),
    "allegro": ("xarm_allegro.yml", np.concatenate([XARM_INIT, ALLEGRO_INIT])),
}
yml_name, init = CONFIGS[args.hand]

tensor_args = TensorDeviceType()
robot_cfg = load_yaml(os.path.join(robot_configs_path, yml_name))["robot_cfg"]
world_cfg = WorldConfig.from_dict({
    "cuboid": {"table": {"dims": [2, 3, 0.2], "pose": [1.1, 0, -0.1 + 0.037, 1, 0, 0, 0]}}
})
rw_config = RobotWorldConfig.load_from_config(
    robot_cfg, world_cfg, collision_activation_distance=0.01, tensor_args=tensor_args,
)
rw = RobotWorld(rw_config)

# init already set above
print(f"Original init: {init}")

q = torch.tensor(init, dtype=torch.float32, device=tensor_args.device).unsqueeze(0)
_, d_self = rw.get_world_self_collision_distance_from_joints(q)
print(f"Original self_collision_cost: {d_self.item():.6f}")

# Also check the self-collision offset tensor
sc_cfg = rw.kinematics.self_collision_config
print(f"\nSelf-collision offset tensor shape: {sc_cfg.offset.shape}")
print(f"Offset values (unique): {torch.unique(sc_cfg.offset).cpu().numpy()}")
print(f"Thread locations shape: {sc_cfg.thread_location.shape}")

# Get sphere positions
state = rw.get_kinematics(q)
spheres = state.link_spheres_tensor[0]  # (n_spheres, 4)
n = spheres.shape[0]
print(f"\nTotal spheres: {n}")

# Manual pairwise check with offset
offsets = sc_cfg.offset.cpu().numpy()  # should be (n, n) or flattened
print(f"Offset tensor shape: {offsets.shape}")

# Build sphere->link mapping from yml
import yaml
sphere_yml_path = os.path.join(robot_configs_path, "spheres", f"xarm_{args.hand}.yml")
with open(sphere_yml_path) as f:
    sphere_yml = yaml.safe_load(f)
sphere_to_link = {}
idx = 0
for link_name, slist in sphere_yml.get("collision_spheres", {}).items():
    for _ in slist:
        sphere_to_link[idx] = link_name
        idx += 1

# Build ignore set from config
ignore_map = robot_cfg["kinematics"].get("self_collision_ignore", {})
ignore_pairs = set()
for a, bs in ignore_map.items():
    for b in bs:
        ignore_pairs.add((a, b))
        ignore_pairs.add((b, a))

# offset is per-sphere self_collision_buffer
activation_dist = 0.01
print(f"\nPairwise check (skipping ignored pairs, using per-sphere offset):")
found = False
for i in range(n):
    for j in range(i+1, n):
        li = sphere_to_link.get(i, "?")
        lj = sphere_to_link.get(j, "?")
        if li == lj:
            continue
        if (li, lj) in ignore_pairs:
            continue
        si = spheres[i].cpu().numpy()
        sj = spheres[j].cpu().numpy()
        dist = np.linalg.norm(si[:3] - sj[:3])
        ri, rj = si[3], sj[3]
        oi, oj = offsets[i], offsets[j]
        gap = dist - (ri + oi) - (rj + oj)
        if gap < activation_dist:
            cost = activation_dist - gap
            print(f"  COLLISION: {li}[{i}](r={ri:.4f}+{oi:.4f}) <-> {lj}[{j}](r={rj:.4f}+{oj:.4f}) "
                  f"dist={dist:.4f} gap={gap:.4f} cost={cost:.6f}")
            found = True
if not found:
    print("  No collisions found (after ignoring pairs and applying offsets)")
