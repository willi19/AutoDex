"""Headless dump of view_bodex's hand pose (no GUI). Compare with dump_mujoco_pose."""
import os, sys, json
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

import view_bodex
view_bodex.obj_path = "/home/mingi/shared_data/AutoDex/object/robothome"
from view_bodex import HAND_URDFS
from autodex.utils.conversion import cart2se3
from scipy.spatial.transform import Rotation as Rot

import yourdfpy
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--hand", default="inspire_f1")
ap.add_argument("--obj", default="Jp_Water")
ap.add_argument("--scene", default="shelf")
ap.add_argument("--scene_id", default="37")
ap.add_argument("--seed", default="0")
ap.add_argument("--version", default="v3")
args = ap.parse_args()

scene_path = f"/home/mingi/shared_data/AutoDex/object/robothome/{args.obj}/scene/{args.scene}/{args.scene_id}.json"
seed_dir = f"{REPO_ROOT}/bodex_outputs/{args.hand}/{args.version}/{args.obj}/{args.scene}/{args.scene_id}/{args.seed}"

cfg = json.load(open(scene_path))
tabletop_pose = cart2se3(cfg["scene"]["mesh"]["target"]["pose"])
wrist_local = np.load(f"{seed_dir}/wrist_se3.npy")
hand_world = tabletop_pose @ wrist_local

print(f"=== Configuration ===")
print(f"obj_pose (tabletop_pose):\n{tabletop_pose}")
print(f"wrist_local:\n{wrist_local}")
print(f"hand_world (= obj_pose @ wrist_local):\n{hand_world}")

# Load URDF and compute base_link world pose as yourdfpy sees it.
urdf_path = HAND_URDFS[args.hand]
print(f"\nURDF: {urdf_path}")
urdf = yourdfpy.URDF.load(urdf_path, build_scene_graph=True, load_meshes=False, load_collision_meshes=False)
print(f"base_link (URDF root): {urdf.base_link}")
print(f"link_map keys: {list(urdf.link_map.keys())[:10]}")

# yourdfpy at zero config: get_transform from base_link to itself = identity
# The visual root frame in viser is at hand_world. yourdfpy puts URDF root (base_link) AT root_frame.
# Each link is then placed relative to base_link via FK.
# Compare: what does yourdfpy think base_link world pose is given root_frame=hand_world?
# Answer: equal to hand_world (since yourdfpy places base_link AT the root frame).

print(f"\n=== Expected (matching mujoco) ===")
print(f"base_link world pos = {hand_world[:3, 3]}")
print(f"base_link world rotmat:\n{hand_world[:3, :3]}")
print(f"\nnote: these should match dump_mujoco_pose's 'base_link in world frame'")
