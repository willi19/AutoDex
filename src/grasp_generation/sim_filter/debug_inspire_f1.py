"""Debug: load one inspire_f1 seed in MuJoCo, dump base_link pose + finger angles
without running viewer. Compares mujoco state to BODex's intended wrist_se3 + grasp_pose."""

import os
import sys
import numpy as np
import mujoco

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from autodex.simulator.hand_object import MjHO
from autodex.utils.conversion import se32cart, cart2se3
from src.grasp_generation.sim_filter.run_sim_filter import (
    HAND_PATHS, INSPIRE_F1_MIMIC_MAP, _expand_mimic_joints, R_DELTA,
)


def main():
    obj = "Lpine_spring_Water"
    seed_dir = "/home/mingi/AutoDex/bodex_outputs/inspire_f1/v3/" + obj + "/box/0/0"
    obj_root = "/home/mingi/shared_data/AutoDex/object/robothome"

    wrist_se3 = np.load(os.path.join(seed_dir, "wrist_se3.npy"))
    pregrasp_active = np.load(os.path.join(seed_dir, "pregrasp_pose.npy"))
    grasp_active = np.load(os.path.join(seed_dir, "grasp_pose.npy"))

    print(f"=== BODex intended ===")
    print(f"wrist_se3 (object frame):\n{wrist_se3}")
    print(f"pregrasp active (6): {pregrasp_active}")
    print(f"grasp active (6): {grasp_active}")

    pregrasp_full = _expand_mimic_joints(pregrasp_active, INSPIRE_F1_MIMIC_MAP)
    grasp_full = _expand_mimic_joints(grasp_active, INSPIRE_F1_MIMIC_MAP)
    print(f"pregrasp expanded (12): {pregrasp_full}")

    # Set object at origin (sim treats wrist_se3 as already in world frame after object reset)
    hand_cfg = HAND_PATHS["inspire_f1"]
    mj = MjHO(obj, hand_cfg["path"], weld_body_name=hand_cfg["weld_body"],
              obj_mass=0.1, debug_viewer=False, obj_root_dir=obj_root)

    print(f"\n=== Mujoco model info ===")
    print(f"  njnt: {mj.model.njnt}, nu: {mj.model.nu}")
    for i in range(mj.model.njnt):
        jnt = mj.model.joint(i)
        print(f"  joint[{i}] {jnt.name}  type={jnt.type}")

    print(f"\n=== Mujoco bodies (hand prefix={mj.hand_prefix}) ===")
    for i in range(mj.model.nbody):
        b = mj.model.body(i)
        if mj.hand_prefix in b.name or b.name in ("world", "mocap_body", "object"):
            print(f"  body[{i}] {b.name}  pos={b.pos}  quat={b.quat}")

    # Apply same transform as eval_single_grasp (no R_DELTA for non-allegro)
    w = wrist_se3.copy()
    wrist_cart = se32cart(w)
    pregrasp_qpos = np.concatenate([wrist_cart, pregrasp_full])
    obj_pose = np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)

    print(f"\n=== Resetting to pregrasp ===")
    print(f"  wrist_cart (xyz+quat): {wrist_cart}")
    mj.reset_pose_qpos(pregrasp_qpos, obj_pose)
    mujoco.mj_forward(mj.model, mj.data)

    print(f"\n=== Mujoco state after reset ===")
    print(f"  qpos[:7] (hand freejoint): {mj.data.qpos[:7]}")
    print(f"  qpos[7:19] (12 finger): {mj.data.qpos[7:19]}")
    print(f"  qpos[-7:] (object): {mj.data.qpos[-7:]}")
    print(f"  mocap_pos: {mj.data.mocap_pos[0]}")
    print(f"  mocap_quat: {mj.data.mocap_quat[0]}")

    # base_link world pose
    base_id = mujoco.mj_name2id(mj.model, mujoco.mjtObj.mjOBJ_BODY, f"{mj.hand_prefix}base_link")
    print(f"\n=== base_link in world frame ===")
    print(f"  pos: {mj.data.xpos[base_id]}")
    print(f"  xmat:\n{mj.data.xmat[base_id].reshape(3,3)}")
    print(f"\n=== INTENDED base_link pos (from wrist_se3): {wrist_se3[:3, 3]}")
    print(f"=== INTENDED base_link rotmat:\n{wrist_se3[:3, :3]}")

    # Compare positions
    delta_pos = mj.data.xpos[base_id] - wrist_se3[:3, 3]
    print(f"\n=== DELTA (mujoco - intended): {delta_pos}, norm={np.linalg.norm(delta_pos):.4f}m ===")


if __name__ == "__main__":
    main()
