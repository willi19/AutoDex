# Standard Library
import time
import logging
from typing import Dict
import os

# Third Party
import torch
import numpy as np
import argparse
import tqdm

# CuRobo
from curobo.geom.sdf.world import WorldConfig
from curobo.wrap.reacher.grasp_solver import GraspSolver, GraspSolverConfig
from curobo.util.world_cfg_generator import get_world_config_dataloader
from curobo.util.logger import setup_logger
from curobo.util_file import (
    get_manip_configs_path,
    join_path,
    load_yaml,
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import random

import transforms3d as t3d

def cart2se3(cart):
    """7D [x,y,z, qw,qx,qy,qz] -> 4x4 SE3 matrix."""
    ret = np.eye(4)
    ret[:3, 3] = cart[0:3]
    ret[:3, :3] = t3d.quaternions.quat2mat(cart[3:7])
    return ret

# Resolve repo root (BODex lives at src/grasp_generation/BODex/)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _setup_logging(output_dir: str, log_dir: str) -> logging.Logger:
    """File + console logger. Log file: {log_dir}/generate.log"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("bodex")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(os.path.join(log_dir, "generate.log"), mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


def save_bodex_output(output_dir: str, save_data: Dict):
    batch_size = save_data["robot_pose"].shape[0]
    for b in tqdm.tqdm(range(batch_size), desc="Saving BODex outputs"):
        output_path = os.path.join(output_dir, save_data['save_prefix'][b])
        os.makedirs(output_path, exist_ok=True)

        num_seed = save_data["robot_pose"].shape[1]
        if num_seed == len(os.listdir(output_path)):
            continue
        obj_name = save_data['manip_name'][b]

        for ns in tqdm.tqdm(range(num_seed), desc=f"Saving seeds for batch {b}"):
            if os.path.exists(os.path.join(output_path, str(ns))):
                continue
            os.makedirs(os.path.join(output_path, str(ns)), exist_ok=True)

            wrist_se3 = cart2se3(save_data["robot_pose"][b, ns, 0, :7])
            pregrasp_pose = save_data["robot_pose"][b, ns, 0, 7:]
            grasp_pose = save_data["robot_pose"][b, ns, 1, 7:]

            obj_se3 = cart2se3(save_data['world_cfg'][b]['mesh'][obj_name]['pose'])
            bodex_info = {
                "contact_point": save_data["contact_point"][b, ns],
                "contact_frame": save_data["contact_frame"][b, ns],
                "contact_force": save_data["contact_force"][b, ns],
                "grasp_error": save_data["grasp_error"][b, ns],
                "dist_error": save_data["dist_error"][b, ns],
                "success": save_data["success"][b, ns],
            }

            np.save(os.path.join(output_path, str(ns), "wrist_se3.npy"), np.linalg.inv(obj_se3) @ wrist_se3)
            np.save(os.path.join(output_path, str(ns), "pregrasp_pose.npy"), pregrasp_pose)
            np.save(os.path.join(output_path, str(ns), "grasp_pose.npy"), grasp_pose)
            np.save(os.path.join(output_path, str(ns), "bodex_info.npy"), bodex_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--manip_cfg_file", type=str, default="fc_leap.yml")
    parser.add_argument("-w", "--parallel_world", type=int, default=20)
    parser.add_argument("-o", "--output_dir", type=str, default=None,
                        help="Output directory (default: bodex_outputs/{exp_name})")
    parser.add_argument("--obj_list_file", type=str, default=None,
                        help="Text file with object names (one per line). Overrides config obj_list.")

    setup_logger("warn")

    args = parser.parse_args()
    manip_config_data = load_yaml(join_path(get_manip_configs_path(), args.manip_cfg_file))

    # Override obj_list from file if provided
    if args.obj_list_file:
        with open(args.obj_list_file) as f:
            obj_list = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        manip_config_data["world"]["obj_list"] = obj_list

    exp_name = manip_config_data["exp_name"]
    robot_name = manip_config_data["robot_file"].replace(".yml", "")  # e.g. "allegro", "inspire"

    # Output under repo root: bodex_outputs/{robot}/{version}
    save_dir = args.output_dir or os.path.join(REPO_ROOT, "bodex_outputs", robot_name, exp_name)
    log_dir = os.path.join(REPO_ROOT, "logging", "grasp_generation")
    logger = _setup_logging(save_dir, log_dir)

    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # output_dir for skip check: save_dir already includes exp_name,
    # but ParadexDataset prepends version again, so pass parent dir
    save_dir_parent = os.path.dirname(save_dir)
    world_generator = get_world_config_dataloader(
        manip_config_data["world"], args.parallel_world,
        manip_config_data["seed_num"], exp_name,
        output_dir=save_dir_parent,
    )

    logger.info(f"START config={args.manip_cfg_file} exp={exp_name} parallel={args.parallel_world} output={save_dir}")

    tst = time.time()
    grasp_solver = None
    n_scenes = 0

    for world_info_dict in tqdm.tqdm(world_generator):
        sst = time.time()
        obj_names = world_info_dict["manip_name"]
        n_scenes += len(obj_names)

        if grasp_solver is None:
            grasp_config = GraspSolverConfig.load_from_robot_config(
                world_model=world_info_dict["world_cfg"],
                manip_name_list=obj_names,
                manip_config_data=manip_config_data,
                obj_gravity_center=world_info_dict["obj_gravity_center"],
                obj_obb_length=world_info_dict["obj_obb_length"],
                use_cuda_graph=False,
                store_debug=False,
            )
            grasp_solver = GraspSolver(grasp_config)
            world_info_dict["world_model"] = grasp_solver.world_coll_checker.world_model
        else:
            world_info_dict["world_model"] = [
                WorldConfig.from_dict(world_cfg) for world_cfg in world_info_dict["world_cfg"]
            ]
            grasp_solver.update_world(
                world_info_dict["world_model"],
                world_info_dict["obj_gravity_center"],
                world_info_dict["obj_obb_length"],
                obj_names,
            )

        result = grasp_solver.solve_batch_env(return_seeds=grasp_solver.num_seeds)

        n_success = result.success.sum().item()
        n_total = result.success.numel()
        elapsed = time.time() - sst

        world_info_dict["robot_pose"] = result.solution.detach().cpu().numpy()
        world_info_dict["contact_point"] = result.contact_point.detach().cpu().numpy()
        world_info_dict["contact_frame"] = result.contact_frame.detach().cpu().numpy()
        world_info_dict["contact_force"] = result.contact_force.detach().cpu().numpy()
        world_info_dict["grasp_error"] = result.grasp_error.detach().cpu().numpy()
        world_info_dict["dist_error"] = result.dist_error.detach().cpu().numpy()
        world_info_dict["success"] = result.success.detach().cpu().numpy()

        save_bodex_output(save_dir, world_info_dict)

        logger.info(f"BATCH objects={obj_names} success={n_success}/{n_total} time={elapsed:.1f}s")

    total_time = time.time() - tst
    logger.info(f"DONE scenes={n_scenes} total_time={total_time:.1f}s")
