"""
Validate scene perception: overlay detected object mesh on camera images.

Captures a scene, runs 6D pose estimation, projects the estimated mesh
onto each camera view, and saves a grid image for visual inspection.

Usage:
    python src/validation/perception/scene.py \
        --obj attached_container --ref_idx 000

    # Use an existing capture directory instead of capturing live
    python src/validation/perception/scene.py \
        --obj attached_container --ref_idx 000 \
        --img_dir ~/shared_data/RSS2026_Mingi/experiment/demo/attached_container/20260310_120000

    # Save output to a specific path
    python src/validation/perception/scene.py \
        --obj attached_container --ref_idx 000 \
        --output overlay_grid.png
"""
import argparse
import datetime
import os

import cv2
import numpy as np
import trimesh

from paradex.image.image_dict import ImageDict
from paradex.calibration.utils import load_c2r, save_current_camparam, save_current_C2R

from autodex.utils.path import project_dir, obj_path, get_object_mesh
from autodex.utils.conversion import cart2se3, se32cart
from autodex.utils.scene import get_scene_image_dict_template


def capture_scene(exp_name, obj_name, dir_idx, ref_idx):
    """Capture images and estimate 6D object pose. Returns (img_dir, scene_cfg)."""
    from paradex.io.camera_system.remote_camera_controller import remote_camera_controller

    rcc = remote_camera_controller("test_lookup_obstacle")

    img_dir = os.path.join(project_dir, "experiment", exp_name, obj_name, dir_idx)
    os.makedirs(img_dir, exist_ok=True)

    rcc.start("image", False,
              os.path.join("shared_data/AutoDex", "experiment", exp_name, obj_name, dir_idx, "raw"))
    rcc.stop()

    save_current_C2R(img_dir)
    save_current_camparam(img_dir)

    ref_dir = os.path.join(project_dir, "..", "shared_data", "AutoDex",
                           "object_pose_template", obj_name, ref_idx)
    scene_cfg = get_scene_image_dict_template(img_dir, ref_dir, obj_name)

    rcc.end()
    return img_dir, scene_cfg


def overlay_and_grid(img_dir, scene_cfg, obj_name, output_path):
    """
    Project detected mesh onto all camera views and save a grid image.

    Uses paradex ImageDict.project_mesh() for rendering and
    ImageDict.merge() for the grid layout.
    """
    c2r = load_c2r(img_dir)
    img_dict = ImageDict.from_path(img_dir)

    # Undistort if not already done
    if not os.path.exists(os.path.join(img_dir, "images")):
        img_dict.undistort()
        img_dict = ImageDict.from_path(img_dir)

    # Load object mesh and transform to camera frame
    obj_pose = cart2se3(scene_cfg["mesh"]["target"]["pose"])
    mesh = get_object_mesh(obj_name)
    mesh.apply_transform(c2r @ obj_pose)

    # Project mesh onto all camera images (green overlay)
    overlaid = img_dict.project_mesh(mesh, color=(0, 255, 0), alpha=0.5)

    # Also overlay table cuboid if present
    if "table" in scene_cfg.get("cuboid", {}):
        table_info = scene_cfg["cuboid"]["table"]
        table_pose = cart2se3(table_info["pose"])
        table_mesh = trimesh.creation.box(extents=table_info["dims"])
        table_mesh.apply_transform(c2r @ table_pose)
        overlaid = overlaid.project_mesh(table_mesh, color=(0, 0, 255), alpha=0.7)

    # Merge into grid with serial labels
    grid = overlaid.merge()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, grid)
    print(f"Overlay grid saved to: {output_path}")
    return grid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, required=True, help="Object name")
    parser.add_argument("--ref_idx", type=str, required=True, help="Reference index for pose estimation")
    parser.add_argument("--img_dir", type=str, default=None,
                        help="Existing capture directory (skip live capture if provided)")
    parser.add_argument("--exp_name", type=str, default="validation",
                        help="Experiment name (only used for live capture)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for overlay grid image")
    args = parser.parse_args()

    if args.img_dir is not None:
        # Use existing capture
        img_dir = args.img_dir
        ref_dir = os.path.join(project_dir, "..", "shared_data", "AutoDex",
                               "object_pose_template", args.obj, args.ref_idx)
        scene_cfg = get_scene_image_dict_template(img_dir, ref_dir, args.obj)
    else:
        # Live capture
        dir_idx = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        img_dir, scene_cfg = capture_scene(args.exp_name, args.obj, dir_idx, args.ref_idx)

    if scene_cfg is None:
        print("Object 6D pose estimation failed — no pose found.")
        exit(1)

    # Add table
    scene_cfg.setdefault("cuboid", {})
    scene_cfg["cuboid"]["table"] = {
        "dims": [2, 3, 0.2],
        "pose": [1.1, 0, -0.1 + 0.037, 1, 0, 0, 0],
    }

    output_path = args.output or os.path.join(img_dir, "overlay_grid.png")
    overlay_and_grid(img_dir, scene_cfg, args.obj, output_path)