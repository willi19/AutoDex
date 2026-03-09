import os
import subprocess
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from paradex.image.image_dict import ImageDict
from paradex.calibration.utils import load_c2r

from autodex.utils.path import project_dir, obj_path, get_object_mesh
from autodex.utils.conversion import cart2se3, se32cart


def overlay_scene(scene_cfg, target_name, img_dict, c2r):
    for mesh_name in scene_cfg["mesh"]:
        pose = scene_cfg["mesh"][mesh_name]["pose"]
        transform = cart2se3(pose)
        obj_name = mesh_name.split("target")[0] if "target" not in mesh_name else target_name
        mesh = get_object_mesh(obj_name)
        mesh.apply_transform(c2r @ transform)
        color = (0, 0, 255) if "target" not in mesh_name else (0, 255, 0)
        img_dict = img_dict.project_mesh(mesh, color)

    for obj_name in scene_cfg["cuboid"]:
        dims = scene_cfg["cuboid"][obj_name]["dims"]
        pose = scene_cfg["cuboid"][obj_name]["pose"]
        transform = cart2se3(pose)
        mesh = trimesh.creation.box(extents=dims)
        mesh.apply_transform(c2r @ transform)
        color = (0, 0, 255)
        img_dict = img_dict.project_mesh(mesh, color)

    return img_dict


def get_object_6d_template(obj_name, exp_name, dir_idx, ref_dir_idx):
    data_dir = os.path.join("../RSS2026_Mingi", "experiment", exp_name, obj_name, dir_idx)
    ref_data_dir = os.path.join("../RSS2026_Mingi", "object_pose_template", obj_name, ref_dir_idx)

    remote_cmd = (
        f"cd ~/shared_data/_object_6d_tracking && "
        f"source ~/anaconda3/etc/profile.d/conda.sh && "
        f"conda activate foundationpose && "
        f"python run/run_sam3_silhouette_from_reference.py "
        f"--config config/sam3_silhouette_from_reference.yaml "
        f"--obj_name {obj_name} "
        f"--data_dir {data_dir} "
        f"--ref_data_dir {ref_data_dir} "
        f"--vis_scale 0.5"
    )

    ssh_cmd = f"ssh -p 77 capture1@192.168.0.101 \"{remote_cmd}\""

    try:
        subprocess.run(ssh_cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"object detection Failed: {e}")

    pose_file = os.path.join(
        project_dir, "experiment", exp_name, obj_name, dir_idx,
        "outputs", f"{obj_name}_pose", "optimized_pose_world.txt"
    )
    if not os.path.exists(pose_file):
        return None

    obj_6d = open(pose_file, "r").readlines()
    obj_T = np.array([list(map(float, line.strip().split())) for line in obj_6d])
    return obj_T


def get_scene_image_dict_template(img_dir, ref_dir, target_name):
    c2r = load_c2r(img_dir)

    img_dict = ImageDict.from_path(img_dir)
    if not os.path.exists(os.path.join(img_dir, "images")):
        img_dict.undistort()
        img_dict = ImageDict.from_path(img_dir)

    scene_cfg = {
        "mesh": {},
        "cuboid": {}
    }
    exp_name = os.path.basename(os.path.dirname(os.path.dirname(img_dir)))
    obj_T = get_object_6d_template(target_name, exp_name, os.path.basename(img_dir), os.path.basename(ref_dir))
    if obj_T is None:
        return None
    scene_cfg["mesh"]["target"] = {
        "pose": se32cart(np.linalg.inv(c2r) @ obj_T).tolist(),
        "file_path": os.path.join(obj_path, target_name, "processed_data", "mesh", "simplified.obj"),
        "urdf_path": os.path.join(obj_path, target_name, "processed_data", "urdf", "coacd.urdf")
    }

    return scene_cfg
