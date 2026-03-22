import os
import random
import numpy as np
import trimesh

home_path = os.path.expanduser("~")
code_path = os.path.join(home_path, "RSS_2026")
shared_dir = os.path.join(home_path, "shared_data")
project_dir = os.path.join(shared_dir, "AutoDex")
bodex_path = os.path.join(code_path, "BODex_outputs")
repo_dir = os.path.join(home_path, "AutoDex")
candidate_path = os.path.join(repo_dir, "candidates", "allegro")

robot_configs_path = os.path.join(project_dir, "content", "configs", "robot")
obj_path = os.path.join(project_dir, "object", "paradex")
urdf_path = os.path.join(project_dir, "content", "assets", "robot", "allegro_description")


def get_object_mesh(obj_name):
    mesh = trimesh.load(os.path.join(obj_path, obj_name, "raw_mesh", f"{obj_name}.obj"))
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh


def load_candidate(obj_name, obj_pose, version, shuffle=True):
    wrist_se3_list = []
    pregrasp_pose_list = []
    grasp_pose_list = []
    scene_info = []

    candidate_obj_path = os.path.join(candidate_path, version, obj_name)

    scene_types = os.listdir(candidate_obj_path)
    if shuffle:
        random.shuffle(scene_types)
    else:
        scene_types = sorted(scene_types)

    for scene_type in scene_types:
        scene_ids = os.listdir(os.path.join(candidate_obj_path, scene_type))
        if shuffle:
            random.shuffle(scene_ids)
        else:
            scene_ids = sorted(scene_ids)

        for scene_id in scene_ids:
            grasp_idxs = os.listdir(os.path.join(candidate_obj_path, scene_type, scene_id))
            if shuffle:
                random.shuffle(grasp_idxs)
            else:
                grasp_idxs = sorted(grasp_idxs)

            for grasp_idx in grasp_idxs:
                base = os.path.join(candidate_obj_path, scene_type, scene_id, grasp_idx)
                pregrasp = np.load(os.path.join(base, "pregrasp_pose.npy"))
                pregrasp_pose_list.append(pregrasp)
                grasp_file = os.path.join(base, "grasp_pose.npy")
                grasp_pose_list.append(np.load(grasp_file) if os.path.exists(grasp_file) else pregrasp)
                wrist_se3_obj = np.load(os.path.join(base, "wrist_se3.npy"))
                wrist_se3_list.append(obj_pose @ wrist_se3_obj)
                scene_info.append((scene_type, scene_id, grasp_idx))

    wrist_se3 = np.array(wrist_se3_list)
    grasp_pose = np.array(grasp_pose_list)
    pregrasp_pose = np.array(pregrasp_pose_list)

    return wrist_se3, pregrasp_pose, grasp_pose, scene_info
