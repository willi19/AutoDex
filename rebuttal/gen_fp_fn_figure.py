"""Generate FP/FN example figures for rebuttal.

Left: grasp render (object + hand, single view from turntable code)
Right: 4 frames from real experiment video (end-3s, end-2s, end-1s, end)

Usage:
    # Single example (first FP)
    python rebuttal/gen_fp_fn_figure.py --type fp --index 0

    # All FP and FN
    python rebuttal/gen_fp_fn_figure.py --all

    # Custom serial and threshold
    python rebuttal/gen_fp_fn_figure.py --all --serial 25305466 --sim_threshold 0.5
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from autodex.utils.conversion import cart2se3
from autodex.utils.path import obj_path, urdf_path, repo_dir

EXPERIMENT_ROOT = os.path.expanduser(
    "~/shared_data/AutoDex/experiment/selected_100/allegro"
)
CANDIDATE_ROOT = os.path.expanduser(
    "~/shared_data/AutoDex/candidates/allegro/selected_100"
)
RESULTS_PATH = os.path.join(repo_dir, "outputs", "sim_dr", "results_40grid.json")
OUTPUT_ROOT = os.path.join(repo_dir, "rebuttal", "figure", "2")

HAND_URDF = os.path.join(urdf_path, "allegro_hand_description_right.urdf")

# Turntable render settings
RENDER_W, RENDER_H = 512, 512
FOV_DEG = 45.0
ELEVATION_DEG = 25.0
PADDING = 1.3
ANGLE_RAD = np.radians(30)  # fixed viewing angle

# Frame offsets from video end (seconds)
FRAME_OFFSETS = [-3, -2, -1, 0]


def get_fp_fn(results_path, sim_threshold=0.5):
    """Load sim DR results, return (fp_list, fn_list)."""
    with open(results_path) as f:
        data = json.load(f)

    fp, fn = [], []
    for r in data["results"]:
        n_succ = sum(1 for t in r["trials"] if t["success"])
        sim_pass = (n_succ / len(r["trials"])) >= sim_threshold
        real = r["real_success"]
        if not real and sim_pass:
            fp.append(r)
        elif real and not sim_pass:
            fn.append(r)
    return fp, fn


def render_grasp_image(obj_name, scene_info):
    """Render single-view grasp image using Open3D offscreen."""
    import open3d as o3d
    from paradex.visualization.robot import RobotModule
    from src.visualization.turntable_grasp import (
        trimesh_to_o3d, compute_auto_camera, compute_turntable_camera,
        COLOR_ROBOT,
    )

    scene_type, scene_id, grasp_idx = scene_info
    cand_dir = os.path.join(CANDIDATE_ROOT, obj_name, scene_type, scene_id, grasp_idx)

    # Load object mesh + pose
    scene_json = os.path.join(obj_path, obj_name, "scene", "table", "4.json")
    if os.path.exists(scene_json):
        with open(scene_json) as f:
            cfg = json.load(f)
        obj_pose = cart2se3(cfg["scene"]["mesh"]["target"]["pose"])
    else:
        obj_pose = np.eye(4)

    mesh_path = os.path.join(obj_path, obj_name, "raw_mesh", f"{obj_name}.obj")
    obj_mesh = trimesh.load(mesh_path, force="mesh")

    # Load hand
    wrist_se3 = np.load(os.path.join(cand_dir, "wrist_se3.npy"))
    grasp_pose = np.load(os.path.join(cand_dir, "grasp_pose.npy"))

    robot = RobotModule(HAND_URDF)
    joint_angles = grasp_pose.flatten()[:robot.num_joints]
    cfg = {name: angle for name, angle in zip(robot.joint_names, joint_angles)}
    robot.update_cfg(cfg)
    robot_mesh = robot.get_robot_mesh(collision_geometry=False)
    robot_mesh.apply_transform(obj_pose @ wrist_se3)

    # Transform object to world
    obj_world = obj_mesh.copy()
    obj_world.apply_transform(obj_pose)

    # Convert to o3d
    obj_o3d = trimesh_to_o3d(obj_world)
    robot_o3d = trimesh_to_o3d(robot_mesh, color=COLOR_ROBOT)

    # Camera
    combined = trimesh.util.concatenate([obj_world, robot_mesh])
    center, cam_dist, elev = compute_auto_camera(
        combined, fov_deg=FOV_DEG, aspect_ratio=1.0,
        elevation_deg=ELEVATION_DEG, padding=PADDING,
    )
    eye, lookat, up = compute_turntable_camera(center, cam_dist, elev, ANGLE_RAD)

    # Render
    renderer = o3d.visualization.rendering.OffscreenRenderer(RENDER_W, RENDER_H)
    renderer.scene.clear_geometry()

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    renderer.scene.add_geometry("object", obj_o3d, mat)
    renderer.scene.add_geometry("robot", robot_o3d, mat)
    renderer.scene.scene.set_sun_light([0.5, -0.5, -1.0], [1.0, 1.0, 1.0], 60000)
    renderer.scene.scene.enable_sun_light(True)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    renderer.setup_camera(FOV_DEG, lookat, eye, up)

    img = np.asarray(renderer.render_to_image())
    del renderer
    return img


def extract_video_frames(exp_dir, serial, offsets_sec=FRAME_OFFSETS):
    """Extract frames at offsets from end of video. Returns list of BGR images."""
    video_path = os.path.join(exp_dir, "videos", f"{serial}.avi")
    if not os.path.exists(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or n_frames <= 0:
        cap.release()
        return None

    last_frame = n_frames - 1
    frames = []
    for offset in offsets_sec:
        frame_idx = max(0, min(last_frame, last_frame + int(offset * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            frames.append(np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), dtype=np.uint8))
    cap.release()
    return frames


def make_figure(grasp_img, video_frames, target_h=512):
    """Compose figure: [grasp_render | frame0 | frame1 | frame2 | frame3].

    grasp_img: RGB HxWx3
    video_frames: list of 4 BGR HxWx3
    """
    # Resize grasp to target_h
    gh, gw = grasp_img.shape[:2]
    scale = target_h / gh
    grasp_resized = cv2.resize(grasp_img, (int(gw * scale), target_h))
    # RGB -> BGR for cv2
    grasp_bgr = cv2.cvtColor(grasp_resized, cv2.COLOR_RGB2BGR)

    # Resize video frames to target_h, keep aspect
    resized_frames = []
    for f in video_frames:
        fh, fw = f.shape[:2]
        s = target_h / fh
        resized_frames.append(cv2.resize(f, (int(fw * s), target_h)))

    # Concatenate horizontally
    parts = [grasp_bgr] + resized_frames
    return np.concatenate(parts, axis=1)


def process_one(entry, serial, output_dir, label):
    """Generate figure for one FP/FN entry."""
    obj_name = entry["obj"]
    dir_idx = entry["dir_idx"]
    scene_info = entry["scene_info"]
    exp_dir = os.path.join(EXPERIMENT_ROOT, obj_name, dir_idx)

    sim_rate = sum(1 for t in entry["trials"] if t["success"]) / len(entry["trials"])

    print(f"  {label}: {obj_name}/{dir_idx} si={scene_info} sim_rate={sim_rate:.2f} real={'succ' if entry['real_success'] else 'fail'}")

    # Render grasp
    try:
        grasp_img = render_grasp_image(obj_name, scene_info)
    except Exception as e:
        print(f"    Render failed: {e}")
        return False

    # Extract video frames
    video_frames = extract_video_frames(exp_dir, serial)
    if video_frames is None:
        print(f"    Video not found for serial {serial}")
        return False

    # Compose
    fig = make_figure(grasp_img, video_frames)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{obj_name}_{dir_idx}_{scene_info[0]}_{scene_info[1]}_{scene_info[2]}.png"
    out_path = os.path.join(output_dir, fname)
    cv2.imwrite(out_path, fig)
    print(f"    Saved: {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["fp", "fn"], default=None,
                        help="Generate only FP or FN (default: both with --all)")
    parser.add_argument("--index", type=int, default=0, help="Index within FP/FN list")
    parser.add_argument("--all", action="store_true", help="Generate all FP and FN figures")
    parser.add_argument("--serial", type=str, default="25305466", help="Camera serial for video frames")
    parser.add_argument("--sim_threshold", type=float, default=0.5)
    parser.add_argument("--results", type=str, default=RESULTS_PATH)
    args = parser.parse_args()

    fp_list, fn_list = get_fp_fn(args.results, args.sim_threshold)
    print(f"FP: {len(fp_list)}, FN: {len(fn_list)}")

    if args.all:
        fp_dir = os.path.join(OUTPUT_ROOT, "false_positive")
        fn_dir = os.path.join(OUTPUT_ROOT, "false_negative")
        for i, entry in enumerate(fp_list):
            process_one(entry, args.serial, fp_dir, f"FP[{i}]")
        for i, entry in enumerate(fn_list):
            process_one(entry, args.serial, fn_dir, f"FN[{i}]")
    elif args.type:
        lst = fp_list if args.type == "fp" else fn_list
        if args.index >= len(lst):
            print(f"Index {args.index} out of range (max {len(lst)-1})")
            return
        subdir = "false_positive" if args.type == "fp" else "false_negative"
        out_dir = os.path.join(OUTPUT_ROOT, subdir)
        process_one(lst[args.index], args.serial, out_dir,
                    f"{'FP' if args.type == 'fp' else 'FN'}[{args.index}]")
    else:
        print("Specify --type fp/fn or --all")


if __name__ == "__main__":
    main()
