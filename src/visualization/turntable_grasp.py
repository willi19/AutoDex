#!/usr/bin/env python3
"""
Turntable rendering of grasp candidates using Open3D offscreen renderer.

Usage:
    # Single grasp
    python turntable_grasp.py --version revalidate --obj soap_dispenser --scene shelf/1/11

    # Top 100 grasps from setcover order
    python turntable_grasp.py --version revalidate --obj soap_dispenser --top 100

    # All 98 objects, top 100 each (reads setcover from order/)
    python turntable_grasp.py --batch-all --top 100

    # Custom render settings
    python turntable_grasp.py --version revalidate --obj soap_dispenser --scene shelf/1/11 --frames 60 --fps 30 --width 1080 --height 1080
"""

import argparse
import json
import os
import sys
import tempfile
import shutil
import subprocess

import numpy as np
import open3d as o3d
import trimesh

from rsslib.conversion import cart2se3
from rsslib.path import candidate_path, obj_path, urdf_path, code_path
from paradex.visualization.robot import RobotModule


COLOR_ROBOT = np.array([153, 128, 224]) / 255.0
COLOR_OBJECT = np.array([180, 180, 180]) / 255.0

ORDER_ROOT = os.path.join(code_path, "order")

# Version priority for --batch-all (first match wins, except revalidate always wins)
BATCH_VERSIONS = ["revalidate", "v2", "v3"]


def get_all_objects() -> list:
    """Get all (version, obj_name) pairs across versions, deduplicating with revalidate priority."""
    obj_version = {}
    for v in BATCH_VERSIONS:
        vdir = os.path.join(ORDER_ROOT, v)
        if not os.path.isdir(vdir):
            continue
        for obj in sorted(os.listdir(vdir)):
            if os.path.exists(os.path.join(vdir, obj, "setcover_order.json")):
                if obj not in obj_version or v == "revalidate":
                    obj_version[obj] = v
    return sorted(obj_version.items(), key=lambda x: x[0])


def load_setcover_grasps(version: str, obj_name: str, top_n: int) -> list:
    """Load top N grasps from setcover_order.json.
    Returns list of (scene_type, scene_id, grasp_name) tuples."""
    json_path = os.path.join(ORDER_ROOT, version, obj_name, "setcover_order.json")
    if not os.path.exists(json_path):
        print(f"Warning: setcover not found: {json_path}")
        return []
    with open(json_path, "r") as f:
        ordered_list = json.load(f)
    # Each entry: [obj_name, version, scene_type, scene_id, grasp_name, orig_idx]
    grasps = []
    for item in ordered_list[:top_n]:
        grasps.append((item[2], item[3], item[4]))
    return grasps


def list_all_grasps(version: str, obj_name: str, scene_type_filter: str = None) -> list:
    """Walk candidates directory to find all grasps.
    Returns list of (scene_type, scene_id, grasp_name) tuples."""
    root = os.path.join(candidate_path, version, obj_name)
    if not os.path.exists(root):
        print(f"Error: candidates path not found: {root}")
        sys.exit(1)

    grasps = []
    scene_types = [scene_type_filter] if scene_type_filter else sorted(os.listdir(root))
    for scene_type in scene_types:
        st_path = os.path.join(root, scene_type)
        if not os.path.isdir(st_path):
            continue
        for scene_id in sorted(os.listdir(st_path)):
            si_path = os.path.join(st_path, scene_id)
            if not os.path.isdir(si_path):
                continue
            for grasp_name in sorted(os.listdir(si_path), key=lambda x: int(x) if x.isdigit() else x):
                gp = os.path.join(si_path, grasp_name)
                if os.path.isdir(gp) and os.path.exists(os.path.join(gp, "wrist_se3.npy")):
                    grasps.append((scene_type, scene_id, grasp_name))
    return grasps


def trimesh_to_o3d(mesh: trimesh.Trimesh, color=None) -> o3d.geometry.TriangleMesh:
    """Convert a trimesh.Trimesh to open3d.geometry.TriangleMesh.
    If color is given, paint uniform. Otherwise bake texture/vertex colors."""
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.compute_vertex_normals()
    if color is not None:
        o3d_mesh.paint_uniform_color(color)
    else:
        # Bake texture to vertex colors (handles UV-mapped textures)
        try:
            color_visual = mesh.visual.to_color()
            vc = np.array(color_visual.vertex_colors)[:, :3] / 255.0
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vc)
        except Exception:
            o3d_mesh.paint_uniform_color([0.7, 0.7, 0.7])
    return o3d_mesh


def load_object_mesh(obj_name: str) -> tuple:
    """Load object mesh and its pose from scene JSON. Returns (trimesh, 4x4 pose)."""
    scene_json_path = os.path.join(obj_path, obj_name, "scene", "table", "4.json")
    if os.path.exists(scene_json_path):
        with open(scene_json_path, "r") as f:
            cfg = json.load(f)
        obj_pose = cart2se3(cfg['scene']['mesh']['target']['pose'])
    else:
        obj_pose = np.eye(4)

    mesh_path = os.path.join(obj_path, obj_name, "raw_mesh", f"{obj_name}.obj")
    mesh = trimesh.load(mesh_path, force="mesh")
    return mesh, obj_pose


def find_grasp_path(obj_name: str, version: str, scene_type: str,
                    scene_id: str, grasp_name: str) -> str:
    """Find grasp data path. Checks tselected_100 first (matches setcover order),
    then falls back to candidates/{version}/."""
    # tselected_100 has the correct top-100 from setcover for 78 objects
    path = os.path.join(candidate_path, "tselected_100", obj_name, scene_type, scene_id, grasp_name)
    if os.path.exists(path):
        return path
    # Fall back to full candidates (needed for 20 v2 objects not in tselected_100)
    path = os.path.join(candidate_path, version, obj_name, scene_type, scene_id, grasp_name)
    if os.path.exists(path):
        return path
    raise FileNotFoundError(
        f"Grasp not found in tselected_100 or candidates/{version}: "
        f"{obj_name}/{scene_type}/{scene_id}/{grasp_name}"
    )


def load_robot_at_grasp(obj_name: str, version: str, scene_type: str,
                        scene_id: str, grasp_name: str, obj_pose: np.ndarray) -> trimesh.Trimesh:
    """Load robot hand mesh at the grasp pose. Returns combined trimesh in world frame."""
    grasp_path = find_grasp_path(obj_name, version, scene_type, scene_id, grasp_name)

    wrist_se3 = np.load(os.path.join(grasp_path, "wrist_se3.npy"))
    grasp_pose = np.load(os.path.join(grasp_path, "grasp_pose.npy"))

    allegro_urdf = os.path.join(urdf_path, "allegro_hand_description_right.urdf")
    robot = RobotModule(allegro_urdf)

    joint_angles = grasp_pose.flatten()[:robot.num_joints]
    cfg = {name: angle for name, angle in zip(robot.joint_names, joint_angles)}
    robot.update_cfg(cfg)

    robot_mesh = robot.get_robot_mesh(collision_geometry=False)
    world_T = obj_pose @ wrist_se3
    robot_mesh.apply_transform(world_T)
    return robot_mesh


def compute_auto_camera(combined_mesh: trimesh.Trimesh, fov_deg: float,
                        aspect_ratio: float, elevation_deg: float = 25.0,
                        padding: float = 1.3) -> tuple:
    """Compute camera orbit params that guarantee the full scene is visible."""
    bsphere = combined_mesh.bounding_sphere
    center = bsphere.primitive.center
    sphere_radius = bsphere.primitive.radius

    vfov = np.radians(fov_deg)
    hfov = 2.0 * np.arctan(np.tan(vfov / 2.0) * aspect_ratio)
    effective_half_fov = min(vfov, hfov) / 2.0

    cam_dist = (sphere_radius * padding) / np.sin(effective_half_fov)
    return center, cam_dist, elevation_deg


def compute_turntable_camera(center: np.ndarray, cam_dist: float,
                             elevation_deg: float, angle_rad: float) -> tuple:
    """Compute camera eye/lookat/up for a turntable frame."""
    elev_rad = np.radians(elevation_deg)
    horiz_r = cam_dist * np.cos(elev_rad)
    vert_h = cam_dist * np.sin(elev_rad)

    eye = np.array([
        center[0] + horiz_r * np.cos(angle_rad),
        center[1] + horiz_r * np.sin(angle_rad),
        center[2] + vert_h,
    ])
    lookat = center.copy()
    up = np.array([0.0, 0.0, 1.0])
    return eye, lookat, up


def render_turntable(obj_mesh_o3d, robot_mesh_o3d, center, cam_dist,
                     elevation_deg, fov_deg, n_frames, width, height_px,
                     output_dir):
    """Render turntable frames using Open3D offscreen renderer."""
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height_px)
    mat_obj = o3d.visualization.rendering.MaterialRecord()
    mat_obj.shader = "defaultLit"

    mat_robot = o3d.visualization.rendering.MaterialRecord()
    mat_robot.shader = "defaultLit"

    renderer.scene.add_geometry("object", obj_mesh_o3d, mat_obj)
    renderer.scene.add_geometry("robot", robot_mesh_o3d, mat_robot)

    renderer.scene.scene.set_sun_light([0.5, -0.5, -1.0], [1.0, 1.0, 1.0], 60000)
    renderer.scene.scene.enable_sun_light(True)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])

    angles = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)

    for i, angle in enumerate(angles):
        eye, lookat, up = compute_turntable_camera(center, cam_dist, elevation_deg, angle)
        renderer.setup_camera(fov_deg, lookat, eye, up)

        img = renderer.render_to_image()
        frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        o3d.io.write_image(frame_path, img)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Rendered frame {i+1}/{n_frames}")

    del renderer


def frames_to_video(frame_dir: str, output_path: str, fps: int):
    """Encode PNG frames to video using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-framerate", str(fps),
        "-i", os.path.join(frame_dir, "frame_%04d.png"),
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr}")


def render_single_grasp(obj_name, version, scene_type, scene_id, grasp_name,
                        output_path, args, obj_mesh=None, obj_pose=None):
    """Render a single grasp turntable video. Returns True on success."""
    try:
        if obj_mesh is None or obj_pose is None:
            obj_mesh, obj_pose = load_object_mesh(obj_name)

        robot_mesh = load_robot_at_grasp(obj_name, version, scene_type, scene_id, grasp_name, obj_pose)

        obj_mesh_world = obj_mesh.copy()
        obj_mesh_world.apply_transform(obj_pose)

        obj_mesh_o3d = trimesh_to_o3d(obj_mesh_world)
        robot_mesh_o3d = trimesh_to_o3d(robot_mesh, color=COLOR_ROBOT)

        combined = trimesh.util.concatenate([obj_mesh_world, robot_mesh])
        aspect_ratio = args.width / args.height
        center, cam_dist, elevation_deg = compute_auto_camera(
            combined, fov_deg=args.fov, aspect_ratio=aspect_ratio,
            elevation_deg=args.elevation, padding=args.padding,
        )

        temp_dir = tempfile.mkdtemp(prefix="turntable_")
        render_turntable(
            obj_mesh_o3d, robot_mesh_o3d, center, cam_dist, elevation_deg,
            args.fov, args.frames, args.width, args.height, temp_dir,
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        frames_to_video(temp_dir, output_path, args.fps)
        shutil.rmtree(temp_dir)
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Turntable rendering of grasp candidates")
    parser.add_argument("--version", type=str, default=None, help="Candidate version (e.g., revalidate)")
    parser.add_argument("--obj", type=str, default=None, help="Object name (e.g., soap_dispenser)")
    parser.add_argument("--scene", type=str, default=None,
                        help="Single grasp: scene_type/scene_id/grasp_name (e.g., shelf/1/11)")
    parser.add_argument("--top", type=int, default=None,
                        help="Render top N grasps from setcover order")
    parser.add_argument("--batch-all", action="store_true",
                        help="Batch render all 98 objects (uses setcover, requires --top)")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory for batch mode (default: data)")
    parser.add_argument("--frames", type=int, default=60, help="Number of frames (default: 60)")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS (default: 30)")
    parser.add_argument("--width", type=int, default=540, help="Render width (default: 540)")
    parser.add_argument("--height", type=int, default=540, help="Render height (default: 540)")
    parser.add_argument("--fov", type=float, default=45.0, help="Vertical FOV in degrees (default: 45)")
    parser.add_argument("--elevation", type=float, default=25.0, help="Camera elevation angle in degrees (default: 25)")
    parser.add_argument("--padding", type=float, default=1.3, help="Camera padding multiplier (default: 1.3)")
    parser.add_argument("--output", type=str, default=None, help="Output video path (single grasp mode)")
    args = parser.parse_args()

    # ---- Batch all objects ----
    if args.batch_all:
        if args.top is None:
            print("Error: --batch-all requires --top N")
            sys.exit(1)

        all_objects = get_all_objects()
        print(f"Batch rendering {len(all_objects)} objects, top {args.top} grasps each")
        output_dir = os.path.abspath(args.output_dir)
        os.makedirs(output_dir, exist_ok=True)

        total_success = 0
        total_fail = 0

        for obj_idx, (obj_name, version) in enumerate(all_objects):
            print(f"\n[{obj_idx+1}/{len(all_objects)}] {version}/{obj_name}")

            grasps = load_setcover_grasps(version, obj_name, args.top)
            if not grasps:
                print(f"  Skipping (no setcover data)")
                continue

            # Load object mesh once per object
            try:
                obj_mesh, obj_pose = load_object_mesh(obj_name)
            except Exception as e:
                print(f"  Error loading object mesh: {e}")
                total_fail += len(grasps)
                continue

            for gi, (scene_type, scene_id, grasp_name) in enumerate(grasps):
                episode_dir = os.path.join(output_dir, obj_name, f"{gi:03d}")
                video_path = os.path.join(episode_dir, "turntable.mp4")
                if os.path.exists(video_path):
                    print(f"  [{gi+1}/{len(grasps)}] Already exists, skipping")
                    total_success += 1
                    continue

                print(f"  [{gi+1}/{len(grasps)}] {scene_type}/{scene_id}/{grasp_name}")
                os.makedirs(episode_dir, exist_ok=True)
                ok = render_single_grasp(
                    obj_name, version, scene_type, scene_id, grasp_name,
                    video_path, args, obj_mesh=obj_mesh, obj_pose=obj_pose,
                )
                if ok:
                    total_success += 1
                else:
                    total_fail += 1

        print(f"\nDone! {total_success} succeeded, {total_fail} failed")
        print(f"Output: {output_dir}")
        return

    # ---- Single object, top N grasps ----
    if args.top is not None:
        if args.version is None or args.obj is None:
            print("Error: --top requires --version and --obj")
            sys.exit(1)

        grasps = load_setcover_grasps(args.version, args.obj, args.top)
        if not grasps:
            print("No grasps found in setcover order")
            sys.exit(1)

        print(f"Rendering top {len(grasps)} grasps for {args.version}/{args.obj}")

        obj_mesh, obj_pose = load_object_mesh(args.obj)
        output_dir = os.path.abspath(args.output_dir)

        success, fail = 0, 0
        for gi, (scene_type, scene_id, grasp_name) in enumerate(grasps):
            episode_dir = os.path.join(output_dir, args.obj, f"{gi:03d}")
            video_path = os.path.join(episode_dir, "turntable.mp4")

            if os.path.exists(video_path):
                print(f"[{gi+1}/{len(grasps)}] Already exists, skipping")
                success += 1
                continue

            print(f"[{gi+1}/{len(grasps)}] {scene_type}/{scene_id}/{grasp_name}")
            os.makedirs(episode_dir, exist_ok=True)
            ok = render_single_grasp(
                args.obj, args.version, scene_type, scene_id, grasp_name,
                video_path, args, obj_mesh=obj_mesh, obj_pose=obj_pose,
            )
            if ok:
                success += 1
            else:
                fail += 1

        print(f"\nDone! {success} succeeded, {fail} failed")
        print(f"Output: {os.path.join(output_dir, args.obj)}")
        return

    # ---- Single grasp mode ----
    if args.scene is None or args.version is None or args.obj is None:
        print("Error: single grasp mode requires --version, --obj, and --scene")
        parser.print_help()
        sys.exit(1)

    scene_parts = args.scene.strip("/").split("/")
    if len(scene_parts) != 3:
        print(f"Error: --scene must be scene_type/scene_id/grasp_name, got '{args.scene}'")
        sys.exit(1)
    scene_type, scene_id, grasp_name = scene_parts

    if args.output is None:
        output_path = os.path.abspath(f"turntable_{args.obj}_{scene_type}_{scene_id}_{grasp_name}.mp4")
    else:
        output_path = os.path.abspath(args.output)

    print(f"Loading {args.obj} grasp: {scene_type}/{scene_id}/{grasp_name}")
    ok = render_single_grasp(
        args.obj, args.version, scene_type, scene_id, grasp_name,
        output_path, args,
    )
    if ok:
        print(f"Saved: {output_path}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
