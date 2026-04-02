#!/usr/bin/env python3
"""Compute object surface visibility/coverage from each camera.

For each experiment run, measures what fraction of the object surface is
visible from each camera, considering:
  1. Self-occlusion (surface normal facing away from camera)
  2. Robot occlusion (arm + hand blocking the view)

Usage:
    conda activate planner
    python rebuttal/compute_coverage.py
    python rebuttal/compute_coverage.py --obj brown_ramen --n_points 2000
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from paradex.visualization.robot import RobotModule

EXPERIMENT_ROOT = os.path.expanduser(
    "~/shared_data/AutoDex/experiment/selected_100/allegro"
)
HAND_URDF = os.path.expanduser(
    "~/shared_data/AutoDex/content/assets/robot/allegro_description/allegro_hand_description_right.urdf"
)
OBJ_MESH_ROOT = os.path.expanduser(
    "~/shared_data/AutoDex/object/paradex"
)


def load_object_mesh(obj_name):
    """Load object mesh (trimesh)."""
    import glob
    import trimesh

    mesh_dir = os.path.join(OBJ_MESH_ROOT, obj_name, "raw_mesh")
    mesh_path = os.path.join(mesh_dir, f"{obj_name}.obj")
    if not os.path.isfile(mesh_path):
        # Fallback: pick first .obj in the directory
        objs = sorted(glob.glob(os.path.join(mesh_dir, "*.obj")))
        if not objs:
            raise FileNotFoundError(f"No .obj mesh in {mesh_dir}")
        mesh_path = objs[0]

    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh


def sample_surface_points(mesh, n_points=1000):
    """Sample points + normals uniformly on mesh surface."""
    points, face_idx = mesh.sample(n_points, return_index=True)
    normals = mesh.face_normals[face_idx]
    return points, normals


def build_hand_mesh(hand_joints_16, wrist_se3):
    """Build hand mesh at wrist_se3 pose with given hand joint angles."""
    robot = RobotModule(HAND_URDF)
    cfg = {name: angle for name, angle in zip(robot.joint_names, hand_joints_16)}
    robot.update_cfg(cfg)
    mesh = robot.get_robot_mesh(collision_geometry=False)
    mesh.apply_transform(wrist_se3)
    return mesh


def compute_visibility(obj_points, obj_normals, obj_mesh_robot, robot_mesh, cam_positions):
    """Compute per-camera visibility of object surface points.

    Checks:
      1. Normal facing camera
      2. Object self-occlusion (ray from cam hits closer object surface first)
      3. Robot occlusion (ray from cam hits robot before reaching the point)

    Args:
        obj_points: (N, 3) object surface points in robot frame
        obj_normals: (N, 3) surface normals in robot frame
        obj_mesh_robot: trimesh.Trimesh of object in robot frame
        robot_mesh: trimesh.Trimesh of robot in robot frame
        cam_positions: dict {serial: (3,) position in robot frame}

    Returns:
        visibility: dict {serial: (N,) bool array}
    """
    import trimesh

    # Combined scene mesh (object + robot) for full occlusion check
    combined = trimesh.util.concatenate([obj_mesh_robot, robot_mesh])
    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(combined)

    visibility = {}
    for serial, cam_pos in cam_positions.items():
        # 1. Normal facing check
        cam_dir = cam_pos[None, :] - obj_points  # (N, 3)
        cam_dist = np.linalg.norm(cam_dir, axis=1, keepdims=True)
        cam_dir_norm = cam_dir / (cam_dist + 1e-10)
        facing = np.sum(obj_normals * cam_dir_norm, axis=1) > 0  # (N,)

        facing_idx = np.where(facing)[0]
        if len(facing_idx) == 0:
            visibility[serial] = facing
            continue

        # 2. Cast rays from CAMERA toward each object point
        #    If the first hit is close to the expected distance, the point is visible.
        #    If something blocks the ray (object self-occlusion or robot), it's not.
        pts = obj_points[facing_idx]
        dists = cam_dist[facing_idx, 0]

        ray_origins = np.tile(cam_pos, (len(pts), 1))
        ray_dirs = pts - cam_pos  # cam -> point
        ray_dirs /= np.linalg.norm(ray_dirs, axis=1, keepdims=True) + 1e-10

        # Find first intersection along each ray
        locs, ray_ids, _ = intersector.intersects_location(
            ray_origins, ray_dirs, multiple_hits=False,
        )

        # A point is visible if the first hit is at (approximately) the point's distance
        visible_mask = np.zeros(len(facing_idx), dtype=bool)
        if len(locs) > 0:
            hit_dists = np.linalg.norm(locs - cam_pos, axis=1)
            expected_dists = dists[ray_ids]
            # Visible if first hit is within 1cm of the expected point
            visible_mask[ray_ids] = np.abs(hit_dists - expected_dists) < 0.01

        vis = facing.copy()
        vis[facing_idx] = visible_mask
        visibility[serial] = vis

    return visibility


def compute_coverage_vs_ncams(visibility, n_repeats=10):
    """Compute coverage as function of number of cameras.

    For each k in 1..n_cams, randomly sample up to n_repeats camera subsets
    of size k and compute coverage. Report max and mean.

    Args:
        visibility: dict {serial: (N,) bool}
        n_repeats: number of random subsets per camera count

    Returns:
        results: list of dicts, one per k:
            {n_cams, max_coverage, mean_coverage, coverages: [...]}
    """
    serials = list(visibility.keys())
    n_points = len(next(iter(visibility.values())))
    n_cams = len(serials)

    # Stack into matrix (n_cams, n_points)
    vis_mat = np.array([visibility[s] for s in serials])

    from math import comb

    results = []
    for k in range(1, n_cams + 1):
        total_combos = comb(n_cams, k)
        n_samples = min(n_repeats, total_combos)

        coverages = []
        seen = set()
        for _ in range(n_samples * 10):  # extra attempts to avoid duplicates
            if len(coverages) >= n_samples:
                break
            subset = tuple(sorted(np.random.choice(n_cams, k, replace=False)))
            if subset in seen:
                continue
            seen.add(subset)
            cov = np.any(vis_mat[list(subset)], axis=0)
            coverages.append(float(np.mean(cov)))

        results.append({
            "n_cams": k,
            "max_coverage": max(coverages),
            "mean_coverage": float(np.mean(coverages)),
            "coverages": coverages,
        })

    return results


def process_single_run(obj_name, run_dir, obj_mesh, n_points):
    """Process a single experiment run.

    Returns:
        dict with coverage results, or None if data is missing/invalid.
    """
    # Load experiment data
    result_path = os.path.join(run_dir, "result.json")
    wrist_path = os.path.join(run_dir, "plan", "wrist_se3.npy")
    squeeze_path = os.path.join(run_dir, "squeeze_hand.npy")
    pose_world_path = os.path.join(run_dir, "pose_world.npy")
    c2r_path = os.path.join(run_dir, "C2R.npy")
    ext_path = os.path.join(run_dir, "cam_param", "extrinsics.json")

    for p in [result_path, wrist_path, squeeze_path, pose_world_path, c2r_path, ext_path]:
        if not os.path.exists(p):
            return None

    with open(result_path) as f:
        result = json.load(f)

    pose_world = np.load(pose_world_path)
    C2R = np.load(c2r_path)
    wrist_se3 = np.load(wrist_path)
    squeeze_hand = np.load(squeeze_path)
    with open(ext_path) as f:
        extrinsics_raw = json.load(f)

    # Convert extrinsics to 4x4
    extrinsics = {}
    for s, T in extrinsics_raw.items():
        T = np.array(T, dtype=np.float64)
        if T.shape == (3, 4):
            T = np.vstack([T, [0, 0, 0, 1]])
        extrinsics[s] = T

    # World-to-robot transform
    W2R = np.linalg.inv(C2R)

    # Object pose in robot frame
    obj_pose_robot = W2R @ pose_world

    # Sample object surface points + normals (in object local frame)
    pts_local, normals_local = sample_surface_points(obj_mesh, n_points)

    # Transform to robot frame
    pts_h = np.hstack([pts_local, np.ones((len(pts_local), 1))])
    pts_robot = (obj_pose_robot @ pts_h.T).T[:, :3]
    R_obj = obj_pose_robot[:3, :3]
    normals_robot = (R_obj @ normals_local.T).T
    normals_robot /= np.linalg.norm(normals_robot, axis=1, keepdims=True) + 1e-10

    # Build hand mesh at wrist pose with squeeze joints
    # plan/wrist_se3.npy is already in robot frame
    # squeeze_hand is in executor order (thumb first) — convert to URDF order (thumb last)
    hand_urdf_order = np.empty(16)
    hand_urdf_order[:12] = squeeze_hand[4:]   # finger0,1,2
    hand_urdf_order[12:] = squeeze_hand[:4]   # thumb
    robot_mesh = build_hand_mesh(hand_urdf_order, wrist_se3)

    # Camera positions in robot frame
    cam_positions = {}
    for s, ext in extrinsics.items():
        # ext is world-to-cam → inv(ext) is cam-to-world
        # Camera origin in world frame: inv(ext)[:3, 3]
        cam_in_world = np.linalg.inv(ext)
        cam_in_robot = W2R @ cam_in_world
        cam_positions[s] = cam_in_robot[:3, 3]

    # Transform object mesh to robot frame for self-occlusion raycasting
    import trimesh
    obj_mesh_robot = obj_mesh.copy()
    obj_mesh_robot.apply_transform(obj_pose_robot)

    # Compute visibility (with robot occlusion)
    vis = compute_visibility(pts_robot, normals_robot, obj_mesh_robot, robot_mesh, cam_positions)

    # Coverage vs number of cameras
    cov_with_robot = compute_coverage_vs_ncams(vis)

    # Per-camera stats
    per_cam = {}
    for s in vis:
        per_cam[s] = {
            "visible_frac": float(np.mean(vis[s])),
            "n_visible": int(np.sum(vis[s])),
        }

    # Coverage without robot occlusion (object self-occlusion only)
    vis_no_robot = compute_visibility(pts_robot, normals_robot, obj_mesh_robot,
                                      trimesh.Trimesh(),  # empty mesh = no robot
                                      cam_positions)

    cov_no_robot = compute_coverage_vs_ncams(vis_no_robot)

    return {
        "obj_name": obj_name,
        "run_dir": os.path.basename(run_dir),
        "success": result.get("success", False),
        "scene_info": result.get("scene_info"),
        "n_cameras": len(cam_positions),
        "n_points": n_points,
        "per_camera": per_cam,
        "coverage_with_robot": cov_with_robot,
        "coverage_no_robot": cov_no_robot,
    }


def render_overlay(run_dir, obj_name, obj_mesh, out_dir):
    """Render arm(blue), hand(red), object(green) overlay on each camera image."""
    import cv2
    import trimesh

    traj_path = os.path.join(run_dir, "plan", "traj.npy")
    pose_world_path = os.path.join(run_dir, "pose_world.npy")
    c2r_path = os.path.join(run_dir, "C2R.npy")
    ext_path = os.path.join(run_dir, "cam_param", "extrinsics.json")
    intr_path = os.path.join(run_dir, "cam_param", "intrinsics.json")
    img_dir = os.path.join(run_dir, "images")

    for p in [traj_path, pose_world_path, c2r_path, ext_path, intr_path, img_dir]:
        if not os.path.exists(p):
            return

    pose_world = np.load(pose_world_path)
    C2R = np.load(c2r_path)
    traj = np.load(traj_path)
    with open(ext_path) as f:
        extr_raw = json.load(f)
    with open(intr_path) as f:
        intr_raw = json.load(f)

    W2R = np.linalg.inv(C2R)
    obj_pose_robot = W2R @ pose_world

    # Robot config (squeeze)
    q = traj[-1].copy()
    squeeze_path = os.path.join(run_dir, "squeeze_hand.npy")
    if os.path.exists(squeeze_path):
        q[6:] = np.load(squeeze_path)

    arm_mesh, hand_mesh = build_robot_meshes_separate(q)

    # Object mesh in robot frame
    obj_mesh_r = obj_mesh.copy()
    obj_mesh_r.apply_transform(obj_pose_robot)

    # For each camera: project meshes and draw colored overlay
    serials = sorted(os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(".png"))
    os.makedirs(out_dir, exist_ok=True)

    for s in serials:
        img = cv2.imread(os.path.join(img_dir, f"{s}.png"))
        if img is None:
            continue
        H, W = img.shape[:2]

        K = np.array(intr_raw[s]["intrinsics_undistort"], dtype=np.float64)
        T = np.array(extr_raw[s], dtype=np.float64)
        if T.shape == (3, 4):
            T = np.vstack([T, [0, 0, 0, 1]])

        # world-to-cam; but meshes are in robot frame
        # cam = T_world2cam @ T_robot2world @ pt_robot
        # T_robot2world = C2R
        P = T @ C2R  # robot-to-cam

        # Project all meshes' vertices to pixel space
        all_meshes = [
            (obj_mesh_r, (0, 200, 0)),     # green
            (arm_mesh, (255, 150, 0)),      # blue (BGR)
            (hand_mesh, (0, 0, 255)),       # red (BGR)
        ]

        # Build per-pixel depth + color via face rasterization
        color_layer = np.zeros_like(img)
        depth_buf = np.full((H, W), np.inf, dtype=np.float64)

        for mesh, color in all_meshes:
            if mesh is None or len(mesh.vertices) == 0:
                continue
            verts = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)
            verts_h = np.hstack([verts, np.ones((len(verts), 1))])
            cam_pts = (P @ verts_h.T).T[:, :3]

            # Project to pixel
            px = (K @ cam_pts.T).T
            uv = px[:, :2] / px[:, 2:3]
            z = cam_pts[:, 2]

            for face in faces:
                zs = z[face]
                if (zs < 0.01).any():
                    continue
                tri_uv = uv[face].astype(np.int32)
                tri_z = zs.mean()

                # Bounding box check
                u_min, v_min = tri_uv.min(axis=0)
                u_max, v_max = tri_uv.max(axis=0)
                if u_max < 0 or v_max < 0 or u_min >= W or v_min >= H:
                    continue

                # Fill triangle on mask
                tri_mask = np.zeros((H, W), dtype=np.uint8)
                cv2.fillPoly(tri_mask, [tri_uv[:, ::-1].reshape(1, 3, 2)], 1)

                # Actually use (u,v) = (col, row) for fillPoly: pts are (x,y)
                tri_mask = np.zeros((H, W), dtype=np.uint8)
                pts = tri_uv.reshape(1, 3, 2)  # (1, 3, 2) with (u, v) = (x, y)
                cv2.fillPoly(tri_mask, pts, 1)
                hit = tri_mask > 0

                # Depth test
                closer = hit & (tri_z < depth_buf)
                depth_buf[closer] = tri_z
                color_layer[closer] = color

        # Blend
        mask = depth_buf < np.inf
        overlay = img.copy()
        overlay[mask] = (img[mask].astype(float) * 0.5 + color_layer[mask].astype(float) * 0.5).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"{s}.png"), overlay)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, default=None, help="Single object name (default: all)")
    parser.add_argument("--n_points", type=int, default=1000, help="Surface sample count")
    parser.add_argument("--output", type=str, default="rebuttal/coverage_results.json")
    parser.add_argument("--max_runs", type=int, default=None, help="Max runs per object")
    parser.add_argument("--overlay", action="store_true", help="Save overlay images to autodex/tmp/")
    args = parser.parse_args()

    if args.obj:
        objects = [args.obj]
    else:
        objects = sorted(os.listdir(EXPERIMENT_ROOT))

    # Collect all (obj_name, run_dir) pairs
    from tqdm import tqdm

    tasks = []
    for obj_name in objects:
        obj_dir = os.path.join(EXPERIMENT_ROOT, obj_name)
        if not os.path.isdir(obj_dir):
            continue
        runs = sorted(os.listdir(obj_dir))
        if args.max_runs:
            runs = runs[:args.max_runs]
        for run_name in runs:
            run_dir = os.path.join(obj_dir, run_name)
            if os.path.isdir(run_dir):
                tasks.append((obj_name, run_dir))

    # Load existing results for resume
    all_results = []
    done_keys = set()
    if os.path.isfile(args.output):
        with open(args.output) as f:
            all_results = json.load(f)
        for r in all_results:
            done_keys.add((r["obj_name"], r["run_dir"]))
        print(f"Resuming: {len(all_results)} already done, {len(tasks) - len(done_keys)} remaining")

    obj_mesh_cache = {}
    pbar = tqdm(tasks, desc="Coverage", unit="run")
    for obj_name, run_dir in pbar:
        run_name = os.path.basename(run_dir)
        if not args.overlay and (obj_name, run_name) in done_keys:
            pbar.set_postfix_str(f"{obj_name}: cached")
            continue

        if obj_name not in obj_mesh_cache:
            obj_mesh_cache[obj_name] = load_object_mesh(obj_name)
        obj_mesh = obj_mesh_cache[obj_name]

        if args.overlay:
            overlay_dir = os.path.join("autodex", "tmp", obj_name, run_name)
            render_overlay(run_dir, obj_name, obj_mesh, overlay_dir)
            pbar.set_postfix_str(f"{obj_name}: overlay saved")
            continue

        result = process_single_run(obj_name, run_dir, obj_mesh, args.n_points)
        if result is None:
            pbar.set_postfix_str(f"{obj_name}: skipped")
            continue

        all_results.append(result)
        cov = result["coverage_with_robot"]
        cov_1 = cov[0]["mean_coverage"] if cov else 0
        cov_24 = cov[-1]["mean_coverage"] if cov else 0
        pbar.set_postfix_str(f"{obj_name} 1c={cov_1:.0%} 24c={cov_24:.0%}")

        # Incremental save
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\nTotal: {len(all_results)} runs processed")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
