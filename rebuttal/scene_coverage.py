"""Scene coverage comparison: selected_100 (ours) vs baseline_100.

For each object:
1. Generate 50 test scenes each for wall, shelf, cluttered (150 total)
2. For each (grasp_version, scene), run collision check + IK
3. Count surviving grasps per scene
4. Report objects ranked by coverage gap (ours - baseline)

Intermediate results are cached in rebuttal/cache/{obj}/{version}/
Test scenes saved in rebuttal/scenes/{obj}/

Usage:
    python rebuttal/scene_coverage.py
    python rebuttal/scene_coverage.py --obj attached_container  # single object
"""
import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from autodex.utils.path import obj_path, candidate_path, robot_configs_path, load_candidate
from autodex.utils.conversion import cart2se3
from autodex.planner.planner import GraspPlanner, _to_curobo_world
from autodex.planner.obstacles import (
    get_cluttered_obstacles,
    TABLE_CUBOID,
)

REBUTTAL_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_DIR = os.path.join(REBUTTAL_DIR, "scenes")
CACHE_DIR = os.path.join(REBUTTAL_DIR, "cache")
TABLE_SURFACE_Z = 0.037

VERSIONS = ["selected_100", "baseline_100"]
N_SCENES_PER_TYPE = 50
SEED = 123


# ── Scene generation (OBB-based, from src/scene_generation/generate_scene.py) ─

def _make_obj_pose(tabletop_pose, x_offset):
    """Place object: use tabletop rotation, set position to (x_offset, 0, z).

    tabletop_pose[2,3] is the object origin height above z=0 plane.
    We shift it up by TABLE_SURFACE_Z so it sits on the table.
    """
    pose = tabletop_pose.copy()
    pose[:3, 3] = [x_offset, 0.0, tabletop_pose[2, 3] + TABLE_SURFACE_Z]
    return pose


def _pose_to_7d(pose):
    """4x4 -> [x, y, z, qw, qx, qy, qz]."""
    from scipy.spatial.transform import Rotation
    t = pose[:3, 3].tolist()
    xyzw = Rotation.from_matrix(pose[:3, :3]).as_quat()
    return t + [float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2])]


def _obb_corners_world(obb_info, pose):
    """Compute 8 OBB corners in world frame, accounting for OBB center offset."""
    obb_tf = np.array(obb_info["obb_transform"])
    R_obb = obb_tf[:3, :3]
    t_obb = obb_tf[:3, 3]
    half_ext = np.array(obb_info["obb"]) / 2.0

    R_w = pose[:3, :3]
    t_w = pose[:3, 3]
    axes_w = R_w @ R_obb
    center_w = R_w @ t_obb + t_w

    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                corners.append(axes_w @ (np.array([sx, sy, sz]) * half_ext) + center_w)
    return np.array(corners)


def _make_scene_cfg(obj_name, obj_pose_4x4, cuboids):
    """Build scene_cfg dict."""
    obj_mesh_path = os.path.join(obj_path, obj_name, "processed_data", "mesh", "simplified.obj")
    obj_urdf_path = os.path.join(obj_path, obj_name, "processed_data", "urdf", "coacd.urdf")
    return {
        "mesh": {
            "target": {
                "pose": _pose_to_7d(obj_pose_4x4),
                "file_path": obj_mesh_path,
                "urdf_path": obj_urdf_path,
                "scale": [1.0, 1.0, 1.0],
            }
        },
        "cuboid": cuboids,
    }


def _gen_wall_scene(obj_name, obj_pose, obb_info, wall_angle_deg, wall_gap):
    """OBB-aware wall: wall is placed at gap distance from OBB boundary along wall_angle direction."""
    corners = _obb_corners_world(obb_info, obj_pose)
    table_z = TABLE_CUBOID["pose"][2] + TABLE_CUBOID["dims"][2] / 2
    max_z = corners[:, 2].max()

    angle_rad = np.radians(wall_angle_deg)
    # Wall normal direction (pointing toward object)
    nx = -np.sin(angle_rad)
    ny = np.cos(angle_rad)

    # Project corners onto wall normal to find max extent
    projections = corners[:, 0] * nx + corners[:, 1] * ny
    max_proj = projections.max()

    wall_thickness = 0.02
    wall_dist = max_proj + wall_gap + wall_thickness / 2
    wall_height = max(max_z - table_z + 0.05, 0.15)

    wall_cx = obj_pose[0, 3] + nx * (wall_dist - max_proj + projections.max() - obj_pose[0, 3] * nx - obj_pose[1, 3] * ny)
    # Simpler: place wall at max_proj + gap along normal direction from origin
    wall_center = np.array([
        nx * wall_dist,
        ny * wall_dist,
        table_z + wall_height / 2,
    ])

    from autodex.planner.obstacles import _quat_from_euler
    wall_quat = _quat_from_euler(yaw=angle_rad)

    cuboids = {
        "table": TABLE_CUBOID,
        "wall": {
            "dims": [0.5, wall_thickness, wall_height],
            "pose": wall_center.tolist() + wall_quat,
        },
    }
    return _make_scene_cfg(obj_name, obj_pose, cuboids)


def _gen_shelf_scene(obj_name, obj_pose, obb_info, shelf_gap, back, sides, top):
    """OBB-aware shelf: walls placed at gap from OBB boundary."""
    corners = _obb_corners_world(obb_info, obj_pose)
    table_z = TABLE_CUBOID["pose"][2] + TABLE_CUBOID["dims"][2] / 2

    min_x, max_x = corners[:, 0].min(), corners[:, 0].max()
    min_y, max_y = corners[:, 1].min(), corners[:, 1].max()
    max_z = corners[:, 2].max()

    THICK = 0.02
    up_z = max_z + shelf_gap + THICK / 2
    wall_height = up_z - THICK / 2 - table_z
    if wall_height < 0.02:
        wall_height = 0.1
    wall_center_z = table_z + wall_height / 2

    y_min_wall = min_y - shelf_gap - THICK
    y_max_wall = max_y + shelf_gap
    wall_len_y = y_max_wall - y_min_wall

    cuboids = {"table": TABLE_CUBOID}

    if back:
        back_x_len = (max_x - min_x) + 2 * shelf_gap + 2 * THICK
        cuboids["shelf_back"] = {
            "dims": [back_x_len, THICK, wall_height],
            "pose": [(max_x + min_x) / 2, y_min_wall + THICK / 2, wall_center_z, 1, 0, 0, 0],
        }
    if sides:
        cuboids["shelf_left"] = {
            "dims": [THICK, wall_len_y, wall_height],
            "pose": [min_x - shelf_gap - THICK / 2, (y_min_wall + y_max_wall) / 2, wall_center_z, 1, 0, 0, 0],
        }
        cuboids["shelf_right"] = {
            "dims": [THICK, wall_len_y, wall_height],
            "pose": [max_x + shelf_gap + THICK / 2, (y_min_wall + y_max_wall) / 2, wall_center_z, 1, 0, 0, 0],
        }
    if top:
        ceil_x_len = (max_x - min_x) + 2 * shelf_gap + 2 * THICK
        cuboids["shelf_top"] = {
            "dims": [ceil_x_len, wall_len_y, THICK],
            "pose": [(max_x + min_x) / 2, (y_min_wall + y_max_wall) / 2, up_z, 1, 0, 0, 0],
        }

    return _make_scene_cfg(obj_name, obj_pose, cuboids)


def _load_obb_info(obj_name):
    obb_path = os.path.join(obj_path, obj_name, "processed_data", "info", "simplified.json")
    if os.path.exists(obb_path):
        with open(obb_path) as f:
            return json.load(f)
    return None


def generate_scenes(obj_name, seed=SEED):
    """Generate 50 wall + 50 shelf + 50 cluttered scenes."""
    rng = np.random.RandomState(seed)

    tabletop_dir = os.path.join(obj_path, obj_name, "processed_data", "info", "tabletop")
    if not os.path.isdir(tabletop_dir):
        print(f"  WARNING: no tabletop dir for {obj_name}")
        return []
    tabletop_files = sorted(os.listdir(tabletop_dir))
    tabletop_poses = [np.load(os.path.join(tabletop_dir, f)) for f in tabletop_files]
    if not tabletop_poses:
        return []

    obb_info = _load_obb_info(obj_name)
    if obb_info is None:
        print(f"  WARNING: no OBB info for {obj_name}")
        return []

    scenes = []

    def _pick_tabletop():
        ti = rng.randint(len(tabletop_poses))
        return ti, tabletop_poses[ti]

    # Wall: OBB-aware placement
    for i in range(N_SCENES_PER_TYPE):
        ti, tp = _pick_tabletop()
        x_off = rng.uniform(0.3, 0.5)
        obj_pose = _make_obj_pose(tp, x_off)
        wall_angle = rng.uniform(0, 360)
        wall_gap = rng.choice([0.02, 0.04, 0.06])
        cfg = _gen_wall_scene(obj_name, obj_pose, obb_info, wall_angle, wall_gap)
        cfg["meta"] = {"type": "wall", "idx": i, "x_offset": x_off,
                        "wall_angle": float(wall_angle), "wall_gap": float(wall_gap),
                        "tabletop_file": tabletop_files[ti]}
        scenes.append(("wall", cfg))

    # Shelf: OBB-aware placement
    for i in range(N_SCENES_PER_TYPE):
        ti, tp = _pick_tabletop()
        x_off = rng.uniform(0.3, 0.5)
        obj_pose = _make_obj_pose(tp, x_off)
        shelf_gap = rng.choice([0.02, 0.04, 0.06])
        back = bool(rng.choice([True, True, False]))
        sides = bool(rng.choice([True, True, False]))
        top = bool(rng.choice([True, True, False]))
        if not (back or sides or top):
            back = True
        cfg = _gen_shelf_scene(obj_name, obj_pose, obb_info, shelf_gap, back, sides, top)
        cfg["meta"] = {"type": "shelf", "idx": i, "x_offset": x_off,
                        "shelf_gap": float(shelf_gap),
                        "back": back, "sides": sides, "top": top}
        scenes.append(("shelf", cfg))

    # Cluttered: uses obj center, OK as-is
    for i in range(N_SCENES_PER_TYPE):
        ti, tp = _pick_tabletop()
        x_off = rng.uniform(0.3, 0.5)
        obj_pose = _make_obj_pose(tp, x_off)
        n_obs = rng.randint(2, 6)
        clutter_seed = int(rng.randint(0, 100000))
        cuboids = get_cluttered_obstacles(obj_pose, n_obstacles=n_obs, seed=clutter_seed)
        cfg = _make_scene_cfg(obj_name, obj_pose, cuboids)
        cfg["meta"] = {"type": "cluttered", "idx": i, "x_offset": x_off,
                        "n_obstacles": n_obs, "clutter_seed": clutter_seed}
        scenes.append(("cluttered", cfg))

    return scenes


def save_scenes(obj_name, scenes):
    """Save scenes to rebuttal/scenes/{obj}/{type}/{idx}.json."""
    for scene_type, cfg in scenes:
        idx = cfg["meta"]["idx"]
        out_dir = os.path.join(SCENE_DIR, obj_name, scene_type)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{idx:03d}.json"), "w") as f:
            json.dump(cfg, f, indent=2)


def load_scenes(obj_name):
    """Load cached scenes from disk."""
    scenes = []
    scene_dir = os.path.join(SCENE_DIR, obj_name)
    if not os.path.isdir(scene_dir):
        return []
    for scene_type in sorted(os.listdir(scene_dir)):
        type_dir = os.path.join(scene_dir, scene_type)
        if not os.path.isdir(type_dir):
            continue
        for fname in sorted(os.listdir(type_dir)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(type_dir, fname)) as f:
                cfg = json.load(f)
            scenes.append((scene_type, cfg))
    return scenes


# ── Coverage check ────────────────────────────────────────────────────────────

def load_grasp_data_direct(obj_name, version):
    """Load wrist_se3 (in object frame) and pregrasp from candidate dir."""
    root = os.path.join(candidate_path, version, obj_name)
    if not os.path.isdir(root):
        return [], np.array([]), np.array([])

    wrist_list, pregrasp_list, info_list = [], [], []
    for scene_type in sorted(os.listdir(root)):
        st_dir = os.path.join(root, scene_type)
        if not os.path.isdir(st_dir):
            continue
        for scene_id in sorted(os.listdir(st_dir)):
            sid_dir = os.path.join(st_dir, scene_id)
            if not os.path.isdir(sid_dir):
                continue
            for grasp_name in sorted(os.listdir(sid_dir)):
                gdir = os.path.join(sid_dir, grasp_name)
                if not os.path.isdir(gdir):
                    continue
                wf = os.path.join(gdir, "wrist_se3.npy")
                pf = os.path.join(gdir, "pregrasp_pose.npy")
                if not (os.path.exists(wf) and os.path.exists(pf)):
                    continue
                wrist_list.append(np.load(wf))
                pregrasp_list.append(np.load(pf))
                info_list.append((scene_type, scene_id, grasp_name))

    if not wrist_list:
        return [], np.array([]), np.array([])
    return info_list, np.array(wrist_list), np.array(pregrasp_list)


def check_coverage(planner, obj_name, scenes, version):
    """For each scene, check how many grasps survive collision + IK.

    Returns: list of int (surviving grasp count per scene).
    Caches per (obj, version, scene_type, scene_idx).
    """
    # Load grasps
    info_list, wrist_se3_obj, pregrasp_arr = load_grasp_data_direct(obj_name, version)
    n_grasps = len(info_list)
    if n_grasps == 0:
        return [0] * len(scenes)

    cache_obj_dir = os.path.join(CACHE_DIR, obj_name, version)
    os.makedirs(cache_obj_dir, exist_ok=True)

    results = []
    for scene_type, scene_cfg in tqdm(scenes, desc=f"  {version}", leave=False):
        idx = scene_cfg["meta"]["idx"]
        cache_file = os.path.join(cache_obj_dir, f"{scene_type}_{idx:03d}.json")

        # Check cache
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                cached = json.load(f)
            results.append(cached["n_surviving"])
            continue

        # Transform wrist to world frame
        obj_pose = cart2se3(scene_cfg["mesh"]["target"]["pose"])
        wrist_world = np.einsum("ij,ajk->aik", obj_pose, wrist_se3_obj)

        # Build world config (no target mesh for collision check)
        world_cfg = _to_curobo_world(scene_cfg)
        world_cfg["mesh"] = {}

        # Collision check
        collided = planner._check_collision(world_cfg, wrist_world, pregrasp_arr)
        valid_mask = ~collided
        valid_idx = np.where(valid_mask)[0]

        # IK check on valid ones
        if len(valid_idx) > 0:
            if planner._ik_solver is None:
                planner._init_ik_solver(world_cfg)
            else:
                from curobo.geom.types import WorldConfig
                planner._ik_solver.update_world(WorldConfig.from_dict(world_cfg))

            import torch
            from autodex.utils.robot_config import INIT_STATE
            from autodex.planner.planner import _to_curobo_pose

            ik_success = np.zeros(len(valid_idx), dtype=bool)
            BATCH = planner.BATCH_SIZE
            for cs in range(0, len(valid_idx), BATCH):
                chunk = valid_idx[cs:cs + BATCH]
                poses = wrist_world[chunk]
                B = len(poses)
                if B < BATCH:
                    poses = np.concatenate([poses, np.tile(poses[:1], (BATCH - B, 1, 1))], axis=0)

                goal = _to_curobo_pose(poses, planner._tensor_args.device)
                B_padded = poses.shape[0]
                retract = torch.tensor(
                    INIT_STATE, dtype=torch.float32, device=planner._tensor_args.device
                ).unsqueeze(0).repeat(B_padded, 1)
                result = planner._ik_solver.solve_batch(goal, retract_config=retract)
                succ = result.success.cpu().numpy()[:B].flatten()
                ik_success[cs:cs + B] = succ

            n_surviving = int(ik_success.sum())
        else:
            n_surviving = 0

        # Save cache
        cache_data = {
            "obj": obj_name,
            "version": version,
            "scene_type": scene_type,
            "scene_idx": idx,
            "n_grasps": n_grasps,
            "n_after_collision": int(valid_mask.sum()),
            "n_surviving": n_surviving,
        }
        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)

        results.append(n_surviving)

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, default=None)
    args = parser.parse_args()

    # Find objects with both versions
    baseline_dir = os.path.join(candidate_path, "baseline_100")
    selected_dir = os.path.join(candidate_path, "selected_100")
    if args.obj:
        obj_list = [args.obj]
    else:
        b_objs = set(os.listdir(baseline_dir)) if os.path.isdir(baseline_dir) else set()
        s_objs = set(os.listdir(selected_dir)) if os.path.isdir(selected_dir) else set()
        obj_list = sorted(b_objs & s_objs)

    print(f"Objects to process: {len(obj_list)}")

    # Init planner (one-time warmup)
    hand_cfg = os.path.join(robot_configs_path, "allegro_floating.yml")
    planner = GraspPlanner(hand_cfg_path=hand_cfg)

    all_results = {}

    for obj_name in tqdm(obj_list, desc="Objects"):
        print(f"\n{'='*60}")
        print(f"Object: {obj_name}")
        print(f"{'='*60}")

        # 1. Generate or load scenes
        scenes = load_scenes(obj_name)
        if not scenes:
            print(f"  Generating {N_SCENES_PER_TYPE * 3} test scenes...")
            scenes = generate_scenes(obj_name)
            if not scenes:
                print(f"  No scenes generated, skipping")
                continue
            save_scenes(obj_name, scenes)
            print(f"  Saved {len(scenes)} scenes to {SCENE_DIR}/{obj_name}/")
        else:
            print(f"  Loaded {len(scenes)} cached scenes")

        # 2. Check coverage for each version
        obj_results = {}
        for version in VERSIONS:
            print(f"\n  --- {version} ---")
            surviving = check_coverage(planner, obj_name, scenes, version)
            obj_results[version] = surviving

            # Per-type summary
            for stype in ["wall", "shelf", "cluttered"]:
                type_vals = [s for (st, _), s in zip(scenes, surviving) if st == stype]
                if type_vals:
                    coverage = sum(1 for v in type_vals if v > 0) / len(type_vals) * 100
                    avg_surv = np.mean(type_vals)
                    print(f"    {stype:10s}: coverage={coverage:5.1f}%  avg_surviving={avg_surv:.1f}")

        # 3. Compute gap
        sel = obj_results.get("selected_100", [])
        base = obj_results.get("baseline_100", [])
        if sel and base:
            sel_coverage = sum(1 for v in sel if v > 0) / len(sel) * 100
            base_coverage = sum(1 for v in base if v > 0) / len(base) * 100
            gap = sel_coverage - base_coverage
            all_results[obj_name] = {
                "selected_100_coverage": sel_coverage,
                "baseline_100_coverage": base_coverage,
                "gap": gap,
                "selected_100_per_type": {},
                "baseline_100_per_type": {},
            }
            for stype in ["wall", "shelf", "cluttered"]:
                for ver_key, ver_vals in [("selected_100", sel), ("baseline_100", base)]:
                    type_vals = [s for (st, _), s in zip(scenes, ver_vals) if st == stype]
                    cov = sum(1 for v in type_vals if v > 0) / len(type_vals) * 100 if type_vals else 0
                    all_results[obj_name][f"{ver_key}_per_type"][stype] = cov

            print(f"\n  >> Coverage: selected_100={sel_coverage:.1f}%  baseline_100={base_coverage:.1f}%  gap={gap:+.1f}%")

    # ── Final ranking ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RANKING BY COVERAGE GAP (selected_100 - baseline_100)")
    print(f"{'='*60}")

    ranked = sorted(all_results.items(), key=lambda x: x[1]["gap"], reverse=True)
    print(f"\n{'Rank':<5} {'Object':<30} {'Ours':>8} {'Base':>8} {'Gap':>8}")
    print("-" * 62)
    for i, (obj, res) in enumerate(ranked[:10], 1):
        print(f"{i:<5} {obj:<30} {res['selected_100_coverage']:>7.1f}% {res['baseline_100_coverage']:>7.1f}% {res['gap']:>+7.1f}%")

    # Per-type breakdown for top 10
    print(f"\nPer-type breakdown (top 10):")
    print(f"{'Object':<25} {'Wall Ours':>9} {'Wall Base':>10} {'Shelf Ours':>11} {'Shelf Base':>11} {'Clut Ours':>10} {'Clut Base':>10}")
    print("-" * 90)
    for obj, res in ranked[:10]:
        sp = res["selected_100_per_type"]
        bp = res["baseline_100_per_type"]
        print(f"{obj:<25} {sp.get('wall',0):>8.1f}% {bp.get('wall',0):>9.1f}% "
              f"{sp.get('shelf',0):>10.1f}% {bp.get('shelf',0):>10.1f}% "
              f"{sp.get('cluttered',0):>9.1f}% {bp.get('cluttered',0):>9.1f}%")

    # Save summary
    summary_path = os.path.join(REBUTTAL_DIR, "coverage_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "ranking": [(obj, res) for obj, res in ranked],
            "n_scenes_per_type": N_SCENES_PER_TYPE,
            "seed": SEED,
        }, f, indent=2)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
