"""Extract per-capture summaries from robothome_demo_inspire_left captures.

For every ``<base_dir>/<obj>/<idx>/`` capture, compute and save:

- ``occluded_centroid`` (3,)       : centroid of charuco corners not detected
                                      by any camera, in robot-base frame —
                                      approximates the object's location
                                      since the object occludes them.
- ``plane_normal``     (3,)        : unit normal of the best-fit charuco
                                      plane (smallest-singular-vector of the
                                      triangulated corner cloud) in robot
                                      frame. Sign flipped so it points
                                      "up" (positive z component).
- ``plane_centroid``   (3,)        : centroid of triangulated corners in
                                      robot frame (anchor for plane normal).
- ``wrist_se3_traj``   (T, 4, 4)   : URDF "wrist" link pose in robot-base
                                      frame, one per arm logging frame.
- ``hand_qpos``        (T, 6)      : finger driver joint angles (radians) in
                                      URDF/RobotModule order
                                      [thumb_1, thumb_2, index_1, middle_1,
                                      ring_1, little_1] resampled to arm
                                      timestamps. Use these directly to drive
                                      the inspire-left URDF.
- ``arm_t``            (T,)        : timestamps for ``wrist_se3_traj``.

Output: ``src/validation/robothome/extracted/<obj>/<idx>.npz``.

Usage:
    /home/mingi/miniconda3/envs/foundationpose/bin/python \
        src/validation/robothome/extract_traj.py \
        [--base-dir /home/mingi/paradex1/capture/robothome_demo_inspire_left] \
        [--force]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import yourdfpy

PARADEX_ROOT = Path("/home/mingi/paradex")
if str(PARADEX_ROOT) not in sys.path:
    sys.path.insert(0, str(PARADEX_ROOT))

from paradex.calibration.utils import load_c2r, load_camparam  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_BASE = Path("/home/mingi/paradex1/capture/robothome_demo_inspire_left")
OUT_DIR = HERE / "extracted"
HAND_URDF_ORDER = [
    "left_thumb_1_joint", "left_thumb_2_joint",
    "left_index_1_joint", "left_middle_1_joint",
    "left_ring_1_joint", "left_little_1_joint",
]
RAW_TO_URDF = [5, 4, 3, 2, 1, 0]


def _import_visualize_capture():
    spec = importlib.util.spec_from_file_location("vc", HERE / "visualize_capture.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fit_plane_normal(pts_robot: np.ndarray):
    """Best-fit plane through ``pts_robot``: return (centroid, unit_normal).
    Normal sign is forced to have positive z (table-up convention)."""
    c = pts_robot.mean(axis=0)
    _, _, vt = np.linalg.svd(pts_robot - c, full_matrices=False)
    n = vt[-1]
    n = n / max(np.linalg.norm(n), 1e-12)
    if n[2] < 0:
        n = -n
    return c, n


def wrist_se3_trajectory(urdf, arm_pos: np.ndarray, arm_joints, hand_drivers,
                         hand_at_arm: np.ndarray, link_name: str = "wrist") -> np.ndarray:
    """FK the URDF at every arm frame; return ``link_name`` pose in robot base."""
    actuated = list(urdf.actuated_joint_names)
    arm_idx = [actuated.index(j) for j in arm_joints]
    hand_idx = [actuated.index(j) for j in hand_drivers]
    T = len(arm_pos)
    out = np.zeros((T, 4, 4), dtype=np.float64)
    cfg = np.zeros(len(actuated))
    for f in range(T):
        cfg[arm_idx] = arm_pos[f]
        cfg[hand_idx] = hand_at_arm[f]
        urdf.update_cfg(cfg)
        out[f] = urdf.get_transform(link_name, urdf.base_link)
    return out


def extract_one(cap_dir: Path, vc, urdf) -> dict:
    """Run extraction for a single capture dir. Reuses charuco_cache.npz if
    present; otherwise computes detect/triangulate fresh."""
    intrinsic, extrinsic = load_camparam(str(cap_dir))
    C2R = load_c2r(str(cap_dir))

    cache_path = cap_dir / "charuco_cache.npz"
    if cache_path.exists():
        d = np.load(cache_path)
        pts_world = d["pts_world"]
        occ = d["occluded_centroid"] if "occluded_centroid" in d.files else None
        if occ is not None and occ.size == 0:
            occ = None
    else:
        # Triangulate from scratch.
        vc._patch_charuco_system("robothome")
        from paradex.image.aruco import merge_charuco_detection
        img_dict = vc.video_frame_image_dict(cap_dir, intrinsic, extrinsic, frame_idx=1)
        img_und = img_dict.undistort()
        img_und.set_camparam(intrinsic, extrinsic)
        ch3d = img_und.triangulate_charuco()
        merged = merge_charuco_detection(ch3d)
        pts_world = np.asarray(merged["checkerCorner"]).reshape(-1, 3)
        occ = vc.undetected_corner_centroid_robot(
            cap_dir, intrinsic, extrinsic, C2R,
            video_frame_idx=1, system_name="robothome",
        )
        np.savez(
            cache_path,
            pts_world=pts_world,
            occluded_centroid=(occ if occ is not None else np.zeros((0,))),
        )

    pts_robot = vc.world_to_robot(pts_world, C2R)
    if len(pts_robot) < 3 or occ is None:
        raise RuntimeError(f"insufficient charuco data in {cap_dir}")
    plane_centroid, plane_normal = fit_plane_normal(pts_robot)

    # Arm + hand qpos (with timestamp-aligned hand).
    arm_pos, arm_t, hand_pos, hand_t = vc.load_arm_hand(cap_dir)
    hand_rad = vc.inspire_int_to_rad(hand_pos)
    t0 = min(arm_t[0], hand_t[0])
    hand_at_arm_raw = np.empty((len(arm_t), 6))
    for j in range(6):
        hand_at_arm_raw[:, j] = np.interp(arm_t - t0, hand_t - t0, hand_rad[:, j])
    hand_at_arm = hand_at_arm_raw[:, RAW_TO_URDF]

    wrist_traj = wrist_se3_trajectory(
        urdf, arm_pos, vc.ARM_JOINTS, HAND_URDF_ORDER, hand_at_arm,
        link_name="wrist",
    )

    return {
        "occluded_centroid": np.asarray(occ, dtype=np.float64),
        "plane_normal": plane_normal.astype(np.float64),
        "plane_centroid": plane_centroid.astype(np.float64),
        "wrist_se3_traj": wrist_traj,
        "hand_qpos": hand_at_arm.astype(np.float64),
        "hand_qpos_order": np.array("urdf"),
        "arm_t": arm_t.astype(np.float64),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--force", action="store_true",
                    help="Recompute even if the output .npz already exists.")
    args = ap.parse_args()

    base = args.base_dir.resolve()
    if not base.exists():
        raise FileNotFoundError(base)

    vc = _import_visualize_capture()
    urdf = yourdfpy.URDF.load(
        str(vc.URDF_PATH), mesh_dir=str(vc.URDF_MESH_DIR),
        build_collision_scene_graph=False, load_collision_meshes=False,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_ok = n_skip = n_fail = 0
    for obj_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for idx_dir in sorted(p for p in obj_dir.iterdir() if p.is_dir()):
            out = OUT_DIR / obj_dir.name / f"{idx_dir.name}.npz"
            if out.exists() and not args.force:
                n_skip += 1
                continue
            try:
                rec = extract_one(idx_dir, vc, urdf)
            except Exception as e:
                print(f"  [fail] {obj_dir.name}/{idx_dir.name}: {e}")
                n_fail += 1
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out, **rec)
            n_ok += 1
            print(f"  [ok]   {obj_dir.name}/{idx_dir.name}: "
                  f"occluded={rec['occluded_centroid']}  "
                  f"plane_n={rec['plane_normal']}  "
                  f"T={len(rec['wrist_se3_traj'])}")
    print(f"\nDone. ok={n_ok} skip={n_skip} fail={n_fail}  out={OUT_DIR}")


if __name__ == "__main__":
    main()
