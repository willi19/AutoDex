#!/usr/bin/env python3
"""Compare FoundPose-based first-frame init vs current PerceptionPipeline.

Reference: existing `pose_world.npy` saved by `src/execution/run_perception.py`
(SAM3 + DA3 + FPose register + cross-view IoU + Sil refine).

For each episode we run `FoundPoseInit.estimate_per_view` on the same
`images/` + `_pipeline_tmp/masks/` and report two selectors:

    fp_quality  — built-in PnP quality + inlier tiebreak
    fp_iou      — cross-view SAM mask IoU on the per-view candidate poses
                  (via autodex.perception.pose_select.select_best_pose_by_iou)

Errors are reported relative to the reference pose:

    trans_err_mm = ‖t_ref − t_test‖
    rot_err_deg  = arccos((trace(R_ref^T R_test) − 1) / 2)

Auto-selects N objects × M episodes, runs FoundPose, dumps a CSV.

Run inside the `gotrack` conda env:

    conda run -n gotrack python src/validation/perception/foundpose_init_compare.py \\
        --experiment-root ~/shared_data/AutoDex/experiment/selected_100/allegro \\
        --output-dir outputs/foundpose_init_compare/selected_100
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Make sure we're not silently picking up an unwanted gl backend during init.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

MESH_ROOT = Path.home() / "shared_data/AutoDex/object/paradex"


# ── data discovery ──

def _list_episodes(obj_dir: Path) -> List[Path]:
    eps = []
    for p in sorted(obj_dir.iterdir()):
        if not p.is_dir():
            continue
        if not (p / "pose_world.npy").exists():
            continue
        if not (p / "_pipeline_tmp" / "masks").exists():
            continue
        if not (p / "images").exists():
            continue
        if not (p / "cam_param" / "intrinsics.json").exists():
            continue
        eps.append(p)
    return eps


def _select_objects(
    experiment_root: Path, n_objects: int, n_episodes: int, seed: int = 0,
) -> List[Tuple[str, List[Path]]]:
    """Random sample n_objects from objects with >= n_episodes valid episodes,
    then random sample n_episodes per object."""
    import random
    rng = random.Random(seed)
    eligible: List[Tuple[str, List[Path]]] = []
    for obj_dir in sorted(experiment_root.iterdir()):
        if not obj_dir.is_dir():
            continue
        eps = _list_episodes(obj_dir)
        if len(eps) >= n_episodes:
            eligible.append((obj_dir.name, eps))
    rng.shuffle(eligible)
    chosen = eligible[:n_objects]
    out: List[Tuple[str, List[Path]]] = []
    for name, eps in chosen:
        sample = rng.sample(eps, n_episodes)
        sample.sort()
        out.append((name, sample))
    return out


def _resolve_mesh_path(obj_name: str) -> Path:
    candidates = [
        MESH_ROOT / obj_name / "raw_mesh" / f"{obj_name}.obj",
        MESH_ROOT / obj_name / "processed_data" / "mesh" / "raw.obj",
        MESH_ROOT / obj_name / "processed_data" / "mesh" / "simplified.obj",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"No mesh for {obj_name} under {MESH_ROOT}")


def _load_episode(ep_dir: Path) -> Dict[str, object]:
    images_dir = ep_dir / "images"
    masks_dir = ep_dir / "_pipeline_tmp" / "masks"
    cam_param_dir = ep_dir / "cam_param"

    with open(cam_param_dir / "intrinsics.json") as f:
        intr_raw = json.load(f)
    with open(cam_param_dir / "extrinsics.json") as f:
        extr_raw = json.load(f)

    serials = sorted(p.stem for p in images_dir.glob("*.png"))
    images_rgb: Dict[str, np.ndarray] = {}
    masks_bool: Dict[str, np.ndarray] = {}
    intrinsics: Dict[str, np.ndarray] = {}
    extrinsics: Dict[str, np.ndarray] = {}

    for s in serials:
        if s not in intr_raw or s not in extr_raw:
            continue
        bgr = cv2.imread(str(images_dir / f"{s}.png"))
        if bgr is None:
            continue
        images_rgb[s] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        m = cv2.imread(str(masks_dir / f"{s}.png"), cv2.IMREAD_GRAYSCALE)
        masks_bool[s] = (m > 127) if m is not None else np.zeros(bgr.shape[:2], dtype=bool)
        intrinsics[s] = np.asarray(intr_raw[s]["intrinsics_undistort"], dtype=np.float64)
        T = np.asarray(extr_raw[s], dtype=np.float64)
        if T.shape == (3, 4):
            T = np.vstack([T, [0, 0, 0, 1]])
        extrinsics[s] = T

    H, W = bgr.shape[:2]
    return {
        "images_rgb": images_rgb,
        "masks_bool": masks_bool,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "H": H, "W": W,
        "serials": list(images_rgb.keys()),
    }


# ── geometry ──

def _pose_errors(p_ref: np.ndarray, p_test: np.ndarray) -> Tuple[float, float]:
    t_err_m = float(np.linalg.norm(p_ref[:3, 3] - p_test[:3, 3]))
    R_ref, R_test = p_ref[:3, :3], p_test[:3, :3]
    cos = (np.trace(R_ref.T @ R_test) - 1.0) / 2.0
    cos = float(np.clip(cos, -1.0, 1.0))
    return t_err_m * 1000.0, float(np.degrees(np.arccos(cos)))


# ── main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=str, required=True,
                        help="e.g. ~/shared_data/AutoDex/experiment/selected_100/allegro")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--n-objects", type=int, default=20)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for object/episode sampling.")
    parser.add_argument("--assets-root", type=str,
                        default=str(REPO_ROOT / "outputs/foundpose_assets"))
    parser.add_argument("--force-onboard", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    from autodex.perception.foundpose_init import FoundPoseInit
    from autodex.perception.pose_select import select_best_pose_by_iou

    experiment_root = Path(args.experiment_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_root = Path(args.assets_root).expanduser().resolve()
    assets_root.mkdir(parents=True, exist_ok=True)

    obj_eps = _select_objects(experiment_root, args.n_objects, args.n_episodes, seed=args.seed)
    print(f"[compare] selected {len(obj_eps)} objects:")
    for obj, eps in obj_eps:
        print(f"  - {obj}: {len(eps)} episodes")

    # Silhouette optimizer is rebuilt per object (mesh changes).
    from autodex.perception.silhouette import SilhouetteOptimizer

    csv_path = output_dir / "results.csv"
    csv_header = [
        "object", "episode",
        "fp_iou_best_view", "fp_iou_mean",
        "trans_err_pre_mm", "rot_err_pre_deg",
        "trans_err_post_mm", "rot_err_post_deg",
        "sil_loss", "sil_sec",
        "fp_compute_sec", "fp_n_views_ok",
        "ref_perception_total_s", "ref_sam3_s", "ref_depth_s",
        "ref_fpose_s", "ref_select_s", "ref_sil_s",
        # New metrics for direct apples-to-apples comparison.
        "iou_pre_mean", "iou_pre_max",
        "iou_post_mean", "iou_post_max",
        "sil_loss_pre",
        "ref_iou_mean", "ref_iou_max",
        "ref_sil_loss",
    ]
    done: set = set()
    if csv_path.exists():
        with open(csv_path) as fr:
            for row in csv.DictReader(fr):
                done.add((row["object"], row["episode"]))
        print(f"[compare] resuming: {len(done)} (obj, ep) already in {csv_path.name}")
        fcsv = open(csv_path, "a", newline="")
        writer = csv.writer(fcsv)
    else:
        fcsv = open(csv_path, "w", newline="")
        writer = csv.writer(fcsv)
        writer.writerow(csv_header)
    try:

        for obj_name, episodes in obj_eps:
            # Resume: skip if all episodes for this object are already done.
            episodes = [ep for ep in episodes if (obj_name, ep.name) not in done]
            if not episodes:
                print(f"[{obj_name}] all {len(done)} eps already done, skipping")
                continue

            try:
                mesh_path = _resolve_mesh_path(obj_name)
            except FileNotFoundError as e:
                print(f"[skip {obj_name}] {e}")
                continue

            asset_dir = assets_root / obj_name
            ref_ep = episodes[0]
            ref_intr_json = ref_ep / "cam_param" / "intrinsics.json"
            with open(ref_intr_json) as f:
                ref_serials = sorted(json.load(f).keys())
            ref_camera_id = ref_serials[0]

            print(f"\n[{obj_name}] init (mesh={mesh_path.name})")
            try:
                fp_init = FoundPoseInit(
                    mesh_path=str(mesh_path),
                    assets_root=str(asset_dir),
                    obj_name=obj_name,
                    object_id=1,
                    device=args.device,
                    reference_intrinsics_json=str(ref_intr_json),
                    reference_camera_id=ref_camera_id,
                    force_onboard=args.force_onboard,
                )
            except Exception as exc:
                print(f"[skip {obj_name}] FoundPoseInit failed: {exc}")
                continue

            # SilhouetteOptimizer also builds mesh_tensors + glctx; reuse them
            # for cross-view IoU select to avoid double rendering setup.
            sil_optimizer = SilhouetteOptimizer(str(mesh_path), device="cuda")
            mesh_tensors = sil_optimizer.mesh_tensors
            glctx = sil_optimizer.glctx

            for ep in episodes:
                p_ref = np.load(ep / "pose_world.npy")
                rj_path = ep / "result.json"
                if rj_path.exists():
                    with open(rj_path) as f:
                        rj = json.load(f)
                    ref_t = rj.get("timing", {}).get("perception_detail", {}) or {}
                else:
                    ref_t = {}

                ep_data = _load_episode(ep)

                t0 = time.perf_counter()
                per_view = fp_init.estimate_per_view(
                    ep_data["images_rgb"], ep_data["masks_bool"],
                    ep_data["intrinsics"], ep_data["extrinsics"],
                )
                fp_compute_sec = time.perf_counter() - t0

                ok_views = {s: r for s, r in per_view.items() if r is not None}
                if not ok_views:
                    print(f"  [{ep.name}] all views failed")
                    continue

                # Cross-view IoU select on per-view candidate poses
                candidates = {s: r["pose_world"] for s, r in ok_views.items()}
                best_iou_serial, p_pre, mean_iou, _ = select_best_pose_by_iou(
                    candidates=candidates,
                    masks=ep_data["masks_bool"],
                    intrinsics=ep_data["intrinsics"],
                    extrinsics=ep_data["extrinsics"],
                    H=ep_data["H"], W=ep_data["W"],
                    glctx=glctx, mesh_tensors=mesh_tensors,
                )
                if p_pre is None:
                    print(f"  [{ep.name}] iou select failed")
                    continue
                t_err_pre, r_err_pre = _pose_errors(p_ref, p_pre)

                # Silhouette refinement on selected pose (current pipeline params).
                sil_views = []
                for s, m in ep_data["masks_bool"].items():
                    if int(m.sum()) < 100:
                        continue
                    sil_views.append({
                        "mask": (m.astype(np.uint8) * 255),
                        "K": ep_data["intrinsics"][s].astype(np.float32),
                        "extrinsic": ep_data["extrinsics"][s].astype(np.float64),
                    })
                t_sil0 = time.perf_counter()
                p_post, sil_loss = sil_optimizer.optimize(
                    p_pre, sil_views, iters=100, lr=0.002, antialias=True,
                )
                sil_sec = time.perf_counter() - t_sil0
                t_err_post, r_err_post = _pose_errors(p_ref, p_post)

                # Persist poses for future backfill (avoids re-running FoundPose+sil
                # just to add new metrics).
                pose_dir = output_dir / "poses" / obj_name
                pose_dir.mkdir(parents=True, exist_ok=True)
                np.savez(pose_dir / f"{ep.name}.npz", pre=p_pre, post=p_post)

                print(f"  [{ep.name}] iou_best={best_iou_serial} mean_iou={mean_iou:.3f} "
                      f"pre t={t_err_pre:.1f}mm r={r_err_pre:.2f}° -> "
                      f"post t={t_err_post:.1f}mm r={r_err_post:.2f}° "
                      f"(sil_loss={sil_loss:.4f}, {sil_sec:.1f}s) | "
                      f"fp={fp_compute_sec:.1f}s nv={len(ok_views)}/{len(per_view)}")

                writer.writerow([
                    obj_name, ep.name,
                    best_iou_serial, f"{mean_iou:.4f}",
                    f"{t_err_pre:.2f}", f"{r_err_pre:.3f}",
                    f"{t_err_post:.2f}", f"{r_err_post:.3f}",
                    f"{sil_loss:.6f}", f"{sil_sec:.2f}",
                    f"{fp_compute_sec:.2f}", len(ok_views),
                    f"{ref_t.get('total', 0):.2f}",
                    f"{ref_t.get('sam3', 0):.2f}",
                    f"{ref_t.get('depth', 0):.2f}",
                    f"{ref_t.get('fpose', 0):.2f}",
                    f"{ref_t.get('select', 0):.2f}",
                    f"{ref_t.get('sil', 0):.2f}",
                ])
                fcsv.flush()

        print(f"\n[done] wrote {csv_path}")
    finally:
        fcsv.close()


if __name__ == "__main__":
    main()
