#!/usr/bin/env python3
"""Backfill IoU + sil_loss columns on an existing foundpose_init_compare CSV.

Adds (or recomputes) these columns for every row in results.csv:

  iou_pre_mean / iou_pre_max     — FoundPose+IoU select pose, cross-view IoU
                                   (mean and best-view max, vs SAM masks)
  iou_post_mean / iou_post_max   — same after sil refine
  sil_loss_pre                   — silhouette MSE of FoundPose pose (before refine)
  ref_iou_mean / ref_iou_max     — reference pose's cross-view IoU
  ref_sil_loss                   — silhouette MSE of reference pose

For each existing row we re-run FoundPose + sil refine on the episode (FoundPose
result wasn't saved, so we have to recompute). With the per-object FoundPose
model + sil optimizer cached this is ~10-15 s/row; 90 rows ≈ 20 min.

Run inside the `gotrack` conda env. Make sure no other big GPU job is running.

Usage:
    python src/validation/perception/foundpose_init_compare_backfill.py \\
        --csv outputs/foundpose_init_compare/selected_100/results.csv \\
        --experiment-root ~/shared_data/AutoDex/experiment/selected_100/allegro \\
        --assets-root outputs/foundpose_assets
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

MESH_ROOT = Path("/home/mingi/shared_data/AutoDex/object/paradex")

NEW_COLUMNS = [
    "iou_pre_mean", "iou_pre_max",
    "iou_post_mean", "iou_post_max",
    "sil_loss_pre",
    "ref_iou_mean", "ref_iou_max",
    "ref_sil_loss",
]


def _resolve_mesh(obj: str) -> Path:
    for sub in [
        MESH_ROOT / obj / "raw_mesh" / f"{obj}.obj",
        MESH_ROOT / obj / "processed_data" / "mesh" / "raw.obj",
        MESH_ROOT / obj / "processed_data" / "mesh" / "simplified.obj",
    ]:
        if sub.exists():
            return sub
    raise FileNotFoundError(f"No mesh for {obj}")


def _load_episode(ep_dir: Path) -> Dict[str, Any]:
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
    last_shape = None
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
        last_shape = bgr.shape
    H, W = last_shape[:2] if last_shape else (0, 0)
    return {
        "images_rgb": images_rgb, "masks_bool": masks_bool,
        "intrinsics": intrinsics, "extrinsics": extrinsics,
        "H": H, "W": W,
    }


def _iou_mean_max(
    pose_world, masks, intrinsics, extrinsics, H, W, glctx, mesh_tensors
) -> Tuple[float, float]:
    from autodex.perception.pose_select import compute_cross_view_iou
    mean, per_view = compute_cross_view_iou(
        pose_world=pose_world, masks=masks,
        intrinsics=intrinsics, extrinsics=extrinsics,
        H=H, W=W, glctx=glctx, mesh_tensors=mesh_tensors,
    )
    if per_view:
        return float(mean), float(max(per_view.values()))
    return 0.0, 0.0


def _build_sil_views(masks_bool, intrinsics, extrinsics):
    out = []
    for s, m in masks_bool.items():
        if int(m.sum()) < 100:
            continue
        out.append({
            "mask": (m.astype(np.uint8) * 255),
            "K": intrinsics[s].astype(np.float32),
            "extrinsic": extrinsics[s].astype(np.float64),
        })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--experiment-root", type=str, required=True)
    parser.add_argument("--assets-root", type=str,
                        default=str(REPO_ROOT / "outputs/foundpose_assets"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--poses-dir", type=str, default=None,
                        help="Sidecar dir with {obj}/{ep}.npz holding 'pre' and 'post' "
                             "pose matrices. Defaults to {csv-dir}/poses/. If a row's "
                             ".npz exists we skip FoundPose+sil and only measure "
                             "IoU/sil_loss (10× faster).")
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    out_path = csv_path.with_name(csv_path.stem + "_backfilled.csv")
    experiment_root = Path(args.experiment_root).expanduser().resolve()
    assets_root = Path(args.assets_root).expanduser().resolve()
    poses_root = (Path(args.poses_dir).resolve()
                  if args.poses_dir else csv_path.parent / "poses")

    rows = list(csv.DictReader(open(csv_path)))
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"[backfill] reading {len(rows)} rows from {csv_path.name}")

    fieldnames = list(rows[0].keys())
    for col in NEW_COLUMNS:
        if col not in fieldnames:
            fieldnames.append(col)

    from autodex.perception.foundpose_init import FoundPoseInit
    from autodex.perception.silhouette import SilhouetteOptimizer
    from autodex.perception.pose_select import select_best_pose_by_iou

    rows_by_obj: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        rows_by_obj[r["object"]].append(r)

    out_rows: List[Dict[str, Any]] = []
    t_start = time.perf_counter()
    n_done = 0

    for obj, obj_rows in rows_by_obj.items():
        try:
            mesh_path = _resolve_mesh(obj)
        except FileNotFoundError as e:
            print(f"[skip {obj}] {e}")
            for r in obj_rows:
                for col in NEW_COLUMNS:
                    r.setdefault(col, "")
                out_rows.append(r)
            continue

        # Need a reference intrinsics for FoundPose Stage A (already onboarded).
        ref_ep_name = obj_rows[0]["episode"]
        ref_ep = experiment_root / obj / ref_ep_name
        ref_intr_json = ref_ep / "cam_param" / "intrinsics.json"
        with open(ref_intr_json) as f:
            ref_serials = sorted(json.load(f).keys())

        print(f"\n[{obj}] loading FoundPose + Sil ({len(obj_rows)} rows) ...")
        try:
            fp_init = FoundPoseInit(
                mesh_path=str(mesh_path),
                assets_root=str(assets_root / obj),
                obj_name=obj, object_id=1,
                device=args.device,
                reference_intrinsics_json=str(ref_intr_json),
                reference_camera_id=ref_serials[0],
            )
        except Exception as exc:
            print(f"[skip {obj}] FoundPoseInit: {exc}")
            for r in obj_rows:
                for col in NEW_COLUMNS:
                    r.setdefault(col, "")
                out_rows.append(r)
            continue
        sil_optimizer = SilhouetteOptimizer(str(mesh_path), device="cuda")
        mesh_tensors = sil_optimizer.mesh_tensors
        glctx = sil_optimizer.glctx

        for r in obj_rows:
            ep_name = r["episode"]
            ep_dir = experiment_root / obj / ep_name
            t_row = time.perf_counter()
            try:
                p_ref = np.load(ep_dir / "pose_world.npy")
                ep_data = _load_episode(ep_dir)
                if not ep_data["images_rgb"]:
                    raise RuntimeError("no images")

                # Fast path: poses dumped by foundpose_init_compare.py.
                pose_npz = poses_root / obj / f"{ep_name}.npz"
                cached = None
                if pose_npz.exists():
                    try:
                        cached = np.load(pose_npz)
                        p_pre_cached = np.asarray(cached["pre"], dtype=np.float64)
                        p_post_cached = np.asarray(cached["post"], dtype=np.float64)
                    except Exception as exc:
                        print(f"  [{ep_name}] cache read failed ({exc}); falling back")
                        cached = None

                if cached is not None:
                    p_pre = p_pre_cached
                else:
                    per_view = fp_init.estimate_per_view(
                        ep_data["images_rgb"], ep_data["masks_bool"],
                        ep_data["intrinsics"], ep_data["extrinsics"],
                    )
                    ok = {s: v for s, v in per_view.items() if v is not None}
                    candidates = {s: v["pose_world"] for s, v in ok.items()}
                    _, p_pre, _, _ = select_best_pose_by_iou(
                        candidates=candidates, masks=ep_data["masks_bool"],
                        intrinsics=ep_data["intrinsics"], extrinsics=ep_data["extrinsics"],
                        H=ep_data["H"], W=ep_data["W"],
                        glctx=glctx, mesh_tensors=mesh_tensors,
                    )
                    if p_pre is None:
                        raise RuntimeError("iou select empty")

                # IoU pre (mean + max)
                iou_pre_mean, iou_pre_max = _iou_mean_max(
                    p_pre, ep_data["masks_bool"], ep_data["intrinsics"],
                    ep_data["extrinsics"], ep_data["H"], ep_data["W"], glctx, mesh_tensors,
                )

                # Sil loss pre (1 iter, lr=0)
                sil_views = _build_sil_views(
                    ep_data["masks_bool"], ep_data["intrinsics"], ep_data["extrinsics"],
                )
                _, sil_loss_pre = sil_optimizer.optimize(
                    p_pre, sil_views, iters=1, lr=0.0, antialias=True,
                )

                # Sil refine → p_post (skip if cached).
                if cached is not None:
                    p_post = p_post_cached
                else:
                    p_post, _ = sil_optimizer.optimize(
                        p_pre, sil_views, iters=100, lr=0.002, antialias=True,
                    )
                iou_post_mean, iou_post_max = _iou_mean_max(
                    p_post, ep_data["masks_bool"], ep_data["intrinsics"],
                    ep_data["extrinsics"], ep_data["H"], ep_data["W"], glctx, mesh_tensors,
                )

                # Reference pose metrics
                ref_iou_mean, ref_iou_max = _iou_mean_max(
                    p_ref, ep_data["masks_bool"], ep_data["intrinsics"],
                    ep_data["extrinsics"], ep_data["H"], ep_data["W"], glctx, mesh_tensors,
                )
                _, ref_sil_loss = sil_optimizer.optimize(
                    p_ref, sil_views, iters=1, lr=0.0, antialias=True,
                )

                r["iou_pre_mean"] = f"{iou_pre_mean:.4f}"
                r["iou_pre_max"] = f"{iou_pre_max:.4f}"
                r["iou_post_mean"] = f"{iou_post_mean:.4f}"
                r["iou_post_max"] = f"{iou_post_max:.4f}"
                r["sil_loss_pre"] = f"{sil_loss_pre:.6f}"
                r["ref_iou_mean"] = f"{ref_iou_mean:.4f}"
                r["ref_iou_max"] = f"{ref_iou_max:.4f}"
                r["ref_sil_loss"] = f"{ref_sil_loss:.6f}"
                row_sec = time.perf_counter() - t_row
                print(f"  [{ep_name}] iou_pre={iou_pre_mean:.3f}/{iou_pre_max:.3f} "
                      f"iou_post={iou_post_mean:.3f}/{iou_post_max:.3f} "
                      f"sil_pre={sil_loss_pre:.4f} "
                      f"ref_iou={ref_iou_mean:.3f}/{ref_iou_max:.3f} "
                      f"ref_sil={ref_sil_loss:.4f} ({row_sec:.1f}s)")
            except Exception as exc:
                print(f"  [{ep_name}] FAILED: {exc}")
                for col in NEW_COLUMNS:
                    r.setdefault(col, "")
            out_rows.append(r)
            n_done += 1
            # Stream-write after each row so a crash mid-run keeps progress.
            with open(out_path, "w", newline="") as fout:
                w = csv.DictWriter(fout, fieldnames=fieldnames)
                w.writeheader()
                for rr in out_rows:
                    w.writerow(rr)

    print(f"\n[backfill] done in {time.perf_counter() - t_start:.1f}s — wrote {out_path}")


if __name__ == "__main__":
    main()
