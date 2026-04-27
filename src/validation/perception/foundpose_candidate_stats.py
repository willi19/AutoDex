#!/usr/bin/env python3
"""Compute per-candidate FoundPose stats (24 candidate × 24-view-mean IoU + sil_loss).

For each episode: run FoundPose on 24 cams → 24 candidate poses. For each
candidate, compute its 24-view-mean IoU vs SAM masks and its mean silhouette
MSE. Then report mean / max across the 24 candidates.

Output: writes a small CSV with per-episode stats and prints aggregate.
"""
from __future__ import annotations
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

MESH_ROOT = Path("/home/mingi/shared_data/AutoDex/object/paradex")
ASSETS_ROOT = REPO_ROOT / "outputs/foundpose_assets"
EXP_ROOT = Path("/home/mingi/shared_data/AutoDex/experiment/selected_100/allegro")
CSV_IN = REPO_ROOT / "outputs/foundpose_init_compare/selected_100/results.csv"
CSV_OUT = REPO_ROOT / "outputs/foundpose_init_compare/selected_100/candidate_stats.csv"


def _resolve_mesh(obj):
    for sub in [MESH_ROOT/obj/"raw_mesh"/f"{obj}.obj",
                MESH_ROOT/obj/"processed_data"/"mesh"/"raw.obj",
                MESH_ROOT/obj/"processed_data"/"mesh"/"simplified.obj"]:
        if sub.exists(): return sub
    raise FileNotFoundError(obj)


def _load_episode(ep):
    img_dir = ep/"images"; mask_dir = ep/"_pipeline_tmp"/"masks"
    intr = json.load(open(ep/"cam_param"/"intrinsics.json"))
    extr = json.load(open(ep/"cam_param"/"extrinsics.json"))
    serials = sorted(p.stem for p in img_dir.glob("*.png"))
    images, masks, K, T = {}, {}, {}, {}
    last_shape = None
    for s in serials:
        if s not in intr or s not in extr: continue
        bgr = cv2.imread(str(img_dir/f"{s}.png"))
        if bgr is None: continue
        images[s] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        m = cv2.imread(str(mask_dir/f"{s}.png"), cv2.IMREAD_GRAYSCALE)
        masks[s] = (m > 127) if m is not None else np.zeros(bgr.shape[:2], bool)
        K[s] = np.asarray(intr[s]["intrinsics_undistort"], np.float64)
        e = np.asarray(extr[s], np.float64)
        if e.shape == (3, 4): e = np.vstack([e, [0, 0, 0, 1]])
        T[s] = e
        last_shape = bgr.shape
    H, W = last_shape[:2]
    return images, masks, K, T, H, W


def main():
    from autodex.perception.foundpose_init import FoundPoseInit
    from autodex.perception.silhouette import SilhouetteOptimizer
    from autodex.perception.pose_select import compute_cross_view_iou
    import torch
    sys.path.insert(0, str(REPO_ROOT/"autodex/perception/thirdparty/FoundationPose"))
    from Utils import nvdiffrast_render
    import torch.nn.functional as F

    rows = list(csv.DictReader(open(CSV_IN)))
    print(f"reading {len(rows)} rows", flush=True)

    by_obj: Dict[str, List[dict]] = {}
    for r in rows:
        by_obj.setdefault(r["object"], []).append(r)

    out_rows = []
    fout = open(CSV_OUT, "w", newline="")
    w = csv.writer(fout)
    w.writerow([
        "object", "episode", "n_candidates",
        "cand_iou_mean", "cand_iou_max", "cand_iou_min",
        "cand_sil_mean", "cand_sil_min", "cand_sil_max",
    ])

    t_start = time.perf_counter()
    for obj, eps in by_obj.items():
        try:
            mesh_path = _resolve_mesh(obj)
        except Exception as e:
            print(f"[skip {obj}] {e}", flush=True); continue

        ref_intr = EXP_ROOT/obj/eps[0]["episode"]/"cam_param"/"intrinsics.json"
        ref_cam = sorted(json.load(open(ref_intr)).keys())[0]
        print(f"\n[{obj}] loading ({len(eps)} eps)", flush=True)
        try:
            fp = FoundPoseInit(
                mesh_path=str(mesh_path), assets_root=str(ASSETS_ROOT/obj),
                obj_name=obj, object_id=1, device="cuda:0",
                reference_intrinsics_json=str(ref_intr), reference_camera_id=ref_cam,
            )
        except Exception as e:
            print(f"[skip {obj}] FoundPose: {e}", flush=True); continue
        sil = SilhouetteOptimizer(str(mesh_path), device="cuda")
        glctx, mt = sil.glctx, sil.mesh_tensors

        for r in eps:
            ep_name = r["episode"]
            ep = EXP_ROOT/obj/ep_name
            try:
                images, masks, K, T, H, W = _load_episode(ep)
            except Exception as e:
                print(f"  [{ep_name}] load fail: {e}", flush=True); continue

            t0 = time.perf_counter()
            per_view = fp.estimate_per_view(images, masks, K, T)
            ok = {s: v for s, v in per_view.items() if v is not None}
            if not ok:
                print(f"  [{ep_name}] no candidates", flush=True); continue

            # For each candidate, compute (a) 24-view-mean IoU, (b) 24-view-mean sil MSE.
            cand_ious, cand_sils = [], []
            sil_views = []
            for s, m in masks.items():
                if int(m.sum()) < 100: continue
                sil_views.append((s, m, K[s], T[s]))

            for src_s, payload in ok.items():
                pose_world = payload["pose_world"]
                # IoU
                iou_mean, _ = compute_cross_view_iou(
                    pose_world=pose_world, masks=masks,
                    intrinsics=K, extrinsics=T,
                    H=H, W=W, glctx=glctx, mesh_tensors=mt,
                )
                cand_ious.append(iou_mean)
                # sil MSE (no optimization, just measure loss for input pose)
                _, loss = sil.optimize(
                    pose_world,
                    [{"mask": (m.astype(np.uint8)*255), "K": K[s].astype(np.float32),
                      "extrinsic": T[s].astype(np.float64)}
                     for (s, m, _, _) in sil_views],
                    iters=1, lr=0.0, antialias=True,
                )
                cand_sils.append(float(loss))

            n = len(cand_ious)
            iou_mean = float(np.mean(cand_ious)); iou_max = float(np.max(cand_ious)); iou_min = float(np.min(cand_ious))
            sil_mean = float(np.mean(cand_sils)); sil_min = float(np.min(cand_sils)); sil_max = float(np.max(cand_sils))
            elapsed = time.perf_counter() - t0
            print(f"  [{ep_name}] n={n}  iou mean/max/min={iou_mean:.3f}/{iou_max:.3f}/{iou_min:.3f}  "
                  f"sil mean/min/max={sil_mean:.4f}/{sil_min:.4f}/{sil_max:.4f}  ({elapsed:.1f}s)", flush=True)
            w.writerow([obj, ep_name, n,
                        f"{iou_mean:.4f}", f"{iou_max:.4f}", f"{iou_min:.4f}",
                        f"{sil_mean:.6f}", f"{sil_min:.6f}", f"{sil_max:.6f}"])
            fout.flush()
            out_rows.append((obj, ep_name, n, iou_mean, iou_max, iou_min, sil_mean, sil_min, sil_max))

    fout.close()
    print(f"\n[done] {len(out_rows)} eps in {time.perf_counter()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
