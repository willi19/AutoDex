#!/usr/bin/env python3
"""Full per-candidate evaluation for FoundPose 24 candidates per episode.

For each episode (from results CSV):
  Run FoundPose on 24 cams → 24 candidate poses.
  For each of 24 candidates, compute 8 metrics:
    iou_mean_pre, iou_max_pre, sil_mean_pre, sil_max_pre   (no refine)
    iou_mean_post, iou_max_post, sil_mean_post, sil_max_post (after sil refine 100 iter)

Output per-candidate CSV: outputs/.../candidate_full.csv (one row per (ep, cand)).

Run inside `gotrack` env. Estimated cost: ~120s/ep × 90 ep ≈ 3 hours.
Use --limit N for sanity check (first N rows).
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
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


def _measure_per_view(pose_world, masks_gpu, K, T, H, W, glctx, mt):
    """Per-view (iou, sil_mse) dicts for one pose. masks_gpu: serial -> {bool, float} GPU tensors."""
    import torch
    sys.path.insert(0, str(REPO_ROOT/"autodex/perception/thirdparty/FoundationPose"))
    from Utils import nvdiffrast_render
    iou_tensors, sil_tensors, serials = [], [], []
    for s, mt_gpu in masks_gpu.items():
        Ki = K[s].astype(np.float32)
        pose_cam = T[s] @ pose_world
        pt = torch.as_tensor(pose_cam, device="cuda", dtype=torch.float32).reshape(1, 4, 4)
        rc, _, _ = nvdiffrast_render(K=Ki, H=H, W=W, ob_in_cams=pt, glctx=glctx,
                                      mesh_tensors=mt, use_light=False)
        sil_bool = rc[0].sum(dim=2) > 0
        inter = (sil_bool & mt_gpu["bool"]).sum().float()
        union = (sil_bool | mt_gpu["bool"]).sum().float()
        iou_tensors.append(torch.where(union > 0, inter / union,
                                        torch.zeros_like(inter)))
        alpha = rc[0, :, :, 0]
        sil_tensors.append(((alpha - mt_gpu["float"]) ** 2).mean())
        serials.append(s)
    iou_arr = torch.stack(iou_tensors).cpu().tolist()
    sil_arr = torch.stack(sil_tensors).cpu().tolist()
    return dict(zip(serials, iou_arr)), dict(zip(serials, sil_arr))


def _stats(d):
    vals = list(d.values())
    return float(np.mean(vals)), float(np.max(vals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-in", type=str,
                        default=str(REPO_ROOT/"outputs/foundpose_init_compare/selected_100/results.csv"))
    parser.add_argument("--csv-out", type=str,
                        default=str(REPO_ROOT/"outputs/foundpose_init_compare/selected_100/candidate_full.csv"))
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()

    from autodex.perception.foundpose_init import FoundPoseInit
    from autodex.perception.silhouette import SilhouetteOptimizer

    rows = list(csv.DictReader(open(args.csv_in)))
    if args.limit > 0: rows = rows[:args.limit]
    print(f"[full] {len(rows)} eps", flush=True)

    by_obj: Dict[str, List[dict]] = {}
    for r in rows: by_obj.setdefault(r["object"], []).append(r)

    fout = open(args.csv_out, "w", newline="")
    w = csv.writer(fout)
    w.writerow([
        "object", "episode", "cand_serial",
        "iou_mean_pre", "iou_max_pre", "sil_mean_pre", "sil_max_pre",
    ])

    t_start = time.perf_counter()
    for obj, eps in by_obj.items():
        try: mesh_path = _resolve_mesh(obj)
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
            ep_name = r["episode"]; ep = EXP_ROOT/obj/ep_name
            try: images, masks, K, T, H, W = _load_episode(ep)
            except Exception as e:
                print(f"  [{ep_name}] load: {e}", flush=True); continue
            t0 = time.perf_counter()
            per_view = fp.estimate_per_view(images, masks, K, T)
            ok = {s: v for s, v in per_view.items() if v is not None}
            if not ok:
                print(f"  [{ep_name}] no candidates", flush=True); continue

            import torch
            masks_gpu = {
                s: {
                    "bool": torch.as_tensor(masks[s], device="cuda"),
                    "float": torch.as_tensor(masks[s], device="cuda", dtype=torch.float32),
                }
                for s in masks
            }

            for src_s, payload in ok.items():
                p_pre = payload["pose_world"]
                iou_pre, sil_pre = _measure_per_view(p_pre, masks_gpu, K, T, H, W, glctx, mt)
                iou_pre_mean, iou_pre_max = _stats(iou_pre)
                sil_pre_mean, sil_pre_max = _stats(sil_pre)
                w.writerow([
                    obj, ep_name, src_s,
                    f"{iou_pre_mean:.4f}", f"{iou_pre_max:.4f}",
                    f"{sil_pre_mean:.6f}", f"{sil_pre_max:.6f}",
                ])
                fout.flush()
            print(f"  [{ep_name}] {len(ok)} cand done ({time.perf_counter()-t0:.1f}s)", flush=True)

    fout.close()
    print(f"\n[full] done in {time.perf_counter()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
