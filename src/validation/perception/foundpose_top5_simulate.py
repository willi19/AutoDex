#!/usr/bin/env python3
"""Task 2: simulate FoundPose pipeline restricted to top-5 cameras.

Step A. From candidate_full.csv, aggregate per-camera mean pre-sil-loss across
        all episodes; pick the 5 cams with smallest mean → fixed top-5 set.
Step B. For each episode (90):
          - re-run FoundPose to get 24 candidates
          - keep only candidates from the top-5 cams (= 5 candidates)
          - cross-view IoU select on 24-view masks → best 1
          - sil refine 100 iter (24 views) → final pose
          - measure 24-view mean IoU + final sil_loss

Output: outputs/.../top5_simulate.csv (per-ep) + console summary of top-5 cams.
Run inside `gotrack` env. ~15 min for 90 eps.
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
from collections import defaultdict
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


def _measure_24view_mean(pose_world, masks, K, T, H, W, glctx, mt):
    """Return (mean_iou, mean_sil_mse) for one pose across all view masks."""
    import torch
    sys.path.insert(0, str(REPO_ROOT/"autodex/perception/thirdparty/FoundationPose"))
    from Utils import nvdiffrast_render
    iou_t, sil_t = [], []
    for s, mask in masks.items():
        Ki = K[s].astype(np.float32)
        pose_cam = T[s] @ pose_world
        pt = torch.as_tensor(pose_cam, device="cuda", dtype=torch.float32).reshape(1, 4, 4)
        rc, _, _ = nvdiffrast_render(K=Ki, H=H, W=W, ob_in_cams=pt, glctx=glctx,
                                      mesh_tensors=mt, use_light=False)
        sil_bool = rc[0].sum(dim=2) > 0
        m_b = torch.as_tensor(mask, device="cuda")
        m_f = m_b.float()
        inter = (sil_bool & m_b).sum().float()
        union = (sil_bool | m_b).sum().float()
        iou_t.append(torch.where(union > 0, inter/union, torch.zeros_like(inter)))
        alpha = rc[0, :, :, 0]
        sil_t.append(((alpha - m_f)**2).mean())
    return float(torch.stack(iou_t).mean().item()), float(torch.stack(sil_t).mean().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cand-csv", type=str,
                        default=str(REPO_ROOT/"outputs/foundpose_init_compare/selected_100/candidate_full.csv"))
    parser.add_argument("--out-csv", type=str,
                        default=str(REPO_ROOT/"outputs/foundpose_init_compare/selected_100/top5_simulate.csv"))
    parser.add_argument("--n-views", type=int, default=5,
                        help="Number of fixed top-N cams (default 5).")
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()

    from autodex.perception.foundpose_init import FoundPoseInit
    from autodex.perception.silhouette import SilhouetteOptimizer
    from autodex.perception.pose_select import select_best_pose_by_iou

    # ── Step A: aggregate per-cam mean pre-sil-loss → pick top N cams.
    cand_rows = list(csv.DictReader(open(args.cand_csv)))
    by_cam: Dict[str, List[float]] = defaultdict(list)
    for r in cand_rows:
        try: by_cam[r["cand_serial"]].append(float(r["sil_mean_pre"]))
        except Exception: pass
    cam_mean_sil = {c: float(np.mean(v)) for c, v in by_cam.items()}
    sorted_cams = sorted(cam_mean_sil.items(), key=lambda kv: kv[1])
    top_cams = [c for c, _ in sorted_cams[: args.n_views]]
    print(f"[step A] {len(by_cam)} cams seen across {len(set((r['object'], r['episode']) for r in cand_rows))} eps")
    print(f"[step A] top-{args.n_views} cams (smallest mean pre-sil-loss):")
    for c, v in sorted_cams[: args.n_views]:
        print(f"  {c}: {v:.6f}")
    print(f"[step A] worst {args.n_views} cams (for reference):")
    for c, v in sorted_cams[-args.n_views:]:
        print(f"  {c}: {v:.6f}")

    # ── Step B: per-episode simulation.
    eps_set = sorted(set((r["object"], r["episode"]) for r in cand_rows))
    if args.limit > 0:
        eps_set = eps_set[: args.limit]
    print(f"\n[step B] simulating top-{args.n_views} pipeline on {len(eps_set)} eps")

    fout = open(args.out_csv, "w", newline="")
    w = csv.writer(fout)
    w.writerow([
        "object", "episode", "n_top_cands", "best_serial",
        "iou_mean_post", "sil_loss_post", "elapsed_s",
        "fp_sec", "select_sec", "sil_sec", "production_sec",
    ])

    by_obj: Dict[str, List[str]] = defaultdict(list)
    for o, e in eps_set:
        by_obj[o].append(e)

    t_start = time.perf_counter()
    for obj, eps in by_obj.items():
        try: mesh_path = _resolve_mesh(obj)
        except Exception as exc:
            print(f"[skip {obj}] {exc}", flush=True); continue
        ref_intr = EXP_ROOT/obj/eps[0]/"cam_param"/"intrinsics.json"
        ref_cam = sorted(json.load(open(ref_intr)).keys())[0]
        print(f"\n[{obj}] loading", flush=True)
        try:
            fp = FoundPoseInit(
                mesh_path=str(mesh_path), assets_root=str(ASSETS_ROOT/obj),
                obj_name=obj, object_id=1, device="cuda:0",
                reference_intrinsics_json=str(ref_intr), reference_camera_id=ref_cam,
            )
        except Exception as exc:
            print(f"[skip {obj}] FoundPose: {exc}", flush=True); continue
        sil = SilhouetteOptimizer(str(mesh_path), device="cuda")
        glctx, mt = sil.glctx, sil.mesh_tensors

        for ep_name in eps:
            ep = EXP_ROOT/obj/ep_name
            try: images, masks, K, T, H, W = _load_episode(ep)
            except Exception as exc:
                print(f"  [{ep_name}] load: {exc}", flush=True); continue

            # Restrict FoundPose to top-N cams only.
            images_top = {s: images[s] for s in top_cams if s in images}
            masks_top = {s: masks[s] for s in top_cams if s in masks}
            K_top = {s: K[s] for s in top_cams if s in K}
            T_top = {s: T[s] for s in top_cams if s in T}

            t_fp0 = time.perf_counter()
            per_view = fp.estimate_per_view(images_top, masks_top, K_top, T_top)
            t_fp = time.perf_counter() - t_fp0
            ok_top = {s: v for s, v in per_view.items() if v is not None}
            if not ok_top:
                print(f"  [{ep_name}] no top-cam candidates", flush=True); continue

            t_sel0 = time.perf_counter()
            candidates = {s: v["pose_world"] for s, v in ok_top.items()}
            best_s, p_pre, _, _ = select_best_pose_by_iou(
                candidates=candidates, masks=masks,
                intrinsics=K, extrinsics=T, H=H, W=W, glctx=glctx, mesh_tensors=mt,
            )
            t_sel = time.perf_counter() - t_sel0
            if p_pre is None:
                print(f"  [{ep_name}] iou select empty", flush=True); continue

            sil_views = []
            for s, m in masks.items():
                if int(m.sum()) < 100: continue
                sil_views.append({
                    "mask": (m.astype(np.uint8)*255),
                    "K": K[s].astype(np.float32),
                    "extrinsic": T[s].astype(np.float64),
                })
            t_sil0 = time.perf_counter()
            p_post, _ = sil.optimize(p_pre, sil_views, iters=100, lr=0.002, antialias=True)
            t_sil = time.perf_counter() - t_sil0

            iou_post, sil_loss_post = _measure_24view_mean(p_post, masks, K, T, H, W, glctx, mt)
            elapsed = time.perf_counter() - t_fp0
            production_time = t_fp + t_sel + t_sil  # excludes final 24-view measurement
            w.writerow([
                obj, ep_name, len(ok_top), best_s,
                f"{iou_post:.4f}", f"{sil_loss_post:.6f}", f"{elapsed:.2f}",
                f"{t_fp:.2f}", f"{t_sel:.2f}", f"{t_sil:.2f}", f"{production_time:.2f}",
            ])
            fout.flush()
            print(f"  [{ep_name}] best={best_s} iou_post={iou_post:.3f} sil_post={sil_loss_post:.4f} "
                  f"fp={t_fp:.2f} sel={t_sel:.2f} sil={t_sil:.2f} prod={production_time:.2f}s", flush=True)

    fout.close()
    print(f"\n[done] {time.perf_counter()-t_start:.1f}s → {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
