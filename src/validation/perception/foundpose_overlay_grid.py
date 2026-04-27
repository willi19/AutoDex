#!/usr/bin/env python3
"""Generate 24-cam grid overlays for 3 pose hypotheses per episode:

  fp_pre   = FoundPose IoU-selected pose (before sil refine)
  fp_post  = sil-refined fp_pre
  ref_post = reference final pose (pose_world.npy, after sil refine)

Picks 1 episode per object from results.csv (9 obj total).

Outputs PNG grids to:
  outputs/foundpose_overlay/{obj}/{ep}/{pose_name}_grid.png

Run inside `gotrack` env.
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
from pathlib import Path
from typing import Dict, List, Tuple

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
OUT_ROOT = REPO_ROOT / "outputs/foundpose_overlay"
RESULTS_CSV = REPO_ROOT / "outputs/foundpose_init_compare/selected_100/results.csv"

# Pre-computed best-5 cams from PerceptionPipeline.
BEST_5_DA3 = ["25322638", "25322645", "24080331", "25322639", "25322643"]


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


def _render_overlay(image_rgb, pose_world, K, T, H, W, glctx, mesh_tensors,
                    color=(0, 200, 0), alpha=0.5):
    """Render mesh at pose_world in this camera, alpha-blend onto image_rgb."""
    import torch
    sys.path.insert(0, str(REPO_ROOT/"autodex/perception/thirdparty/FoundationPose"))
    from Utils import nvdiffrast_render
    K = np.asarray(K, np.float32)
    pose_cam = T @ pose_world
    pt = torch.as_tensor(pose_cam, device="cuda", dtype=torch.float32).reshape(1, 4, 4)
    rc, _, _ = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=pt, glctx=glctx,
                                  mesh_tensors=mesh_tensors, use_light=False)
    sil = rc[0].sum(dim=2) > 0
    sil_np = sil.detach().cpu().numpy()
    out = image_rgb.copy()
    color_arr = np.array(color, dtype=np.float32)
    out[sil_np] = (out[sil_np] * (1 - alpha) + color_arr * alpha).astype(np.uint8)
    return out


def _make_grid(images_dict: Dict[str, np.ndarray], cols=6, label_prefix="",
               highlight_key: str = None, highlight_color=(0, 255, 255), border_px=6):
    """Arrange images in cols×rows grid, downscale by 2.

    If highlight_key is given, draw a colored border around that cell.
    """
    keys = sorted(images_dict.keys())
    if not keys: return None
    h, w = images_dict[keys[0]].shape[:2]
    h2, w2 = h // 2, w // 2
    rows = (len(keys) + cols - 1) // cols
    grid = np.zeros((rows * h2, cols * w2, 3), dtype=np.uint8)
    for i, k in enumerate(keys):
        r, c = i // cols, i % cols
        small = cv2.resize(images_dict[k], (w2, h2))
        if k == highlight_key:
            cv2.rectangle(small, (0, 0), (w2 - 1, h2 - 1), highlight_color, border_px)
        cv2.putText(small, f"{label_prefix}{k}", (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        grid[r*h2:(r+1)*h2, c*w2:(c+1)*w2] = small
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", type=str, default=str(RESULTS_CSV))
    parser.add_argument("--out-root", type=str, default=str(OUT_ROOT))
    parser.add_argument("--n-eps-per-obj", type=int, default=1)
    args = parser.parse_args()

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    # Pick 1 ep per obj from results.csv
    rows = list(csv.DictReader(open(args.results_csv)))
    by_obj: Dict[str, List[str]] = {}
    for r in rows:
        by_obj.setdefault(r["object"], []).append(r["episode"])
    selected = [(o, eps[:args.n_eps_per_obj]) for o, eps in by_obj.items()]
    print(f"[overlay] {len(selected)} obj × {args.n_eps_per_obj} ep each")

    from autodex.perception.foundpose_init import FoundPoseInit
    from autodex.perception.silhouette import SilhouetteOptimizer
    from autodex.perception.pose_select import select_best_pose_by_iou

    t_start = time.perf_counter()
    for obj, eps in selected:
        try: mesh_path = _resolve_mesh(obj)
        except Exception as e:
            print(f"[skip {obj}] {e}"); continue

        ref_intr_json = EXP_ROOT/obj/eps[0]/"cam_param"/"intrinsics.json"
        ref_cam = sorted(json.load(open(ref_intr_json)).keys())[0]

        print(f"\n[{obj}] loading models")
        try:
            fp = FoundPoseInit(
                mesh_path=str(mesh_path), assets_root=str(ASSETS_ROOT/obj),
                obj_name=obj, object_id=1, device="cuda:0",
                reference_intrinsics_json=str(ref_intr_json), reference_camera_id=ref_cam,
            )
        except Exception as e:
            print(f"[skip {obj}] FoundPose: {e}"); continue
        sil_opt = SilhouetteOptimizer(str(mesh_path), device="cuda")
        glctx, mt = sil_opt.glctx, sil_opt.mesh_tensors

        for ep_name in eps:
            ep = EXP_ROOT/obj/ep_name
            try: images, masks, K, T, H, W = _load_episode(ep)
            except Exception as e:
                print(f"  [{ep_name}] load: {e}"); continue
            p_ref_post = np.load(ep/"pose_world.npy").astype(np.float64)

            t0 = time.perf_counter()

            # ── FoundPose path ──
            per_view = fp.estimate_per_view(images, masks, K, T)
            ok = {s: v for s, v in per_view.items() if v is not None}
            cands_fp = {s: v["pose_world"] for s, v in ok.items()}
            _, p_fp_pre, _, _ = select_best_pose_by_iou(
                candidates=cands_fp, masks=masks,
                intrinsics=K, extrinsics=T, H=H, W=W, glctx=glctx, mesh_tensors=mt,
            )
            sil_views = [
                {"mask": (m.astype(np.uint8)*255), "K": K[s].astype(np.float32),
                 "extrinsic": T[s].astype(np.float64)}
                for s, m in masks.items() if int(m.sum()) >= 100
            ]
            p_fp_post, _ = sil_opt.optimize(p_fp_pre, sil_views, iters=100, lr=0.002, antialias=True)

            # ── Render overlays ──
            ep_out = out_root / obj / ep_name
            ep_out.mkdir(parents=True, exist_ok=True)

            # Per-candidate grids (24 source cams, each candidate rendered in all 24 cams)
            for src_s, payload in ok.items():
                pose = payload["pose_world"]
                ovs = {}
                for s in sorted(images.keys()):
                    ov = _render_overlay(images[s], pose, K[s], T[s], H, W,
                                          glctx, mt, color=(0, 200, 0), alpha=0.5)
                    ovs[s] = cv2.cvtColor(ov, cv2.COLOR_RGB2BGR)
                grid = _make_grid(ovs, cols=6, label_prefix="", highlight_key=src_s)
                if grid is not None:
                    cv2.imwrite(str(ep_out/f"fp_pre_{src_s}_grid.png"), grid)

            # Selected best (post-sil) and ref grids
            single_pose_dict = {
                "fp_post": (p_fp_post, (0, 0, 255)),     # red
                "ref_post": (p_ref_post, (200, 0, 200)), # magenta
            }
            for name, (pose, color) in single_pose_dict.items():
                if pose is None:
                    continue
                ovs = {}
                for s in sorted(images.keys()):
                    ov = _render_overlay(images[s], pose, K[s], T[s], H, W,
                                          glctx, mt, color=color, alpha=0.5)
                    ovs[s] = cv2.cvtColor(ov, cv2.COLOR_RGB2BGR)
                grid = _make_grid(ovs, cols=6, label_prefix="")
                if grid is not None:
                    cv2.imwrite(str(ep_out/f"{name}_grid.png"), grid)

            elapsed = time.perf_counter() - t0
            print(f"  [{ep_name}] grids saved → {ep_out} ({elapsed:.1f}s)")

    print(f"\n[done] {time.perf_counter() - t_start:.1f}s total")


if __name__ == "__main__":
    main()
