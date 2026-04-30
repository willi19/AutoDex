#!/usr/bin/env python3
"""Run FoundPose on the first frame of every camera, overlay the per-view pose.

Each camera is treated independently: FoundPose estimates a world-frame pose
from that view, and we render the mesh under that pose back into the same
image. Useful for sanity-checking FoundPose accuracy on a single experiment.

Requires:
  - Onboarded assets:        outputs/foundpose_assets/{obj}/object_repre/...
  - Undistorted first frame: {capture_dir}/images/{serial}.png
  - Masks (SAM3/YOLO-E):     {capture_dir}/_pipeline_tmp/masks/{serial}.png
  - cam_param/{intr,extr}.json

Run inside the `gotrack` env (FoundPose stack lives there):
    /home/mingi/miniconda3/envs/gotrack/bin/python \
        src/validation/execution/eval_perception/overlay_foundpose_init.py \
        --capture_dir /home/mingi/shared_data/AutoDex/experiment/selected_100/inspire/attached_container/20260405_235218 \
        --obj attached_container
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

MESH_BASE = Path("/home/mingi/shared_data/AutoDex/object/paradex")
ASSETS_ROOT = REPO_ROOT / "outputs/foundpose_assets"


def _resolve_mesh(obj):
    for sub in [
        MESH_BASE / obj / "raw_mesh" / f"{obj}.obj",
        MESH_BASE / obj / "processed_data" / "mesh" / "raw.obj",
        MESH_BASE / obj / "processed_data" / "mesh" / "simplified.obj",
    ]:
        if sub.exists():
            return sub
    raise FileNotFoundError(f"No mesh for {obj}")


def _load_episode(ep: Path):
    img_dir = ep / "images"
    if not img_dir.exists():
        img_dir = ep / "raw" / "images"
    mask_dir = ep / "_pipeline_tmp" / "masks"
    intr = json.load(open(ep / "cam_param" / "intrinsics.json"))
    extr = json.load(open(ep / "cam_param" / "extrinsics.json"))
    serials = sorted(p.stem for p in img_dir.glob("*.png"))
    images, masks, K, T = {}, {}, {}, {}
    last_shape = None
    for s in serials:
        if s not in intr or s not in extr:
            continue
        bgr = cv2.imread(str(img_dir / f"{s}.png"))
        if bgr is None:
            continue
        images[s] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        m_path = mask_dir / f"{s}.png"
        m = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE) if m_path.exists() else None
        masks[s] = (m > 127) if m is not None else np.zeros(bgr.shape[:2], bool)
        K[s] = np.asarray(intr[s]["intrinsics_undistort"], np.float64)
        e = np.asarray(extr[s], np.float64)
        if e.shape == (3, 4):
            e = np.vstack([e, [0, 0, 0, 1]])
        T[s] = e
        last_shape = bgr.shape
    H, W = last_shape[:2]
    return images, masks, K, T, H, W


def _render_overlay(image_rgb, pose_world, K, T, H, W, glctx, mt,
                     color=(0, 200, 0), alpha=0.5):
    import torch
    sys.path.insert(0, str(REPO_ROOT / "autodex/perception/thirdparty/FoundationPose"))
    from Utils import nvdiffrast_render
    K = np.asarray(K, np.float32)
    pose_cam = T @ pose_world
    pt = torch.as_tensor(pose_cam, device="cuda", dtype=torch.float32).reshape(1, 4, 4)
    rc, _, _ = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=pt, glctx=glctx,
                                  mesh_tensors=mt, use_light=False)
    sil = (rc[0].sum(dim=2) > 0).detach().cpu().numpy()
    out = image_rgb.copy()
    color_arr = np.array(color, dtype=np.float32)
    out[sil] = (out[sil] * (1 - alpha) + color_arr * alpha).astype(np.uint8)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture_dir", required=True)
    p.add_argument("--obj", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    ep = Path(args.capture_dir)
    mesh_path = _resolve_mesh(args.obj)
    print(f"[mesh] {mesh_path}")

    images, masks, K, T, H, W = _load_episode(ep)
    print(f"[load] {len(images)} cams, {H}x{W}")

    from autodex.perception.foundpose_init import FoundPoseInit
    from autodex.perception.silhouette import SilhouetteOptimizer

    ref_intr_json = ep / "cam_param" / "intrinsics.json"
    ref_cam = sorted(json.load(open(ref_intr_json)).keys())[0]

    fp = FoundPoseInit(
        mesh_path=str(mesh_path),
        assets_root=str(ASSETS_ROOT / args.obj),
        obj_name=args.obj, object_id=1, device="cuda:0",
        reference_intrinsics_json=str(ref_intr_json),
        reference_camera_id=ref_cam,
    )
    sil_opt = SilhouetteOptimizer(str(mesh_path), device="cuda")
    glctx, mt = sil_opt.glctx, sil_opt.mesh_tensors

    print("[fp] estimating per-view ...")
    per_view = fp.estimate_per_view(images, masks, K, T)
    ok = {s: v for s, v in per_view.items() if v is not None}
    for s in sorted(per_view.keys()):
        print(f"  {s}: {'OK' if per_view[s] is not None else 'FAIL'}")

    # ── IoU best pose + silhouette refinement (matches foundpose_overlay_grid.py) ──
    from autodex.perception.pose_select import select_best_pose_by_iou
    cands_fp = {s: v["pose_world"] for s, v in ok.items()}
    best_s, p_fp_pre, _, _ = select_best_pose_by_iou(
        candidates=cands_fp, masks=masks,
        intrinsics=K, extrinsics=T, H=H, W=W, glctx=glctx, mesh_tensors=mt,
    )
    print(f"[fp] best by IoU: {best_s}")

    sil_views = [{
        "mask": (masks[best_s].astype(np.uint8) * 255),
        "K": K[best_s].astype(np.float32),
        "extrinsic": T[best_s].astype(np.float64),
    }]
    print(f"[sil] refining over best view {best_s} ...")
    p_fp_post, sil_loss = sil_opt.optimize(p_fp_pre, sil_views, iters=100, lr=0.002, antialias=True)
    print(f"[sil] refine done: final_loss={sil_loss:.6f}")

    def _grid(pose, color, label_prefix):
        ovs = {}
        for s in sorted(images.keys()):
            ov = _render_overlay(images[s], pose, K[s], T[s], H, W,
                                  glctx, mt, color=color, alpha=0.5)
            ov_bgr = cv2.cvtColor(ov, cv2.COLOR_RGB2BGR)
            cv2.putText(ov_bgr, f"{label_prefix}{s}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            ovs[s] = ov_bgr
        cols = 6
        rows = (len(ovs) + cols - 1) // cols
        scale = 0.3
        keys = sorted(ovs.keys())
        oh, ow = ovs[keys[0]].shape[:2]
        th, tw = int(oh * scale), int(ow * scale)
        g = np.full((rows * th, cols * tw, 3), 40, dtype=np.uint8)
        for i, k in enumerate(keys):
            r, c = divmod(i, cols)
            g[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = cv2.resize(ovs[k], (tw, th))
        return g

    pre_grid = _grid(p_fp_pre, (0, 200, 0), "pre ")
    post_grid = _grid(p_fp_post, (0, 0, 255), "post ")

    out_pre = args.out or str(ep / "foundpose_init_overlay_pre.png")
    out_post = str(ep / "foundpose_init_overlay_post.png")
    cv2.imwrite(out_pre, pre_grid)
    cv2.imwrite(out_post, post_grid)
    print(f"[done] saved {out_pre}")
    print(f"[done] saved {out_post}")

    # Save raw per-view poses too
    poses_out = ep / "foundpose_init_poses.json"
    dump = {"per_view": {}, "best_by_iou": best_s,
            "fp_pre": p_fp_pre.tolist(), "fp_post": p_fp_post.tolist()}
    for s, v in per_view.items():
        if v is None:
            dump["per_view"][s] = None
            continue
        dump["per_view"][s] = {
            "pose_world": v["pose_world"].tolist(),
            "quality": float(v.get("quality", 0)),
            "inliers": int(v.get("inliers", 0)),
            "template_id": int(v.get("template_id", -1)),
        }
    with open(poses_out, "w") as f:
        json.dump(dump, f, indent=2)
    print(f"[done] saved {poses_out}")


if __name__ == "__main__":
    main()
