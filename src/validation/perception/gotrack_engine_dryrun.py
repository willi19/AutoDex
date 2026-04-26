#!/usr/bin/env python3
"""Local dry-run for autodex.perception.gotrack_engine.GoTrackEngine.

Loads one experiment episode (already has 24 cam undistorted images +
SAM3 masks + reference pose_world.npy from PerceptionPipeline.run), and
calls GoTrackEngine.process_frame using the reference pose as prior.

Verifies that:
  - GoTrackEngine constructs (model load, anchor bank load, renderer setup)
  - process_frame returns the expected per-cam payload (uv_curr, conf, etc.)
  - shapes / dtypes are sane

Run inside `gotrack` env. Single GPU; do NOT run while another big model is
loaded on the same GPU (~3-5GB needed).

Usage:
    python src/validation/perception/gotrack_engine_dryrun.py \\
        --obj attached_container \\
        --episode 20260330_164351
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GOTRACK_ROOT = REPO_ROOT / "autodex/perception/thirdparty/MV-GoTrack"
ANCHOR_DIR = GOTRACK_ROOT / "anchor_banks"
MESH_BASE = Path("/home/mingi/shared_data/AutoDex/object/paradex")
EXP_BASE = Path("/home/mingi/shared_data/AutoDex/experiment/selected_100/allegro")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, required=True,
                        help="Object name (e.g. attached_container).")
    parser.add_argument("--episode", type=str, default=None,
                        help="Episode timestamp dir under "
                             "experiment/selected_100/allegro/{obj}/. "
                             "Default: pick first available.")
    parser.add_argument("--n-cams", type=int, default=4,
                        help="Take only first N cams to mimic single-PC setup.")
    args = parser.parse_args()

    obj = args.obj
    obj_dir = EXP_BASE / obj
    if not obj_dir.exists():
        raise FileNotFoundError(f"No experiment for {obj} at {obj_dir}")

    if args.episode:
        ep = obj_dir / args.episode
    else:
        eps = [p for p in sorted(obj_dir.iterdir())
               if p.is_dir() and (p / "pose_world.npy").exists()
               and (p / "_pipeline_tmp" / "masks").exists()]
        if not eps:
            raise FileNotFoundError(f"No usable episodes under {obj_dir}")
        ep = eps[0]
    print(f"[dryrun] obj={obj}  episode={ep.name}")

    # Mesh + anchor bank.
    mesh_path = MESH_BASE / obj / "raw_mesh" / f"{obj}.obj"
    bank_path = ANCHOR_DIR / f"{obj}.npz"
    for p, label in [(mesh_path, "mesh"), (bank_path, "anchor bank")]:
        if not p.exists():
            raise FileNotFoundError(f"{label} missing: {p}")

    # Load camera params.
    with open(ep / "cam_param" / "intrinsics.json") as f:
        intr_raw = json.load(f)
    with open(ep / "cam_param" / "extrinsics.json") as f:
        extr_raw = json.load(f)

    img_dir = ep / "images"
    serials = sorted(p.stem for p in img_dir.glob("*.png"))
    serials = [s for s in serials if s in intr_raw and s in extr_raw][: args.n_cams]
    print(f"[dryrun] using {len(serials)} cams: {serials}")

    images_bgr: Dict[str, np.ndarray] = {}
    cams_meta: List = []

    from autodex.perception.gotrack_engine import GoTrackEngine, CameraIntrinsics

    for s in serials:
        img = cv2.imread(str(img_dir / f"{s}.png"))
        if img is None:
            continue
        H, W = img.shape[:2]
        K = np.asarray(intr_raw[s]["intrinsics_undistort"], dtype=np.float64)
        T = np.asarray(extr_raw[s], dtype=np.float64)
        if T.shape == (3, 4):
            T = np.vstack([T, [0, 0, 0, 1]])
        images_bgr[s] = img
        cams_meta.append(CameraIntrinsics(
            serial=s, K=K, extrinsic_cw=T, width=W, height=H,
        ))

    # Reference pose_world from PerceptionPipeline (we use as the prior).
    p_ref = np.load(ep / "pose_world.npy")
    print(f"[dryrun] prior pose_world translation = {p_ref[:3, 3].tolist()}")

    # Build engine.
    print("[dryrun] building GoTrackEngine ...")
    t0 = time.perf_counter()
    engine = GoTrackEngine(
        mesh_path=str(mesh_path),
        anchor_bank_path=str(bank_path),
        cameras=cams_meta,
        object_id=1,
        object_name=obj,
        mesh_scale=1.0,
        unit_scale_mode="auto",
        num_iters=1,
        first_frame_num_iters=5,
        mask_free=True,
        skip_pnp=True,
    )
    print(f"[dryrun] engine ready ({time.perf_counter() - t0:.1f}s)")

    # Process one frame.
    print("[dryrun] processing 1 frame ...")
    t0 = time.perf_counter()
    per_cam = engine.process_frame(
        prior_pose_world=p_ref,
        frames_bgr=images_bgr,
        masks=None,
        frame_index=0,
    )
    elapsed = time.perf_counter() - t0
    print(f"[dryrun] process_frame took {elapsed:.2f}s for {len(per_cam)} cams")

    for s, payload in per_cam.items():
        keys_present = sorted(payload.keys())
        uv = payload.get("uv_curr")
        conf = payload.get("confidence")
        valid = payload.get("valid_mask")
        sel = payload.get("selected_mask")
        ci = payload.get("crop_intrinsic")
        Tw = payload.get("T_world_from_crop_cam")
        print(f"  {s}: status={payload.get('status')}  keys={keys_present}")
        if uv is not None:
            print(f"    uv_curr      shape={uv.shape}  dtype={uv.dtype}")
        if conf is not None:
            print(f"    confidence   shape={conf.shape}  mean={float(np.mean(conf)):.3f}")
        if valid is not None:
            print(f"    valid_mask   shape={valid.shape}  n_true={int(valid.sum())}")
        if sel is not None:
            print(f"    selected_mask shape={sel.shape}  n_true={int(sel.sum())}")
        if ci is not None:
            print(f"    crop_intrinsic shape={ci.shape}")
        if Tw is not None:
            print(f"    T_world_from_crop_cam shape={Tw.shape}")
        break  # only print first cam

    print("[dryrun] OK")


if __name__ == "__main__":
    main()
