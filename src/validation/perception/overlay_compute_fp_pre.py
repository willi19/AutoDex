#!/usr/bin/env python3
"""Compute FoundPose 24 candidate poses (pre-sil) per episode and save to disk.

For each (obj, ep) selected from results.csv (1 ep per obj):
  Run FoundPoseInit.estimate_per_view (24 cam)
  Save dict {cam_serial: 4x4 pose_world} to:
    outputs/foundpose_overlay/{obj}/{ep}/fp_pre.npz

Run inside `gotrack` env.
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
from pathlib import Path
from typing import Dict, List

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
    return images, masks, K, T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", type=str, default=str(RESULTS_CSV))
    parser.add_argument("--out-root", type=str, default=str(OUT_ROOT))
    parser.add_argument("--n-eps-per-obj", type=int, default=1)
    args = parser.parse_args()

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(args.results_csv)))
    by_obj: Dict[str, List[str]] = {}
    for r in rows:
        by_obj.setdefault(r["object"], []).append(r["episode"])
    selected = [(o, eps[:args.n_eps_per_obj]) for o, eps in by_obj.items()]
    print(f"[fp_pre] {len(selected)} obj × {args.n_eps_per_obj} ep")

    from autodex.perception.foundpose_init import FoundPoseInit

    t_start = time.perf_counter()
    for obj, eps in selected:
        try: mesh_path = _resolve_mesh(obj)
        except Exception as e:
            print(f"[skip {obj}] {e}"); continue

        ref_intr_json = EXP_ROOT/obj/eps[0]/"cam_param"/"intrinsics.json"
        ref_cam = sorted(json.load(open(ref_intr_json)).keys())[0]

        print(f"\n[{obj}] loading FoundPose")
        try:
            fp = FoundPoseInit(
                mesh_path=str(mesh_path), assets_root=str(ASSETS_ROOT/obj),
                obj_name=obj, object_id=1, device="cuda:0",
                reference_intrinsics_json=str(ref_intr_json), reference_camera_id=ref_cam,
            )
        except Exception as e:
            print(f"[skip {obj}] FoundPose: {e}"); continue

        for ep_name in eps:
            ep = EXP_ROOT/obj/ep_name
            try: images, masks, K, T = _load_episode(ep)
            except Exception as e:
                print(f"  [{ep_name}] load: {e}"); continue

            t0 = time.perf_counter()
            per_view = fp.estimate_per_view(images, masks, K, T)
            elapsed = time.perf_counter() - t0
            ok_count = sum(1 for v in per_view.values() if v is not None)

            # Save: dict serial -> 4x4 pose_world
            poses = {s: v["pose_world"] for s, v in per_view.items() if v is not None}
            ep_out = out_root / obj / ep_name
            ep_out.mkdir(parents=True, exist_ok=True)
            np.savez(ep_out/"fp_pre.npz", **{f"cam_{s}": p for s, p in poses.items()})
            print(f"  [{ep_name}] {ok_count}/24 candidates ({elapsed:.1f}s) → fp_pre.npz")

    print(f"\n[done] {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
