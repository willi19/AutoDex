#!/usr/bin/env python3
"""Compute reference path 24 candidate poses (DA3 depth + FoundationPose register on each cam).

Two-phase to avoid OOM (DA3 + FoundationPose can't both fit on 24GB):
  Phase A: load DA3, compute depth for all 9 ep × 24 cam, save .npz, free DA3
  Phase B: load FoundationPose, read depth + register per cam, save ref_pre.npz

Run inside `foundationpose` env (DA3 + FoundationPose + nvdiffrast all available).
"""
from __future__ import annotations
import argparse, csv, gc, json, os, sys, time
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
    parser.add_argument("--skip-depth", action="store_true",
                        help="Skip Phase A (assume depth.npz already exists per ep).")
    args = parser.parse_args()

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(args.results_csv)))
    by_obj: Dict[str, List[str]] = {}
    for r in rows:
        by_obj.setdefault(r["object"], []).append(r["episode"])
    selected = [(o, eps[:args.n_eps_per_obj]) for o, eps in by_obj.items()]
    print(f"[ref_pre] {len(selected)} obj × {args.n_eps_per_obj} ep")

    # ── Phase A: DA3 depth ──
    if not args.skip_depth:
        sys.path.insert(0, str(REPO_ROOT/"autodex/perception/thirdparty/Depth-Anything-3/src"))
        from depth_anything_3.api import DepthAnything3
        from autodex.perception.depth import get_depth_da3
        import torch

        print("[Phase A] loading DA3 ...")
        da3 = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE").to("cuda").eval()
        print("[Phase A] DA3 loaded")

        for obj, eps in selected:
            for ep_name in eps:
                ep = EXP_ROOT/obj/ep_name
                ep_out = out_root / obj / ep_name
                ep_out.mkdir(parents=True, exist_ok=True)
                depth_npz = ep_out / "depth.npz"
                if depth_npz.exists():
                    print(f"  [{obj}/{ep_name}] depth cached, skip")
                    continue
                try: images, masks, K, T = _load_episode(ep)
                except Exception as e:
                    print(f"  [{obj}/{ep_name}] load fail: {e}"); continue
                serials = sorted(images.keys())
                imgs = [images[s] for s in serials]
                K_arr = np.asarray([K[s] for s in serials], dtype=np.float32)
                T_arr = np.asarray([T[s] for s in serials], dtype=np.float32)
                t0 = time.perf_counter()
                depth_list = get_depth_da3(imgs, intrinsics=K_arr, extrinsics=T_arr, model=da3)
                np.savez(depth_npz, **{f"d_{s}": np.asarray(d, np.float32)
                                       for s, d in zip(serials, depth_list)})
                print(f"  [{obj}/{ep_name}] depth saved ({time.perf_counter()-t0:.1f}s)")

        # Free DA3
        del da3; gc.collect(); torch.cuda.empty_cache()
        print("[Phase A] DA3 unloaded")
    else:
        print("[Phase A] skipped")

    # ── Phase B: FoundationPose register (per-cam subprocess) ──
    import subprocess, tempfile

    REGISTER_SCRIPT = Path(__file__).parent / "overlay_register_one_cam.py"

    t_start = time.perf_counter()
    print(f"\n[Phase B] register per cam (subprocess) starting...", flush=True)
    for obj, eps in selected:
        try: mesh_path = _resolve_mesh(obj)
        except Exception as e:
            print(f"[skip {obj}] {e}", flush=True); continue

        for ep_name in eps:
            ep = EXP_ROOT/obj/ep_name
            ep_out = out_root / obj / ep_name
            depth_npz = ep_out / "depth.npz"
            if not depth_npz.exists():
                print(f"  [{ep_name}] missing depth.npz, skip", flush=True); continue
            try: images, masks, K, T = _load_episode(ep)
            except Exception as e:
                print(f"  [{ep_name}] load: {e}", flush=True); continue
            serials = sorted(images.keys())
            depth_data = np.load(depth_npz)

            print(f"\n[{obj}/{ep_name}] {len(serials)} cams to register", flush=True)
            t0 = time.perf_counter()
            poses = {}
            tmp_dir = tempfile.mkdtemp(prefix=f"refpre_{obj}_{ep_name}_")

            for s in serials:
                print(f"    [{s}] registering ...", flush=True)
                key = f"d_{s}"
                if key not in depth_data.files: continue
                if int(masks[s].sum()) < 100: continue
                # Dump inputs to disk for subprocess
                img_p = ep / "images" / f"{s}.png"
                mask_p = ep / "_pipeline_tmp" / "masks" / f"{s}.png"
                d_path = f"{tmp_dir}/depth_{s}.npy"
                K_path = f"{tmp_dir}/K_{s}.json"
                pose_out = f"{tmp_dir}/pose_{s}.npy"
                np.save(d_path, np.asarray(depth_data[key], np.float32))
                json.dump(K[s].tolist(), open(K_path, "w"))

                cmd = [
                    sys.executable, str(REGISTER_SCRIPT),
                    "--mesh", str(mesh_path),
                    "--image-path", str(img_p),
                    "--depth-path", d_path,
                    "--mask-path", str(mask_p),
                    "--K-json", K_path,
                    "--out", pose_out,
                ]
                try:
                    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                    if res.returncode != 0:
                        print(f"    [{s}] register failed (rc={res.returncode}): "
                              f"{res.stderr.strip().splitlines()[-1] if res.stderr else ''}")
                        continue
                    pose_cam = np.load(pose_out)
                    poses[s] = np.linalg.inv(T[s]) @ pose_cam
                except subprocess.TimeoutExpired:
                    print(f"    [{s}] register timeout")
                    continue

            # Cleanup tmp
            import shutil; shutil.rmtree(tmp_dir, ignore_errors=True)

            elapsed = time.perf_counter() - t0
            np.savez(ep_out/"ref_pre.npz", **{f"cam_{s}": p for s, p in poses.items()})
            print(f"  [{ep_name}] {len(poses)}/{len(serials)} candidates ({elapsed:.1f}s) → ref_pre.npz")

    print(f"\n[done] Phase B {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
