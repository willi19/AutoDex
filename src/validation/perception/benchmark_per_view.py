#!/usr/bin/env python3
"""FoundationPose register 1-cam benchmark across 10 episodes.

Other stages are already measured in foundpose_init_compare's results.csv.
Only FoundationPose register-on-each-cam time is unmeasured for 24-cam case.

Median ×24 = full 24-cam pipeline estimate.
Run inside `foundationpose` env.
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
import statistics
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

MESH_ROOT = Path("/home/mingi/shared_data/AutoDex/object/paradex")
EXP_ROOT = Path("/home/mingi/shared_data/AutoDex/experiment/selected_100/allegro")
RESULTS_CSV = REPO_ROOT / "outputs/foundpose_init_compare/selected_100/results.csv"
ASSETS_ROOT = REPO_ROOT / "outputs/foundpose_assets"


def _resolve_mesh(obj):
    for sub in [MESH_ROOT/obj/"raw_mesh"/f"{obj}.obj",
                MESH_ROOT/obj/"processed_data"/"mesh"/"raw.obj",
                MESH_ROOT/obj/"processed_data"/"mesh"/"simplified.obj"]:
        if sub.exists(): return sub
    raise FileNotFoundError(obj)


def _load_one_view(ep_dir):
    img_dir = ep_dir/"images"
    intr = json.load(open(ep_dir/"cam_param"/"intrinsics.json"))
    extr = json.load(open(ep_dir/"cam_param"/"extrinsics.json"))
    serials = sorted([p.stem for p in img_dir.glob("*.png")
                       if p.stem in intr and p.stem in extr])
    s = serials[0]
    bgr = cv2.imread(str(img_dir/f"{s}.png"))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    K = np.asarray(intr[s]["intrinsics_undistort"], np.float32)
    e = np.asarray(extr[s], np.float64)
    if e.shape == (3, 4): e = np.vstack([e, [0, 0, 0, 1]])
    T = e
    mask_p = ep_dir/"_pipeline_tmp"/"masks"/f"{s}.png"
    mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
    mask_bool = mask > 127 if mask is not None else np.zeros((H, W), bool)
    return s, rgb, K, T, mask_bool, H, W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eps", type=int, default=10)
    ap.add_argument("--downscale", type=float, default=0.5)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(RESULTS_CSV)))
    seen_objs = set()
    eps = []
    for r in rows:
        if r["object"] not in seen_objs:
            eps.append((r["object"], r["episode"]))
            seen_objs.add(r["object"])
        if len(eps) >= args.n_eps:
            break
    print(f"benchmarking on {len(eps)} ep, 1 view each\n")

    import torch, gc
    from autodex.perception.pose import PoseTracker

    times = []
    for obj, ep_name in eps:
        ep_dir = EXP_ROOT/obj/ep_name
        s, rgb, K, T, mask_bool, H, W = _load_one_view(ep_dir)
        mesh_path = _resolve_mesh(obj)

        # Use cached depth from overlay_compute_ref_pre Phase A if available;
        # otherwise compute on the fly. Cached path:
        depth_npz = REPO_ROOT/"outputs/foundpose_overlay"/obj/ep_name/"depth.npz"
        if depth_npz.exists():
            depth_data = np.load(depth_npz)
            depth_np = np.asarray(depth_data[f"d_{s}"], dtype=np.float32)
            if depth_np.shape[:2] != (H, W):
                depth_np = cv2.resize(depth_np, (W, H), interpolation=cv2.INTER_NEAREST)
        else:
            print(f"  [{obj}/{ep_name}] no cached depth, skip"); continue

        # Match perception_daemon.py FPose handler exactly:
        nH, nW = int(H*args.downscale), int(W*args.downscale)
        rgb_ds = cv2.resize(rgb, (nW, nH))
        depth_ds = cv2.resize(depth_np, (nW, nH), interpolation=cv2.INTER_NEAREST)
        mask_ds = cv2.resize(mask_bool.astype(np.uint8), (nW, nH), interpolation=cv2.INTER_NEAREST)
        K_ds = K.copy(); K_ds[0, :] *= args.downscale; K_ds[1, :] *= args.downscale
        depth_ds[(depth_ds < 0.001) | (depth_ds >= 100)] = 0

        tracker = PoseTracker(str(mesh_path), device_id=0)
        # Skip warmup (OOM-prone, 5+ GiB single alloc); first call includes lazy init.
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tracker.init(rgb=rgb_ds, depth=depth_ds, mask=mask_ds,
                      K=K_ds.astype(np.float32), iteration=5)
        torch.cuda.synchronize()
        t_fpose = time.perf_counter() - t0
        times.append(t_fpose)
        del tracker; gc.collect(); torch.cuda.empty_cache()

        print(f"  [{obj}/{ep_name}] fpose={t_fpose*1000:5.0f}ms")

    if not times:
        print("\nno measurements")
        return

    med = statistics.median(times)
    mean = statistics.mean(times)
    print(f"\n=== FoundationPose register 1 cam, n={len(times)} ===")
    print(f"  median: {med*1000:.0f}ms (×24 = {med*24:.2f}s)")
    print(f"  mean:   {mean*1000:.0f}ms (×24 = {mean*24:.2f}s)")
    out_dir = REPO_ROOT / "outputs/foundpose_overlay"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark_per_view.json"
    json.dump({"n": len(times), "per_cam_seconds": times,
               "median_s": med, "mean_s": mean,
               "median_x24_s": med*24, "mean_x24_s": mean*24},
              open(out_path, "w"), indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
