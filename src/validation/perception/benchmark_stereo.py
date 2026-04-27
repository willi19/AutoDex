#!/usr/bin/env python3
"""FoundationStereo TRT benchmark per stereo pair, across 10 episodes.

Runs the same per-pair pipeline that batch_depth.py / batch_depth_auto.py uses:
  rectify maps + TRT disparity inference (no R1 un-rectify, no PLY).
Times the TRT inference itself (excludes file IO).

Run inside `foundation_stereo` env (tensorrt + pycuda available).
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

EXP_ROOT = Path("/home/mingi/shared_data/AutoDex/experiment/selected_100/allegro")
RESULTS_CSV = REPO_ROOT / "outputs/foundpose_init_compare/selected_100/results.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eps", type=int, default=10)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(RESULTS_CSV)))
    seen = set(); eps = []
    for r in rows:
        if r["object"] not in seen:
            eps.append((r["object"], r["episode"]))
            seen.add(r["object"])
        if len(eps) >= args.n_eps:
            break
    print(f"benchmarking FoundationStereo on {len(eps)} ep, 1 pair each\n")

    from autodex.perception.depth import StereoDepthTRT, find_all_stereo_pairs, build_rectify_maps
    print("loading TRT engine ...")
    trt = StereoDepthTRT()
    print("TRT loaded\n")

    times_infer = []
    times_total = []  # rectify + infer

    for obj, ep_name in eps:
        ep_dir = EXP_ROOT / obj / ep_name
        # Match PerceptionPipeline.run() loading: K = float32 from undistort, T = 4x4
        with open(ep_dir/"cam_param"/"intrinsics.json") as f:
            intr_raw = json.load(f)
        with open(ep_dir/"cam_param"/"extrinsics.json") as f:
            extr_raw = json.load(f)
        serials = sorted(p.stem for p in (ep_dir/"images").glob("*.png"))
        intrinsics = {s: np.array(intr_raw[s]["intrinsics_undistort"], np.float32) for s in serials if s in intr_raw}
        extrinsics = {}
        for s in serials:
            if s not in extr_raw: continue
            T = np.array(extr_raw[s], np.float64)
            if T.shape == (3, 4): T = np.vstack([T, [0, 0, 0, 1]])
            extrinsics[s] = T

        # find_all_stereo_pairs(capture_dir, serials, intrinsics, extrinsics) — capture_dir first
        try:
            pairs = find_all_stereo_pairs(ep_dir, serials, intrinsics, extrinsics)
        except Exception as exc:
            print(f"  [{obj}/{ep_name}] pair selection fail: {exc}"); continue
        if not pairs:
            print(f"  [{obj}/{ep_name}] no pairs"); continue

        # pair tuple is (left, right, baseline_m)
        left_s, right_s = pairs[0][0], pairs[0][1]
        left = cv2.imread(str(ep_dir/"images"/f"{left_s}.png"))
        right = cv2.imread(str(ep_dir/"images"/f"{right_s}.png"))
        left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
        K_l, K_r = intrinsics[left_s], intrinsics[right_s]
        T_l, T_r = extrinsics[left_s], extrinsics[right_s]
        H, W = left_rgb.shape[:2]

        t0 = time.perf_counter()
        # build_rectify_maps takes (W, H) and returns tuple
        result = build_rectify_maps(K_l, K_r, T_l, T_r, (W, H))
        if result is None:
            print(f"  [{obj}/{ep_name}] rectify fail"); continue
        map_left, map_right = result[0], result[1]
        left_rect = cv2.remap(left_rgb, map_left[0], map_left[1], cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_rgb, map_right[0], map_right[1], cv2.INTER_LINEAR)
        t_rectify = time.perf_counter() - t0

        t0 = time.perf_counter()
        disp = trt._run_trt(left_rect, right_rect)
        t_infer = time.perf_counter() - t0

        times_infer.append(t_infer)
        times_total.append(t_rectify + t_infer)

        print(f"  [{obj}/{ep_name}] pair=({left_s},{right_s})  rectify={t_rectify*1000:5.0f}ms  infer={t_infer*1000:5.0f}ms")

    if not times_infer:
        print("\nno measurements")
        return

    med_infer = statistics.median(times_infer)
    med_total = statistics.median(times_total)
    print(f"\n=== Median across {len(times_infer)} ep ===")
    print(f"  TRT infer only:     {med_infer*1000:5.0f}ms / pair")
    print(f"  rectify + infer:    {med_total*1000:5.0f}ms / pair")
    print(f"\n(For 24 cams via 12 pairs: total ≈ {med_total*12:.2f}s)")
    out_dir = REPO_ROOT / "outputs/foundpose_overlay"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark_stereo.json"
    json.dump({"n": len(times_infer),
               "infer_seconds_per_pair": times_infer,
               "total_seconds_per_pair": times_total,
               "median_infer_s": med_infer,
               "median_total_s": med_total,
               "median_total_12pair_s": med_total * 12},
              open(out_path, "w"), indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
