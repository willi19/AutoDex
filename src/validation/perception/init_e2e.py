#!/usr/bin/env python3
"""End-to-end test for distributed FoundPose first-frame init.

Run this on the robot PC. Capture PCs (capture1-6) must have init_daemon
running in `gotrack_cu128`:

    # On each capture PC:
    conda activate gotrack_cu128
    python src/execution/daemon/init_daemon.py \\
        --port-mask 5006 --port-pose 5007 --port-cmd 6893

Then on robot PC:

    python src/execution/run_init_test.py --obj attached_container \\
        --prompt "object on the checkerboard"

Optional: --n-trials 3 to repeat for latency stats.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_PC_LIST = ["capture1", "capture2", "capture3", "capture4", "capture5", "capture6"]
DEFAULT_PC_IPS = {
    "capture1": "192.168.0.101", "capture2": "192.168.0.102",
    "capture3": "192.168.0.103", "capture4": "192.168.0.104",
    "capture5": "192.168.0.105", "capture6": "192.168.0.106",
}


def _latest_calib_dir() -> Path:
    base = Path.home() / "shared_data/cam_param"
    candidates = sorted(d for d in base.iterdir() if d.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No calibration found under {base}")
    return candidates[-1]


def _load_calib(calib_dir: Path):
    with open(calib_dir / "intrinsics.json") as f:
        intr_raw = json.load(f)
    with open(calib_dir / "extrinsics.json") as f:
        extr_raw = json.load(f)

    intrinsics_full = {}
    extrinsics_full = {}
    for s, d in intr_raw.items():
        K_orig = np.asarray(d["original_intrinsics"], dtype=np.float64).reshape(3, 3)
        K_undist = np.asarray(d["intrinsics_undistort"], dtype=np.float64).reshape(3, 3)
        dist = np.asarray(d["dist_params"], dtype=np.float64).reshape(-1)
        intrinsics_full[s] = {
            "K_orig": K_orig, "K_undist": K_undist, "dist_params": dist,
            "width": int(d["width"]), "height": int(d["height"]),
        }
    for s, ext in extr_raw.items():
        a = np.asarray(ext, dtype=np.float64).reshape(-1)
        if a.size == 12:
            a = np.vstack([a.reshape(3, 4), [0, 0, 0, 1]])
        else:
            a = a.reshape(4, 4)
        extrinsics_full[s] = a
    return intrinsics_full, extrinsics_full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="object on the checkerboard")
    parser.add_argument("--n-trials", type=int, default=1)
    parser.add_argument("--mesh-base", type=str,
                        default=str(Path.home() / "shared_data/AutoDex/object/paradex"))
    parser.add_argument("--assets-base", type=str,
                        default=str(Path.home() / "shared_data/AutoDex/foundpose_assets"))
    parser.add_argument("--calib-dir", type=str, default=None)
    parser.add_argument("--pc-list", type=str, nargs="+", default=DEFAULT_PC_LIST)
    parser.add_argument("--pc-ips", type=str, nargs="+", default=None,
                        help="One IP per pc-list entry; default uses internal map.")
    parser.add_argument("--port-mask", type=int, default=5006)
    parser.add_argument("--port-pose", type=int, default=5007)
    parser.add_argument("--port-cmd", type=int, default=6893)
    parser.add_argument("--sil-iters", type=int, default=100)
    parser.add_argument("--sil-lr", type=float, default=0.002)
    parser.add_argument("--out-pose", type=str, default=None,
                        help="Path to save final pose .npy.")
    args = parser.parse_args()

    if args.pc_ips is None:
        args.pc_ips = [DEFAULT_PC_IPS[p] for p in args.pc_list]

    mesh_path = Path(args.mesh_base) / args.obj / "raw_mesh" / f"{args.obj}.obj"
    assets_root = Path(args.assets_base) / args.obj
    if not mesh_path.exists():
        raise FileNotFoundError(f"mesh not found: {mesh_path}")
    if not (assets_root / "object_repre/v1" / args.obj / "1/repre.pth").exists():
        raise FileNotFoundError(
            f"FoundPose repre missing for {args.obj}: "
            f"{assets_root}/object_repre/v1/{args.obj}/1/repre.pth")

    calib_dir = Path(args.calib_dir) if args.calib_dir else _latest_calib_dir()
    logger.info(f"calib: {calib_dir}")
    intrinsics_full, extrinsics_full = _load_calib(calib_dir)
    H = next(iter(intrinsics_full.values()))["height"]
    W = next(iter(intrinsics_full.values()))["width"]
    logger.info(f"{len(intrinsics_full)} cameras, image {H}x{W}")

    from autodex.perception.init_orchestrator import InitOrchestrator

    orch = InitOrchestrator(
        pc_list=args.pc_list,
        capture_ips=args.pc_ips,
        port_mask=args.port_mask, port_pose=args.port_pose, port_cmd=args.port_cmd,
    )

    try:
        t0 = time.perf_counter()
        orch.init_object(
            obj_name=args.obj,
            mesh_path=str(mesh_path),
            assets_root=str(assets_root),
            intrinsics_full=intrinsics_full,
            extrinsics_full=extrinsics_full,
            image_hw=(H, W),
        )
        logger.info(f"init_object took {time.perf_counter()-t0:.1f}s "
                    f"(loads SAM3+FoundPose on each capture PC; first time is slow)")

        # Trials
        latencies = []
        for trial in range(args.n_trials):
            print(f"\n========== trial {trial+1}/{args.n_trials} ==========")
            pose, timing = orch.trigger_init(
                prompt=args.prompt,
                sil_iters=args.sil_iters, sil_lr=args.sil_lr,
            )
            print(json.dumps(timing, indent=2, default=str))
            if pose is not None:
                latencies.append(timing.get("total_s", -1))
                if args.out_pose:
                    out = Path(args.out_pose)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    np.save(out, pose)
                    logger.info(f"saved pose -> {out}")
            else:
                logger.warning("init failed: %s", timing.get("reason"))

        if latencies:
            arr = np.array(latencies)
            print(f"\n=== latency over {len(arr)} trials: "
                  f"median {np.median(arr):.2f}s  mean {arr.mean():.2f}s  "
                  f"min {arr.min():.2f}s  max {arr.max():.2f}s ===")
    finally:
        orch.close()


if __name__ == "__main__":
    main()
