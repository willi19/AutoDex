#!/usr/bin/env python3
"""Batch stereo depth estimation using FoundationStereo (TensorRT).

Reads stereo video pairs from local cache, camera params from network FS,
runs FoundationStereo TRT, and saves depth/{serial}.avi in cache.

Pre-download videos:
    python src/perception/download_videos.py --base ... --serials 22684755 23263780

Run depth:
    conda activate foundation_stereo
    python -u src/perception/batch_depth.py \
        --base /home/mingi/paradex1/capture/eccv2026/inspire_f1 \
        --left_serial 22684755 --right_serial 23263780

Upload results:
    python src/perception/upload_results.py --base ...
"""

import os
import gc
import time
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

CACHE_ROOT = os.path.expanduser("~/video_cache")
NETWORK_PREFIX = "/home/mingi/paradex1/capture"

_FS_ROOT = Path(__file__).resolve().parents[2] / "autodex/perception/thirdparty/FoundationStereo"
_DEFAULT_ENGINE = _FS_ROOT / "output/foundation_stereo_448x672.engine"


def _get_cache_base(base_dir):
    base = str(Path(base_dir).resolve())
    if base.startswith(NETWORK_PREFIX):
        rel = base[len(NETWORK_PREFIX):].lstrip("/")
    else:
        rel = Path(base_dir).name
    return os.path.join(CACHE_ROOT, rel)


# ── Camera / rectification ───────────────────────────────────────────────────

def load_cam_param(capture_dir: Path):
    """Load intrinsics (undistorted) and extrinsics keyed by serial string."""
    param_dir = capture_dir / "cam_param"
    with open(param_dir / "intrinsics.json") as f:
        intr_raw = json.load(f)
    with open(param_dir / "extrinsics.json") as f:
        extr_raw = json.load(f)

    intrinsics = {s: np.array(v["intrinsics_undistort"], dtype=np.float64) for s, v in intr_raw.items()}
    extrinsics = {s: np.array(extr_raw[s], dtype=np.float64) for s in intr_raw}
    return intrinsics, extrinsics


def build_rectify_maps(K_left, K_right, T_left, T_right, image_size):
    """Compute stereo rectification maps.

    Returns: (map_left, map_right, f_rect, cx, cy, baseline)
    """
    W, H = image_size

    def to_4x4(T):
        if T.shape == (3, 4):
            T4 = np.eye(4, dtype=np.float64)
            T4[:3, :] = T
            return T4
        return T.astype(np.float64)

    T_l = to_4x4(T_left)
    T_r = to_4x4(T_right)
    T_rel = T_r @ np.linalg.inv(T_l)

    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
        K_left, None, K_right, None, (W, H),
        T_rel[:3, :3], T_rel[:3, 3],
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )

    f_rect = float(P1[0, 0])
    cx = float(P1[0, 2])
    cy = float(P1[1, 2])
    baseline = abs(float(P2[0, 3])) / (f_rect + 1e-9)

    map_left = cv2.initUndistortRectifyMap(K_left, None, R1, P1, (W, H), cv2.CV_32FC1)
    map_right = cv2.initUndistortRectifyMap(K_right, None, R2, P2, (W, H), cv2.CV_32FC1)

    return map_left, map_right, f_rect, cx, cy, baseline


# ── Depth encode ─────────────────────────────────────────────────────────────

def encode_depth_uint16(depth: np.ndarray) -> np.ndarray:
    """Encode depth (m) -> BGR uint8 for lossless video (FFV1).

    B = low byte, G = high byte of uint16 millimeters. R = 0.
    """
    depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
    low = (depth_mm & 0xFF).astype(np.uint8)
    high = (depth_mm >> 8).astype(np.uint8)
    return np.stack([low, high, np.zeros_like(low)], axis=-1)


# ── TRT engine ───────────────────────────────────────────────────────────────

def load_trt_engine(engine_path: str):
    """Load TRT engine + allocate GPU buffers. Returns (context, buffers, shape)."""
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    trt_shape = engine.get_tensor_shape("left")  # (1, 3, H_trt, W_trt)
    H_trt, W_trt = int(trt_shape[2]), int(trt_shape[3])

    disp_arr = np.zeros((1, 1, H_trt, W_trt), dtype=np.float32)
    d_left = cuda.mem_alloc(int(np.prod([1, 3, H_trt, W_trt])) * 4)
    d_right = cuda.mem_alloc(int(np.prod([1, 3, H_trt, W_trt])) * 4)
    d_disp = cuda.mem_alloc(disp_arr.nbytes)
    stream = cuda.Stream()

    buffers = {
        "d_left": d_left, "d_right": d_right, "d_disp": d_disp,
        "disp_arr": disp_arr, "stream": stream,
    }
    return context, buffers, (H_trt, W_trt)


def run_trt_inference(context, buffers, left_rgb, right_rgb, H_trt, W_trt):
    """Run TRT on one rectified RGB pair. Returns disparity at TRT resolution."""
    import pycuda.driver as cuda

    left_trt = cv2.resize(left_rgb, (W_trt, H_trt), interpolation=cv2.INTER_LINEAR)
    right_trt = cv2.resize(right_rgb, (W_trt, H_trt), interpolation=cv2.INTER_LINEAR)
    left_arr = np.ascontiguousarray(left_trt.astype(np.float32).transpose(2, 0, 1)[None])
    right_arr = np.ascontiguousarray(right_trt.astype(np.float32).transpose(2, 0, 1)[None])

    stream = buffers["stream"]
    cuda.memcpy_htod_async(buffers["d_left"], left_arr, stream)
    cuda.memcpy_htod_async(buffers["d_right"], right_arr, stream)
    context.set_tensor_address("left", int(buffers["d_left"]))
    context.set_tensor_address("right", int(buffers["d_right"]))
    context.set_tensor_address("disp", int(buffers["d_disp"]))
    context.execute_async_v3(stream.handle)
    stream.synchronize()
    cuda.memcpy_dtoh(buffers["disp_arr"], buffers["d_disp"])

    return buffers["disp_arr"].squeeze()  # (H_trt, W_trt)


def disp_to_depth(disp_trt, f_rect, baseline, W_orig, H_orig, W_trt):
    """Convert TRT disparity to depth at original resolution."""
    f_trt = f_rect * W_trt / W_orig
    valid = disp_trt >= 0.5
    depth_trt = np.zeros_like(disp_trt, dtype=np.float32)
    depth_trt[valid] = f_trt * baseline / np.maximum(disp_trt[valid], 0.1)
    depth = cv2.resize(depth_trt, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
    return depth


# ── Task collection ──────────────────────────────────────────────────────────

def collect_tasks(base_dir, left_serial, right_serial):
    """Collect tasks from local cache — stereo pairs without depth output."""
    cache_base = Path(_get_cache_base(base_dir))
    if not cache_base.is_dir():
        return []
    tasks = []
    for obj_dir in sorted(cache_base.iterdir()):
        if not obj_dir.is_dir():
            continue
        for idx_dir in sorted(obj_dir.iterdir()):
            if not idx_dir.is_dir():
                continue
            video_dir = idx_dir / "videos"
            if not video_dir.is_dir():
                continue
            left_path = video_dir / f"{left_serial}.avi"
            right_path = video_dir / f"{right_serial}.avi"
            if not left_path.exists() or not right_path.exists():
                continue
            # Skip if both depths already exist with correct frame count
            n_src = int(cv2.VideoCapture(str(left_path)).get(cv2.CAP_PROP_FRAME_COUNT))
            depth_left = idx_dir / "depth" / f"{left_serial}.avi"
            depth_right = idx_dir / "depth" / f"{right_serial}.avi"
            has_left = depth_left.exists()
            has_right = depth_right.exists()
            # Delete truncated depth files
            if has_left:
                n_dl = int(cv2.VideoCapture(str(depth_left)).get(cv2.CAP_PROP_FRAME_COUNT))
                if n_dl < n_src:
                    depth_left.unlink()
                    has_left = False
            if has_right:
                n_dr = int(cv2.VideoCapture(str(depth_right)).get(cv2.CAP_PROP_FRAME_COUNT))
                if n_dr < n_src:
                    depth_right.unlink()
                    has_right = False
            if has_left and has_right:
                continue
            # Need cam_param from network FS
            rel = str(idx_dir.relative_to(cache_base))
            net_dir = Path(base_dir) / rel
            if not (net_dir / "cam_param").is_dir():
                continue
            tasks.append((str(left_path), str(right_path), str(idx_dir), str(net_dir),
                          obj_dir.name, idx_dir.name))
    return tasks


# ── Process one capture ──────────────────────────────────────────────────────

def process_one_capture(trt_context, trt_buffers, trt_shape,
                        left_path, right_path, cache_dir, net_dir,
                        left_serial, right_serial):
    """Process one stereo pair. Saves depth/{left_serial}.avi and depth/{right_serial}.avi."""
    intrinsics, extrinsics = load_cam_param(Path(net_dir))

    K_left = intrinsics[left_serial]
    K_right = intrinsics[right_serial]
    T_left = extrinsics[left_serial]
    T_right = extrinsics[right_serial]

    cap_l = cv2.VideoCapture(left_path)
    cap_r = cv2.VideoCapture(right_path)

    n_frames = int(cap_l.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap_l.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  {W}x{H}, {n_frames} frames, {fps:.1f} fps", flush=True)

    map_left, map_right, f_rect, _, _, baseline = build_rectify_maps(
        K_left, K_right, T_left, T_right, (W, H)
    )
    print(f"  f={f_rect:.1f}px  baseline={baseline:.4f}m", flush=True)

    if baseline < 0.01 or f_rect > 20000:
        cap_l.release()
        cap_r.release()
        print("  SKIP degenerate rectification", flush=True)
        return

    H_trt, W_trt = trt_shape

    depth_dir = os.path.join(cache_dir, "depth")
    os.makedirs(depth_dir, exist_ok=True)

    left_out = os.path.join(depth_dir, f"{left_serial}.avi")
    right_out = os.path.join(depth_dir, f"{right_serial}.avi")
    need_left = not os.path.exists(left_out)
    need_right = not os.path.exists(right_out)

    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    writer_l = cv2.VideoWriter(left_out, fourcc, fps, (W, H)) if need_left else None
    writer_r = cv2.VideoWriter(right_out, fourcc, fps, (W, H)) if need_right else None

    for idx in tqdm(range(n_frames), desc="  depth", unit="f"):
        ret_l, bgr_l = cap_l.read()
        ret_r, bgr_r = cap_r.read()
        if not ret_l or not ret_r:
            break

        rgb_l = cv2.cvtColor(bgr_l, cv2.COLOR_BGR2RGB)
        rgb_r = cv2.cvtColor(bgr_r, cv2.COLOR_BGR2RGB)
        left_rect = cv2.remap(rgb_l, map_left[0], map_left[1], cv2.INTER_LINEAR)
        right_rect = cv2.remap(rgb_r, map_right[0], map_right[1], cv2.INTER_LINEAR)

        # Left depth: TRT(left, right)
        if writer_l is not None:
            disp_l = run_trt_inference(trt_context, trt_buffers, left_rect, right_rect, H_trt, W_trt)
            depth_l = disp_to_depth(disp_l, f_rect, baseline, W, H, W_trt)
            writer_l.write(encode_depth_uint16(depth_l))

        # Right depth: TRT(right, left) — swap inputs
        if writer_r is not None:
            disp_r = run_trt_inference(trt_context, trt_buffers, right_rect, left_rect, H_trt, W_trt)
            depth_r = disp_to_depth(disp_r, f_rect, baseline, W, H, W_trt)
            writer_r.write(encode_depth_uint16(depth_r))

    cap_l.release()
    cap_r.release()
    if writer_l is not None:
        writer_l.release()
        print(f"  Saved: {left_out}", flush=True)
    if writer_r is not None:
        writer_r.release()
        print(f"  Saved: {right_out}", flush=True)
    print("  Done.", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base dir (network FS path)")
    parser.add_argument("--left_serial", required=True)
    parser.add_argument("--right_serial", required=True)
    parser.add_argument("--engine", type=str, default=str(_DEFAULT_ENGINE),
                        help="Path to TRT engine file")
    args = parser.parse_args()
    print("Starting batch depth estimation (TRT)...", flush=True)

    tasks = collect_tasks(args.base, args.left_serial, args.right_serial)
    if not tasks:
        print("Nothing to do (all captures already have depth).")
        return

    print(f"{len(tasks)} captures to process", flush=True)

    print(f"Loading TRT engine: {args.engine}", flush=True)
    t0 = time.time()
    trt_context, trt_buffers, trt_shape = load_trt_engine(args.engine)
    print(f"TRT ready: {trt_shape[1]}x{trt_shape[0]}  ({time.time() - t0:.1f}s)", flush=True)

    done = 0
    total = len(tasks)
    for left_path, right_path, cache_dir, net_dir, obj_name, idx_name in tasks:
        done += 1
        print(f"[{done}/{total}] {obj_name}/{idx_name}", flush=True)
        try:
            process_one_capture(
                trt_context, trt_buffers, trt_shape,
                left_path, right_path, cache_dir, net_dir,
                args.left_serial, args.right_serial,
            )
        except Exception as e:
            import traceback
            print(f"  Error: {e}", flush=True)
            traceback.print_exc()

    print(f"All done! {done}/{total} captures processed.", flush=True)


if __name__ == "__main__":
    main()
