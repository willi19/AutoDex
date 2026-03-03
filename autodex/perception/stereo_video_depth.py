"""Stereo depth from video pairs using FoundationStereo (TensorRT).

Iterates over all {obj_name}/{idx}/ directories under --base, stereo-rectifies
each frame pair, runs FoundationStereo TRT, and saves depth/{left_serial}.avi.

Videos are assumed to be already undistorted (uses intrinsics_undistort).
Input is resized to TRT engine resolution (448x672), disparity is scaled back.

Usage:
    python -m autodex.perception.stereo_video_depth \
        --base /home/mingi/paradex1/capture/eccv2026/inspire_f1 \
        --left_serial 22641005 \
        --right_serial 22641023
"""

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

_THIS_DIR = Path(__file__).parent
_FS_ROOT = _THIS_DIR / "thirdparty/FoundationStereo"
_DEFAULT_ENGINE = _FS_ROOT / "output/foundation_stereo_448x672.engine"

logging.basicConfig(level=logging.INFO, format="[stereo_depth] %(message)s")


# ── Camera / rectification ────────────────────────────────────────────────────

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


# ── TRT engine ────────────────────────────────────────────────────────────────

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


# ── Depth encode/decode ───────────────────────────────────────────────────────

def encode_depth_uint16(depth: np.ndarray) -> np.ndarray:
    """Encode depth (m) → BGR uint8 for lossless video (FFV1).

    B = low byte, G = high byte of uint16 millimeters. R = 0.
    """
    depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
    low = (depth_mm & 0xFF).astype(np.uint8)
    high = (depth_mm >> 8).astype(np.uint8)
    return np.stack([low, high, np.zeros_like(low)], axis=-1)


def decode_depth_uint16(bgr: np.ndarray) -> np.ndarray:
    """Decode BGR frame back to depth in meters."""
    depth_mm = bgr[:, :, 1].astype(np.uint16) * 256 + bgr[:, :, 0].astype(np.uint16)
    return depth_mm.astype(np.float32) / 1000.0


# ── Per-capture processing ────────────────────────────────────────────────────

def process_one_capture(
    capture_dir: str,
    left_serial: str,
    right_serial: str,
    trt_context,
    trt_buffers,
    trt_shape: tuple[int, int],
):
    """Process one capture directory. Saves depth/{left_serial}.avi under capture_dir."""
    capture_dir = Path(capture_dir)
    intrinsics, extrinsics = load_cam_param(capture_dir)

    K_left = intrinsics[left_serial]
    K_right = intrinsics[right_serial]
    T_left = extrinsics[left_serial]
    T_right = extrinsics[right_serial]

    # Open videos
    cap_l = cv2.VideoCapture(str(capture_dir / "videos" / f"{left_serial}.avi"))
    cap_r = cv2.VideoCapture(str(capture_dir / "videos" / f"{right_serial}.avi"))

    n_frames = int(cap_l.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap_l.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logging.info(f"  {W}x{H}, {n_frames} frames, {fps:.1f} fps")

    # Build rectification maps
    map_left, map_right, f_rect, _, _, baseline = build_rectify_maps(
        K_left, K_right, T_left, T_right, (W, H)
    )
    logging.info(f"  f={f_rect:.1f}px  baseline={baseline:.4f}m")

    if baseline < 0.01 or f_rect > 20000:
        cap_l.release()
        cap_r.release()
        logging.warning("  SKIP degenerate rectification")
        return

    H_trt, W_trt = trt_shape
    f_trt = f_rect * W_trt / W

    # Output writer — save at original resolution
    depth_dir = capture_dir / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)
    out_path = depth_dir / f"{left_serial}.avi"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (W, H))

    for idx in range(n_frames):
        ret_l, bgr_l = cap_l.read()
        ret_r, bgr_r = cap_r.read()
        if not ret_l or not ret_r:
            break

        rgb_l = cv2.cvtColor(bgr_l, cv2.COLOR_BGR2RGB)
        rgb_r = cv2.cvtColor(bgr_r, cv2.COLOR_BGR2RGB)
        left_rect = cv2.remap(rgb_l, map_left[0], map_left[1], cv2.INTER_LINEAR)
        right_rect = cv2.remap(rgb_r, map_right[0], map_right[1], cv2.INTER_LINEAR)

        t1 = time.perf_counter()
        disp_trt = run_trt_inference(trt_context, trt_buffers, left_rect, right_rect, H_trt, W_trt)

        # Disparity → depth at TRT resolution, then resize to original
        valid = disp_trt >= 0.5
        depth_trt = np.zeros_like(disp_trt, dtype=np.float32)
        depth_trt[valid] = f_trt * baseline / np.maximum(disp_trt[valid], 0.1)
        depth = cv2.resize(depth_trt, (W, H), interpolation=cv2.INTER_LINEAR)
        dt = time.perf_counter() - t1

        writer.write(encode_depth_uint16(depth))

        if idx % 30 == 0:
            pos = depth[depth > 0]
            if len(pos) > 0:
                logging.info(f"  Frame {idx}/{n_frames}: [{pos.min():.3f}, {pos.max():.3f}]m  ({dt:.2f}s)")

    cap_l.release()
    cap_r.release()
    writer.release()
    logging.info(f"  Saved: {out_path}")


# ── Batch main ────────────────────────────────────────────────────────────────

def discover_dirs(base: Path):
    """Find all {obj_name}/{idx} dirs with videos/ and cam_param/."""
    dirs = []
    for obj_dir in sorted(base.iterdir()):
        if not obj_dir.is_dir():
            continue
        for idx_dir in sorted(obj_dir.iterdir()):
            if not idx_dir.is_dir():
                continue
            if (idx_dir / "videos").is_dir() and (idx_dir / "cam_param").is_dir():
                dirs.append(idx_dir)
    return dirs


def main():
    parser = argparse.ArgumentParser(description="Stereo depth (TRT) for all capture dirs")
    parser.add_argument("--base", type=str,
                        default="/home/mingi/paradex1/capture/eccv2026/inspire_f1",
                        help="Base directory containing {obj_name}/{idx}/ subdirs")
    parser.add_argument("--left_serial", type=str, required=True)
    parser.add_argument("--right_serial", type=str, required=True)
    parser.add_argument("--engine", type=str, default=str(_DEFAULT_ENGINE),
                        help="Path to TRT engine file")
    args = parser.parse_args()

    # Load TRT engine once
    t0 = time.perf_counter()
    trt_context, trt_buffers, trt_shape = load_trt_engine(args.engine)
    logging.info(f"TRT engine loaded: {trt_shape[1]}x{trt_shape[0]}  ({time.perf_counter() - t0:.2f}s)")

    base = Path(args.base)
    dirs = discover_dirs(base)
    logging.info(f"Found {len(dirs)} capture directories under {base}")

    for i, capture_dir in enumerate(dirs):
        out_path = capture_dir / "depth" / f"{args.left_serial}.avi"
        if out_path.exists():
            logging.info(f"[{i+1}/{len(dirs)}] SKIP {capture_dir.relative_to(base)} (already done)")
            continue

        logging.info(f"[{i+1}/{len(dirs)}] {capture_dir.relative_to(base)}")
        try:
            process_one_capture(
                capture_dir=str(capture_dir),
                left_serial=args.left_serial,
                right_serial=args.right_serial,
                trt_context=trt_context,
                trt_buffers=trt_buffers,
                trt_shape=trt_shape,
            )
        except Exception as e:
            logging.error(f"  FAILED: {e}")
            continue


if __name__ == "__main__":
    main()
