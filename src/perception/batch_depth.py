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

    Returns: (map_left, map_right, R1, R2, f_rect, cx, cy, baseline)
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

    return map_left, map_right, R1, R2, f_rect, cx, cy, baseline


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


def disp_to_depth_aligned(disp_trt, f_rect, cx_rect, cy_rect, baseline,
                          R_rect, K_orig, W_orig, H_orig, H_trt, W_trt):
    """Convert TRT disparity to depth aligned with original (unrectified) camera.

    Follows step2_depth.py:
      1. Backproject disparity to 3D in rectified frame (at TRT resolution)
      2. Rotate from rectified to original camera frame (R_rect.T)
      3. Project to depth map using original camera intrinsics
    """
    # Non-uniform resize: fx and fy scale differently
    fx_trt = f_rect * W_trt / W_orig
    fy_trt = f_rect * H_trt / H_orig
    cx_trt = cx_rect * W_trt / W_orig
    cy_trt = cy_rect * H_trt / H_orig

    valid = disp_trt >= 0.5
    disp_v = np.maximum(disp_trt[valid], 0.1)
    depth_v = fx_trt * baseline / disp_v  # disparity is horizontal → use fx

    u_g, v_g = np.meshgrid(np.arange(W_trt, dtype=np.float32),
                            np.arange(H_trt, dtype=np.float32))
    pts_rect = np.stack([
        (u_g[valid] - cx_trt) * depth_v / fx_trt,
        (v_g[valid] - cy_trt) * depth_v / fy_trt,
        depth_v,
    ], axis=1)

    # Rectified → original camera frame (R_rect: orig→rect, so R_rect.T: rect→orig)
    pts_orig = (R_rect.T @ pts_rect.T).T

    # Project to depth map using original intrinsics
    in_front = pts_orig[:, 2] > 0.01
    pts_v = pts_orig[in_front]

    fx, fy = K_orig[0, 0], K_orig[1, 1]
    cx_o, cy_o = K_orig[0, 2], K_orig[1, 2]
    px = (fx * pts_v[:, 0] / pts_v[:, 2] + cx_o).astype(int)
    py = (fy * pts_v[:, 1] / pts_v[:, 2] + cy_o).astype(int)
    in_img = (px >= 0) & (px < W_orig) & (py >= 0) & (py < H_orig)

    depth_map = np.zeros((H_orig, W_orig), dtype=np.float32)
    z_vals = pts_v[in_img, 2]
    order = np.argsort(z_vals)[::-1]  # far → near, so near overwrites
    depth_map[py[in_img][order], px[in_img][order]] = z_vals[order]

    return depth_map


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

    n_left = int(cap_l.get(cv2.CAP_PROP_FRAME_COUNT))
    n_right = int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(n_left, n_right)
    fps = cap_l.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if n_left != n_right:
        print(f"  {W}x{H}, left={n_left} right={n_right} -> using {n_frames}, {fps:.1f} fps", flush=True)
    else:
        print(f"  {W}x{H}, {n_frames} frames, {fps:.1f} fps", flush=True)

    map_left, map_right, R1, R2, f_rect, cx_rect, cy_rect, baseline = build_rectify_maps(
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

    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    writer_l = cv2.VideoWriter(left_out, fourcc, fps, (W, H))
    writer_r = cv2.VideoWriter(right_out, fourcc, fps, (W, H))

    for idx in tqdm(range(n_frames), desc="  depth", unit="f"):
        ret_l, bgr_l = cap_l.read()
        ret_r, bgr_r = cap_r.read()
        if not ret_l or not ret_r:
            break

        rgb_l = cv2.cvtColor(bgr_l, cv2.COLOR_BGR2RGB)
        rgb_r = cv2.cvtColor(bgr_r, cv2.COLOR_BGR2RGB)
        left_rect = cv2.remap(rgb_l, map_left[0], map_left[1], cv2.INTER_LINEAR)
        right_rect = cv2.remap(rgb_r, map_right[0], map_right[1], cv2.INTER_LINEAR)

        # Left depth: TRT(left, right) → aligned to original left camera
        disp_l = run_trt_inference(trt_context, trt_buffers, left_rect, right_rect, H_trt, W_trt)
        depth_l = disp_to_depth_aligned(disp_l, f_rect, cx_rect, cy_rect, baseline,
                                        R1, K_left, W, H, H_trt, W_trt)
        writer_l.write(encode_depth_uint16(depth_l))

        # Right depth: TRT(right, left) → aligned to original right camera
        disp_r = run_trt_inference(trt_context, trt_buffers, right_rect, left_rect, H_trt, W_trt)
        depth_r = disp_to_depth_aligned(disp_r, f_rect, cx_rect, cy_rect, baseline,
                                        R2, K_right, W, H, H_trt, W_trt)
        writer_r.write(encode_depth_uint16(depth_r))

        # Overlay on second frame (idx==1) so we can check early
        if idx == 1:
            for serial, depth_arr, bgr in [
                (left_serial, depth_l, bgr_l),
                (right_serial, depth_r, bgr_r),
            ]:
                save_depth_overlay(cache_dir, net_dir, serial,
                                   intrinsics, extrinsics,
                                   depth=depth_arr, src_bgr=bgr)

    cap_l.release()
    cap_r.release()
    writer_l.release()
    writer_r.release()
    print(f"  Saved: {left_out}", flush=True)
    print(f"  Saved: {right_out}", flush=True)


# ── Cross-view depth overlay ────────────────────────────────────────────────

def _read_frame_depth(depth_video_path, frame_idx=1):
    """Read specific frame of depth video, decode to float32 meters."""
    cap = cv2.VideoCapture(depth_video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, bgr = cap.read()
    cap.release()
    if not ret:
        return None
    depth_mm = bgr[:, :, 1].astype(np.uint16) * 256 + bgr[:, :, 0].astype(np.uint16)
    return depth_mm.astype(np.float32) / 1000.0


def _read_frame_rgb(video_path, frame_idx=1):
    """Read specific frame of RGB video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, bgr = cap.read()
    cap.release()
    if not ret:
        return None
    return bgr  # keep BGR for overlay


def _to_4x4(T):
    if T.shape == (3, 4):
        T4 = np.eye(4, dtype=np.float64)
        T4[:3, :] = T
        return T4
    return T.astype(np.float64)


def save_depth_overlay(cache_dir, net_dir, src_serial, intrinsics, extrinsics,
                       depth=None, src_bgr=None):
    """Backproject src depth to world, reproject to all cameras, save overlays.

    Saves to depth_overlay_{src_serial}/ with per-camera PNGs and grid.png.
    If depth/src_bgr are provided, uses them directly; otherwise reads from video.
    """
    import math

    if depth is None:
        depth_path = os.path.join(cache_dir, "depth", f"{src_serial}.avi")
        depth = _read_frame_depth(depth_path)
    if depth is None:
        print(f"  Depth overlay ({src_serial}): no depth, skipping", flush=True)
        return

    H, W = depth.shape
    K_src = intrinsics[src_serial]
    T_src = _to_4x4(extrinsics[src_serial])

    # Read source RGB for pixel colors (if not provided)
    if src_bgr is None:
        src_rgb_path = os.path.join(cache_dir, "videos", f"{src_serial}.avi")
        src_bgr = _read_frame_rgb(src_rgb_path)
    elif src_bgr.shape[2] == 3 and len(src_bgr.shape) == 3:
        src_bgr = src_bgr  # already BGR from cap.read()

    # Backproject depth to 3D camera frame
    fx, fy = K_src[0, 0], K_src[1, 1]
    cx, cy = K_src[0, 2], K_src[1, 2]
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    valid = depth > 0.01
    z = depth[valid]
    pts_cam = np.stack([
        (u[valid] - cx) * z / fx,
        (v[valid] - cy) * z / fy,
        z,
    ], axis=1)

    # Camera → world
    T_src_inv = np.linalg.inv(T_src)
    pts_world = (T_src_inv[:3, :3] @ pts_cam.T).T + T_src_inv[:3, 3]

    # Get source colors
    if src_bgr is not None:
        colors = src_bgr.reshape(-1, 3)[valid.ravel()]  # BGR
    else:
        colors = np.full((len(pts_world), 3), 128, dtype=np.uint8)

    print(f"  Depth overlay ({src_serial}): {len(pts_world)} world points", flush=True)

    # Reproject to each camera that has a video
    video_dir = os.path.join(cache_dir, "videos")
    overlay_dir = os.path.join(cache_dir, f"depth_overlay_{src_serial}")
    os.makedirs(overlay_dir, exist_ok=True)

    overlay_images = []
    overlay_serials = []

    for vf in sorted(Path(video_dir).iterdir()):
        if vf.suffix != ".avi":
            continue
        tgt_serial = vf.stem
        if tgt_serial not in intrinsics:
            continue

        tgt_bgr = _read_frame_rgb(str(vf))
        if tgt_bgr is None:
            continue

        K_tgt = intrinsics[tgt_serial]
        T_tgt = _to_4x4(extrinsics[tgt_serial])

        # World → target camera
        pts_tgt = (T_tgt[:3, :3] @ pts_world.T).T + T_tgt[:3, 3]
        in_front = pts_tgt[:, 2] > 0.01
        pts_v = pts_tgt[in_front]
        colors_v = colors[in_front]

        px = (K_tgt[0, 0] * pts_v[:, 0] / pts_v[:, 2] + K_tgt[0, 2]).astype(int)
        py = (K_tgt[1, 1] * pts_v[:, 1] / pts_v[:, 2] + K_tgt[1, 2]).astype(int)
        Ht, Wt = tgt_bgr.shape[:2]
        in_img = (px >= 0) & (px < Wt) & (py >= 0) & (py < Ht)

        canvas = tgt_bgr.copy()
        if in_img.any():
            z_vals = pts_v[in_img, 2]
            order = np.argsort(z_vals)[::-1]  # far→near
            canvas[py[in_img][order], px[in_img][order]] = colors_v[in_img][order]

        cv2.imwrite(os.path.join(overlay_dir, f"{tgt_serial}.png"), canvas)
        # Downscale for grid
        small = cv2.resize(canvas, (Wt // 4, Ht // 4))
        overlay_images.append(small)
        overlay_serials.append(tgt_serial)

    # Save grid
    if overlay_images:
        n = len(overlay_images)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        h, w = overlay_images[0].shape[:2]
        grid = np.ones((rows * h, cols * w, 3), dtype=np.uint8) * 255
        for idx, (img, serial) in enumerate(zip(overlay_images, overlay_serials)):
            r, c = divmod(idx, cols)
            grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
            cv2.putText(grid, serial, (c * w + 5, r * h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        grid_path = os.path.join(overlay_dir, "grid.png")
        cv2.imwrite(grid_path, grid)
        print(f"  Depth overlay: {n} cameras -> {grid_path}", flush=True)


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
        # Print why this episode needs processing
        dl = Path(cache_dir) / "depth" / f"{args.left_serial}.avi"
        dr = Path(cache_dir) / "depth" / f"{args.right_serial}.avi"
        reason = []
        if not dl.exists():
            reason.append("left_depth MISSING")
        elif dl.stat().st_size == 0:
            reason.append("left_depth EMPTY")
        if not dr.exists():
            reason.append("right_depth MISSING")
        elif dr.stat().st_size == 0:
            reason.append("right_depth EMPTY")
        print(f"[{done}/{total}] {obj_name}/{idx_name} ({', '.join(reason)})", flush=True)
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
