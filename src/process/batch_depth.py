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


def _disp_to_depth_left(disp_trt, f_rect, cx_rect, cy_rect, baseline,
                        map_left, W_orig, H_orig, H_trt, W_trt):
    """Convert left disparity to depth map aligned to the original left camera.

    Uses the rectification map directly: map_left tells us where each rectified
    pixel came from in the original image. No 3D back-projection needed.
    """
    f_trt = f_rect * W_trt / W_orig

    # Compute depth at TRT resolution
    valid = disp_trt >= 0.5
    depth_trt = np.zeros_like(disp_trt)
    depth_trt[valid] = f_trt * baseline / np.maximum(disp_trt[valid], 0.1)

    # Resize depth to original rectified resolution
    depth_rect = cv2.resize(depth_trt, (W_orig, H_orig),
                            interpolation=cv2.INTER_NEAREST)
    valid_rect = depth_rect > 0.001

    # Use rectification map to place depth at original pixel positions.
    # map_left[0] = x_orig, map_left[1] = y_orig for each rectified pixel
    map_x, map_y = map_left
    depth_map = np.zeros((H_orig, W_orig), dtype=np.float32)

    px_orig = map_x[valid_rect].astype(int)
    py_orig = map_y[valid_rect].astype(int)
    z_vals = depth_rect[valid_rect]

    in_img = (px_orig >= 0) & (px_orig < W_orig) & (py_orig >= 0) & (py_orig < H_orig)
    order = np.argsort(z_vals[in_img])[::-1]  # far-to-near so near overwrites
    depth_map[py_orig[in_img][order], px_orig[in_img][order]] = z_vals[in_img][order]
    return depth_map


def _depth_left_to_right(depth_left, K_left, T_left, K_right, T_right,
                         W, H):
    """Reproject left depth map to right camera view using extrinsics.

    Left depth (proven correct) → back-project with K_left → left cam 3D
    → world → right cam 3D → project with K_right.
    """
    valid = depth_left > 0.001
    if not valid.any():
        return np.zeros_like(depth_left)

    fx_l, fy_l = K_left[0, 0], K_left[1, 1]
    cx_l, cy_l = K_left[0, 2], K_left[1, 2]

    u_g, v_g = np.meshgrid(np.arange(W, dtype=np.float32),
                            np.arange(H, dtype=np.float32))
    z = depth_left[valid]
    x = (u_g[valid] - cx_l) * z / fx_l
    y = (v_g[valid] - cy_l) * z / fy_l
    pts_left = np.stack([x, y, z], axis=1)  # (N, 3) in left cam frame

    # Left cam → world → right cam
    T_l = _to_4x4_local(T_left)
    T_r = _to_4x4_local(T_right)
    T_l_inv = np.linalg.inv(T_l)
    pts_world = (T_l_inv[:3, :3] @ pts_left.T).T + T_l_inv[:3, 3]
    pts_right = (T_r[:3, :3] @ pts_world.T).T + T_r[:3, 3]

    # Project to right camera image
    in_front = pts_right[:, 2] > 0.01
    pts_v = pts_right[in_front]

    fx_r, fy_r = K_right[0, 0], K_right[1, 1]
    cx_r, cy_r = K_right[0, 2], K_right[1, 2]
    px = (fx_r * pts_v[:, 0] / pts_v[:, 2] + cx_r).astype(int)
    py = (fy_r * pts_v[:, 1] / pts_v[:, 2] + cy_r).astype(int)
    in_img = (px >= 0) & (px < W) & (py >= 0) & (py < H)

    depth_map = np.zeros((H, W), dtype=np.float32)
    z_vals = pts_v[in_img, 2]
    order = np.argsort(z_vals)[::-1]  # far-to-near so near overwrites far
    depth_map[py[in_img][order], px[in_img][order]] = z_vals[order]
    return depth_map


def _to_4x4_local(T):
    if T.shape == (3, 4):
        T4 = np.eye(4, dtype=np.float64)
        T4[:3, :] = T
        return T4
    return T.astype(np.float64)


# ── Task collection ──────────────────────────────────────────────────────────

def collect_tasks(base_dir, left_serial, right_serial):
    """Fast scan: collect all captures that have stereo videos and cam_param.

    No depth check here — checked per-episode right before processing so that
    parallel instances with overlapping ranges naturally skip each other's work.
    """
    cache_base = Path(_get_cache_base(base_dir))
    if not cache_base.is_dir():
        return []
    tasks = []
    global_idx = 0
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
            rel = str(idx_dir.relative_to(cache_base))
            net_dir = Path(base_dir) / rel
            if not (net_dir / "cam_param").is_dir():
                continue
            global_idx += 1
            tasks.append((str(left_path), str(right_path), str(idx_dir), str(net_dir),
                          obj_dir.name, idx_dir.name, global_idx))
    return tasks, global_idx


def _depth_is_valid(cache_dir, left_serial, right_serial, left_path, right_path):
    """Return True if depth already exists with correct frame count."""
    depth_dir = Path(cache_dir) / "depth"
    depth_file = next(
        ((depth_dir / f"{s}.avi") for s in (left_serial, right_serial)
         if (depth_dir / f"{s}.avi").exists() and (depth_dir / f"{s}.avi").stat().st_size > 0),
        None
    )
    if depth_file is None:
        return False
    cap_l = cv2.VideoCapture(left_path)
    cap_r = cv2.VideoCapture(right_path)
    n_expected = min(int(cap_l.get(cv2.CAP_PROP_FRAME_COUNT)),
                     int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap_l.release(); cap_r.release()
    cap_d = cv2.VideoCapture(str(depth_file))
    n_depth = int(cap_d.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_d.release()
    return n_depth == n_expected


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

    # Auto-detect left/right: stereoRectify expects right camera to the right of left.
    # P2[0,3] = -f * Tx. If Tx > 0, right is to the right (correct).
    # If Tx < 0 (P2[0,3] > 0), cameras are swapped → swap them.
    def _to_4x4(T):
        if T.shape == (3, 4):
            T4 = np.eye(4, dtype=np.float64)
            T4[:3, :] = T
            return T4
        return T.astype(np.float64)
    T_rel_check = _to_4x4(T_right) @ np.linalg.inv(_to_4x4(T_left))
    _, _, _, P2_check, _, _, _ = cv2.stereoRectify(
        K_left, None, K_right, None, (100, 100),
        T_rel_check[:3, :3], T_rel_check[:3, 3],
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )
    if P2_check[0, 3] > 0:
        print(f"  Auto-swap: {left_serial} <-> {right_serial} (cameras were in wrong order)", flush=True)
        left_serial, right_serial = right_serial, left_serial
        left_path, right_path = right_path, left_path
        K_left, K_right = K_right, K_left
        T_left, T_right = T_right, T_left

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
    print(f"  f={f_rect:.1f}px  baseline={baseline:.4f}m  "
          f"K_left fx={K_left[0,0]:.1f} fy={K_left[1,1]:.1f}  "
          f"R1 angle={np.degrees(np.arccos(np.clip((np.trace(R1)-1)/2, -1, 1))):.2f}deg  "
          f"R2 angle={np.degrees(np.arccos(np.clip((np.trace(R2)-1)/2, -1, 1))):.2f}deg", flush=True)

    if baseline < 0.01 or f_rect > 20000:
        cap_l.release()
        cap_r.release()
        print("  SKIP degenerate rectification", flush=True)
        return

    H_trt, W_trt = trt_shape

    depth_dir = os.path.join(cache_dir, "depth")
    os.makedirs(depth_dir, exist_ok=True)

    left_out = os.path.join(depth_dir, f"{left_serial}.avi")

    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    writer_l = cv2.VideoWriter(left_out, fourcc, fps, (W, H))

    for idx in tqdm(range(n_frames), desc="  depth", unit="f"):
        ret_l, bgr_l = cap_l.read()
        ret_r, bgr_r = cap_r.read()
        if not ret_l or not ret_r:
            break

        rgb_l = cv2.cvtColor(bgr_l, cv2.COLOR_BGR2RGB)
        rgb_r = cv2.cvtColor(bgr_r, cv2.COLOR_BGR2RGB)
        left_rect = cv2.remap(rgb_l, map_left[0], map_left[1], cv2.INTER_LINEAR)
        right_rect = cv2.remap(rgb_r, map_right[0], map_right[1], cv2.INTER_LINEAR)

        disp_l = run_trt_inference(trt_context, trt_buffers, left_rect, right_rect, H_trt, W_trt)

        # Left disparity → depth for left camera (un-rectify with R1)
        depth_left = _disp_to_depth_left(disp_l, f_rect, cx_rect, cy_rect, baseline,
                                         map_left, W, H, H_trt, W_trt)
        writer_l.write(encode_depth_uint16(depth_left))


    cap_l.release()
    cap_r.release()
    writer_l.release()
    print(f"  Saved: {left_out}", flush=True)


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
                       disp_trt, f_rect, cx_rect, cy_rect, baseline,
                       R_rect, T_src, src_bgr, H_trt, W_trt):
    """Disparity → world points → reproject to all cameras. Follows step2_depth.py exactly.

    Bypasses disp_to_depth_aligned. Also saves point cloud as .npz.
    """
    import math

    H, W = src_bgr.shape[:2]

    # Disparity → 3D in rectified frame (at TRT resolution) — matches step2
    f_trt = f_rect * W_trt / W
    cx_trt = cx_rect * W_trt / W
    cy_trt = cy_rect * H_trt / H

    valid = disp_trt >= 0.5
    disp_v = np.maximum(disp_trt[valid], 0.1)
    depth_v = f_trt * baseline / disp_v

    u_g, v_g = np.meshgrid(np.arange(W_trt, dtype=np.float32),
                            np.arange(H_trt, dtype=np.float32))
    pts_rect = np.stack([
        (u_g[valid] - cx_trt) * depth_v / f_trt,
        (v_g[valid] - cy_trt) * depth_v / f_trt,
        depth_v,
    ], axis=1)

    # Rectified → original camera frame
    pts_cam = (R_rect.T @ pts_rect.T).T

    # Original camera → world
    T_src_4x4 = _to_4x4(T_src)
    T_src_inv = np.linalg.inv(T_src_4x4)
    pts_world = (T_src_inv[:3, :3] @ pts_cam.T).T + T_src_inv[:3, 3]

    # Source pixel colors: map TRT grid → original image pixels
    # Use rectification maps would be ideal, but just use nearest for speed
    K_src = intrinsics[src_serial]
    fx_o, fy_o = K_src[0, 0], K_src[1, 1]
    cx_o, cy_o = K_src[0, 2], K_src[1, 2]
    in_front = pts_cam[:, 2] > 0.01
    px_src = (fx_o * pts_cam[in_front, 0] / pts_cam[in_front, 2] + cx_o).astype(int)
    py_src = (fy_o * pts_cam[in_front, 1] / pts_cam[in_front, 2] + cy_o).astype(int)
    in_src = (px_src >= 0) & (px_src < W) & (py_src >= 0) & (py_src < H)

    # Filter to valid points with colors
    pts_world_v = pts_world[in_front][in_src]
    colors = src_bgr[py_src[in_src], px_src[in_src]]  # BGR

    print(f"  Depth overlay ({src_serial}): {len(pts_world_v)} world points, "
          f"depth range {depth_v.min():.3f}-{depth_v.max():.3f}m", flush=True)

    # Save point cloud as PLY (viewable in MeshLab)
    overlay_dir = os.path.join(cache_dir, "overlay_debug")
    os.makedirs(overlay_dir, exist_ok=True)
    ply_path = os.path.join(overlay_dir, f"pointcloud_{src_serial}.ply")
    with open(ply_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(pts_world_v)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for pt, c in zip(pts_world_v, colors):
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {c[2]} {c[1]} {c[0]}\n")
    print(f"  Saved pointcloud: {ply_path}", flush=True)

    # Reproject to each camera that has a video
    video_dir = os.path.join(cache_dir, "videos")
    overlay_images = []
    overlay_labels = []

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
        pts_tgt = (T_tgt[:3, :3] @ pts_world_v.T).T + T_tgt[:3, 3]
        in_front_t = pts_tgt[:, 2] > 0.01
        pts_v = pts_tgt[in_front_t]
        colors_v = colors[in_front_t]

        px = (K_tgt[0, 0] * pts_v[:, 0] / pts_v[:, 2] + K_tgt[0, 2]).astype(int)
        py = (K_tgt[1, 1] * pts_v[:, 1] / pts_v[:, 2] + K_tgt[1, 2]).astype(int)
        Ht, Wt = tgt_bgr.shape[:2]
        in_img = (px >= 0) & (px < Wt) & (py >= 0) & (py < Ht)

        # Points-only (black background)
        canvas_pts = np.zeros_like(tgt_bgr)
        if in_img.any():
            z_vals = pts_v[in_img, 2]
            order = np.argsort(z_vals)[::-1]
            canvas_pts[py[in_img][order], px[in_img][order]] = colors_v[in_img][order]

        cv2.imwrite(os.path.join(overlay_dir, f"from_{src_serial}_on_{tgt_serial}_pts.png"), canvas_pts)
        cv2.imwrite(os.path.join(overlay_dir, f"from_{src_serial}_on_{tgt_serial}_orig.png"), tgt_bgr)

        small_pts = cv2.resize(canvas_pts, (Wt // 4, Ht // 4))
        small_orig = cv2.resize(tgt_bgr, (Wt // 4, Ht // 4))
        overlay_images.append(np.hstack([small_pts, small_orig]))
        overlay_labels.append(f"{src_serial}->{tgt_serial}")

    # Save grid
    if overlay_images:
        n = len(overlay_images)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        h, w = overlay_images[0].shape[:2]
        grid = np.ones((rows * h, cols * w, 3), dtype=np.uint8) * 255
        for idx, (img, label) in enumerate(zip(overlay_images, overlay_labels)):
            r, c = divmod(idx, cols)
            grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
            cv2.putText(grid, label, (c * w + 5, r * h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        grid_path = os.path.join(overlay_dir, f"grid_{src_serial}.png")
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
    parser.add_argument("--max_episodes", type=int, default=0,
                        help="Max episodes to process (0 = all)")
    parser.add_argument("--last_episodes", type=int, default=0,
                        help="Process only the last N episodes (0 = all)")
    args = parser.parse_args()
    print("Starting batch depth estimation (TRT)...", flush=True)

    tasks, total_global = collect_tasks(args.base, args.left_serial, args.right_serial)
    if not tasks:
        print("Nothing to do.")
        return

    if args.last_episodes > 0:
        tasks = tasks[-args.last_episodes:]
    elif args.max_episodes > 0:
        tasks = tasks[:args.max_episodes]
    print(f"{len(tasks)} captures to process (total in dir: {total_global})", flush=True)

    print(f"Loading TRT engine: {args.engine}", flush=True)
    t0 = time.time()
    trt_context, trt_buffers, trt_shape = load_trt_engine(args.engine)
    print(f"TRT ready: {trt_shape[1]}x{trt_shape[0]}  ({time.time() - t0:.1f}s)", flush=True)

    done = 0
    skipped = 0
    total = len(tasks)
    for left_path, right_path, cache_dir, net_dir, obj_name, idx_name, gidx in tasks:
        done += 1
        if _depth_is_valid(cache_dir, args.left_serial, args.right_serial, left_path, right_path):
            skipped += 1
            print(f"[{done}/{total}] #{gidx}/{total_global} {obj_name}/{idx_name} SKIP (depth valid)", flush=True)
            continue
        print(f"[{done}/{total}] #{gidx}/{total_global} {obj_name}/{idx_name}", flush=True)
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

    processed = done - skipped
    print(f"All done! {processed} processed, {skipped} skipped (already valid) out of {total}.", flush=True)


if __name__ == "__main__":
    main()
