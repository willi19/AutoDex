#!/usr/bin/env python3
"""Stereo depth estimation with auto pair selection.

Supports two modes:
  --capture_dir : process a single episode (all stereo pairs)
  --base        : batch all episodes under a directory (with progress/ETA)

Usage:
    conda activate foundation_stereo

    # Single episode
    python -u src/process/depth.py \
        --capture_dir /home/mingi/shared_data/RSS2026_Mingi/experiment/selected_100/apple/20260206_181110

    # Single episode, fixed pair
    python -u src/process/depth.py \
        --capture_dir ... --serials 22684755 23263780

    # Batch all episodes
    python -u src/process/depth.py \
        --base /home/mingi/shared_data/RSS2026_Mingi/experiment/selected_100

    # Quick test (first frame only)
    python -u src/process/depth.py --capture_dir ... --num_frames 1

    # Regenerate overlay only
    python -u src/process/depth.py --capture_dir ... --overlay_only
"""

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

AUTODEX_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AUTODEX_ROOT))

from autodex.perception.depth import (
    load_cam_param,
    _auto_order_stereo,
    _to_4x4,
    encode_depth_uint16,
    decode_depth_uint16,
)


# ── Stereo rectification ─────────────────────────────────────────────────────

def _compute_rectify_params(K_left, K_right, T_left, T_right, image_size):
    """Compute stereo rectification geometry (no maps, no allocation).

    Returns dict with: R1, R2, f_rect, Tx_phys, baseline, and the
    oversized canvas params (W_big, H_big, cx_big, cy_big) needed to
    find the valid intersection region.
    """
    W, H = image_size

    T_l = _to_4x4(T_left)
    T_r = _to_4x4(T_right)
    T_rel = T_r @ np.linalg.inv(T_l)

    R1, R2, P1_cv, P2_cv, _, _, _ = cv2.stereoRectify(
        K_left, None, K_right, None, (W, H),
        T_rel[:3, :3], T_rel[:3, 3],
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )

    f_rect_cv = float(P1_cv[0, 0])
    Tx_phys = float(P2_cv[0, 3]) / (f_rect_cv + 1e-9)
    baseline = abs(Tx_phys)
    f_orig = max(float(K_left[0, 0]), float(K_right[0, 0]))

    return {
        "R1": R1, "R2": R2,
        "f_rect": f_orig, "Tx_phys": Tx_phys, "baseline": baseline,
        "P1_cv": P1_cv, "P2_cv": P2_cv, "f_rect_cv": f_rect_cv,
    }


def _find_valid_region(K_left, K_right, R1, R2, f_rect, Tx_phys, image_size):
    """Find the intersection of both cameras' valid regions on an oversized canvas.

    Returns (x0, y0, x1, y1) in big-canvas coords, plus (W_big, H_big).
    Returns None if no valid intersection.
    """
    W, H = image_size

    corners = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float64)
    all_projected = []
    for K, R in [(K_left, R1), (K_right, R2)]:
        pts = corners.reshape(-1, 1, 2)
        P_id = np.array([[f_rect, 0, 0], [0, f_rect, 0], [0, 0, 1]], dtype=np.float64)
        projected = cv2.undistortPoints(pts, K, None, R=R, P=P_id)
        all_projected.append(projected.reshape(-1, 2))
    all_projected = np.vstack(all_projected)
    margin = 100
    x_range = all_projected[:, 0].max() - all_projected[:, 0].min() + 2 * margin
    y_range = all_projected[:, 1].max() - all_projected[:, 1].min() + 2 * margin
    W_big = max(int(np.ceil(x_range)), W * 3)
    H_big = max(int(np.ceil(y_range)), H * 3)

    P_big = np.array([
        [f_rect, 0, W_big / 2.0, 0],
        [0, f_rect, H_big / 2.0, 0],
        [0, 0, 1, 0],
    ], dtype=np.float64)
    P_big_r = P_big.copy()
    P_big_r[0, 3] = f_rect * Tx_phys

    map_l_big = cv2.initUndistortRectifyMap(K_left, None, R1, P_big, (W_big, H_big), cv2.CV_32FC1)
    map_r_big = cv2.initUndistortRectifyMap(K_right, None, R2, P_big_r, (W_big, H_big), cv2.CV_32FC1)

    valid_l = ((map_l_big[0] >= 0) & (map_l_big[0] < W) &
               (map_l_big[1] >= 0) & (map_l_big[1] < H))
    valid_r = ((map_r_big[0] >= 0) & (map_r_big[0] < W) &
               (map_r_big[1] >= 0) & (map_r_big[1] < H))

    # Union — preserve all content from both cameras, nothing gets cropped.
    valid_both = valid_l | valid_r
    ys, xs = np.where(valid_both)
    if len(xs) == 0:
        return None

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return (x0, y0, x1, y1, W_big, H_big)


def build_rectify_maps(K_left, K_right, T_left, T_right, image_size,
                       capture_dir=None):
    """Compute stereo rectification maps cropped to valid intersection.

    Output maps remap raw images directly to the intersection of both cameras'
    valid regions — no black borders, no content cut off.

    Returns: (map_left, map_right, R1, R2, f_rect, cx, cy, baseline,
              rect_size, disp_offset)
        rect_size = (W_out, H_out) — final output size.
        disp_offset = always 0.0 (same cx for both cameras).
    """
    W, H = image_size
    params = _compute_rectify_params(K_left, K_right, T_left, T_right, (W, H))
    R1, R2 = params["R1"], params["R2"]
    f_rect = params["f_rect"]
    Tx_phys = params["Tx_phys"]
    baseline = params["baseline"]

    # Find valid intersection region on oversized canvas
    valid = _find_valid_region(K_left, K_right, R1, R2, f_rect, Tx_phys, (W, H))
    if valid is None:
        P1_cv, P2_cv = params["P1_cv"], params["P2_cv"]
        f_cv = params["f_rect_cv"]
        map_left = cv2.initUndistortRectifyMap(K_left, None, R1, P1_cv, (W, H), cv2.CV_32FC1)
        map_right = cv2.initUndistortRectifyMap(K_right, None, R2, P2_cv, (W, H), cv2.CV_32FC1)
        return (map_left, map_right, R1, R2, f_cv,
                float(P1_cv[0, 2]), float(P1_cv[1, 2]),
                baseline, (W, H), 0.0)

    vx0, vy0, vx1, vy1, W_big, H_big = valid
    cx_big = W_big / 2.0
    cy_big = H_big / 2.0

    # Workspace crop: union of both cameras' bbox projections.
    # Clipped to the valid union region so no camera content is lost within bbox.
    ws_crop = _workspace_crop(
        K_left, T_left, R1, K_right, T_right, R2,
        f_rect, cx_big, cy_big, vx0, vy0, vx1, vy1,
        capture_dir)

    if ws_crop is not None:
        cx0, cy0, cx1, cy1 = ws_crop
    else:
        cx0, cy0, cx1, cy1 = vx0, vy0, vx1, vy1
    W_out = cx1 - cx0
    H_out = cy1 - cy0

    # Build P matrices with crop baked in (same cx for both)
    cx_out = cx_big - cx0
    cy_out = cy_big - cy0

    P_out = np.array([
        [f_rect, 0, cx_out, 0],
        [0, f_rect, cy_out, 0],
        [0, 0, 1, 0],
    ], dtype=np.float64)
    P_out_r = P_out.copy()
    P_out_r[0, 3] = f_rect * Tx_phys

    map_left = cv2.initUndistortRectifyMap(K_left, None, R1, P_out, (W_out, H_out), cv2.CV_32FC1)
    map_right = cv2.initUndistortRectifyMap(K_right, None, R2, P_out_r, (W_out, H_out), cv2.CV_32FC1)

    print(f"    Rectify+crop: valid {vx1-vx0}x{vy1-vy0} -> out {W_out}x{H_out}", flush=True)

    return (map_left, map_right, R1, R2, f_rect,
            cx_out, cy_out, baseline, (W_out, H_out), 0.0)


# ── TRT engine ───────────────────────────────────────────────────────────────

_FS_ROOT = Path(__file__).resolve().parents[2] / "autodex/perception/thirdparty/FoundationStereo"
_DEFAULT_ENGINE = _FS_ROOT / "output/foundation_stereo_448x672.engine"


def load_trt_engine(engine_path: str):
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    trt_shape = engine.get_tensor_shape("left")
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

    return buffers["disp_arr"].squeeze()


# ── Disparity → depth ────────────────────────────────────────────────────────

def disp_to_depth_left(disp_trt, f_rect, baseline,
                       K_left, R1, cx_rect, cy_rect,
                       W_orig, H_orig, W_rect, H_rect, H_trt, W_trt):
    """Convert left disparity to depth map aligned to the original left camera.

    Uses inverse mapping (cv2.remap) to avoid moire/aliasing artifacts.
    Applies Z_orig = Z_rect / rz correction for rectification rotation.
    """
    f_trt = f_rect * W_trt / W_rect

    valid = disp_trt >= 0.5
    depth_trt = np.zeros_like(disp_trt)
    depth_trt[valid] = f_trt * baseline / np.maximum(disp_trt[valid], 0.1)

    depth_rect = cv2.resize(depth_trt, (W_rect, H_rect),
                            interpolation=cv2.INTER_NEAREST)

    u_grid, v_grid = np.meshgrid(
        np.arange(W_orig, dtype=np.float64),
        np.arange(H_orig, dtype=np.float64),
    )
    K_inv = np.linalg.inv(K_left.astype(np.float64))
    x_norm = K_inv[0, 0] * u_grid + K_inv[0, 1] * v_grid + K_inv[0, 2]
    y_norm = K_inv[1, 0] * u_grid + K_inv[1, 1] * v_grid + K_inv[1, 2]
    z_norm = np.ones_like(u_grid)
    R1_64 = R1.astype(np.float64)
    rx = R1_64[0, 0] * x_norm + R1_64[0, 1] * y_norm + R1_64[0, 2] * z_norm
    ry = R1_64[1, 0] * x_norm + R1_64[1, 1] * y_norm + R1_64[1, 2] * z_norm
    rz = R1_64[2, 0] * x_norm + R1_64[2, 1] * y_norm + R1_64[2, 2] * z_norm
    inv_map_x = (f_rect * rx / rz + cx_rect).astype(np.float32)
    inv_map_y = (f_rect * ry / rz + cy_rect).astype(np.float32)

    depth_rect_sampled = cv2.remap(depth_rect, inv_map_x, inv_map_y,
                                    cv2.INTER_NEAREST)
    # Z_orig = Z_rect / rz  (see CLAUDE.md: Rectified Z vs Original Z)
    rz_safe = np.where(np.abs(rz) > 1e-6, rz, 1.0)
    depth_map = depth_rect_sampled / rz_safe.astype(np.float32)
    depth_map[depth_rect_sampled < 0.001] = 0
    return depth_map


# ── Find stereo pairs ────────────────────────────────────────────────────────

def find_all_stereo_pairs(capture_dir, serials, intrinsics, extrinsics):
    """Find stereo pairs using rig-based adjacency grouping.

    Groups by (focal_group, z_level), sorts by angle, pairs adjacent cameras.
    Returns: [(left_serial, right_serial, baseline_m), ...]
    """
    from collections import defaultdict

    MAX_ANGLE_GAP = 40.0

    def focal_group(serial):
        f = float(intrinsics[serial][0, 0])
        if f < 2500:
            return "wide"
        elif f < 4000:
            return "mid"
        return "tele"

    c2r_path = capture_dir / "C2R.npy"
    if c2r_path.exists():
        C2R_inv = np.linalg.inv(np.load(str(c2r_path)))
    else:
        C2R_inv = np.eye(4)

    rig_data = {}
    positions = {}
    for s in serials:
        T = _to_4x4(extrinsics[s])
        pos_world = -T[:3, :3].T @ T[:3, 3]
        pos_rig = C2R_inv[:3, :3] @ pos_world + C2R_inv[:3, 3]
        angle = float(np.degrees(np.arctan2(pos_rig[1], pos_rig[0])))
        z = float(pos_rig[2])
        z_level = "low" if z < 0.7 else ("mid" if z < 1.1 else "high")
        rig_data[s] = {"angle": angle, "z_level": z_level}
        positions[s] = pos_world

    buckets = defaultdict(list)
    for s in serials:
        buckets[(focal_group(s), rig_data[s]["z_level"])].append(s)
    for key in buckets:
        buckets[key].sort(key=lambda s: rig_data[s]["angle"])

    raw_pairs = []
    for (grp, zlev), ss in sorted(buckets.items()):
        print(f"  {grp}/{zlev}: {len(ss)} cameras — {ss}", flush=True)
        for i in range(len(ss) - 1):
            s1, s2 = ss[i], ss[i + 1]
            gap = abs(rig_data[s1]["angle"] - rig_data[s2]["angle"])
            if gap <= MAX_ANGLE_GAP:
                raw_pairs.append((s1, s2))

    pairs = []
    for s1, s2 in raw_pairs:
        baseline_m = float(np.linalg.norm(positions[s1] - positions[s2]))
        swapped = _auto_order_stereo(
            intrinsics[s1], intrinsics[s2],
            extrinsics[s1], extrinsics[s2],
        )
        if swapped:
            pairs.append((s2, s1, baseline_m))
        else:
            pairs.append((s1, s2, baseline_m))

    return pairs


# ── Process one stereo pair (full video) ─────────────────────────────────────

def _workspace_crop(K_left, T_left, R1, K_right, T_right, R2,
                    f_rect, cx_big, cy_big,
                    vx0, vy0, vx1, vy1,
                    capture_dir):
    """Compute workspace crop in big-canvas coordinates.

    Same crop for both cameras. Union of both cameras' workspace projections
    so the full workspace bbox is visible in both views.

    Returns (cx0, cy0, cx1, cy1) in big-canvas coords, or None.
    """
    if capture_dir is None:
        return None
    c2r_path = capture_dir / "C2R.npy"
    if not c2r_path.exists():
        return None

    ws_min = np.array([0.35, -0.30, 0.0])
    ws_max = np.array([0.80, 0.21, 0.4])
    corners_robot = np.array([[x, y, z]
                              for x in [ws_min[0], ws_max[0]]
                              for y in [ws_min[1], ws_max[1]]
                              for z in [ws_min[2], ws_max[2]]])

    C2R = np.load(str(c2r_path))
    C2R_4x4 = _to_4x4(C2R)

    margin = 50
    per_cam_px = []
    per_cam_py = []
    for T_cam, R_rect in [(T_left, R1), (T_right, R2)]:
        T_c = _to_4x4(T_cam)
        robot_to_cam = T_c @ C2R_4x4
        corners_cam = (robot_to_cam[:3, :3] @ corners_robot.T).T + robot_to_cam[:3, 3]

        in_front = corners_cam[:, 2] > 0.01
        if not in_front.any():
            return None
        corners_cam = corners_cam[in_front]

        R_64 = R_rect.astype(np.float64)
        corners_rect = (R_64 @ corners_cam.T).T
        in_front_rect = corners_rect[:, 2] > 0.01
        if not in_front_rect.any():
            return None
        corners_rect = corners_rect[in_front_rect]

        px = f_rect * corners_rect[:, 0] / corners_rect[:, 2] + cx_big
        py = f_rect * corners_rect[:, 1] / corners_rect[:, 2] + cy_big
        per_cam_px.append(px)
        per_cam_py.append(py)

    if len(per_cam_px) != 2:
        return None

    # Union of both cameras' projections — full bbox visible in both views
    all_px = np.concatenate(per_cam_px)
    all_py = np.concatenate(per_cam_py)
    cx0 = max(vx0, int(all_px.min()) - margin)
    cx1 = min(vx1, int(all_px.max()) + margin)
    cy0 = max(vy0, int(all_py.min()) - margin)
    cy1 = min(vy1, int(all_py.max()) + margin)

    if cx1 <= cx0 or cy1 <= cy0:
        return None

    return (cx0, cy0, cx1, cy1)


def process_pair_video(trt_context, trt_buffers, trt_shape,
                       capture_dir, left_serial, right_serial,
                       intrinsics, extrinsics, num_frames=None):
    """Process one stereo pair for full video. Saves depth/{left_serial}.avi."""
    H_trt, W_trt = trt_shape
    video_dir = capture_dir / "videos"
    left_path = str(video_dir / f"{left_serial}.avi")
    right_path = str(video_dir / f"{right_serial}.avi")

    K_left = intrinsics[left_serial]
    K_right = intrinsics[right_serial]
    T_left = extrinsics[left_serial]
    T_right = extrinsics[right_serial]

    cap_l = cv2.VideoCapture(left_path)
    cap_r = cv2.VideoCapture(right_path)
    n_frames = min(int(cap_l.get(cv2.CAP_PROP_FRAME_COUNT)),
                   int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT)))
    if num_frames is not None:
        n_frames = min(n_frames, num_frames)
    fps = cap_l.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))

    result = build_rectify_maps(K_left, K_right, T_left, T_right, (W, H),
                                capture_dir=capture_dir)
    map_left, map_right, R1, R2, f_rect, cx_rect, cy_rect, baseline, rect_size, disp_offset = result
    W_rect, H_rect = rect_size

    print(f"  {left_serial}<->{right_serial}: {W}x{H} -> rect {W_rect}x{H_rect}, {n_frames}f, "
          f"f={f_rect:.0f}px, bl={baseline:.4f}m", flush=True)

    if baseline < 0.01 or f_rect <= 0:
        cap_l.release()
        cap_r.release()
        print("  SKIP tiny baseline", flush=True)
        return None

    aspect = max(W_rect, H_rect) / max(min(W_rect, H_rect), 1)
    if aspect > 2.5:
        cap_l.release()
        cap_r.release()
        print(f"  SKIP bad aspect ratio {aspect:.1f}:1 ({W_rect}x{H_rect})", flush=True)
        return None

    depth_dir = capture_dir / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(depth_dir / f"{left_serial}.avi")

    debug_dir = capture_dir / "depth_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    # Maps now output directly to cropped size — no separate crop step needed.
    W_crop, H_crop = W_rect, H_rect
    cx_crop, cy_crop = cx_rect, cy_rect

    for idx in tqdm(range(n_frames), desc=f"  {left_serial}", unit="f"):
        ret_l, bgr_l = cap_l.read()
        ret_r, bgr_r = cap_r.read()
        if not ret_l or not ret_r:
            break

        rgb_l = cv2.cvtColor(bgr_l, cv2.COLOR_BGR2RGB)
        rgb_r = cv2.cvtColor(bgr_r, cv2.COLOR_BGR2RGB)
        left_crop = cv2.remap(rgb_l, map_left[0], map_left[1], cv2.INTER_LINEAR)
        right_crop = cv2.remap(rgb_r, map_right[0], map_right[1], cv2.INTER_LINEAR)

        if idx == 0:
            orig_pair = np.hstack([bgr_l, bgr_r])
            cv2.putText(orig_pair,
                        f"ORIGINAL  L={left_serial}  R={right_serial}  {W}x{H}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            left_rect_bgr = cv2.cvtColor(left_crop, cv2.COLOR_RGB2BGR)
            right_rect_bgr = cv2.cvtColor(right_crop, cv2.COLOR_RGB2BGR)
            rect_pair = np.hstack([left_rect_bgr, right_rect_bgr])
            for y in range(0, rect_pair.shape[0], max(rect_pair.shape[0] // 16, 1)):
                cv2.line(rect_pair, (0, y), (rect_pair.shape[1], y), (0, 255, 0), 1)
            cv2.putText(rect_pair,
                        f"RECTIFIED+CROP  f={f_rect:.0f}  bl={baseline:.4f}m  "
                        f"{W_crop}x{H_crop}  disp_off={disp_offset:.0f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            target_w = max(orig_pair.shape[1], rect_pair.shape[1])
            if orig_pair.shape[1] != target_w:
                scale = target_w / orig_pair.shape[1]
                orig_pair = cv2.resize(orig_pair, (target_w, int(orig_pair.shape[0] * scale)))
            if rect_pair.shape[1] != target_w:
                scale = target_w / rect_pair.shape[1]
                rect_pair = cv2.resize(rect_pair, (target_w, int(rect_pair.shape[0] * scale)))

            debug_img = np.vstack([orig_pair, rect_pair])
            cv2.imwrite(str(debug_dir / f"{left_serial}_{right_serial}.jpg"), debug_img)

            # TRT debug
            left_trt_dbg = cv2.resize(left_rect_bgr, (W_trt, H_trt), interpolation=cv2.INTER_LINEAR)
            right_trt_dbg = cv2.resize(right_rect_bgr, (W_trt, H_trt), interpolation=cv2.INTER_LINEAR)
            trt_pair = np.hstack([left_trt_dbg, right_trt_dbg])
            for y in range(0, H_trt, max(H_trt // 16, 1)):
                cv2.line(trt_pair, (0, y), (trt_pair.shape[1], y), (0, 255, 0), 1)
            cv2.putText(trt_pair,
                        f"TRT input  {W_crop}x{H_crop}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(str(debug_dir / f"{left_serial}_{right_serial}_trt.jpg"), trt_pair)

        disp = run_trt_inference(trt_context, trt_buffers, left_crop, right_crop, H_trt, W_trt)
        depth = disp_to_depth_left(disp, f_rect, baseline,
                                   K_left, R1, cx_crop, cy_crop,
                                   W, H, W_crop, H_crop, H_trt, W_trt)
        writer.write(encode_depth_uint16(depth))

    cap_l.release()
    cap_r.release()
    writer.release()
    print(f"  Saved: {out_path}", flush=True)
    return out_path


# ── Multiview overlay ────────────────────────────────────────────────────────

def generate_overlay(capture_dir, intrinsics, extrinsics, frame_idx=0):
    """Generate multiview depth overlay for quality checking."""
    depth_dir = capture_dir / "depth"
    video_dir = capture_dir / "videos"
    overlay_dir = capture_dir / "depth_overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    depth_serials = []
    for p in sorted(depth_dir.glob("*.avi")):
        if p.stem in intrinsics and p.stat().st_size > 0:
            depth_serials.append(p.stem)

    if not depth_serials:
        print("No depth videos found.", flush=True)
        return

    all_serials = sorted(s for s in intrinsics if (video_dir / f"{s}.avi").exists())

    print(f"Overlay: {len(depth_serials)} depth sources, {len(all_serials)} cameras", flush=True)

    for src_serial in depth_serials:
        depth_path = str(depth_dir / f"{src_serial}.avi")
        cap_d = cv2.VideoCapture(depth_path)
        cap_d.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, bgr_d = cap_d.read()
        cap_d.release()
        if not ret:
            continue
        depth_map = decode_depth_uint16(bgr_d)

        cap_rgb = cv2.VideoCapture(str(video_dir / f"{src_serial}.avi"))
        cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, src_bgr = cap_rgb.read()
        H, W = src_bgr.shape[:2]
        cap_rgb.release()

        valid_pct = (depth_map > 0.001).sum() / (H * W) * 100
        d_valid = depth_map[depth_map > 0.001]
        if len(d_valid) == 0:
            print(f"  {src_serial}: no valid depth", flush=True)
            continue
        print(f"  {src_serial}: {valid_pct:.1f}% valid, "
              f"range {d_valid.min():.2f}-{d_valid.max():.2f}m", flush=True)

        K_src = intrinsics[src_serial]
        T_src = extrinsics[src_serial]
        valid = depth_map > 0.001
        fx, fy = K_src[0, 0], K_src[1, 1]
        cx, cy = K_src[0, 2], K_src[1, 2]
        u_g, v_g = np.meshgrid(np.arange(W, dtype=np.float32),
                                np.arange(H, dtype=np.float32))
        z = depth_map[valid]
        pts_cam = np.stack([
            (u_g[valid] - cx) * z / fx,
            (v_g[valid] - cy) * z / fy,
            z,
        ], axis=1)

        T_s = _to_4x4(T_src)
        T_s_inv = np.linalg.inv(T_s)
        pts_world = (T_s_inv[:3, :3] @ pts_cam.T).T + T_s_inv[:3, 3]
        colors = src_bgr[v_g[valid].astype(int), u_g[valid].astype(int)]

        overlay_images = []
        overlay_labels = []
        for tgt_serial in all_serials:
            tgt_cap = cv2.VideoCapture(str(video_dir / f"{tgt_serial}.avi"))
            tgt_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, tgt_bgr = tgt_cap.read()
            tgt_cap.release()
            if not ret:
                continue

            K_tgt = intrinsics[tgt_serial]
            T_tgt = _to_4x4(extrinsics[tgt_serial])
            Ht, Wt = tgt_bgr.shape[:2]

            pts_tgt = (T_tgt[:3, :3] @ pts_world.T).T + T_tgt[:3, 3]
            in_front = pts_tgt[:, 2] > 0.01
            pts_v = pts_tgt[in_front]
            colors_v = colors[in_front]

            px = (K_tgt[0, 0] * pts_v[:, 0] / pts_v[:, 2] + K_tgt[0, 2]).astype(int)
            py = (K_tgt[1, 1] * pts_v[:, 1] / pts_v[:, 2] + K_tgt[1, 2]).astype(int)
            in_img = (px >= 0) & (px < Wt) & (py >= 0) & (py < Ht)

            canvas = np.zeros_like(tgt_bgr)
            if in_img.any():
                z_vals = pts_v[in_img, 2]
                order = np.argsort(z_vals)[::-1]
                canvas[py[in_img][order], px[in_img][order]] = colors_v[in_img][order]

            small_pts = cv2.resize(canvas, (Wt // 4, Ht // 4))
            small_orig = cv2.resize(tgt_bgr, (Wt // 4, Ht // 4))
            overlay_images.append(np.hstack([small_pts, small_orig]))
            overlay_labels.append(f"{src_serial}->{tgt_serial}")

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
            grid_path = str(overlay_dir / f"grid_{src_serial}_f{frame_idx}.png")
            cv2.imwrite(grid_path, grid)
            print(f"  Grid: {grid_path}", flush=True)


# ── Process one capture dir ──────────────────────────────────────────────────

def process_capture(capture_dir, engine_path, serials=None,
                    num_frames=None, overlay_only=False, overlay_frame=0):
    """Process a single capture directory."""
    capture_dir = Path(capture_dir)
    intrinsics, extrinsics = load_cam_param(capture_dir)
    all_serials = list(intrinsics.keys())
    print(f"{len(all_serials)} cameras", flush=True)

    if serials:
        # Fixed pair mode
        s1, s2 = serials[0], serials[1]
        baseline_m = float(np.linalg.norm(
            -_to_4x4(extrinsics[s1])[:3, :3].T @ _to_4x4(extrinsics[s1])[:3, 3]
            - (-_to_4x4(extrinsics[s2])[:3, :3].T @ _to_4x4(extrinsics[s2])[:3, 3])
        ))
        swapped = _auto_order_stereo(
            intrinsics[s1], intrinsics[s2],
            extrinsics[s1], extrinsics[s2],
        )
        if swapped:
            pairs = [(s2, s1, baseline_m)]
        else:
            pairs = [(s1, s2, baseline_m)]
    else:
        # Auto pair mode
        pairs = find_all_stereo_pairs(capture_dir, all_serials, intrinsics, extrinsics)

    print(f"\n{len(pairs)} stereo pairs:", flush=True)
    for left_s, right_s, bl in pairs:
        print(f"  {left_s} <-> {right_s}  baseline={bl:.4f}m", flush=True)

    if not overlay_only:
        print(f"\nLoading TRT engine: {engine_path}", flush=True)
        trt_context, trt_buffers, trt_shape = load_trt_engine(engine_path)
        print(f"TRT ready: {trt_shape[1]}x{trt_shape[0]}", flush=True)

        t_start = time.perf_counter()
        for i, (left_s, right_s, bl) in enumerate(pairs):
            depth_path = capture_dir / "depth" / f"{left_s}.avi"
            if num_frames is None and depth_path.exists() and depth_path.stat().st_size > 0:
                video_path = capture_dir / "videos" / f"{left_s}.avi"
                cap_v = cv2.VideoCapture(str(video_path))
                n_expected = int(cap_v.get(cv2.CAP_PROP_FRAME_COUNT))
                cap_v.release()
                cap_d = cv2.VideoCapture(str(depth_path))
                n_depth = int(cap_d.get(cv2.CAP_PROP_FRAME_COUNT))
                cap_d.release()
                if n_depth == n_expected:
                    print(f"\n[{i+1}/{len(pairs)}] {left_s}<->{right_s} SKIP (done)", flush=True)
                    continue

            print(f"\n[{i+1}/{len(pairs)}] Processing pair {left_s} <-> {right_s}", flush=True)
            process_pair_video(
                trt_context, trt_buffers, trt_shape,
                capture_dir, left_s, right_s,
                intrinsics, extrinsics,
                num_frames=num_frames,
            )

        elapsed = time.perf_counter() - t_start
        print(f"\nDepth done! {len(pairs)} pairs, {elapsed:.1f}s total", flush=True)

    print(f"\n=== Generating overlay (frame {overlay_frame}) ===", flush=True)
    generate_overlay(capture_dir, intrinsics, extrinsics, frame_idx=overlay_frame)
    print("\nDone!", flush=True)


# ── Batch mode helpers ───────────────────────────────────────────────────────

def _format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h{m:02d}m"


def _find_capture_dirs(base):
    """Find all capture directories (contain cam_param/ and videos/)."""
    base = Path(base)
    return sorted(
        p.parent for p in base.rglob("cam_param")
        if p.is_dir() and (p.parent / "videos").is_dir()
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stereo depth estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
modes:
  --capture_dir DIR    Process a single episode
  --base DIR           Batch all episodes under DIR (with progress/ETA)

examples:
  %(prog)s --capture_dir /path/to/apple/20260206_181110
  %(prog)s --capture_dir /path/to/episode --serials 22684755 23263780
  %(prog)s --base /path/to/selected_100
  %(prog)s --capture_dir /path/to/episode --num_frames 1
  %(prog)s --capture_dir /path/to/episode --overlay_only
""",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--capture_dir", type=str, help="Single episode directory")
    group.add_argument("--base", type=str, help="Batch: parent of all episodes")

    parser.add_argument("--serials", nargs=2, metavar="S",
                        help="Fixed stereo pair (two serial numbers, auto-detects left/right)")
    parser.add_argument("--engine", type=str, default=str(_DEFAULT_ENGINE))
    parser.add_argument("--overlay_only", action="store_true",
                        help="Skip depth computation, only generate overlay")
    parser.add_argument("--overlay_frame", type=int, default=0,
                        help="Frame index for overlay visualization")
    parser.add_argument("--num_frames", type=int, default=None,
                        help="Process only first N frames (default: all)")
    args = parser.parse_args()

    if args.capture_dir:
        # Single episode mode
        process_capture(
            args.capture_dir, args.engine,
            serials=args.serials,
            num_frames=args.num_frames,
            overlay_only=args.overlay_only,
            overlay_frame=args.overlay_frame,
        )
    else:
        # Batch mode
        dirs = _find_capture_dirs(args.base)
        total = len(dirs)
        print(f"Found {total} capture directories")
        print("=" * 60)

        failed = []
        t_global = time.perf_counter()

        for i, capture_dir in enumerate(dirs):
            obj_name = capture_dir.parent.name
            idx = capture_dir.name
            elapsed_global = time.perf_counter() - t_global

            if i > 0:
                avg = elapsed_global / i
                remaining = avg * (total - i)
                eta_str = f"ETA {_format_time(remaining)}"
            else:
                eta_str = "ETA --"

            print(f"\n{'=' * 60}")
            print(f"[{i+1}/{total}] {obj_name}/{idx}  "
                  f"(elapsed {_format_time(elapsed_global)}, {eta_str})")
            print("-" * 60)

            t_start = time.perf_counter()
            try:
                process_capture(
                    capture_dir, args.engine,
                    serials=args.serials,
                    num_frames=args.num_frames,
                    overlay_only=args.overlay_only,
                    overlay_frame=args.overlay_frame,
                )
            except Exception as e:
                failed.append((capture_dir, str(e)))
                print(f"  FAILED: {e}")

            dt = time.perf_counter() - t_start
            print(f"  Completed in {_format_time(dt)}")

        total_time = time.perf_counter() - t_global
        print(f"\n{'=' * 60}")
        print(f"All done! {total} captures in {_format_time(total_time)}")
        if failed:
            print(f"\n{len(failed)} FAILED:")
            for d, reason in failed:
                print(f"  {d.parent.name}/{d.name}: {reason}")


if __name__ == "__main__":
    main()
