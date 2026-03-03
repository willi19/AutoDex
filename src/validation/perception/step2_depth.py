#!/usr/bin/env python3
"""
Step 2: Depth estimation validation (DA3 / FoundationStereo)

Reads camera_data.npz from step1. Saves depth maps and cross-camera
pointcloud reprojection overlays to verify multi-view consistency.

Output structure:
    output_dir/
    ├── depth/               # uint16 PNG depth maps (mm)
    └── depth_overlay/
        └── {tgt_serial}/    # from_{src_serial}.png per source camera

Usage (DA3):
    conda activate sam3
    python src/validation/perception/step2_depth.py \
        --output_dir ~/shared_data/.../20260214_231802/6d_output

Usage (FoundationStereo):
    conda activate foundationpose
    python src/validation/perception/step2_depth.py \
        --output_dir ~/shared_data/.../20260214_231802/6d_output \
        --method stereo \
        --stereo_model /path/to/model.engine \
        --baseline 0.12
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

AUTODEX_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(AUTODEX_ROOT))

from autodex.perception import get_depth_da3, get_depth_stereo
from autodex.perception.depth import get_depth_stereo_pytorch

logging.basicConfig(level=logging.INFO, format="[depth] [%(levelname)s] %(message)s")


FS_CKPT = (AUTODEX_ROOT / "autodex/perception/thirdparty/FoundationStereo"
           / "pretrained_models/23-51-11/model_best_bp2.pth")


def load_camera_data(data_dir: Path):
    cam = np.load(str(data_dir / "camera_data.npz"), allow_pickle=True)
    serials = list(cam["serials"])
    intrinsics = cam["intrinsics"]
    extrinsics = cam["extrinsics"]
    return serials, intrinsics, extrinsics


def find_best_stereo_pair(serials, extrinsics, intrinsics):
    """Select the single best stereo pair for stereo depth estimation.

    Camera position in world = -R^T @ t  (extrinsic: pts_cam = R @ pts_world + t)

    Filters (applied in relaxing order):
      - baseline 0.03–0.50 m
      - cos_sim > 0.77  (cameras looking in similar directions)
      - perp_score > 0.30  (baseline mostly perpendicular to optical axis,
        avoids degenerate stereoRectify with inflated focal length)
      - f_ratio < 2.0  (similar focal lengths, avoids stereoRectify failure)
      score = cos_sim  (highest alignment within constraints wins)

    Falls back with relaxed constraints if nothing found.
    Returns (left_serial, right_serial, baseline_m) or None.
    """
    positions = {}
    fwd_dirs = {}
    focal_lengths = {}
    for i, s in enumerate(serials):
        R = extrinsics[i][:3, :3]
        t = extrinsics[i][:3, 3]
        positions[s] = -R.T @ t
        fwd = R[2, :]
        fwd_dirs[s] = fwd / (np.linalg.norm(fwd) + 1e-9)
        focal_lengths[s] = float(intrinsics[i][0, 0])

    def _find(min_b, max_b, min_cos, min_perp=0.30, max_f_ratio=2.0):
        best, best_score = None, -1.0
        for i, s1 in enumerate(serials):
            for j, s2 in enumerate(serials):
                if i >= j:
                    continue
                b = float(np.linalg.norm(positions[s1] - positions[s2]))
                if b < min_b or b > max_b:
                    continue
                cs = float(np.dot(fwd_dirs[s1], fwd_dirs[s2]))
                if cs < min_cos:
                    continue
                baseline_dir = positions[s2] - positions[s1]
                baseline_dir /= np.linalg.norm(baseline_dir) + 1e-9
                mean_fwd = fwd_dirs[s1] + fwd_dirs[s2]
                mean_fwd /= np.linalg.norm(mean_fwd) + 1e-9
                perp = 1.0 - abs(float(np.dot(baseline_dir, mean_fwd)))
                if perp < min_perp:
                    continue
                f1, f2 = focal_lengths[s1], focal_lengths[s2]
                if max(f1, f2) / (min(f1, f2) + 1e-9) > max_f_ratio:
                    continue
                if cs > best_score:
                    best_score = cs
                    best = (s1, s2, b)
        return best

    return (_find(0.03, 0.50, 0.77)
            or _find(0.03, 2.00, 0.50, min_perp=0.20, max_f_ratio=3.0))


def run_da3(output_dir, seg_dir, serials, intrinsics, extrinsics, model_id, device):
    images_dir = seg_dir / "images"
    image_paths = [str(images_dir / f"{s}.png") for s in serials]

    logging.info(f"Running DA3 ({model_id}) on {len(serials)} images...")

    t_model = time.perf_counter()
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(model_id)
    model = model.to(device=device)
    model.eval()
    model_load_s = time.perf_counter() - t_model

    t_infer = time.perf_counter()
    prediction = model.inference(
        image=image_paths,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
    )
    inference_s = time.perf_counter() - t_infer
    logging.info(f"  Inference: {inference_s:.2f}s")

    sub_timing = {
        "model_load_s": round(model_load_s, 2),
        "inference_s": round(inference_s, 2),
    }
    return [prediction.depth[i] for i in range(len(serials))], sub_timing


def run_stereo_pytorch(seg_dir, output_dir, serials, intrinsics, extrinsics, ckpt_dir):
    """Run FoundationStereo on the single best stereo pair, reproject to all cameras.

    Steps:
      1. Select best pair (baseline × view-alignment score)
      2. Stereo rectify the pair
      3. Run FoundationStereo once on the rectified pair
      4. Backproject disparity → 3D (world frame)
      5. Reproject to all 24 cameras
    """
    import torch

    pair = find_best_stereo_pair(serials, extrinsics)
    if pair is None:
        logging.warning("No suitable stereo pair found")
        return serials, {s: None for s in serials}, {}

    left_s, right_s, _ = pair
    serial_to_idx = {s: i for i, s in enumerate(serials)}
    left_idx = serial_to_idx[left_s]
    right_idx = serial_to_idx[right_s]

    images_dir = seg_dir / "images"

    # ── Load images ──────────────────────────────────────────────────────
    t_io = time.perf_counter()
    left_bgr = cv2.imread(str(images_dir / f"{left_s}.png"))
    right_bgr = cv2.imread(str(images_dir / f"{right_s}.png"))
    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
    H, W = left_rgb.shape[:2]
    io_s = time.perf_counter() - t_io

    K_left = intrinsics[left_idx].copy().astype(np.float64)
    K_right = intrinsics[right_idx].copy().astype(np.float64)
    T_left = extrinsics[left_idx]
    T_right = extrinsics[right_idx]

    # ── Stereo rectify ───────────────────────────────────────────────────
    t_rect = time.perf_counter()
    # Relative pose: T_right = T_rel @ T_left
    T_rel = T_right @ np.linalg.inv(T_left)
    R_rel = T_rel[:3, :3].astype(np.float64)
    t_rel = T_rel[:3, 3].astype(np.float64)

    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
        K_left, None, K_right, None, (W, H), R_rel, t_rel,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )
    # Rectified focal length and baseline
    f_rect = float(P1[0, 0])
    cx_rect, cy_rect = float(P1[0, 2]), float(P1[1, 2])
    baseline_rect = abs(float(P2[0, 3])) / (f_rect + 1e-9)

    map1_l, map2_l = cv2.initUndistortRectifyMap(K_left, None, R1, P1, (W, H), cv2.CV_32FC1)
    map1_r, map2_r = cv2.initUndistortRectifyMap(K_right, None, R2, P2, (W, H), cv2.CV_32FC1)
    left_rect = cv2.remap(left_rgb, map1_l, map2_l, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_rgb, map1_r, map2_r, cv2.INTER_LINEAR)
    rectify_s = time.perf_counter() - t_rect

    logging.info(f"Best pair: {left_s} ↔ {right_s}  ({W}x{H})")
    logging.info(f"  io={io_s:.2f}s  rectify={rectify_s:.2f}s  f={f_rect:.1f}px  baseline={baseline_rect:.4f}m")

    # ── Load model ──────────────────────────────────────────────────────
    t_model = time.perf_counter()
    from autodex.perception.depth import _setup_foundation_stereo_path
    _setup_foundation_stereo_path()
    from omegaconf import OmegaConf
    from core.foundation_stereo import FoundationStereo
    from core.utils.utils import InputPadder

    cfg = OmegaConf.load(str(Path(ckpt_dir).parent / "cfg.yaml"))
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    cfg["valid_iters"] = 32
    cfg["hiera"] = 0
    model = FoundationStereo(cfg)
    ckpt = torch.load(ckpt_dir, map_location="cuda")
    model.load_state_dict(ckpt["model"])
    model.cuda()
    model.eval()
    model_load_s = time.perf_counter() - t_model

    # ── Inference ───────────────────────────────────────────────────────
    t_infer = time.perf_counter()
    img0 = torch.as_tensor(left_rect.copy()).cuda().float()[None].permute(0, 3, 1, 2)
    img1 = torch.as_tensor(right_rect.copy()).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)
    with torch.no_grad(), torch.cuda.amp.autocast(True):
        disp = model.forward(img0, img1, iters=32, test_mode=True)
    disp = padder.unpad(disp.float()).data.cpu().numpy().reshape(H, W)
    inference_s = time.perf_counter() - t_infer
    logging.info(f"  model_load={model_load_s:.2f}s  inference={inference_s:.2f}s")

    # ── Disparity → 3D point cloud (world frame) ────────────────────────
    valid = disp >= 0.5
    disp_v = np.maximum(disp[valid], 0.1)
    depth_rect = f_rect * baseline_rect / disp_v

    u_grid, v_grid = np.meshgrid(np.arange(W, dtype=np.float32),
                                  np.arange(H, dtype=np.float32))
    x_rect = ((u_grid[valid] - cx_rect) * depth_rect / f_rect)
    y_rect = ((v_grid[valid] - cy_rect) * depth_rect / f_rect)
    pts_rect = np.stack([x_rect, y_rect, depth_rect], axis=1)  # (N, 3) rectified left cam

    # rectified left cam → original left cam  (R1: orig→rect, so R1^T: rect→orig)
    pts_left = (R1.T @ pts_rect.T).T

    # original left cam → world  (T_left: world→left, inv gives left→world)
    T_left_inv = np.linalg.inv(T_left)
    pts_world = (T_left_inv[:3, :3] @ pts_left.T).T + T_left_inv[:3, 3]

    # ── Reproject to all cameras ─────────────────────────────────────────
    t_proj = time.perf_counter()
    depths = {}
    for i, serial in enumerate(serials):
        R_c = extrinsics[i][:3, :3]
        t_c = extrinsics[i][:3, 3]
        K_c = intrinsics[i]

        pts_cam = (R_c @ pts_world.T).T + t_c
        in_front = pts_cam[:, 2] > 0.01
        pts_v = pts_cam[in_front]

        px = (K_c[0, 0] * pts_v[:, 0] / pts_v[:, 2] + K_c[0, 2]).astype(int)
        py = (K_c[1, 1] * pts_v[:, 1] / pts_v[:, 2] + K_c[1, 2]).astype(int)
        in_img = (px >= 0) & (px < W) & (py >= 0) & (py < H)

        depth_cam = np.zeros((H, W), dtype=np.float32)
        z_vals = pts_v[in_img, 2]
        order = np.argsort(z_vals)[::-1]  # far → near, so near overwrites
        depth_cam[py[in_img][order], px[in_img][order]] = z_vals[order]
        depths[serial] = depth_cam

    reproject_s = time.perf_counter() - t_proj
    logging.info(f"  reproject={reproject_s:.2f}s  (to {len(serials)} cameras)")

    sub_timing = {
        "n_pair": 1,
        "left_serial": left_s,
        "right_serial": right_s,
        "io_s": round(io_s, 2),
        "rectify_s": round(rectify_s, 2),
        "model_load_s": round(model_load_s, 2),
        "inference_s": round(inference_s, 2),
        "reproject_s": round(reproject_s, 2),
    }
    return serials, depths, sub_timing


def run_stereo_trt(seg_dir, output_dir, serials, intrinsics, extrinsics, engine_path):
    """Run FoundationStereo TRT engine on best stereo pair, reproject to all cameras."""
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401

    pair = find_best_stereo_pair(serials, extrinsics, intrinsics)
    if pair is None:
        logging.warning("No suitable stereo pair found")
        return serials, {s: None for s in serials}, {}
    left_s, right_s, _ = pair

    serial_to_idx = {s: i for i, s in enumerate(serials)}
    images_dir = seg_dir / "images"
    H, W = cv2.imread(str(images_dir / f"{serials[0]}.png")).shape[:2]

    # ── Load TRT engine (once) ────────────────────────────────────────────
    t_model = time.perf_counter()
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    trt_shape = engine.get_tensor_shape("left")  # (1, 3, H_trt, W_trt)
    H_trt, W_trt = int(trt_shape[2]), int(trt_shape[3])
    model_load_s = time.perf_counter() - t_model
    logging.info(f"  TRT engine loaded: {H_trt}x{W_trt}  model_load={model_load_s:.2f}s")

    # Preallocate GPU buffers
    disp_arr = np.zeros((1, 1, H_trt, W_trt), dtype=np.float32)
    d_left = cuda.mem_alloc(int(np.prod([1, 3, H_trt, W_trt])) * 4)
    d_right = cuda.mem_alloc(int(np.prod([1, 3, H_trt, W_trt])) * 4)
    d_disp = cuda.mem_alloc(disp_arr.nbytes)
    stream = cuda.Stream()

    def _run_pair(left_s, right_s):
        """Run TRT stereo on one pair, return (world_pts, f_rect, baseline_rect)."""
        left_idx = serial_to_idx[left_s]
        right_idx = serial_to_idx[right_s]
        K_left = intrinsics[left_idx].copy().astype(np.float64)
        K_right = intrinsics[right_idx].copy().astype(np.float64)
        T_left = extrinsics[left_idx]
        T_right = extrinsics[right_idx]

        # Rectify
        T_rel = T_right @ np.linalg.inv(T_left)
        R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
            K_left, None, K_right, None, (W, H),
            T_rel[:3, :3].astype(np.float64), T_rel[:3, 3].astype(np.float64),
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
        )
        f_rect = float(P1[0, 0])
        cx_rect, cy_rect = float(P1[0, 2]), float(P1[1, 2])
        baseline_rect = abs(float(P2[0, 3])) / (f_rect + 1e-9)
        logging.info(f"  Pair {left_s}↔{right_s}: f={f_rect:.1f}px  baseline={baseline_rect:.4f}m")
        # Sanity: skip degenerate rectifications
        if baseline_rect < 0.01 or f_rect > 20000:
            logging.warning(f"    → SKIP (degenerate rectification)")
            return np.zeros((0, 3), dtype=np.float32)

        map1_l, map2_l = cv2.initUndistortRectifyMap(K_left, None, R1, P1, (W, H), cv2.CV_32FC1)
        map1_r, map2_r = cv2.initUndistortRectifyMap(K_right, None, R2, P2, (W, H), cv2.CV_32FC1)
        left_bgr = cv2.imread(str(images_dir / f"{left_s}.png"))
        right_bgr = cv2.imread(str(images_dir / f"{right_s}.png"))
        left_rect = cv2.remap(cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB), map1_l, map2_l, cv2.INTER_LINEAR)
        right_rect = cv2.remap(cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB), map1_r, map2_r, cv2.INTER_LINEAR)

        # TRT inference
        left_trt = cv2.resize(left_rect, (W_trt, H_trt), interpolation=cv2.INTER_LINEAR)
        right_trt = cv2.resize(right_rect, (W_trt, H_trt), interpolation=cv2.INTER_LINEAR)
        left_arr = np.ascontiguousarray(left_trt.astype(np.float32).transpose(2, 0, 1)[None])
        right_arr = np.ascontiguousarray(right_trt.astype(np.float32).transpose(2, 0, 1)[None])

        cuda.memcpy_htod_async(d_left, left_arr, stream)
        cuda.memcpy_htod_async(d_right, right_arr, stream)
        context.set_tensor_address("left", int(d_left))
        context.set_tensor_address("right", int(d_right))
        context.set_tensor_address("disp", int(d_disp))
        context.execute_async_v3(stream.handle)
        stream.synchronize()
        cuda.memcpy_dtoh(disp_arr, d_disp)

        # Work at TRT resolution for efficiency (~10× fewer world points than full-res)
        f_trt = f_rect * W_trt / W
        cx_trt = cx_rect * W_trt / W
        cy_trt = cy_rect * H_trt / H

        disp_trt = disp_arr.squeeze()  # (H_trt, W_trt)
        valid = disp_trt >= 0.5
        disp_v = np.maximum(disp_trt[valid], 0.1)
        depth_trt_v = f_trt * baseline_rect / disp_v
        u_g, v_g = np.meshgrid(np.arange(W_trt, dtype=np.float32), np.arange(H_trt, dtype=np.float32))
        pts_rect = np.stack([
            (u_g[valid] - cx_trt) * depth_trt_v / f_trt,
            (v_g[valid] - cy_trt) * depth_trt_v / f_trt,
            depth_trt_v,
        ], axis=1)
        pts_left = (R1.T @ pts_rect.T).T
        T_left_inv = np.linalg.inv(T_left)
        pts_world = (T_left_inv[:3, :3] @ pts_left.T).T + T_left_inv[:3, 3]
        return pts_world

    # ── Run TRT on the single best pair ──────────────────────────────────
    t_io = time.perf_counter()
    pts_world = _run_pair(left_s, right_s)
    inference_s = time.perf_counter() - t_io
    logging.info(f"  {len(pts_world)} world points")

    # ── Reproject to all cameras ──────────────────────────────────────────
    t_proj = time.perf_counter()
    depths = {}
    for i, serial in enumerate(serials):
        R_c = extrinsics[i][:3, :3]
        t_c = extrinsics[i][:3, 3]
        K_c = intrinsics[i]

        pts_cam = (R_c @ pts_world.T).T + t_c
        in_front = pts_cam[:, 2] > 0.01
        pts_v = pts_cam[in_front]

        px = (K_c[0, 0] * pts_v[:, 0] / pts_v[:, 2] + K_c[0, 2]).astype(int)
        py = (K_c[1, 1] * pts_v[:, 1] / pts_v[:, 2] + K_c[1, 2]).astype(int)
        in_img = (px >= 0) & (px < W) & (py >= 0) & (py < H)

        depth_cam = np.zeros((H, W), dtype=np.float32)
        z_vals = pts_v[in_img, 2]
        order = np.argsort(z_vals)[::-1]
        depth_cam[py[in_img][order], px[in_img][order]] = z_vals[order]
        depths[serial] = depth_cam

    reproject_s = time.perf_counter() - t_proj
    logging.info(f"  reproject={reproject_s:.2f}s  (to {len(serials)} cameras)")

    sub_timing = {
        "left_serial": left_s,
        "right_serial": right_s,
        "trt_resolution": f"{W_trt}x{H_trt}",
        "model_load_s": round(model_load_s, 2),
        "inference_s": round(inference_s, 2),
        "reproject_s": round(reproject_s, 2),
    }
    return serials, depths, sub_timing


def run_stereo(seg_dir, output_dir, serials, intrinsics, stereo_pairs, model_path, baseline):
    import onnxruntime as ort
    import tensorrt as trt

    FOUNDATION_STEREO_PATH = str(
        AUTODEX_ROOT / "autodex/perception/thirdparty/object-6d-tracking/thirdparty/FoundationStereo"
    )
    if FOUNDATION_STEREO_PATH not in sys.path:
        sys.path.insert(0, FOUNDATION_STEREO_PATH)
        sys.path.insert(0, FOUNDATION_STEREO_PATH + "/onnx-tensorrt")

    from onnx_tensorrt import tensorrt_engine

    is_onnx = model_path.endswith(".onnx")
    if is_onnx:
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        model = ort.InferenceSession(model_path, sess_options=session_options,
                                     providers=["CUDAExecutionProvider"])
    else:
        with open(model_path, "rb") as f:
            engine_data = f.read()
        engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(engine_data)
        model = tensorrt_engine.Engine(engine)

    depths = []
    images_dir = seg_dir / "images"
    for i, serial in enumerate(serials):
        left_path, right_path = stereo_pairs[serial]
        left = cv2.imread(left_path)
        right = cv2.imread(right_path)
        K = intrinsics[i]

        depth = get_depth_stereo(left, right, model, K, baseline)
        depths.append(depth)
        logging.info(f"  {serial}: done")

    return depths


def save_depths(output_dir, seg_dir, serials, depths, intrinsics):
    depth_dir = output_dir / "depth"
    depth_dir.mkdir(exist_ok=True)

    images_dir = seg_dir / "images"
    for i, serial in enumerate(serials):
        img = cv2.imread(str(images_dir / f"{serial}.png"))
        H, W = img.shape[:2]

        depth = depths[i]
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
        depth_uint16 = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(depth_dir / f"{serial}.png"), depth_uint16)

    logging.info(f"Saved {len(serials)} depth maps to {depth_dir}")


def save_depth_overlay(output_dir, seg_dir, serials, intrinsics, extrinsics):
    """Cross-camera pointcloud reprojection to verify multi-view consistency.

    Each from_{src}.png shows the target camera's original image with
    source camera's point cloud painted using original source pixel colors.
    """
    images_dir = seg_dir / "images"
    depth_dir = output_dir / "depth"
    overlay_dir = output_dir / "depth_overlay"
    overlay_dir.mkdir(exist_ok=True)

    # Backproject all cameras to world, keeping original pixel colors (BGR)
    world_points, world_colors = {}, {}
    for i, serial in enumerate(serials):
        img_bgr = cv2.imread(str(images_dir / f"{serial}.png"))
        H, W = img_bgr.shape[:2]
        depth = cv2.imread(str(depth_dir / f"{serial}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

        K = intrinsics[i]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        z = depth
        pts_cam = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=-1).reshape(-1, 3)
        colors_flat = img_bgr.reshape(-1, 3)  # (H*W, 3) BGR

        valid = pts_cam[:, 2] > 0
        pts_cam_v = pts_cam[valid]
        colors_v = colors_flat[valid]

        R, t = extrinsics[i][:3, :3], extrinsics[i][:3, 3]
        # pts_cam = R @ pts_world + t  →  pts_world = R^T @ (pts_cam - t)
        world_points[serial] = (R.T @ (pts_cam_v - t).T).T
        world_colors[serial] = colors_v

    # Project each source onto each target
    for i, tgt in enumerate(serials):
        tgt_dir = overlay_dir / tgt
        tgt_dir.mkdir(exist_ok=True)
        img_tgt = cv2.imread(str(images_dir / f"{tgt}.png"))
        H, W = img_tgt.shape[:2]
        K, R, t = intrinsics[i], extrinsics[i][:3, :3], extrinsics[i][:3, 3]

        for j, src in enumerate(serials):
            pts_w = world_points[src]
            colors_src = world_colors[src]  # (N, 3) BGR

            if len(pts_w) == 0:
                cv2.imwrite(str(tgt_dir / f"from_{src}.png"), img_tgt)
                continue

            pts_cam = (R @ pts_w.T).T + t
            in_front = pts_cam[:, 2] > 0
            pts_v = pts_cam[in_front]
            colors_v = colors_src[in_front]

            px = (K[0, 0] * pts_v[:, 0] / pts_v[:, 2] + K[0, 2]).astype(int)
            py = (K[1, 1] * pts_v[:, 1] / pts_v[:, 2] + K[1, 2]).astype(int)
            in_img = (px >= 0) & (px < W) & (py >= 0) & (py < H)

            canvas = img_tgt.copy()
            if in_img.any():
                # Paint far-to-near so near points overwrite far ones
                z_vals = pts_v[in_img, 2]
                order = np.argsort(z_vals)[::-1]
                py_v, px_v = py[in_img][order], px[in_img][order]
                canvas[py_v, px_v] = colors_v[in_img][order]

            cv2.imwrite(str(tgt_dir / f"from_{src}.png"), canvas)

    logging.info(f"Depth overlay saved to {overlay_dir}")


def main(args):
    import json as _json

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seg_dir = Path(args.seg_dir) if args.seg_dir else output_dir

    serials, intrinsics, extrinsics = load_camera_data(seg_dir)
    logging.info(f"Loaded {len(serials)} cameras")

    t0 = time.perf_counter()
    if args.method == "da3":
        depths_list, sub_timing = run_da3(output_dir, seg_dir, serials, intrinsics, extrinsics, args.model_id, args.device)
        depths = {s: depths_list[i] for i, s in enumerate(serials)}
        active_serials = serials

    elif args.method == "stereo":
        if args.trt_engine:
            active_serials, depths, sub_timing = run_stereo_trt(seg_dir, output_dir, serials, intrinsics, extrinsics, args.trt_engine)
        else:
            ckpt = args.stereo_model or str(FS_CKPT)
            active_serials, depths, sub_timing = run_stereo_pytorch(seg_dir, output_dir, serials, intrinsics, extrinsics, ckpt)
        # all cameras get depth via reprojection; filter out any that failed
        active_serials = [s for s in active_serials if depths.get(s) is not None]

    else:
        raise ValueError(f"Unknown method: {args.method}")

    elapsed = time.perf_counter() - t0

    # save_depths expects list; pass active serials and their depths
    depth_list = [depths[s] for s in active_serials]
    K_list = np.array([intrinsics[{s: i for i, s in enumerate(serials)}[s]] for s in active_serials])
    save_depths(output_dir, seg_dir, active_serials, depth_list, K_list)
    save_depth_overlay(output_dir, seg_dir, active_serials,
                       K_list, np.array([extrinsics[{s: i for i, s in enumerate(serials)}[s]] for s in active_serials]))

    # Save source info for step4_compare
    if args.method == "stereo":
        best = find_best_stereo_pair(serials, extrinsics, intrinsics)
        pair_map = {best[0]: best[1]} if best else {}
        source_info = {"method": "stereo", "pairs": pair_map,
                       "source_serial": best[0] if best else ""}
    else:
        source_info = {"method": "da3"}
    (output_dir / "source_info.json").write_text(_json.dumps(source_info, indent=2))

    timing_path = output_dir / "timing.json"
    timing = _json.loads(timing_path.read_text()) if timing_path.exists() else {}
    timing["step2_depth"] = {
        "method": args.method,
        "total_s": round(elapsed, 2),
        "n_cameras": len(active_serials),
        **sub_timing,
    }
    timing_path.write_text(_json.dumps(timing, indent=2))
    logging.info(f"Step 2 done. ({elapsed:.1f}s) → {timing_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to write depth maps (e.g. validation_output/depth/da3)")
    parser.add_argument("--seg_dir", type=str, default=None,
                        help="Segmentation dir with images/ and camera_data.npz (default: output_dir)")
    parser.add_argument("--method", type=str, default="da3", choices=["da3", "stereo"])
    parser.add_argument("--model_id", type=str, default="depth-anything/DA3-LARGE",
                        help="DA3 model ID (HuggingFace)")
    parser.add_argument("--stereo_model", type=str, default=None,
                        help="Path to model_best_bp2.pth (default: thirdparty/FoundationStereo/pretrained_models/23-51-11/)")
    parser.add_argument("--trt_engine", type=str, default=None,
                        help="Path to TRT engine (.engine). If set with --method stereo, uses TRT instead of PyTorch.")
    parser.add_argument("--device", type=str, default="cuda")
    main(parser.parse_args())