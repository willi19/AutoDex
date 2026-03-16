#!/usr/bin/env python3
"""Benchmark timing and GPU memory for a single perception model.

Use benchmark.sh to run all models (each in its correct conda env).

Usage:
    # Run all:
    bash src/validation/execution/benchmark.sh <capture_dir> <mesh_path>

    # Run one model directly:
    conda activate foundationpose
    python -u src/validation/execution/benchmark.py \
        --model yoloe \
        --capture_dir /path/to/episode \
        --mesh /path/to/mesh.obj
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import cv2
import numpy as np

AUTODEX_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(AUTODEX_ROOT))

ALL_MODELS = ["yoloe", "sam3", "sam3_image", "da3", "stereo_pytorch", "stereo_trt", "fpose", "silhouette"]


# ── Data loading (shared by all benchmarks) ──────────────────────────────────

def gpu_mem_mb():
    """GPU memory used by this process (via nvidia-smi, not PyTorch tracker)."""
    import subprocess, os
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            text=True,
        )
        pid = os.getpid()
        for line in out.strip().splitlines():
            parts = line.split(",")
            if int(parts[0].strip()) == pid:
                return float(parts[1].strip())
    except Exception:
        pass
    # Fallback to PyTorch
    import torch
    return torch.cuda.memory_allocated() / 1024 / 1024


def load_test_data(capture_dir, serial=None):
    """Load first-frame RGB, depth, mask, and intrinsics from capture dir."""
    import json
    capture_dir = Path(capture_dir)

    video_dir = capture_dir / "videos"
    images_dir = capture_dir / "images"
    if serial is None:
        if video_dir.exists():
            serials = sorted(p.stem for p in video_dir.glob("*.avi"))
        else:
            serials = sorted(p.stem for p in images_dir.glob("*.png"))
        serial = serials[0]

    # RGB
    rgb = None
    video_path = video_dir / f"{serial}.avi"
    if video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        ret, bgr = cap.read()
        cap.release()
        if ret:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb is None:
        png_path = images_dir / f"{serial}.png"
        if png_path.exists():
            bgr = cv2.imread(str(png_path))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb is None:
        raise FileNotFoundError(f"No video or image for {serial}")

    # Depth
    depth = None
    depth_path = capture_dir / "depth" / f"{serial}.avi"
    if not depth_path.exists():
        depth_files = sorted((capture_dir / "depth").glob("*.avi"))
        if depth_files:
            depth_path = depth_files[0]
    if depth_path.exists():
        from autodex.perception.depth import decode_depth_uint16
        cap = cv2.VideoCapture(str(depth_path))
        ret, bgr_d = cap.read()
        cap.release()
        if ret:
            depth = decode_depth_uint16(bgr_d)

    # Mask
    mask = None
    mask_first = capture_dir / "obj_mask_first" / f"{serial}.png"
    mask_video = capture_dir / "obj_mask" / f"{serial}.avi"
    if mask_first.exists():
        img = cv2.imread(str(mask_first), cv2.IMREAD_GRAYSCALE)
        mask = (img > 127).astype(np.uint8)
    elif mask_video.exists():
        cap = cv2.VideoCapture(str(mask_video))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            m = (gray > 127).astype(np.uint8)
            if m.sum() >= 100:
                mask = m
                break
        cap.release()

    # Intrinsics
    with open(capture_dir / "cam_param" / "intrinsics.json") as f:
        intr_raw = json.load(f)
    K = np.array(intr_raw[serial]["intrinsics_undistort"], dtype=np.float32)

    print(f"Test data: serial={serial}, image={rgb.shape}, "
          f"depth={'ok' if depth is not None else 'MISSING'}, "
          f"mask={'ok' if mask is not None else 'MISSING'}")

    return rgb, depth, mask, K, serial


# ── Individual benchmarks ────────────────────────────────────────────────────

def bench_yoloe(rgb):
    import torch
    print("=" * 60)
    print("YOLOE (single-frame mask)")
    print("-" * 60)

    torch.cuda.reset_peak_memory_stats()
    mem_before = gpu_mem_mb()

    t0 = time.perf_counter()
    from autodex.perception import YoloeSegmentor
    seg = YoloeSegmentor(gpu=0)
    load_time = time.perf_counter() - t0
    mem_after_load = gpu_mem_mb()

    # Warmup
    _ = seg.segment(rgb, "object")
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    mask = seg.segment(rgb, "object on the checkerboard")
    torch.cuda.synchronize()
    infer_time = time.perf_counter() - t0

    mem_peak = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"  {'Load:':<10} {load_time:>7.2f}s")
    print(f"  {'Infer:':<10} {infer_time:>7.3f}s")
    print(f"  {'GPU mem:':<10} {mem_before:>7.0f}MB -> {mem_after_load:.0f}MB "
          f"(+{mem_after_load - mem_before:.0f}MB, peak={mem_peak:.0f}MB)")
    print(f"  Mask: {'found' if mask is not None else 'NOT FOUND'}")
    print()

    del seg
    gc.collect()
    torch.cuda.empty_cache()


def bench_sam3(rgb):
    import torch
    print("=" * 60)
    print("SAM3 (single-frame mask)")
    print("-" * 60)

    torch.cuda.reset_peak_memory_stats()
    mem_before = gpu_mem_mb()

    t0 = time.perf_counter()
    from autodex.perception import Sam3Segmentor
    seg = Sam3Segmentor(gpu=0)
    load_time = time.perf_counter() - t0
    mem_after_load = gpu_mem_mb()

    t0 = time.perf_counter()
    mask = seg.segment(rgb, "object on the checkerboard")
    torch.cuda.synchronize()
    infer_time = time.perf_counter() - t0

    mem_peak = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"  {'Load:':<10} {load_time:>7.2f}s")
    print(f"  {'Infer:':<10} {infer_time:>7.3f}s")
    print(f"  {'GPU mem:':<10} {mem_before:>7.0f}MB -> {mem_after_load:.0f}MB "
          f"(+{mem_after_load - mem_before:.0f}MB, peak={mem_peak:.0f}MB)")
    print(f"  Mask: {'found' if mask is not None else 'NOT FOUND'}")
    print()

    del seg
    gc.collect()
    torch.cuda.empty_cache()


def bench_sam3_image(rgb):
    import torch
    print("=" * 60)
    print("SAM3 Image Model (single-frame)")
    print("-" * 60)

    torch.cuda.reset_peak_memory_stats()
    mem_before = gpu_mem_mb()

    t0 = time.perf_counter()
    from autodex.perception import Sam3ImageSegmentor
    seg = Sam3ImageSegmentor(gpu=0)
    load_time = time.perf_counter() - t0
    mem_after_load = gpu_mem_mb()

    # Warmup
    _ = seg.segment(rgb, "object")
    torch.cuda.synchronize()

    # Single image
    t0 = time.perf_counter()
    mask = seg.segment(rgb, "object on the checkerboard")
    torch.cuda.synchronize()
    infer_time = time.perf_counter() - t0

    # Batch of 4
    t0 = time.perf_counter()
    for _ in range(4):
        seg.segment(rgb, "object on the checkerboard")
    torch.cuda.synchronize()
    seq4_time = time.perf_counter() - t0

    # Batch of 23
    t0 = time.perf_counter()
    for _ in range(23):
        seg.segment(rgb, "object on the checkerboard")
    torch.cuda.synchronize()
    seq23_time = time.perf_counter() - t0

    mem_peak = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"  {'Load:':<10} {load_time:>7.2f}s")
    print(f"  {'Infer x1:':<10} {infer_time:>7.3f}s")
    print(f"  {'Infer x4:':<10} {seq4_time:>7.3f}s  ({seq4_time/4:.3f}s/img)")
    print(f"  {'Infer x23:':<10} {seq23_time:>7.3f}s  ({seq23_time/23:.3f}s/img)")
    print(f"  {'GPU mem:':<10} {mem_before:>7.0f}MB -> {mem_after_load:.0f}MB "
          f"(+{mem_after_load - mem_before:.0f}MB, peak={mem_peak:.0f}MB)")
    print(f"  Mask: {'found' if mask is not None else 'NOT FOUND'}")
    print()

    del seg
    gc.collect()
    torch.cuda.empty_cache()


def bench_da3(rgb, K):
    import torch
    print("=" * 60)
    print("Depth-Anything-3 (monocular depth)")
    print("-" * 60)

    torch.cuda.reset_peak_memory_stats()
    mem_before = gpu_mem_mb()

    t0 = time.perf_counter()
    from autodex.perception import get_depth_da3
    depths = get_depth_da3([rgb], intrinsics=K[np.newaxis])
    torch.cuda.synchronize()
    load_and_first = time.perf_counter() - t0
    mem_after_load = gpu_mem_mb()

    t0 = time.perf_counter()
    depths = get_depth_da3([rgb], intrinsics=K[np.newaxis])
    torch.cuda.synchronize()
    infer_time = time.perf_counter() - t0

    # Batch of 4
    t0 = time.perf_counter()
    get_depth_da3([rgb] * 4, intrinsics=np.tile(K, (4, 1, 1)))
    torch.cuda.synchronize()
    batch4_time = time.perf_counter() - t0

    # Batch of 23 (all cameras)
    t0 = time.perf_counter()
    get_depth_da3([rgb] * 23, intrinsics=np.tile(K, (23, 1, 1)))
    torch.cuda.synchronize()
    batch23_time = time.perf_counter() - t0

    mem_peak = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"  {'Load+1st:':<10} {load_and_first:>7.2f}s")
    print(f"  {'Infer x1:':<10} {infer_time:>7.3f}s")
    print(f"  {'Infer x4:':<10} {batch4_time:>7.3f}s  ({batch4_time/4:.3f}s/img)")
    print(f"  {'Infer x23:':<10} {batch23_time:>7.3f}s  ({batch23_time/23:.3f}s/img)")
    print(f"  {'GPU mem:':<10} {mem_before:>7.0f}MB -> {mem_after_load:.0f}MB "
          f"(+{mem_after_load - mem_before:.0f}MB, peak={mem_peak:.0f}MB)")
    print(f"  Depth range: {depths[0].min():.2f} - {depths[0].max():.2f}m")
    print()

    gc.collect()
    torch.cuda.empty_cache()


def load_stereo_pair(capture_dir):
    """Load a stereo pair (first two cameras) with calibration."""
    import json
    capture_dir = Path(capture_dir)
    video_dir = capture_dir / "videos"
    serials = sorted(p.stem for p in video_dir.glob("*.avi"))
    if len(serials) < 2:
        return None

    with open(capture_dir / "cam_param" / "intrinsics.json") as f:
        intr_raw = json.load(f)
    with open(capture_dir / "cam_param" / "extrinsics.json") as f:
        extr_raw = json.load(f)

    # Pick first two serials that have extrinsics
    left_s, right_s = serials[0], serials[1]

    def read_first_frame(s):
        cap = cv2.VideoCapture(str(video_dir / f"{s}.avi"))
        ret, bgr = cap.read()
        cap.release()
        if ret:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return None

    left_rgb = read_first_frame(left_s)
    right_rgb = read_first_frame(right_s)
    if left_rgb is None or right_rgb is None:
        return None

    K_left = np.array(intr_raw[left_s]["intrinsics_undistort"], dtype=np.float32)
    K_right = np.array(intr_raw[right_s]["intrinsics_undistort"], dtype=np.float32)
    T_left = np.array(extr_raw[left_s], dtype=np.float64)
    T_right = np.array(extr_raw[right_s], dtype=np.float64)

    # Compute baseline
    if T_left.shape == (3, 4):
        T_left = np.vstack([T_left, [0, 0, 0, 1]])
    if T_right.shape == (3, 4):
        T_right = np.vstack([T_right, [0, 0, 0, 1]])
    rel = T_right @ np.linalg.inv(T_left)
    baseline = float(np.linalg.norm(rel[:3, 3]))

    print(f"Stereo pair: {left_s} + {right_s}, baseline={baseline:.4f}m, "
          f"image={left_rgb.shape}")

    return {
        "left_rgb": left_rgb, "right_rgb": right_rgb,
        "K_left": K_left, "K_right": K_right,
        "T_left": T_left, "T_right": T_right,
        "baseline": baseline,
        "left_serial": left_s, "right_serial": right_s,
    }


def bench_stereo_pytorch(capture_dir):
    """Benchmark FoundationStereo PyTorch."""
    import torch
    print("=" * 60)
    print("FoundationStereo (PyTorch)")
    print("-" * 60)

    stereo = load_stereo_pair(capture_dir)
    if stereo is None:
        print("  SKIP: need at least 2 cameras\n")
        return

    torch.cuda.reset_peak_memory_stats()
    mem_before = gpu_mem_mb()

    t0 = time.perf_counter()
    from autodex.perception.depth import get_depth_stereo_pytorch
    depth = get_depth_stereo_pytorch(
        stereo["left_rgb"], stereo["right_rgb"],
        stereo["K_left"], stereo["baseline"],
    )
    torch.cuda.synchronize()
    load_and_first = time.perf_counter() - t0
    mem_after = gpu_mem_mb()

    # Second run (model cached? no — it reloads each time unless passed in)
    # So this measures pure load+infer each time
    t0 = time.perf_counter()
    depth2 = get_depth_stereo_pytorch(
        stereo["left_rgb"], stereo["right_rgb"],
        stereo["K_left"], stereo["baseline"],
    )
    torch.cuda.synchronize()
    second_time = time.perf_counter() - t0

    mem_peak = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"  {'1st (load+infer):':<22} {load_and_first:>7.2f}s")
    print(f"  {'2nd (load+infer):':<22} {second_time:>7.2f}s")
    print(f"  {'GPU mem:':<22} {mem_before:>7.0f}MB -> {mem_after:.0f}MB "
          f"(+{mem_after - mem_before:.0f}MB, peak={mem_peak:.0f}MB)")
    print(f"  Depth range: {depth.min():.2f} - {depth.max():.2f}m")
    print()

    gc.collect()
    torch.cuda.empty_cache()


def bench_stereo_trt(capture_dir):
    """Benchmark FoundationStereo TensorRT."""
    print("=" * 60)
    print("FoundationStereo (TensorRT)")
    print("-" * 60)

    stereo = load_stereo_pair(capture_dir)
    if stereo is None:
        print("  SKIP: need at least 2 cameras\n")
        return

    mem_before = gpu_mem_mb()

    t0 = time.perf_counter()
    from autodex.perception.depth import StereoDepthTRT
    trt_model = StereoDepthTRT()
    load_time = time.perf_counter() - t0
    mem_after_load = gpu_mem_mb()

    # TRT needs rectified images — for benchmark, just run raw disparity
    # (not geometrically correct but measures timing)
    t0 = time.perf_counter()
    disp = trt_model._run_trt(stereo["left_rgb"], stereo["right_rgb"])
    trt_time = time.perf_counter() - t0

    # Run 10x for average
    t0 = time.perf_counter()
    for _ in range(10):
        disp = trt_model._run_trt(stereo["left_rgb"], stereo["right_rgb"])
    avg_time = (time.perf_counter() - t0) / 10

    mem_after = gpu_mem_mb()

    print(f"  {'Load engine:':<22} {load_time:>7.2f}s")
    print(f"  {'Infer x1:':<22} {trt_time:>7.3f}s")
    print(f"  {'Infer avg (x10):':<22} {avg_time:>7.3f}s")
    print(f"  {'GPU mem:':<22} {mem_before:>7.0f}MB -> {mem_after:.0f}MB "
          f"(+{mem_after - mem_before:.0f}MB)")
    print(f"  TRT resolution: {trt_model.W_trt}x{trt_model.H_trt}")
    print()

    del trt_model
    gc.collect()


def bench_fpose(rgb, depth, mask, K, mesh_path):
    import torch
    print("=" * 60)
    print("FoundationPose (6D pose)")
    print("-" * 60)

    if depth is None:
        print("  SKIP: no depth data available\n")
        return
    if mask is None:
        print("  SKIP: no mask data available\n")
        return

    torch.cuda.reset_peak_memory_stats()
    mem_before = gpu_mem_mb()

    t0 = time.perf_counter()
    from autodex.perception import PoseTracker
    tracker = PoseTracker(str(mesh_path), device_id=0)
    load_time = time.perf_counter() - t0
    mem_after_load = gpu_mem_mb()

    # Downscale
    ds = 0.5
    h, w = rgb.shape[:2]
    nw, nh = int(w * ds), int(h * ds)
    rgb_ds = cv2.resize(rgb, (nw, nh))
    depth_ds = cv2.resize(depth, (nw, nh), interpolation=cv2.INTER_NEAREST)
    mask_ds = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    K_ds = K.copy()
    K_ds[0] *= ds
    K_ds[1] *= ds

    # Register (5 iter)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    tracker.init(rgb_ds, depth_ds, mask_ds, K_ds, iteration=5)
    torch.cuda.synchronize()
    reg5_time = time.perf_counter() - t0
    mem_reg_peak = torch.cuda.max_memory_allocated() / 1024 / 1024

    # Register (2 iter)
    tracker.reset()
    t0 = time.perf_counter()
    tracker.init(rgb_ds, depth_ds, mask_ds, K_ds, iteration=2)
    torch.cuda.synchronize()
    reg2_time = time.perf_counter() - t0

    # Track (2 iter)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    tracker.track(rgb_ds, depth_ds, K_ds, iteration=2)
    torch.cuda.synchronize()
    track_time = time.perf_counter() - t0
    mem_track_peak = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"  {'Load:':<22} {load_time:>7.2f}s")
    print(f"  {'Register (5 iter):':<22} {reg5_time:>7.3f}s  (240 candidates)")
    print(f"  {'Register (2 iter):':<22} {reg2_time:>7.3f}s  (240 candidates)")
    print(f"  {'Track (2 iter):':<22} {track_time:>7.3f}s  (1 candidate)")
    print(f"  {'GPU mem (idle):':<22} {mem_after_load:>7.0f}MB  (+{mem_after_load - mem_before:.0f}MB)")
    print(f"  {'GPU mem (register):':<22} peak {mem_reg_peak:.0f}MB")
    print(f"  {'GPU mem (track):':<22} peak {mem_track_peak:.0f}MB")
    print()

    del tracker
    gc.collect()
    torch.cuda.empty_cache()


def bench_silhouette(rgb, mask, K, mesh_path):
    import torch
    print("=" * 60)
    print("Silhouette rendering (nvdiffrast mesh overlay)")
    print("-" * 60)

    if mask is None:
        print("  SKIP: no mask data available\n")
        return

    torch.cuda.reset_peak_memory_stats()
    mem_before = gpu_mem_mb()

    _FP_ROOT = AUTODEX_ROOT / "autodex/perception/thirdparty/_object_6d_tracking/FoundationPose"
    fp_path = str(_FP_ROOT)
    if fp_path not in sys.path:
        sys.path.insert(0, fp_path)

    import trimesh
    import nvdiffrast.torch as dr
    from Utils import make_mesh_tensors, nvdiffrast_render

    t0 = time.perf_counter()
    mesh = trimesh.load(str(mesh_path), force="mesh")
    mesh_tensors = make_mesh_tensors(mesh)
    glctx = dr.RasterizeCudaContext()
    load_time = time.perf_counter() - t0
    mem_after_load = gpu_mem_mb()

    pose = np.eye(4, dtype=np.float32)
    pose[2, 3] = 0.5
    h, w = rgb.shape[:2]
    K_f32 = K.astype(np.float32)
    pose_t = torch.as_tensor(pose, device="cuda", dtype=torch.float32).reshape(1, 4, 4)

    # Warmup
    nvdiffrast_render(K=K_f32, H=h, W=w, ob_in_cams=pose_t, glctx=glctx,
                      mesh_tensors=mesh_tensors, use_light=False)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    nvdiffrast_render(K=K_f32, H=h, W=w, ob_in_cams=pose_t, glctx=glctx,
                      mesh_tensors=mesh_tensors, use_light=False)
    torch.cuda.synchronize()
    render_time = time.perf_counter() - t0

    poses_batch = pose_t.repeat(10, 1, 1)
    poses_batch[:, 2, 3] = torch.linspace(0.3, 0.7, 10, device="cuda")
    t0 = time.perf_counter()
    nvdiffrast_render(K=K_f32, H=h, W=w, ob_in_cams=poses_batch, glctx=glctx,
                      mesh_tensors=mesh_tensors, use_light=False)
    torch.cuda.synchronize()
    batch_time = time.perf_counter() - t0

    mem_peak = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"  {'Load mesh+ctx:':<22} {load_time:>7.3f}s")
    print(f"  {'Render x1:':<22} {render_time:>7.3f}s")
    print(f"  {'Render x10:':<22} {batch_time:>7.3f}s  ({batch_time/10:.3f}s/pose)")
    print(f"  {'GPU mem:':<22} {mem_before:>7.0f}MB -> {mem_after_load:.0f}MB "
          f"(+{mem_after_load - mem_before:.0f}MB, peak={mem_peak:.0f}MB)")
    print()

    del mesh_tensors, glctx
    gc.collect()
    torch.cuda.empty_cache()


# ── Single-model runner (called by orchestrator) ─────────────────────────────

def run_single(model, capture_dir, mesh, serial):
    """Run one benchmark in-process. Called when --model is specified."""
    import torch
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print()

    rgb, depth, mask, K, serial = load_test_data(capture_dir, serial=serial)
    print()

    if model == "yoloe":
        bench_yoloe(rgb)
    elif model == "sam3":
        bench_sam3(rgb)
    elif model == "sam3_image":
        bench_sam3_image(rgb)
    elif model == "da3":
        bench_da3(rgb, K)
    elif model == "stereo_pytorch":
        bench_stereo_pytorch(capture_dir)
    elif model == "stereo_trt":
        bench_stereo_trt(capture_dir)
    elif model == "fpose":
        bench_fpose(rgb, depth, mask, K, mesh)
    elif model == "silhouette":
        bench_silhouette(rgb, mask, K, mesh)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark a single perception model. Use benchmark.sh to run all.")
    parser.add_argument("--model", type=str, required=True, choices=ALL_MODELS)
    parser.add_argument("--capture_dir", type=str, required=True)
    parser.add_argument("--mesh", type=str, required=True)
    parser.add_argument("--serial", type=str, default=None)
    args = parser.parse_args()

    run_single(args.model, args.capture_dir, args.mesh, args.serial)


if __name__ == "__main__":
    main()
