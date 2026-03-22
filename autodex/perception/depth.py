"""Depth estimation: stereo (FoundationStereo TRT/PyTorch/ONNX) and monocular (DA3).

New class-based API:
    StereoDepthTRT  — loads TRT engine once, auto pair selection, proper un-rectification

Legacy function API (preserved for backwards compatibility):
    get_depth_stereo_pytorch, get_depth_stereo, get_depth_da3
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

_JUNC0NG = Path(__file__).parent / "thirdparty/object-6d-tracking"
_THIRDPARTY = Path(__file__).parent / "thirdparty"
_FS_ROOT = Path(__file__).parent / "thirdparty/FoundationStereo"
_DEFAULT_ENGINE = _FS_ROOT / "output/foundation_stereo_448x672.engine"

logger = logging.getLogger(__name__)


# ── Utilities ────────────────────────────────────────────────────────────────


def _to_4x4(T: np.ndarray) -> np.ndarray:
    if T.shape == (3, 4):
        T4 = np.eye(4, dtype=np.float64)
        T4[:3, :] = T
        return T4
    return T.astype(np.float64)


def encode_depth_uint16(depth: np.ndarray) -> np.ndarray:
    """Encode depth (m) -> BGR uint8 for lossless video (FFV1).

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


def load_cam_param(capture_dir: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load intrinsics (undistorted) and extrinsics keyed by serial string."""
    param_dir = capture_dir / "cam_param"
    with open(param_dir / "intrinsics.json") as f:
        intr_raw = json.load(f)
    with open(param_dir / "extrinsics.json") as f:
        extr_raw = json.load(f)

    intrinsics = {s: np.array(v["intrinsics_undistort"], dtype=np.float64) for s, v in intr_raw.items()}
    extrinsics = {s: np.array(extr_raw[s], dtype=np.float64) for s in intr_raw}
    return intrinsics, extrinsics


def _compute_rectify_params(K_left, K_right, T_left, T_right, image_size):
    """Compute stereo rectification geometry (no maps, no allocation)."""
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
    """Find the union of both cameras' valid regions on an oversized canvas."""
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

    valid_both = valid_l | valid_r  # Union
    ys, xs = np.where(valid_both)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return (x0, y0, x1, y1, W_big, H_big)


def _workspace_crop(K_left, T_left, R1, K_right, T_right, R2,
                    f_rect, cx_big, cy_big,
                    vx0, vy0, vx1, vy1,
                    capture_dir):
    """Compute workspace crop in big-canvas coordinates.

    Same crop for both cameras. Union of both cameras' workspace projections.
    Returns (cx0, cy0, cx1, cy1) in big-canvas coords, or None.
    """
    from pathlib import Path
    if capture_dir is None:
        return None
    c2r_path = Path(capture_dir) / "C2R.npy"
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
    all_px = np.concatenate(per_cam_px)
    all_py = np.concatenate(per_cam_py)
    cx0 = max(vx0, int(all_px.min()) - margin)
    cx1 = min(vx1, int(all_px.max()) + margin)
    cy0 = max(vy0, int(all_py.min()) - margin)
    cy1 = min(vy1, int(all_py.max()) + margin)
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    return (cx0, cy0, cx1, cy1)


def build_rectify_maps(K_left, K_right, T_left, T_right, image_size,
                       capture_dir=None):
    """Compute stereo rectification maps cropped to valid region + workspace.

    Output maps remap raw images directly to the intersection of both cameras'
    valid regions, optionally cropped to robot workspace.

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

    logger.info(f"    Rectify+crop: valid {vx1-vx0}x{vy1-vy0} -> out {W_out}x{H_out}")

    return (map_left, map_right, R1, R2, f_rect,
            cx_out, cy_out, baseline, (W_out, H_out), 0.0)


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
    rz_safe = np.where(np.abs(rz) > 1e-6, rz, 1.0)
    depth_map = depth_rect_sampled / rz_safe.astype(np.float32)
    depth_map[depth_rect_sampled < 0.001] = 0
    return depth_map


def find_all_stereo_pairs(capture_dir, serials, intrinsics, extrinsics):
    """Find stereo pairs using rig-based adjacency grouping.

    Groups by (focal_group, z_level), sorts by angle, pairs adjacent cameras.
    Returns: [(left_serial, right_serial, baseline_m), ...]
    """
    from collections import defaultdict
    from pathlib import Path

    MAX_ANGLE_GAP = 40.0

    def focal_group(serial):
        f = float(intrinsics[serial][0, 0])
        if f < 2500:
            return "wide"
        elif f < 4000:
            return "mid"
        return "tele"

    c2r_path = Path(capture_dir) / "C2R.npy"
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
        logger.info(f"  {grp}/{zlev}: {len(ss)} cameras — {ss}")
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


def find_best_stereo_partner(
    target_serial: str,
    serials: List[str],
    intrinsics: Dict[str, np.ndarray],
    extrinsics: Dict[str, np.ndarray],
    min_baseline: float = 0.03,
    max_baseline: float = 0.50,
    min_cos_sim: float = 0.77,
    min_perp: float = 0.30,
    max_f_ratio: float = 3.0,
    max_rect_f_ratio: float = 2.0,
) -> Optional[Tuple[str, float]]:
    """Find the best stereo partner for a given camera.

    Args:
        target_serial: serial of the camera to find a partner for
        serials: list of all camera serials
        intrinsics: {serial: K (3,3)}
        extrinsics: {serial: T (3,4) or (4,4)}  world-to-cam
        max_rect_f_ratio: max ratio of rectified focal length to original.
            Pairs where stereoRectify inflates f beyond this are rejected
            (indicates cameras need too much rotation to align epipolar lines).

    Returns:
        (partner_serial, baseline_m) or None
    """
    positions = {}
    fwd_dirs = {}
    focal_lengths = {}
    for s in serials:
        T = _to_4x4(extrinsics[s])
        R = T[:3, :3]
        t = T[:3, 3]
        positions[s] = -R.T @ t  # camera position in world
        fwd = R[2, :]
        fwd_dirs[s] = fwd / (np.linalg.norm(fwd) + 1e-9)
        focal_lengths[s] = float(intrinsics[s][0, 0])

    def _check_rect_quality(s1, s2, max_f_ratio, max_rot_deg=28.0):
        """Check if stereoRectify produces reasonable result.

        Rejects pairs where:
        - Rectified focal length is inflated too much (image gets zoomed/cropped)
        - Either camera needs rotation > max_rot_deg to align epipolar lines
        """
        K1, K2 = intrinsics[s1], intrinsics[s2]
        T1, T2 = extrinsics[s1], extrinsics[s2]
        W = int(max(K1[0, 2], K2[0, 2]) * 2)
        H = int(max(K1[1, 2], K2[1, 2]) * 2)
        T_rel = _to_4x4(T2) @ np.linalg.inv(_to_4x4(T1))
        R1, R2, P1, _, _, _, _ = cv2.stereoRectify(
            K1, None, K2, None, (W, H),
            T_rel[:3, :3], T_rel[:3, 3],
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
        )
        f_rect = float(P1[0, 0])
        orig_f = max(focal_lengths[s1], focal_lengths[s2])
        if f_rect <= 0 or f_rect / orig_f > max_f_ratio:
            return False
        # Check rotation angles — large rotation means image gets severely warped
        for R in (R1, R2):
            angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
            if angle > max_rot_deg:
                return False
        return True

    def _find(min_b, max_b, min_cos, min_p, max_fr, max_rf):
        best, best_score = None, -1.0
        s1 = target_serial
        for s2 in serials:
            if s2 == s1:
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
            if perp < min_p:
                continue
            f1, f2 = focal_lengths[s1], focal_lengths[s2]
            if max(f1, f2) / (min(f1, f2) + 1e-9) > max_fr:
                continue
            if cs > best_score:
                # Expensive check last: actually run stereoRectify
                if not _check_rect_quality(s1, s2, max_rf):
                    continue
                best_score = cs
                best = (s2, b)
        return best

    return (
        _find(min_baseline, max_baseline, min_cos_sim, min_perp, max_f_ratio, max_rect_f_ratio)
        or _find(min_baseline, 1.0, 0.50, 0.20, 2.0, 2.0)
    )


def _auto_order_stereo(
    K_a: np.ndarray, K_b: np.ndarray, T_a: np.ndarray, T_b: np.ndarray,
) -> bool:
    """Return True if (a, b) should be swapped to make a proper left-right pair.

    stereoRectify expects right camera to the right of left.
    P2[0,3] = -f*Tx. If P2[0,3] > 0, cameras need swapping.
    """
    T_rel = _to_4x4(T_b) @ np.linalg.inv(_to_4x4(T_a))
    _, _, _, P2, _, _, _ = cv2.stereoRectify(
        K_a, None, K_b, None, (100, 100),
        T_rel[:3, :3], T_rel[:3, 3],
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )
    return P2[0, 3] > 0


# ── StereoDepthTRT ───────────────────────────────────────────────────────────


class StereoDepthTRT:
    """Stereo depth estimation using FoundationStereo TensorRT engine.

    Loads the TRT engine once, supports:
    - Auto stereo pair selection per camera
    - Auto left/right ordering
    - Proper un-rectification: disparity → 3D world points → per-camera depth
    """

    def __init__(self, engine_path: str = None):
        """Load TRT engine and allocate GPU buffers.

        Args:
            engine_path: path to .engine file (default: thirdparty/FoundationStereo/output/foundation_stereo_448x672.engine)
        """
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401

        if engine_path is None:
            engine_path = str(_DEFAULT_ENGINE)

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        self._context = engine.create_execution_context()

        trt_shape = engine.get_tensor_shape("left")  # (1, 3, H_trt, W_trt)
        self.H_trt = int(trt_shape[2])
        self.W_trt = int(trt_shape[3])

        disp_arr = np.zeros((1, 1, self.H_trt, self.W_trt), dtype=np.float32)
        self._buffers = {
            "d_left": cuda.mem_alloc(int(np.prod([1, 3, self.H_trt, self.W_trt])) * 4),
            "d_right": cuda.mem_alloc(int(np.prod([1, 3, self.H_trt, self.W_trt])) * 4),
            "d_disp": cuda.mem_alloc(disp_arr.nbytes),
            "disp_arr": disp_arr,
            "stream": cuda.Stream(),
        }
        logger.info(f"TRT engine loaded: {self.W_trt}x{self.H_trt}")

    def _run_trt(self, left_rgb: np.ndarray, right_rgb: np.ndarray) -> np.ndarray:
        """Run TRT inference on a rectified RGB pair. Returns disparity at TRT resolution."""
        import pycuda.driver as cuda

        left_trt = cv2.resize(left_rgb, (self.W_trt, self.H_trt), interpolation=cv2.INTER_LINEAR)
        right_trt = cv2.resize(right_rgb, (self.W_trt, self.H_trt), interpolation=cv2.INTER_LINEAR)
        left_arr = np.ascontiguousarray(left_trt.astype(np.float32).transpose(2, 0, 1)[None])
        right_arr = np.ascontiguousarray(right_trt.astype(np.float32).transpose(2, 0, 1)[None])

        stream = self._buffers["stream"]
        cuda.memcpy_htod_async(self._buffers["d_left"], left_arr, stream)
        cuda.memcpy_htod_async(self._buffers["d_right"], right_arr, stream)
        self._context.set_tensor_address("left", int(self._buffers["d_left"]))
        self._context.set_tensor_address("right", int(self._buffers["d_right"]))
        self._context.set_tensor_address("disp", int(self._buffers["d_disp"]))
        self._context.execute_async_v3(stream.handle)
        stream.synchronize()
        cuda.memcpy_dtoh(self._buffers["disp_arr"], self._buffers["d_disp"])

        return self._buffers["disp_arr"].squeeze()  # (H_trt, W_trt)

    def _disp_to_world_points(
        self,
        disp_trt: np.ndarray,
        f_rect: float,
        cx_rect: float,
        cy_rect: float,
        baseline: float,
        R1: np.ndarray,
        T_left: np.ndarray,
        W_orig: int,
        H_orig: int,
    ) -> np.ndarray:
        """Convert TRT disparity to 3D world points.

        Pipeline: disparity → 3D rectified → R1.T → left cam frame → T_left_inv → world
        Returns: (N, 3) world points
        """
        # Scale rectified intrinsics to TRT resolution
        fx_trt = f_rect * self.W_trt / W_orig
        fy_trt = f_rect * self.H_trt / H_orig
        cx_trt = cx_rect * self.W_trt / W_orig
        cy_trt = cy_rect * self.H_trt / H_orig

        valid = disp_trt >= 0.5
        disp_v = np.maximum(disp_trt[valid], 0.1)
        depth_v = fx_trt * baseline / disp_v  # disparity is horizontal → use fx

        u_g, v_g = np.meshgrid(
            np.arange(self.W_trt, dtype=np.float32),
            np.arange(self.H_trt, dtype=np.float32),
        )
        pts_rect = np.stack([
            (u_g[valid] - cx_trt) * depth_v / fx_trt,
            (v_g[valid] - cy_trt) * depth_v / fy_trt,
            depth_v,
        ], axis=1)

        # Rectified → original left camera frame
        pts_left = (R1.T @ pts_rect.T).T

        # Left camera → world
        T_left_inv = np.linalg.inv(_to_4x4(T_left))
        pts_world = (T_left_inv[:3, :3] @ pts_left.T).T + T_left_inv[:3, 3]

        return pts_world

    @staticmethod
    def _project_to_depth_map(
        pts_world: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        H: int,
        W: int,
    ) -> np.ndarray:
        """Project world points to a depth map for a given camera.

        Args:
            pts_world: (N, 3) world coordinates
            K: (3, 3) intrinsics
            T: (3, 4) or (4, 4) world-to-cam extrinsics
            H, W: output image size

        Returns:
            depth_map: (H, W) float32 in meters
        """
        T4 = _to_4x4(T)
        pts_cam = (T4[:3, :3] @ pts_world.T).T + T4[:3, 3]
        in_front = pts_cam[:, 2] > 0.01
        pts_v = pts_cam[in_front]

        px = (K[0, 0] * pts_v[:, 0] / pts_v[:, 2] + K[0, 2]).astype(int)
        py = (K[1, 1] * pts_v[:, 1] / pts_v[:, 2] + K[1, 2]).astype(int)
        in_img = (px >= 0) & (px < W) & (py >= 0) & (py < H)

        depth_map = np.zeros((H, W), dtype=np.float32)
        z_vals = pts_v[in_img, 2]
        order = np.argsort(z_vals)[::-1]  # far → near, so near overwrites
        depth_map[py[in_img][order], px[in_img][order]] = z_vals[order]
        return depth_map

    def estimate_pair(
        self,
        left_rgb: np.ndarray,
        right_rgb: np.ndarray,
        K_left: np.ndarray,
        K_right: np.ndarray,
        T_left: np.ndarray,
        T_right: np.ndarray,
        target_serials: Optional[List[str]] = None,
        target_intrinsics: Optional[Dict[str, np.ndarray]] = None,
        target_extrinsics: Optional[Dict[str, np.ndarray]] = None,
        debug_dir: Optional[str] = None,
        pair_label: Optional[str] = None,
    ) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """Run stereo depth on one pair, with auto left/right ordering.

        Args:
            left_rgb, right_rgb: RGB images (H, W, 3)
            K_left, K_right: (3, 3) intrinsics
            T_left, T_right: world-to-cam extrinsics
            target_serials: if given, reproject world points to these cameras
            target_intrinsics: {serial: K} for target cameras
            target_extrinsics: {serial: T} for target cameras
            debug_dir: if given, save debug images (rectified pair, disparity colormap)
            pair_label: label for debug filenames (e.g. "25305461_25322646")

        Returns:
            pts_world: (N, 3) world points
            depths: {serial: (H, W)} depth maps if targets given, else None
        """
        H, W = left_rgb.shape[:2]

        # Auto left/right ordering
        swapped = _auto_order_stereo(K_left, K_right, T_left, T_right)
        if swapped:
            logger.info("Auto-swap: cameras were in wrong left-right order")
            left_rgb, right_rgb = right_rgb, left_rgb
            K_left, K_right = K_right, K_left
            T_left, T_right = T_right, T_left

        # Rectify
        map_left, map_right, R1, R2, f_rect, cx_rect, cy_rect, baseline, out_size, disp_offset = \
            build_rectify_maps(K_left, K_right, T_left, T_right, (W, H))
        W_rect, H_rect = out_size

        if baseline < 0.01 or f_rect > 20000:
            logger.warning(f"Degenerate rectification: f={f_rect:.1f}, baseline={baseline:.4f}")
            return np.zeros((0, 3), dtype=np.float32), None

        left_rect = cv2.remap(left_rgb, map_left[0], map_left[1], cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_rgb, map_right[0], map_right[1], cv2.INTER_LINEAR)

        # TRT inference
        disp_trt = self._run_trt(left_rect, right_rect)

        # Save debug images
        if debug_dir is not None:
            self._save_pair_debug(
                left_rgb, right_rgb, left_rect, right_rect, disp_trt,
                debug_dir, pair_label or "pair", swapped,
                f_rect, baseline,
            )

        # Disparity → world points (use rectified image size, not original)
        pts_world = self._disp_to_world_points(
            disp_trt, f_rect, cx_rect, cy_rect, baseline,
            R1, T_left, W_rect, H_rect,
        )
        logger.info(f"  f={f_rect:.1f}px  baseline={baseline:.4f}m  {len(pts_world)} world points")

        # Reproject to target cameras
        depths = None
        if target_serials is not None and target_intrinsics is not None and target_extrinsics is not None:
            depths = {}
            for s in target_serials:
                depths[s] = self._project_to_depth_map(
                    pts_world, target_intrinsics[s], target_extrinsics[s], H, W,
                )

        return pts_world, depths

    @staticmethod
    def _save_pair_debug(
        left_orig, right_orig, left_rect, right_rect, disp_trt,
        debug_dir, label, swapped, f_rect, baseline,
    ):
        """Save debug images: original pair, rectified pair with epipolar lines, disparity colormap."""
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

        H, W = left_rect.shape[:2]

        # --- Original pair side-by-side ---
        orig_pair = np.hstack([
            cv2.cvtColor(left_orig, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(right_orig, cv2.COLOR_RGB2BGR),
        ])
        h_orig = orig_pair.shape[0]
        # Shrink for reasonable file size
        scale = min(1.0, 1200.0 / orig_pair.shape[1])
        if scale < 1.0:
            orig_pair = cv2.resize(orig_pair, None, fx=scale, fy=scale)
        cv2.putText(orig_pair, f"Original L/R ({'swapped' if swapped else 'as-is'})",
                     (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(str(debug_dir / f"{label}_1_original.jpg"), orig_pair)

        # --- Rectified pair with epipolar lines ---
        rect_pair = np.hstack([
            cv2.cvtColor(left_rect, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(right_rect, cv2.COLOR_RGB2BGR),
        ])
        # Draw horizontal epipolar lines
        for y in range(0, H, H // 12):
            color = tuple(int(c) for c in np.random.randint(80, 255, 3))
            cv2.line(rect_pair, (0, y), (2 * W, y), color, 1)
        scale = min(1.0, 1200.0 / rect_pair.shape[1])
        if scale < 1.0:
            rect_pair = cv2.resize(rect_pair, None, fx=scale, fy=scale)
        cv2.putText(rect_pair, f"Rectified f={f_rect:.0f} bl={baseline:.4f}m",
                     (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(str(debug_dir / f"{label}_2_rectified.jpg"), rect_pair)

        # --- Disparity colormap ---
        disp = disp_trt.copy()
        valid = disp >= 0.5
        if valid.any():
            d_min, d_max = disp[valid].min(), np.percentile(disp[valid], 98)
            d_norm = np.clip((disp - d_min) / (d_max - d_min + 1e-6), 0, 1)
            d_norm[~valid] = 0
            cmap = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            cmap[~valid] = 0
        else:
            cmap = np.zeros((disp.shape[0], disp.shape[1], 3), dtype=np.uint8)
        scale = min(1.0, 600.0 / cmap.shape[1])
        if scale < 1.0:
            cmap = cv2.resize(cmap, None, fx=scale, fy=scale)
        cv2.putText(cmap, "Disparity", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(str(debug_dir / f"{label}_3_disparity.jpg"), cmap)

    def estimate_capture(
        self,
        capture_dir: str,
        frame_reader=None,
        debug_dir: str = None,
    ) -> Dict[str, np.ndarray]:
        """Estimate depth for all cameras in a capture directory.

        For each camera, finds the best stereo partner, runs stereo depth,
        and reprojects world points to get a per-camera depth map.

        Args:
            capture_dir: path with cam_param/ and videos/ or images/
            frame_reader: callable(serial) -> RGB image (H, W, 3).
                          If None, reads from {capture_dir}/videos/{serial}.avi frame 0
                          or {capture_dir}/images/{serial}.png.
            debug_dir: if given, save debug images per stereo pair
                       (original pair, rectified pair with epipolar lines, disparity colormap)

        Returns:
            depths: {serial: depth_map (H, W) float32 meters}
        """
        capture_dir = Path(capture_dir)
        intrinsics, extrinsics = load_cam_param(capture_dir)
        serials = list(intrinsics.keys())

        if frame_reader is None:
            frame_reader = self._default_frame_reader(capture_dir, serials)

        # Read one image to get resolution
        sample_img = frame_reader(serials[0])
        H, W = sample_img.shape[:2]

        # For each camera, find best partner and accumulate unique pairs
        pairs = {}  # {(left_s, right_s): set of target serials}
        partner_map = {}  # {serial: (left_s, right_s)}
        for s in serials:
            result = find_best_stereo_partner(s, serials, intrinsics, extrinsics)
            if result is None:
                logger.warning(f"  No stereo partner for {s}")
                continue
            partner_s, baseline_m = result
            # Canonicalize pair order for deduplication
            pair_key = tuple(sorted([s, partner_s]))
            if pair_key not in pairs:
                pairs[pair_key] = set()
            pairs[pair_key].add(s)
            partner_map[s] = pair_key

        logger.info(f"{len(serials)} cameras, {len(pairs)} unique stereo pairs")

        # Run each unique pair once, reproject to all cameras
        all_world_pts = []
        for (s_a, s_b), targets in pairs.items():
            img_a = frame_reader(s_a)
            img_b = frame_reader(s_b)
            pts_world, _ = self.estimate_pair(
                img_a, img_b,
                intrinsics[s_a], intrinsics[s_b],
                extrinsics[s_a], extrinsics[s_b],
                debug_dir=debug_dir,
                pair_label=f"{s_a}_{s_b}",
            )
            if len(pts_world) > 0:
                all_world_pts.append(pts_world)

        if not all_world_pts:
            logger.warning("No valid stereo pairs produced points")
            return {s: np.zeros((H, W), dtype=np.float32) for s in serials}

        # Merge all world points and project to each camera
        pts_world = np.concatenate(all_world_pts, axis=0)
        logger.info(f"Total: {len(pts_world)} world points from {len(all_world_pts)} pairs")

        depths = {}
        for s in serials:
            depths[s] = self._project_to_depth_map(
                pts_world, intrinsics[s], extrinsics[s], H, W,
            )

        return depths

    @staticmethod
    def _default_frame_reader(capture_dir: Path, serials: List[str]):
        """Create a frame reader that tries images/ then videos/ frame 0."""
        images_dir = capture_dir / "images"
        videos_dir = capture_dir / "videos"

        def reader(serial: str) -> np.ndarray:
            # Try PNG image first
            img_path = images_dir / f"{serial}.png"
            if img_path.exists():
                bgr = cv2.imread(str(img_path))
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            # Fall back to video frame 0
            vid_path = videos_dir / f"{serial}.avi"
            if vid_path.exists():
                cap = cv2.VideoCapture(str(vid_path))
                ret, bgr = cap.read()
                cap.release()
                if ret:
                    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            raise FileNotFoundError(f"No image or video for {serial} in {capture_dir}")

        return reader


# ── Legacy function API ──────────────────────────────────────────────────────


def _setup_foundation_stereo_path():
    path = str(_FS_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


def get_depth_stereo_pytorch(
    left_img: np.ndarray,
    right_img: np.ndarray,
    K: np.ndarray,
    baseline: float,
    model=None,
    ckpt_dir: Optional[str] = None,
    valid_iters: int = 32,
) -> np.ndarray:
    """Estimate metric depth from stereo pair using FoundationStereo (PyTorch).

    Args:
        left_img: left RGB image (H, W, 3)
        right_img: right RGB image (H, W, 3)
        K: camera intrinsic matrix (3, 3) for left camera
        baseline: stereo baseline in meters
        model: pre-loaded FoundationStereo model (loads from ckpt_dir if None)
        ckpt_dir: path to model_best_bp2.pth (default: thirdparty/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth)
        valid_iters: number of flow-field updates

    Returns:
        depth: metric depth map in meters (H, W), same resolution as input
    """
    _setup_foundation_stereo_path()
    from omegaconf import OmegaConf
    from core.foundation_stereo import FoundationStereo
    from core.utils.utils import InputPadder

    if ckpt_dir is None:
        ckpt_dir = str(_FS_ROOT / "pretrained_models/23-51-11/model_best_bp2.pth")

    if model is None:
        cfg = OmegaConf.load(str(Path(ckpt_dir).parent / "cfg.yaml"))
        if "vit_size" not in cfg:
            cfg["vit_size"] = "vitl"
        cfg["valid_iters"] = valid_iters
        cfg["hiera"] = 0
        model = FoundationStereo(cfg)
        ckpt = torch.load(ckpt_dir, map_location="cuda")
        model.load_state_dict(ckpt["model"])
        model.cuda()
        model.eval()

    orig_h, orig_w = left_img.shape[:2]

    img0 = torch.as_tensor(left_img.copy()).cuda().float()[None].permute(0, 3, 1, 2)
    img1 = torch.as_tensor(right_img.copy()).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)

    with torch.no_grad(), torch.cuda.amp.autocast(True):
        disp = model.forward(img0, img1, iters=valid_iters, test_mode=True)
    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(orig_h, orig_w)

    depth = K[0, 0] * baseline / (disp + 1e-10)
    return depth


def get_depth_stereo(
    left_img: np.ndarray,
    right_img: np.ndarray,
    model,
    K: np.ndarray,
    baseline: float,
    height: int = None,
    width: int = None,
) -> np.ndarray:
    """Estimate depth from stereo pair using FoundationStereo (TensorRT/ONNX).

    Args:
        left_img: left RGB image (H, W, 3)
        right_img: right RGB image (H, W, 3)
        model: pre-loaded TensorRT or ONNX model
        K: camera intrinsic matrix (3, 3)
        baseline: stereo baseline in meters
        height: resize height for model input (None = keep original)
        width: resize width for model input (None = keep original)

    Returns:
        depth: depth map in meters (H, W)
    """
    foundation_stereo_path = str(_JUNC0NG / "thirdparty/FoundationStereo")
    if foundation_stereo_path not in sys.path:
        sys.path.insert(0, foundation_stereo_path)

    orig_h, orig_w = left_img.shape[:2]

    if height and width:
        left_resized = cv2.resize(left_img, (width, height))
        right_resized = cv2.resize(right_img, (width, height))
        scale_x = width / orig_w
        scale_y = height / orig_h
    else:
        left_resized = left_img
        right_resized = right_img
        scale_x = scale_y = 1.0

    H, W = left_resized.shape[:2]

    left_tensor = torch.as_tensor(left_resized.copy()).float()[None].permute(0, 3, 1, 2)
    right_tensor = torch.as_tensor(right_resized.copy()).float()[None].permute(0, 3, 1, 2)

    is_onnx = not hasattr(model, "run") or hasattr(model, "get_inputs")
    if is_onnx:
        disp = model.run(None, {"left": left_tensor.numpy(), "right": right_tensor.numpy()})[0]
    else:
        disp = model.run([left_tensor.numpy(), right_tensor.numpy()])[0]

    disp = disp.squeeze()
    if disp.ndim == 1:
        disp = disp.reshape(H, W)

    fx_scaled = K[0, 0] * scale_x
    depth = fx_scaled * baseline / (disp + 1e-10)

    if height and width:
        depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    return depth


def get_depth_da3(
    images: List[np.ndarray],
    intrinsics: Optional[np.ndarray] = None,
    extrinsics: Optional[np.ndarray] = None,
    model=None,
    process_res: int = 504,
) -> List[np.ndarray]:
    """Estimate depth from monocular images using Depth-Anything-3.

    Args:
        images: list of RGB images (H, W, 3)
        intrinsics: camera intrinsics (N, 3, 3) or None
        extrinsics: camera extrinsics (N, 4, 4) or None — needed for multi-view metric scale alignment
        model: pre-loaded DepthAnything3 model (loads da3-large if None)
        process_res: processing resolution

    Returns:
        depth_maps: list of depth maps (H, W) in meters
    """
    da3_src = str(Path(__file__).parent / "thirdparty/Depth-Anything-3/src")
    if da3_src not in sys.path:
        sys.path.insert(0, da3_src)

    if model is None:
        from depth_anything_3.api import DepthAnything3
        model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE")
        model = model.to("cuda")
        model.eval()

    try:
        prediction = model.inference(
            image=images,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            process_res=process_res,
        )
    except Exception:
        if extrinsics is not None:
            logger.warning("DA3 alignment failed with extrinsics, retrying without")
            prediction = model.inference(
                image=images,
                intrinsics=intrinsics,
                process_res=process_res,
            )
        else:
            raise

    return [d.cpu().numpy() if hasattr(d, 'cpu') else np.asarray(d) for d in prediction.depth]