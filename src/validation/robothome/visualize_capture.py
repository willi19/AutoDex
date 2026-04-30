"""Visualize a robothome_demo_inspire_left capture: robot trajectory + ChArUco board.

What this does (per ``--capture-dir``):

1. Read the first frame of every ``videos/<serial>.avi``.
2. Undistort with ``cam_param/intrinsics.json``.
3. Detect ChArUco corners (``paradex.image.aruco.detect_charuco``),
   triangulate to 3D in the camera-world frame, reproject to every view.
   Detected corners (red) and reprojected corners (green) are drawn and
   the merged grid is saved to ``<capture_dir>/charuco_debug.jpg``.
4. Launch a viser viewer in the robot base frame:
     - FR3 + Inspire-left URDF animated from ``raw/arm/position.npy`` and
       ``raw/hand/position.npy`` (timeline = arm timestamps; hand
       interpolated to match).
     - Triangulated ChArUco corners as a point cloud (transformed by C2R).
     - Camera frames (extrinsics inverted, then C2R).

No object pose for now — pose detection is not wired up yet, just robot
+ board.

Usage:
    python src/validation/robothome/visualize_capture.py \
        --capture-dir /home/mingi/paradex1/capture/robothome_demo_inspire_left/oreo/1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import viser
import yourdfpy
from scipy.spatial.transform import Rotation as R
from viser.extras import ViserUrdf

# paradex isn't installed; add it to path.
PARADEX_ROOT = Path("/home/mingi/paradex")
if str(PARADEX_ROOT) not in sys.path:
    sys.path.insert(0, str(PARADEX_ROOT))

import json as _json  # noqa: E402

import paradex.image.aruco as _paruco  # noqa: E402
from cv2 import aruco as _cv2aruco  # noqa: E402


def _patch_charuco_system(system_name: str = "robothome"):
    """Replace paradex's globally cached charuco boards/detectors.

    `paradex.image.aruco` caches boards/detectors at import time from
    ``system/current/charuco_info.json``. The robothome captures use a
    different board layout (4 boards × 11×8, IDs 0–175), so we swap the
    config in place and invalidate the caches before any detection runs.
    """
    cfg_path = PARADEX_ROOT / "system" / system_name / "charuco_info.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"charuco config not found: {cfg_path}")
    new_cfg = _json.load(open(cfg_path))
    _paruco.boardinfo_dict.clear()
    _paruco.boardinfo_dict.update(new_cfg)
    _paruco._charuco_detector_cache.clear()
    _paruco._charuco_board_cache.clear()
    for name, bi in new_cfg.items():
        board = _cv2aruco.CharucoBoard(
            (bi["numX"], bi["numY"]),
            float(bi["checkerLength"]),
            float(bi["markerLength"]),
            _paruco.aruco_dict[bi["dict_type"]],
            np.array(bi["markerIDs"]),
        )
        if bi.get("setLegacyPattern", False):
            board.setLegacyPattern(True)
        else:
            board.setLegacyPattern(False)
        _paruco._charuco_board_cache[name] = board


from paradex.calibration.utils import load_camparam, load_c2r  # noqa: E402
from paradex.image.aruco import detect_charuco, merge_charuco_detection  # noqa: E402
from paradex.image.image_dict import ImageDict  # noqa: E402

HERE = Path(__file__).resolve().parent
# robothome_demo_inspire_left captures use xarm6 + inspire-left, not FR3.
# Local copy of paradex1/.../xarm_inspire_left_new.urdf with the
# `arm_to_hand` 180° X-flip removed (real mounting has hand z aligned with
# link6 z). Mesh refs ("xarm/meshes/...", "inspire/meshes_left/...") still
# resolve from the original paradex rsc location below.
URDF_PATH = HERE / "xarm_inspire_left.urdf"
URDF_MESH_DIR = Path("/home/mingi/paradex1/capture/eccv2026/paradex/rsc/robot")

ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]

# Inspire-left controller order matches the captured int array:
# [pinky, ring, middle, index, thumb_pitch, thumb_yaw], 1000=open, 0=closed.
HAND_DRIVERS = [
    "left_little_1_joint",
    "left_ring_1_joint",
    "left_middle_1_joint",
    "left_index_1_joint",
    "left_thumb_2_joint",   # thumb pitch (limit 0.55)
    "left_thumb_1_joint",   # thumb yaw  (limit 1.15)
]
INSPIRE_LIMITS = np.array([1.6, 1.6, 1.6, 1.6, 0.55, 1.15])

# Capture-dir name → object mesh dir under
# ~/shared_data/AutoDex/object/{robothome,paradex}/<name>/visual_mesh/<name>.obj
CAPTURE_TO_OBJECT = {
    "choco": "chocoSong-i",
    "oreo": "Oreo",
    "ghana": "ghanaCrush2",
    "nuts": "mixedNuts",
}
OBJECT_ROOTS = [
    Path.home() / "shared_data" / "AutoDex" / "object" / "robothome",
    Path.home() / "shared_data" / "AutoDex" / "object" / "paradex",
]


def resolve_object_mesh(name: str | None) -> Path | None:
    if not name:
        return None
    for root in OBJECT_ROOTS:
        for sub in ("visual_mesh", "raw_mesh"):
            p = root / name / sub / f"{name}.obj"
            if p.exists():
                return p
    return None


def load_tabletop_poses(name: str | None) -> dict[str, np.ndarray]:
    """{<idx>: 4x4 SE3} of pre-computed lying-on-table poses for ``name``."""
    out: dict[str, np.ndarray] = {}
    if not name:
        return out
    for root in OBJECT_ROOTS:
        d = root / name / "processed_data" / "info" / "tabletop"
        if d.is_dir():
            for p in sorted(d.glob("*.npy")):
                out[p.stem] = np.load(p)
            if out:
                return out
    return out


def inspire_int_to_rad(ints: np.ndarray) -> np.ndarray:
    """(T, 6) ints in [0, 1000] → (T, 6) joint angles for HAND_DRIVERS."""
    f = (1000.0 - np.asarray(ints, dtype=np.float64)) / 1000.0
    return f * INSPIRE_LIMITS[None, :]


def video_frame_image_dict(
    capture_dir: Path,
    intrinsic,
    extrinsic,
    frame_idx: int = 1,
) -> ImageDict:
    """Read frame ``frame_idx`` from every video matching a calibrated serial.

    Defaults to ``frame_idx=1`` because video frame 0 is sometimes a partial
    frame depending on the recorder. Read sequentially (cv2.CAP_PROP_POS_FRAMES
    is unreliable on MJPG/AVI).
    """
    images = {}
    for serial in sorted(intrinsic.keys()):
        vid = capture_dir / "videos" / f"{serial}.avi"
        if not vid.exists():
            continue
        cap = cv2.VideoCapture(str(vid))
        frame = None
        for _ in range(frame_idx + 1):
            ok, frame = cap.read()
            if not ok:
                frame = None
                break
        cap.release()
        if frame is None:
            print(f"[charuco] could not read frame {frame_idx}: {vid.name}")
            continue
        images[serial] = frame
    if not images:
        raise RuntimeError(f"No frames loaded from {capture_dir}/videos at idx={frame_idx}")
    return ImageDict(images, intrinsic=intrinsic, extrinsic=extrinsic, path=None)


def arm_index_for_video_frame(capture_dir: Path, video_frame_idx: int, arm_t: np.ndarray) -> int:
    """Match a video frame to its closest arm-state index by timestamp."""
    video_ts = np.load(capture_dir / "raw/timestamps/timestamp.npy")
    if video_frame_idx >= len(video_ts):
        return min(video_frame_idx, len(arm_t) - 1)
    target = float(video_ts[video_frame_idx])
    return int(np.argmin(np.abs(arm_t - target)))


def project_robot_skeleton(
    img_dict_und: ImageDict,
    urdf: "yourdfpy.URDF",
    qpos_full: np.ndarray,
    C2R: np.ndarray,
):
    """Draw the URDF link origins (orange dots) + parent-child edges
    (yellow lines) on every image in ``img_dict_und``. Returns a new
    ImageDict; existing markers (e.g. charuco) are preserved.

    pixel = K @ ext @ C2R @ [link_origin_in_robot_base; 1]
    where C2R = T_cam_robot, ext = (3,4) world->cam.
    """
    from paradex.calibration.utils import get_cammtx
    urdf.update_cfg(qpos_full)
    base = urdf.base_link
    link_pts_robot = {ln: urdf.get_transform(ln, base)[:3, 3]
                      for ln in urdf.link_map}
    edges = [(j.parent, j.child) for j in urdf.joint_map.values()]

    proj_mtxs = get_cammtx(img_dict_und.intrinsic, img_dict_und.extrinsic)
    out = {}
    for serial, img in img_dict_und.images.items():
        if serial not in proj_mtxs:
            out[serial] = img.copy()
            continue
        P = proj_mtxs[serial] @ C2R  # (3,4) robot_pt → pixel
        h, w = img.shape[:2]
        pix = {}
        for ln, p in link_pts_robot.items():
            uvw = P @ np.array([p[0], p[1], p[2], 1.0])
            if uvw[2] <= 1e-6:
                continue
            uv = uvw[:2] / uvw[2]
            pix[ln] = uv
        canvas = img.copy()
        for parent, child in edges:
            if parent in pix and child in pix:
                p1 = tuple(np.round(pix[parent]).astype(int))
                p2 = tuple(np.round(pix[child]).astype(int))
                cv2.line(canvas, p1, p2, (0, 220, 255), 2, cv2.LINE_AA)
        for ln, uv in pix.items():
            xy = tuple(np.round(uv).astype(int))
            cv2.circle(canvas, xy, 5, (0, 165, 255), -1, cv2.LINE_AA)
        out[serial] = canvas
    return ImageDict(out, intrinsic=img_dict_und.intrinsic,
                     extrinsic=img_dict_und.extrinsic, path=img_dict_und.path)


def detect_triangulate_reproject(
    capture_dir: Path,
    system_name: str = "robothome",
    video_frame_idx: int = 1,
    robot_urdf: "yourdfpy.URDF | None" = None,
    robot_qpos: "np.ndarray | None" = None,
    robot_C2R: "np.ndarray | None" = None,
    use_mesh_overlay: bool = True,
):
    """Charuco detect + triangulate + reproject pipeline. Returns
    (points_world (N,3), intrinsic, extrinsic).

    If ``robot_urdf`` + ``robot_qpos`` + ``robot_C2R`` are provided, the
    URDF mesh (or stick figure fallback) is overlaid on the same per-camera
    debug images.
    """
    _patch_charuco_system(system_name)
    intrinsic, extrinsic = load_camparam(str(capture_dir))
    img_dict = video_frame_image_dict(capture_dir, intrinsic, extrinsic,
                                      frame_idx=video_frame_idx)
    print(f"[charuco] loaded {len(img_dict)} frames at video idx {video_frame_idx}")

    img_dict_und = img_dict.undistort()
    img_dict_und.set_camparam(intrinsic, extrinsic)

    # 1) Triangulate (uses intrinsics_undistort via get_cammtx).
    charuco_3d = img_dict_und.triangulate_charuco()
    merged = merge_charuco_detection(charuco_3d)
    pts_3d = np.asarray(merged["checkerCorner"]).reshape(-1, 3)
    print(f"[charuco] triangulated {len(pts_3d)} 3D corners")

    # 2) Detect 2D on the undistorted images for the debug overlay.
    charuco_2d = img_dict_und.apply(detect_charuco)
    detected_2d = {
        s: merge_charuco_detection(d)["checkerCorner"].reshape(-1, 2)
        for s, d in charuco_2d.items()
    }

    # 3) Reproject the 3D point cloud onto each view.
    if len(pts_3d) > 0:
        proj_2d = img_dict_und.project_pointcloud(pts_3d)
    else:
        proj_2d = {s: np.zeros((0, 2)) for s in img_dict_und.serial_list}

    # 4) Compose overlay: charuco red/green + robot mesh (nvdiffrast) or skeleton fallback.
    overlay = img_dict_und.draw_keypoint(detected_2d, color=(0, 0, 255), radius=8, thickness=3)
    overlay = overlay.draw_keypoint(proj_2d, color=(0, 255, 0), radius=4, thickness=-1)
    if robot_urdf is not None and robot_qpos is not None and robot_C2R is not None:
        rendered = None
        if use_mesh_overlay:
            try:
                rendered = render_robot_mesh_overlay(
                    overlay.images, robot_urdf, robot_qpos,
                    intrinsic, extrinsic, robot_C2R,
                )
            except Exception as e:
                print(f"[mesh-overlay] falling back to skeleton: {e}")
        if rendered is not None:
            overlay = ImageDict(rendered, intrinsic=overlay.intrinsic,
                                extrinsic=overlay.extrinsic, path=overlay.path)
        else:
            overlay = project_robot_skeleton(overlay, robot_urdf, robot_qpos, robot_C2R)

    out_dir = capture_dir / "charuco_debug"
    out_dir.mkdir(exist_ok=True)
    for serial, img in overlay.images.items():
        cv2.imwrite(str(out_dir / f"{serial}.jpg"), img)
    grid = overlay.merge()
    out_grid = capture_dir / "charuco_debug.jpg"
    cv2.imwrite(str(out_grid), grid)
    print(f"[charuco] debug grid -> {out_grid}")
    print(f"[charuco] per-cam   -> {out_dir}")

    return pts_3d, intrinsic, extrinsic


def render_robot_mesh_overlay(
    images_bgr: dict,
    urdf: "yourdfpy.URDF",
    qpos_full: np.ndarray,
    intrinsic: dict,
    extrinsic: dict,
    C2R: np.ndarray,
):
    """nvdiffrast-based robot mesh overlay over each image (per-camera).

    Lazy-imports the GPU renderer from ``src/visualization/overlay_robot_video.py``.
    Returns ``{serial: bgr_image}`` with the URDF mesh blended on top.
    """
    import importlib.util as _il
    sv_path = Path(__file__).resolve().parents[2] / "visualization" / "overlay_robot_video.py"
    spec = _il.spec_from_file_location("_overlay_rrv", sv_path)
    mod = _il.module_from_spec(spec)
    spec.loader.exec_module(mod)  # imports nvdiffrast etc.
    RobotOverlayRenderer = mod.RobotOverlayRenderer

    # Per-camera intrinsic/extrinsic_robot_frame.
    H, W = next(iter(images_bgr.values())).shape[:2]
    serials = sorted(images_bgr.keys())
    intr_subset = {s: intrinsic[s] for s in serials}
    extr_robot = {}
    for s in serials:
        ext4 = np.eye(4)
        ext4[:3, :] = extrinsic[s]            # world->cam
        extr_robot[s] = (ext4 @ C2R)[:3, :]   # robot->cam (3,4)

    # Pull URDF link meshes + transforms at the given qpos.
    urdf.update_cfg(qpos_full)
    scene = urdf.scene
    link_names_ordered = list(scene.geometry.keys())
    scene_meshes = [scene.geometry[ln] for ln in link_names_ordered]
    link_labels = {ln: None for ln in link_names_ordered}  # all "ARM_COLOR"
    link_poses = [scene.graph.get(ln)[0] for ln in link_names_ordered]

    renderer = RobotOverlayRenderer(
        scene_meshes, link_names_ordered, link_labels,
        intr_subset, extr_robot, H, W,
    )
    frames_list = [images_bgr[s] for s in renderer.serials]
    overlays = renderer.render(link_poses, frames_list)
    return {s: overlays[i] for i, s in enumerate(renderer.serials)}


def mat_to_pos_wxyz(T):
    T = np.asarray(T, dtype=np.float64)
    pos = tuple(T[:3, 3].tolist())
    qxyzw = R.from_matrix(T[:3, :3]).as_quat()
    return pos, (float(qxyzw[3]), float(qxyzw[0]), float(qxyzw[1]), float(qxyzw[2]))


def _fit_charuco_plane(pts: np.ndarray, padding: float = 0.0):
    """Best-fit plane through ``pts`` (Nx3), returned as (verts(4,3), faces(2,3))
    sized to the point cloud's in-plane extent. Returns (None, None) on rank deficiency."""
    if len(pts) < 3:
        return None, None
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    # Plane basis: two largest singular vectors; normal is the smallest.
    u_axis, v_axis = vt[0], vt[1]
    proj = (pts - c) @ np.stack([u_axis, v_axis], axis=1)  # Nx2
    u_lo, u_hi = proj[:, 0].min() - padding, proj[:, 0].max() + padding
    v_lo, v_hi = proj[:, 1].min() - padding, proj[:, 1].max() + padding
    corners = []
    for uu, vv in [(u_lo, v_lo), (u_hi, v_lo), (u_hi, v_hi), (u_lo, v_hi)]:
        corners.append(c + uu * u_axis + vv * v_axis)
    verts = np.stack(corners).astype(np.float32)
    # Two front-facing tris + two reverse-winding tris so the plane is
    # visible from both sides in viser (no back-face culling surprise).
    faces = np.array(
        [[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]],
        dtype=np.int32,
    )
    return verts, faces


def _add_cube_at(server, name: str, size: float = 0.05, color=(220, 190, 100)):
    """Add a simple axis-aligned cube mesh at the given scene path."""
    s = size / 2.0
    # 8 corners of a cube centered at origin
    v = np.array([
        [-s, -s, -s], [+s, -s, -s], [+s, +s, -s], [-s, +s, -s],
        [-s, -s, +s], [+s, -s, +s], [+s, +s, +s], [-s, +s, +s],
    ], dtype=np.float32)
    f = np.array([
        [0,1,2],[0,2,3],  # -z
        [4,6,5],[4,7,6],  # +z
        [0,5,1],[0,4,5],  # -y
        [2,7,3],[2,6,7],  # +y
        [0,3,7],[0,7,4],  # -x
        [1,5,6],[1,6,2],  # +x
    ], dtype=np.int32)
    server.scene.add_mesh_simple(name, vertices=v, faces=f, color=color, opacity=0.9)


def undetected_corner_centroid_robot(
    capture_dir: Path,
    intrinsic: dict,
    extrinsic: dict,
    C2R: np.ndarray,
    video_frame_idx: int = 1,
    system_name: str = "robothome",
):
    """For each board with detections, fit world←board transform from
    triangulated corners, project all expected corners through it, and
    return the centroid (in robot frame) of the corners that were NOT
    detected. Returns ``None`` if every board's corners are fully detected.
    """
    _patch_charuco_system(system_name)
    import paradex.image.aruco as _pa
    img_dict = video_frame_image_dict(capture_dir, intrinsic, extrinsic, frame_idx=video_frame_idx)
    img_und = img_dict.undistort()
    img_und.set_camparam(intrinsic, extrinsic)

    # Per-camera detected IDs (anywhere). A corner is "occluded by object"
    # if no camera sees it, not just if triangulation failed (which fires
    # for any corner seen in <2 views).
    per_cam = img_und.apply(detect_charuco)
    det_anywhere: dict[str, set[int]] = {}
    for serial, det_dict in per_cam.items():
        for bid, payload in det_dict.items():
            ids = np.asarray(payload.get("checkerIDs", [])).reshape(-1).astype(int)
            det_anywhere.setdefault(bid, set()).update(ids.tolist())

    # Triangulated corners (need 3D positions for the world←board fit).
    charuco_3d = img_und.triangulate_charuco()

    undetected_world = []
    n_occluded_total = 0
    for bid, board in _pa._charuco_board_cache.items():
        all_corners_board = np.asarray(board.getChessboardCorners()).reshape(-1, 3)
        n_total = len(all_corners_board)
        tri = charuco_3d.get(bid, {})
        tri_ids = np.asarray(tri.get("checkerIDs", [])).reshape(-1).astype(int)
        tri_world = np.asarray(tri.get("checkerCorner", [])).reshape(-1, 3)
        seen = det_anywhere.get(bid, set())
        if len(tri_ids) < 3 or len(seen) >= n_total:
            continue
        # Umeyama similarity (scale included; robothome charuco config is
        # unitless while world is metric).
        src = all_corners_board[tri_ids]
        dst = tri_world
        sc = src.mean(0); dc = dst.mean(0)
        src_c = src - sc; dst_c = dst - dc
        H = src_c.T @ dst_c
        U, D, Vt = np.linalg.svd(H)
        S = np.eye(3); S[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
        Rm = (Vt.T @ S @ U.T)
        n = len(src_c)
        var_src = (src_c * src_c).sum() / n
        scale = (D * np.array([1.0, 1.0, S[2, 2]])).sum() / max(n * var_src, 1e-12)
        tm = dc - scale * Rm @ sc
        all_world = scale * (Rm @ all_corners_board.T).T + tm
        # Truly undetected: not seen by ANY camera.
        occluded_mask = np.array([i not in seen for i in range(n_total)])
        n_occluded_total += int(occluded_mask.sum())
        if occluded_mask.any():
            undetected_world.append(all_world[occluded_mask])
    if not undetected_world:
        return None
    pts_world = np.concatenate(undetected_world, axis=0)
    pts_robot = world_to_robot(pts_world, C2R)
    print(f"[object] occluded corners: {n_occluded_total}")
    return pts_robot.mean(axis=0)


def world_to_robot(pts_world: np.ndarray, C2R: np.ndarray) -> np.ndarray:
    """Camera-world point cloud → robot-base frame.

    Despite the filename, ``C2R.npy`` holds ``T_cam_robot`` (robot frame
    expressed in cam-world coords) — see ``paradex/src/calibration/handeye``
    where it is saved as ``robot_wrt_cam`` and consumed via ``inv()``. So the
    cam→robot map is ``inv(C2R)``.
    """
    if len(pts_world) == 0:
        return pts_world.reshape(0, 3)
    R2C = np.linalg.inv(C2R)
    h = np.concatenate([pts_world, np.ones((len(pts_world), 1))], axis=1)
    return (R2C @ h.T).T[:, :3]


def cam_pose_in_robot(extrinsic_3x4: np.ndarray, C2R: np.ndarray) -> np.ndarray:
    """extrinsic is (3,4) world->cam. Return cam pose (4,4) in robot base frame."""
    T_wc = np.eye(4)
    T_wc[:3, :4] = extrinsic_3x4
    T_cw = np.linalg.inv(T_wc)              # cam pose in cam-world frame
    return np.linalg.inv(C2R) @ T_cw         # cam pose in robot base frame


def _stack_object_array(arr):
    """Return float64 stack of an object dtype array of per-frame ndarrays."""
    return np.stack([np.asarray(x, dtype=np.float64) for x in arr])


def load_arm_hand(capture_dir: Path):
    arm_pos = _stack_object_array(np.load(capture_dir / "raw/arm/position.npy", allow_pickle=True))
    arm_t = np.asarray([float(x) for x in np.load(capture_dir / "raw/arm/time.npy", allow_pickle=True)],
                       dtype=np.float64)
    hand_pos = np.asarray(np.load(capture_dir / "raw/hand/position.npy"), dtype=np.int64)
    hand_t = np.asarray(np.load(capture_dir / "raw/hand/time.npy"), dtype=np.float64)
    return arm_pos, arm_t, hand_pos, hand_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", required=True, type=Path)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--no-charuco", action="store_true",
                    help="Skip the charuco detect/triangulate/reproject step.")
    ap.add_argument("--charuco-system", default="robothome",
                    help="paradex/system/<name>/charuco_info.json to use for board defs.")
    ap.add_argument("--video-frame-idx", type=int, default=1,
                    help="Which video frame to use for charuco/overlay. Default 1 "
                         "(skip the often-partial frame 0).")
    ap.add_argument("--no-mesh-overlay", action="store_true",
                    help="Skip nvdiffrast mesh overlay; use skeleton stick figure.")
    ap.add_argument("--force-charuco", action="store_true",
                    help="Force re-run of charuco detect/triangulate/overlay even "
                         "if charuco_cache.npz exists.")
    ap.add_argument("--object-name", default=None,
                    help="Object name for the draggable mesh. Default: auto-derive "
                         "from capture-dir's parent name via CAPTURE_TO_OBJECT.")
    ap.add_argument("--object-mesh", default=None,
                    help="Direct path to the object mesh. Overrides --object-name lookup.")
    args = ap.parse_args()

    cap_dir = args.capture_dir.resolve()
    if not cap_dir.exists():
        raise FileNotFoundError(cap_dir)

    C2R = load_c2r(str(cap_dir))

    # ---- URDF + per-frame full qpos (need frame-0 cfg before charuco overlay) ----
    urdf = yourdfpy.URDF.load(
        str(URDF_PATH),
        mesh_dir=str(URDF_MESH_DIR),
        build_collision_scene_graph=False,
        load_collision_meshes=False,
    )
    actuated = list(urdf.actuated_joint_names)
    arm_idx = [actuated.index(j) for j in ARM_JOINTS]
    hand_idx = [actuated.index(j) for j in HAND_DRIVERS]

    arm_pos, arm_t, hand_pos, hand_t = load_arm_hand(cap_dir)
    assert arm_pos.shape[1] == len(ARM_JOINTS), (
        f"arm position has {arm_pos.shape[1]} joints, URDF has {len(ARM_JOINTS)} arm joints"
    )
    hand_rad = inspire_int_to_rad(hand_pos)

    # Resample hand to arm timestamps (arm is the master timeline).
    t0 = min(arm_t[0], hand_t[0])
    arm_rel = arm_t - t0
    hand_rel = hand_t - t0
    hand_at_arm = np.empty((len(arm_rel), 6))
    for j in range(6):
        hand_at_arm[:, j] = np.interp(arm_rel, hand_rel, hand_rad[:, j])

    T = len(arm_pos)
    full_q = np.zeros((T, len(actuated)))
    for f in range(T):
        full_q[f, arm_idx] = arm_pos[f]
        full_q[f, hand_idx] = hand_at_arm[f]
    print(f"[traj] arm frames={len(arm_pos)}  duration={arm_rel[-1] - arm_rel[0]:.2f}s")

    # ---- charuco + robot-mesh overlay (cached) ----
    cache_path = cap_dir / "charuco_cache.npz"
    use_cache = (not args.no_charuco) and cache_path.exists() and not args.force_charuco
    cached_occluded = None
    if args.no_charuco:
        intrinsic, extrinsic = load_camparam(str(cap_dir))
        pts_world = np.zeros((0, 3))
    elif use_cache:
        d = np.load(cache_path)
        pts_world = d["pts_world"]
        if "occluded_centroid" in d.files:
            v = d["occluded_centroid"]
            cached_occluded = None if not v.size else np.asarray(v, dtype=np.float64)
        intrinsic, extrinsic = load_camparam(str(cap_dir))
        print(f"[cache] loaded {len(pts_world)} charuco corners from {cache_path}")
    else:
        overlay_arm_idx = arm_index_for_video_frame(cap_dir, args.video_frame_idx, arm_t)
        print(f"[overlay] video frame {args.video_frame_idx} -> arm idx {overlay_arm_idx}")
        pts_world, intrinsic, extrinsic = detect_triangulate_reproject(
            cap_dir,
            system_name=args.charuco_system,
            video_frame_idx=args.video_frame_idx,
            robot_urdf=urdf,
            robot_qpos=full_q[overlay_arm_idx],
            robot_C2R=C2R,
            use_mesh_overlay=not args.no_mesh_overlay,
        )
        cached_occluded = undetected_corner_centroid_robot(
            cap_dir, intrinsic, extrinsic, C2R,
            video_frame_idx=args.video_frame_idx,
            system_name=args.charuco_system,
        )
        np.savez(
            cache_path,
            pts_world=pts_world,
            occluded_centroid=(cached_occluded if cached_occluded is not None
                               else np.zeros((0,))),
        )
        print(f"[cache] saved -> {cache_path}")

    pts_robot = world_to_robot(pts_world, C2R)

    # ---- viser ----
    server = viser.ViserServer(port=args.port)
    print(f"[viser] http://localhost:{args.port}")

    viser_urdf = ViserUrdf(server, urdf, root_node_name="/robot")

    # Cameras as little frames in robot frame.
    for serial, ext in extrinsic.items():
        T_cam = cam_pose_in_robot(ext, C2R)
        pos, wxyz = mat_to_pos_wxyz(T_cam)
        server.scene.add_frame(
            f"/cameras/{serial}", wxyz=wxyz, position=pos,
            axes_length=0.05, axes_radius=0.002, show_axes=True,
        )

    # ChArUco point cloud (red) + best-fit plane (translucent red, padded
    # 5cm beyond the corner extent so it's clearly visible in viser).
    if len(pts_robot) >= 3:
        colors = np.tile(np.array([[230, 60, 60]], dtype=np.uint8), (len(pts_robot), 1))
        server.scene.add_point_cloud(
            "/charuco/points", points=pts_robot.astype(np.float32),
            colors=colors, point_size=0.006,
        )
        plane_v, plane_f = _fit_charuco_plane(pts_robot, padding=0.05)
        if plane_v is not None:
            server.scene.add_mesh_simple(
                "/charuco/plane", vertices=plane_v, faces=plane_f,
                color=(230, 60, 60), opacity=0.55,
            )

    # Draggable target object. Initial position = centroid of the charuco
    # corners that were NOT detected (occluded by the object → centroid is
    # near the object). Falls back to overall charuco centroid, else origin.
    obj_default_pos = cached_occluded
    if obj_default_pos is not None:
        print(f"[object] init pos = undetected-corner centroid "
              f"{tuple(round(float(x), 3) for x in obj_default_pos)}")
    else:
        obj_default_pos = (pts_robot.mean(axis=0) if len(pts_robot) > 0
                           else np.array([0.5, 0.0, 0.05]))
    # Resolve mesh: --object-mesh wins; else --object-name; else auto from
    # capture-dir's parent (e.g., ".../choco/0" → "choco" → "blackChocoBar").
    mesh_path: Path | None = None
    if args.object_mesh and Path(args.object_mesh).exists():
        mesh_path = Path(args.object_mesh)
    else:
        obj_name = args.object_name or CAPTURE_TO_OBJECT.get(cap_dir.parent.name)
        mesh_path = resolve_object_mesh(obj_name)
        if mesh_path is None and obj_name:
            print(f"[object] no mesh found for '{obj_name}'; using cube placeholder")
    bbox_center = np.zeros(3, dtype=np.float64)
    v_centered: np.ndarray | None = None
    faces_arr: np.ndarray | None = None
    if mesh_path is not None:
        import trimesh as _tm
        m = _tm.load(str(mesh_path), force="mesh", process=False)
        v = np.asarray(m.vertices, dtype=np.float64)
        bbox_center = (v.min(axis=0) + v.max(axis=0)) / 2
        v_centered = (v - bbox_center).astype(np.float32)
        faces_arr = np.asarray(m.faces, dtype=np.int32)

    # Gizmo size matches mesh extent so the bars stay visually attached to
    # the mesh (default 0.05 if no mesh).
    if v_centered is not None and len(v_centered) > 0:
        ext = v_centered.max(axis=0) - v_centered.min(axis=0)
        gizmo_scale = float(np.linalg.norm(ext) * 0.6)
    else:
        gizmo_scale = 0.05
    obj_gizmo = server.scene.add_transform_controls(
        "/object", scale=gizmo_scale,
        position=tuple(np.asarray(obj_default_pos, dtype=np.float32).tolist()),
    )
    if v_centered is not None:
        server.scene.add_mesh_simple(
            "/object/mesh", vertices=v_centered, faces=faces_arr,
            color=(220, 190, 100), opacity=0.9,
        )
        print(f"[object] mesh: {mesh_path}  gizmo_scale={gizmo_scale:.3f}")
    else:
        _add_cube_at(server, "/object/mesh", size=0.05)

    # Position + rotation sliders in robot frame. Bidirectional with the
    # gizmo: changing a slider drives the gizmo, and the tabletop dropdown
    # drives both. (Dragging the gizmo bars doesn't sync back into the
    # sliders — viser doesn't expose a reliable on_update for that.)
    obj_name = args.object_name or CAPTURE_TO_OBJECT.get(cap_dir.parent.name)
    tabletops = load_tabletop_poses(obj_name)

    init_pos = np.asarray(obj_default_pos, dtype=np.float64).copy()
    init_R = np.eye(3)
    if tabletops:
        # Default to the first tabletop's rotation.
        init_R = tabletops[next(iter(tabletops.keys()))][:3, :3].copy()
    init_rxyz = R.from_matrix(init_R).as_euler("xyz", degrees=True)

    obj_state = {"suppress": False}

    with server.gui.add_folder("Object pose (robot frame)"):
        x_sl = server.gui.add_slider("x (m)",  -0.5, 1.5, 0.001, float(init_pos[0]))
        y_sl = server.gui.add_slider("y (m)",  -0.8, 0.8, 0.001, float(init_pos[1]))
        z_sl = server.gui.add_slider("z (m)",  -0.2, 0.6, 0.001, float(init_pos[2]))
        rx_sl = server.gui.add_slider("roll (deg)",  -180.0, 180.0, 0.5, float(init_rxyz[0]))
        ry_sl = server.gui.add_slider("pitch (deg)", -180.0, 180.0, 0.5, float(init_rxyz[1]))
        rz_sl = server.gui.add_slider("yaw (deg)",   -180.0, 180.0, 0.5, float(init_rxyz[2]))

    def _push_to_gizmo():
        if obj_state["suppress"]:
            return
        Rm = R.from_euler("xyz", [rx_sl.value, ry_sl.value, rz_sl.value],
                          degrees=True).as_matrix()
        qxyzw = R.from_matrix(Rm).as_quat()
        obj_gizmo.position = (float(x_sl.value), float(y_sl.value), float(z_sl.value))
        obj_gizmo.wxyz = (float(qxyzw[3]), float(qxyzw[0]),
                          float(qxyzw[1]), float(qxyzw[2]))

    for sl in (x_sl, y_sl, z_sl, rx_sl, ry_sl, rz_sl):
        sl.on_update(lambda _e: _push_to_gizmo())

    def _set_pose(pos: np.ndarray, R_mat: np.ndarray):
        """Set sliders + gizmo atomically in robot frame."""
        rxyz = R.from_matrix(R_mat).as_euler("xyz", degrees=True)
        obj_state["suppress"] = True
        try:
            x_sl.value, y_sl.value, z_sl.value = float(pos[0]), float(pos[1]), float(pos[2])
            rx_sl.value, ry_sl.value, rz_sl.value = float(rxyz[0]), float(rxyz[1]), float(rxyz[2])
        finally:
            obj_state["suppress"] = False
        _push_to_gizmo()

    def _apply_tabletop(key: str):
        """Tabletop only sets rotation. Position stays where the user has
        it (i.e. at the undetected-corner centroid initially) so the mesh
        center stays at the occluded centre."""
        R_tt = tabletops[key][:3, :3]
        cur_pos = np.array([x_sl.value, y_sl.value, z_sl.value], dtype=np.float64)
        _set_pose(cur_pos, R_tt)

    if tabletops:
        tt_keys = list(tabletops.keys())
        tt_dd = server.gui.add_dropdown(
            "Tabletop pose", ["(none)"] + tt_keys, initial_value=tt_keys[0],
        )

        @tt_dd.on_update
        def _(_):
            if tt_dd.value != "(none)":
                _apply_tabletop(tt_dd.value)

        # Initial state already reflects the first tabletop's rotation via
        # init_R; just push to the gizmo.
        _push_to_gizmo()
        print(f"[object] tabletop poses: {tt_keys} (default {tt_keys[0]} applied)")
    else:
        _push_to_gizmo()

    # GUI: frame slider + play/pause + speed.
    play = server.gui.add_checkbox("Playing", initial_value=True)
    fps = server.gui.add_slider("FPS", min=1, max=120, step=1, initial_value=30)
    t_slider = server.gui.add_slider(
        "Frame", min=0, max=max(1, T - 1), step=1, initial_value=0,
    )

    state = {"idx": 0, "dirty": True}

    def render(idx: int):
        idx = int(np.clip(idx, 0, T - 1))
        viser_urdf.update_cfg(full_q[idx])

    @t_slider.on_update
    def _(_):
        # User-driven scrub: take the slider as ground truth.
        state["idx"] = int(t_slider.value)
        state["dirty"] = True

    render(0)

    # Drive the URDF from the main loop. Programmatic slider writes don't
    # always retrigger on_update, so we own state["idx"] explicitly here.
    last = time.time()
    while True:
        now = time.time()
        dt = now - last
        last = now
        if play.value and T > 1:
            advance = max(1, int(round(dt * fps.value)))
            state["idx"] = (state["idx"] + advance) % T
            t_slider.value = state["idx"]
            state["dirty"] = True
        if state["dirty"]:
            render(state["idx"])
            state["dirty"] = False
        time.sleep(1.0 / 120.0)


if __name__ == "__main__":
    main()
