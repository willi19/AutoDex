"""Render pregrasp-frame thumbnails with planned grasp hand + object overlay.

For each camera, seek to the video frame closest to the pregrasp timestamp,
overlay (a) the planned grasp pose hand mesh (from candidate dir) and
(b) the initial object pose mesh on the frame. Saves 24 per-camera PNGs +
1 grid PNG.

Hand uses floating URDF so we render only the hand (no arm), placing the
wrist via pose_world @ wrist_se3_obj. Fingers are colored individually.

Prints `[thumb_progress] N/TOTAL` lines for parent-process tqdm parsing.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr

PARADEX_ROOT = Path.home() / "paradex"
sys.path.insert(0, str(PARADEX_ROOT))
from paradex.image.projection import intr_opencv_to_opengl_proj
from paradex.image.grid import make_image_grid
from paradex.visualization.robot import RobotModule


ROBOT_VIDEO_OFFSET_S = 0.03
ALPHA_HAND = 0.55
ALPHA_OBJ = 0.5

# Per-finger BGR colors (OpenCV is BGR)
FINGER_COLORS_BGR = {
    "thumb":  (  0, 140, 255),  # orange
    "index":  (255, 200,   0),  # cyan-ish
    "middle": (100, 255,   0),  # lime
    "ring":   (200,   0, 255),  # magenta
    "pinky":  (  0, 220, 255),  # yellow-ish
}
HAND_FALLBACK_BGR = (40, 200, 40)
OBJECT_BGR = (255, 80, 80)  # blue

FINGER_PREFIX_MAP = {
    "right_thumb_": "thumb", "right_index_": "index", "right_middle_": "middle",
    "right_ring_": "ring", "right_little_": "pinky",
    "left_thumb_": "thumb", "left_index_": "index", "left_middle_": "middle",
    "left_ring_": "ring", "left_little_": "pinky",
}
ALLEGRO_LINK_LABELS = {}
for i in range(4):
    ALLEGRO_LINK_LABELS[f"link_{i}.0.obj"] = "index"
    ALLEGRO_LINK_LABELS[f"link_{i}.0.obj_1"] = "middle"
    ALLEGRO_LINK_LABELS[f"link_{i}.0.obj_2"] = "ring"
ALLEGRO_LINK_LABELS["link_3.0_tip.obj"] = "index"
ALLEGRO_LINK_LABELS["link_3.0_tip.obj_1"] = "middle"
ALLEGRO_LINK_LABELS["link_3.0_tip.obj_2"] = "ring"
for name in ["link_12.0_right.obj", "link_12.0_left.obj",
             "link_13.0.obj", "link_14.0.obj", "link_15.0.obj", "link_15.0_tip.obj"]:
    ALLEGRO_LINK_LABELS[name] = "thumb"


def _label_for_link(link_name):
    if link_name in ALLEGRO_LINK_LABELS:
        return ALLEGRO_LINK_LABELS[link_name]
    for prefix, label in FINGER_PREFIX_MAP.items():
        if link_name.startswith(prefix):
            return label
    return None


_GLCAM_IN_CVCAM = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1],
], dtype=np.float32)


def se3_to_floating_qpos(T):
    """Convert 4x4 SE(3) to floating URDF first 6 qpos: [x, y, z, rx, ry, rz].

    Verified against RobotModule FK: scipy 'XYZ' intrinsic gives diff=0."""
    from scipy.spatial.transform import Rotation as R
    rxyz = R.from_matrix(T[:3, :3]).as_euler("XYZ")
    return np.array([T[0, 3], T[1, 3], T[2, 3], rxyz[0], rxyz[1], rxyz[2]], dtype=np.float64)


class MultiMeshOverlayRenderer:
    """Rasterize multiple per-link meshes (hand) + one object mesh in one pass.

    Each "link" is a separately-colored region; meshes are concatenated and
    a per-vertex link id is used to colorize at blend time.
    """

    def __init__(self, link_meshes, link_colors_bgr, link_alphas,
                 intrinsics, extrinsics_cw, H, W, device="cuda"):
        self.device = device
        self.serials = sorted(intrinsics.keys())
        self.N = len(self.serials)
        self.H, self.W = H, W
        self.glctx = dr.RasterizeCudaContext()

        glcam = torch.from_numpy(_GLCAM_IN_CVCAM).to(device)
        cam_extrs, proj_list = [], []
        for s in self.serials:
            ext = np.eye(4, dtype=np.float32)
            ext[:3, :] = extrinsics_cw[s][:3, :]
            cam_extrs.append(torch.from_numpy(ext).to(device))
            proj = intr_opencv_to_opengl_proj(intrinsics[s], W, H, near=0.01, far=5).astype(np.float32)
            proj_list.append(torch.from_numpy(proj).to(device))
        self.mtx = (torch.stack(proj_list) @ glcam[None] @ torch.stack(cam_extrs)).contiguous()

        per_v, per_f, per_lid = [], [], []
        v_off = 0
        self.link_vert_ranges = []
        for i, mesh in enumerate(link_meshes, start=1):
            v = torch.as_tensor(np.asarray(mesh.vertices, dtype=np.float32), device=device)
            f = torch.as_tensor(np.asarray(mesh.faces, dtype=np.int32), device=device)
            nv = v.shape[0]
            per_v.append(v)
            per_f.append(f + v_off)
            per_lid.append(torch.full((nv,), float(i), dtype=torch.float32, device=device))
            self.link_vert_ranges.append((v_off, v_off + nv))
            v_off += nv

        self.base_verts = torch.cat(per_v, dim=0)
        self.faces = torch.cat(per_f, dim=0)
        self.vert_lid = torch.cat(per_lid, dim=0)[:, None]
        self.n_links = len(link_meshes)
        self.V = v_off

        color_lut = np.zeros((self.n_links + 1, 3), dtype=np.float32)
        alpha_lut = np.zeros((self.n_links + 1,), dtype=np.float32)
        for i, (bgr, a) in enumerate(zip(link_colors_bgr, link_alphas), start=1):
            color_lut[i] = bgr
            alpha_lut[i] = a
        self.color_lut = torch.from_numpy(color_lut).to(device)
        self.alpha_lut = torch.from_numpy(alpha_lut).to(device)[:, None]

    def render(self, link_poses, frames_bgr_list):
        """link_poses: list of 4x4 numpy, same order as link_meshes at init."""
        device = self.device
        poses = torch.as_tensor(np.stack(link_poses), dtype=torch.float32, device=device)
        verts_world = torch.empty((self.V, 3), dtype=torch.float32, device=device)
        for i, (start, end) in enumerate(self.link_vert_ranges):
            v = self.base_verts[start:end]
            v_h = torch.cat([v, torch.ones(v.shape[0], 1, device=device)], dim=1)
            verts_world[start:end] = (v_h @ poses[i].T)[:, :3]

        v_homo = torch.cat([verts_world, torch.ones(self.V, 1, device=device)], dim=1)
        pos_clip = torch.einsum("nij,vj->nvi", self.mtx, v_homo).contiguous()
        rast_out, _ = dr.rasterize(self.glctx, pos_clip, self.faces, resolution=(self.H, self.W))
        id_map, _ = dr.interpolate(self.vert_lid, rast_out, self.faces)
        id_map = torch.flip(id_map, dims=[1])
        ids = torch.clamp(torch.round(id_map[..., 0]).long(), 0, self.n_links)
        colors = self.color_lut[ids]
        alphas = self.alpha_lut[ids]

        frames_np = np.stack(frames_bgr_list)
        frames_gpu = torch.from_numpy(frames_np).to(device).float()
        overlay = frames_gpu * (1.0 - alphas) + colors * alphas
        overlay_u8 = overlay.clamp(0, 255).to(torch.uint8).cpu().numpy()
        return [overlay_u8[i] for i in range(self.N)]


def load_cam_param(cam_param_dir):
    intr_raw = json.load(open(cam_param_dir / "intrinsics.json"))
    extr_raw = json.load(open(cam_param_dir / "extrinsics.json"))
    intrinsics, extrinsics = {}, {}
    for s in intr_raw:
        intrinsics[s] = np.array(intr_raw[s]["intrinsics_undistort"], dtype=np.float32).reshape(3, 3)
        T = np.array(extr_raw[s], dtype=np.float64).reshape(-1)
        T = np.vstack([T.reshape(3, 4), [0, 0, 0, 1]]) if T.size == 12 else T.reshape(4, 4)
        extrinsics[s] = T
    return intrinsics, extrinsics


def find_pregrasp_frame(result_json_path, timestamps_npy_path):
    with open(result_json_path) as f:
        result = json.load(f)
    pregrasp_iso = None
    for s in result["timing"]["execution_states"]:
        if s["state"] == "pregrasp":
            pregrasp_iso = s["time"]
            break
    if pregrasp_iso is None:
        raise RuntimeError(f"no pregrasp state in {result_json_path}")
    pregrasp_epoch = datetime.fromisoformat(pregrasp_iso).timestamp() + ROBOT_VIDEO_OFFSET_S
    ts = np.load(timestamps_npy_path)
    idx = int(np.argmin(np.abs(ts - pregrasp_epoch)))
    dt = float(ts[idx] - pregrasp_epoch)
    return idx, dt, pregrasp_iso, result.get("scene_info")


def read_frame(video_path, frame_idx, H, W):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return np.zeros((H, W, 3), dtype=np.uint8)
    return frame


def build_hand_link_meshes(hand_type, candidate_dir, pose_world):
    """Forward kinematics on floating URDF with planned grasp pose.
    Returns: (link_meshes, link_poses, link_colors_bgr, link_alphas)."""
    urdf_base = Path.home() / "AutoDex" / "autodex" / "planner" / "src" / "curobo" / "content" / "assets" / "robot"
    urdf_path = str(urdf_base / f"{hand_type}_description" / f"{hand_type}_floating.urdf")
    robot = RobotModule(urdf_path)
    n_dof = robot.get_num_joints()

    grasp_pose = np.load(candidate_dir / "grasp_pose.npy")  # (n_hand_dof,)
    wrist_se3_obj = np.load(candidate_dir / "wrist_se3.npy")  # (4,4) in obj frame
    wrist_world = pose_world @ wrist_se3_obj
    floating_qpos = se3_to_floating_qpos(wrist_world)

    qpos = np.zeros(n_dof, dtype=np.float64)
    qpos[:6] = floating_qpos
    qpos[6:6 + len(grasp_pose)] = grasp_pose
    robot.update_cfg(qpos)

    scene = robot.scene
    link_names = list(scene.geometry.keys())
    meshes, poses, colors, alphas = [], [], [], []
    for ln in link_names:
        label = _label_for_link(ln)
        bgr = FINGER_COLORS_BGR.get(label, HAND_FALLBACK_BGR)
        meshes.append(scene.geometry[ln])
        poses.append(scene.graph.get(ln)[0])
        colors.append(bgr)
        alphas.append(ALPHA_HAND)
    return meshes, poses, colors, alphas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", required=True)
    ap.add_argument("--cam_param_dir", required=True)
    ap.add_argument("--pose_world", required=True)
    ap.add_argument("--object_mesh", required=True)
    ap.add_argument("--candidate_dir", required=True,
                    help="path to candidate dir with grasp_pose.npy + wrist_se3.npy")
    ap.add_argument("--result_json", required=True)
    ap.add_argument("--timestamps", required=True)
    ap.add_argument("--hand", required=True, choices=["allegro", "inspire"])
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    videos_dir = Path(args.videos_dir)
    cam_param_dir = Path(args.cam_param_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    intrinsics, extrinsics = load_cam_param(cam_param_dir)
    available = {p.stem for p in videos_dir.glob("*.avi")}
    serials = sorted(s for s in intrinsics if s in available)
    intrinsics = {s: intrinsics[s] for s in serials}
    extrinsics = {s: extrinsics[s] for s in serials}
    if not serials:
        print("[error] no serials match between cam_param and videos", flush=True)
        sys.exit(1)

    frame_idx, dt, pregrasp_iso, scene_info = find_pregrasp_frame(args.result_json, args.timestamps)
    print(f"[thumb] pregrasp={pregrasp_iso} frame_idx={frame_idx} dt={dt*1000:.1f}ms scene={scene_info}", flush=True)

    pose_world = np.load(args.pose_world)

    cap0 = cv2.VideoCapture(str(videos_dir / f"{serials[0]}.avi"))
    W = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap0.release()

    hand_meshes, hand_poses, hand_colors, hand_alphas = build_hand_link_meshes(
        args.hand, Path(args.candidate_dir), pose_world)

    obj_mesh = trimesh.load(args.object_mesh, process=False)
    if isinstance(obj_mesh, trimesh.Scene):
        obj_mesh = trimesh.util.concatenate(list(obj_mesh.geometry.values()))

    all_meshes = hand_meshes + [obj_mesh]
    all_poses = hand_poses + [pose_world]
    all_colors = hand_colors + [OBJECT_BGR]
    all_alphas = hand_alphas + [ALPHA_OBJ]

    renderer = MultiMeshOverlayRenderer(all_meshes, all_colors, all_alphas,
                                         intrinsics, extrinsics, H, W)
    ordered = renderer.serials

    total = len(ordered) + 1
    frames = []
    for i, s in enumerate(ordered):
        frames.append(read_frame(videos_dir / f"{s}.avi", frame_idx, H, W))
        print(f"[thumb_progress] {i+1}/{total}", flush=True)

    overlays = renderer.render(all_poses, frames)
    for s, img in zip(ordered, overlays):
        cv2.imwrite(str(out_dir / f"thumb_{s}.png"), img)

    grid = make_image_grid([cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in overlays])
    grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_dir / "thumb_grid.png"), grid_bgr)
    print(f"[thumb_progress] {total}/{total}", flush=True)
    print(f"[thumb] wrote {len(ordered)} thumbs + grid to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
