"""Batch generate robot overlay videos using paradex library.

Renders TWO versions per camera:
  - overlay_actual_{serial}.mp4  — robot at recorded joint position
  - overlay_target_{serial}.mp4  — robot at commanded joint action

Pipeline per object:
  1. Prefetch next object's NAS → local cache (background)
  2. Render all experiments for current object (GPU via paradex BatchRenderer)
  3. Upload finished output cache → NAS (background)

Usage:
    python src/visualization/overlay_robot_video.py --hand allegro
    python src/visualization/overlay_robot_video.py --hand inspire --obj banana
    python src/visualization/overlay_robot_video.py --hand allegro --obj banana --ep 20260405_073417
"""
import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import nvdiffrast.torch as dr

# Paradex library
PARADEX_ROOT = Path.home() / "paradex"
sys.path.insert(0, str(PARADEX_ROOT))

from paradex.calibration.utils import load_camparam
from paradex.visualization.robot import RobotModule
from paradex.image.projection import intr_opencv_to_opengl_proj
from paradex.video.util import change_to_h264

from tqdm import tqdm


# ── Paths ────────────────────────────────────────────────────────────────────

EXP_BASE = Path.home() / "shared_data" / "AutoDex" / "experiment" / "selected_100"
OUTPUT_BASE = Path.home() / "shared_data" / "AutoDex" / "overlay_video"
LOCAL_CACHE = Path.home() / "cache" / "overlay_robot_video"


# ── Color config ─────────────────────────────────────────────────────────────

FINGER_COLORS = {
    "thumb":  (255, 140,   0),
    "index":  (  0, 200, 255),
    "middle": (  0, 255, 100),
    "ring":   (255,   0, 200),
    "pinky":  (255, 220,   0),
}
ARM_COLOR = (40, 200, 40)
ARM_ALPHA = 0.35
FINGER_ALPHA = 0.55

FINGER_PREFIX_MAP = {
    "right_thumb_": "thumb", "right_index_": "index", "right_middle_": "middle",
    "right_ring_": "ring", "right_little_": "pinky",
    "thumb_tip": "thumb", "index_tip": "index", "middle_tip": "middle",
    "ring_tip": "ring", "little_tip": "pinky",
    "left_thumb_": "thumb", "left_index_": "index", "left_middle_": "middle",
    "left_ring_": "ring", "left_little_": "pinky",
}

# Allegro real robot → URDF joint order


ALLEGRO_LINK_LABELS = {}
# trimesh reuses meshes across fingers with _1, _2 suffixes:
#   no suffix = index (links 0-3), _1 = middle (links 4-7), _2 = ring (links 8-11)
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


# ── GPU renderer (robot overlay, keeps everything on GPU) ───────────────────

_GLCAM_IN_CVCAM = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1],
], dtype=np.float32)


class RobotOverlayRenderer:
    """Render robot mesh overlay onto multi-view video frames entirely on GPU.

    Build once per episode (fixed intrinsics/extrinsics, mesh topology).
    Call render() per frame with link poses and BGR frames.
    """

    def __init__(self, scene_meshes, link_names_ordered, link_labels,
                 intrinsic, extrinsic_robot_frame, H, W, device="cuda"):
        """
        scene_meshes: list of trimesh meshes (one per link)
        link_names_ordered: list of link names, same order as scene_meshes
        link_labels: dict {link_name: label_or_None}
        intrinsic: {serial: {intrinsics_undistort, width, height}}
        extrinsic_robot_frame: {serial: 3x4 cam_from_robot}
        """
        self.device = device
        self.serials = sorted(intrinsic.keys())
        self.N = len(self.serials)
        self.H = H
        self.W = W
        self.glctx = dr.RasterizeCudaContext()

        # ---- Camera matrices (precomputed) ----
        glcam = torch.from_numpy(_GLCAM_IN_CVCAM).to(device)
        cam_extrs = []
        proj_list = []
        for s in self.serials:
            ext = np.eye(4, dtype=np.float32)
            ext[:3, :] = extrinsic_robot_frame[s]
            cam_extrs.append(torch.from_numpy(ext).to(device))
            K = intrinsic[s]["intrinsics_undistort"]
            proj = intr_opencv_to_opengl_proj(K, W, H, near=0.01, far=5).astype(np.float32)
            proj_list.append(torch.from_numpy(proj).to(device))
        cam_extrs = torch.stack(cam_extrs)       # (N, 4, 4)
        projs = torch.stack(proj_list)           # (N, 4, 4)
        # mtx: clip = proj @ flip_z @ extr — precompute
        self.mtx = (projs @ glcam[None] @ cam_extrs).contiguous()  # (N, 4, 4)

        # ---- Mesh data on GPU ----
        # Per-link: base verts, faces, per-vertex link_id
        self.link_ids = []       # ordered link index (1-based; 0 = bg)
        per_link_verts = []
        per_link_faces = []
        per_link_lid = []
        vert_offset = 0
        self.link_vert_ranges = []  # [(start, end), ...]
        colors_uint8 = []
        alphas = []
        for i, (mesh, link_name) in enumerate(zip(scene_meshes, link_names_ordered), start=1):
            v = torch.as_tensor(np.asarray(mesh.vertices, dtype=np.float32), device=device)
            f = torch.as_tensor(np.asarray(mesh.faces, dtype=np.int32), device=device)
            nv = v.shape[0]
            per_link_verts.append(v)
            per_link_faces.append(f + vert_offset)
            per_link_lid.append(torch.full((nv,), float(i), dtype=torch.float32, device=device))
            self.link_vert_ranges.append((vert_offset, vert_offset + nv))
            vert_offset += nv
            label = link_labels.get(link_name)
            if label is None:
                colors_uint8.append(ARM_COLOR)
                alphas.append(ARM_ALPHA)
            else:
                colors_uint8.append(FINGER_COLORS[label])
                alphas.append(FINGER_ALPHA)

        self.base_verts = torch.cat(per_link_verts, dim=0)       # (V, 3)
        self.faces = torch.cat(per_link_faces, dim=0)            # (F, 3)
        self.vert_lid = torch.cat(per_link_lid, dim=0)[:, None]  # (V, 1)
        self.n_links = len(scene_meshes)
        self.V = vert_offset

        # LUT: index 0 = background (transparent), 1..N = links
        # Colors are stored in BGR (match input video) so no cvtColor needed
        color_lut = np.zeros((self.n_links + 1, 3), dtype=np.float32)
        alpha_lut = np.zeros((self.n_links + 1,), dtype=np.float32)
        for i, (rgb, a) in enumerate(zip(colors_uint8, alphas), start=1):
            color_lut[i] = [rgb[2], rgb[1], rgb[0]]  # RGB → BGR
            alpha_lut[i] = a
        self.color_lut = torch.from_numpy(color_lut).to(device)        # (L+1, 3)
        self.alpha_lut = torch.from_numpy(alpha_lut).to(device)[:, None]  # (L+1, 1)

    def render(self, link_poses_list, frames_bgr_list):
        """
        link_poses_list: list of 4x4 np/torch (same order as scene_meshes at init)
        frames_bgr_list: list of (H,W,3) uint8 numpy, one per serial (same order as self.serials)
        Returns: list of (H,W,3) uint8 numpy overlays.
        """
        device = self.device
        # Stack link poses and apply per-link transform to base verts (all on GPU)
        poses = torch.as_tensor(np.stack(link_poses_list), dtype=torch.float32, device=device)  # (L, 4, 4)
        verts_world = torch.empty((self.V, 3), dtype=torch.float32, device=device)
        for i, (start, end) in enumerate(self.link_vert_ranges):
            v = self.base_verts[start:end]  # (nv, 3)
            v_h = torch.cat([v, torch.ones(v.shape[0], 1, device=device)], dim=1)  # (nv, 4)
            verts_world[start:end] = (v_h @ poses[i].T)[:, :3]

        # Transform to clip space per camera: pos_clip = mtx @ [v, 1]
        v_homo = torch.cat([verts_world, torch.ones(self.V, 1, device=device)], dim=1)   # (V, 4)
        pos_clip = torch.einsum("nij,vj->nvi", self.mtx, v_homo).contiguous()             # (N, V, 4)

        rast_out, _ = dr.rasterize(self.glctx, pos_clip, self.faces, resolution=(self.H, self.W))
        # id_map per pixel (float), shape (N, H, W, 1)
        id_map, _ = dr.interpolate(self.vert_lid, rast_out, self.faces)
        id_map = torch.flip(id_map, dims=[1])  # match paradex convention

        ids = torch.clamp(torch.round(id_map[..., 0]).long(), 0, self.n_links)  # (N, H, W)
        colors = self.color_lut[ids]          # (N, H, W, 3)
        alphas = self.alpha_lut[ids]          # (N, H, W, 1)

        # Upload frames and blend
        frames_np = np.stack(frames_bgr_list)  # (N, H, W, 3) uint8
        frames_gpu = torch.from_numpy(frames_np).to(device).float()  # (N, H, W, 3)
        overlay = frames_gpu * (1.0 - alphas) + colors * alphas
        overlay_u8 = overlay.clamp(0, 255).to(torch.uint8)
        overlay_np = overlay_u8.cpu().numpy()  # single D2H transfer
        return [overlay_np[i] for i in range(self.N)]


# ── Joint sequence helpers ───────────────────────────────────────────────────

def build_qpos_sequences(exp_dir, video_times, hand_type, **unused):
    """Load pre-synced qpos from {exp}/arm/state.npy etc.

    Run src/process/precompute_synced_qpos.py first to generate these files.
    """
    from paradex.utils.file_io import load_robot_traj, load_robot_target_traj

    req = [exp_dir / "arm" / "state.npy", exp_dir / "arm" / "action.npy",
           exp_dir / "hand" / "state.npy", exp_dir / "hand" / "action.npy"]
    missing = [p for p in req if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Synced qpos not found in {exp_dir}. "
            f"Run: python src/process/precompute_synced_qpos.py --hand {hand_type}"
        )
    actual = load_robot_traj(str(exp_dir))
    target = load_robot_target_traj(str(exp_dir))
    n = len(video_times)
    return actual[:n], target[:n]


# ── Skip logic ───────────────────────────────────────────────────────────────

def _is_valid_video(path, min_frames=1):
    """Quick validity check: file exists, can open, frame count > threshold."""
    if not path.exists():
        return False
    # Silence ffmpeg stderr ("moov atom not found" etc) for corrupted/partial files
    stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 2)
    try:
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    finally:
        os.dup2(stderr_fd, 2)
        os.close(devnull_fd)
        os.close(stderr_fd)
    return n >= min_frames


def nas_has_all(nas_out_dir, serials):
    for s in serials:
        if not (nas_out_dir / f"overlay_actual_{s}.mp4").exists():
            return False
        if not (nas_out_dir / f"overlay_target_{s}.mp4").exists():
            return False
    return True


def obj_all_done_on_nas(hand, obj, exps_with_serials):
    for exp_name, serials in exps_with_serials:
        nas_out = OUTPUT_BASE / hand / obj / exp_name
        if not nas_has_all(nas_out, serials):
            return False
    return True


# ── Background download/upload ───────────────────────────────────────────────

def _is_videos_cached(local_videos_dir):
    if not local_videos_dir.is_dir():
        return False
    return any(local_videos_dir.glob("*.avi"))


def download_episode(nas_ep, local_ep):
    """Copy ONLY videos/ from one episode; everything else read from NAS directly."""
    local_videos = local_ep / "videos"
    if _is_videos_cached(local_videos):
        return
    if local_videos.exists():
        shutil.rmtree(local_videos)
    local_ep.mkdir(parents=True, exist_ok=True)
    nas_videos = nas_ep / "videos"
    if nas_videos.is_dir():
        shutil.copytree(str(nas_videos), str(local_videos))


def download_object(nas_dir, local_dir):
    local_dir.mkdir(parents=True, exist_ok=True)
    for ep_dir in sorted(nas_dir.iterdir()):
        if not ep_dir.is_dir():
            continue
        download_episode(ep_dir, local_dir / ep_dir.name)


def upload_output(local_out_dir, nas_out_dir):
    if not local_out_dir.exists():
        return
    nas_out_dir.parent.mkdir(parents=True, exist_ok=True)
    for src_ep in local_out_dir.iterdir():
        dst_ep = nas_out_dir / src_ep.name
        dst_ep.mkdir(parents=True, exist_ok=True)
        for f in src_ep.iterdir():
            dst = dst_ep / f.name
            if not dst.exists():
                shutil.copy2(str(f), str(dst))
    shutil.rmtree(local_out_dir)


def upload_episode(local_ep_dir, nas_ep_dir):
    """Upload a single episode's output dir to NAS and delete local copy."""
    if not local_ep_dir.exists():
        return
    nas_ep_dir.mkdir(parents=True, exist_ok=True)
    for f in local_ep_dir.iterdir():
        dst = nas_ep_dir / f.name
        if not dst.exists():
            shutil.copy2(str(f), str(dst))
    shutil.rmtree(local_ep_dir)


# ── Per-experiment rendering ─────────────────────────────────────────────────

def process_experiment(exp_dir, nas_ep, hand_type, arm_name, out_base, nas_out_base,
                       serial_filter=None, arm_time_offset=0.09, frame_pbar=None,
                       per_frame_timing=False):
    """exp_dir: local with videos/ only. nas_ep: NAS path for cam_param/raw/C2R."""
    if not (nas_ep / "C2R.npy").exists():
        return 0
    if not (exp_dir / "videos").is_dir():
        return 0

    intrinsic, extrinsic_from_camparam = load_camparam(str(nas_ep))
    c2r = np.load(nas_ep / "C2R.npy")

    all_serials = sorted(intrinsic.keys())
    if serial_filter:
        all_serials = [s for s in all_serials if s in serial_filter]
    all_serials = [s for s in all_serials if (exp_dir / "videos" / f"{s}.avi").exists()]
    if not all_serials:
        return 0

    todo = []
    for s in all_serials:
        nas_a = nas_out_base / f"overlay_actual_{s}.mp4"
        nas_t = nas_out_base / f"overlay_target_{s}.mp4"
        local_a = out_base / f"overlay_actual_{s}.mp4"
        local_t = out_base / f"overlay_target_{s}.mp4"
        if nas_a.exists() and nas_t.exists():
            continue
        if _is_valid_video(local_a, 1) and _is_valid_video(local_t, 1):
            nas_out_base.mkdir(parents=True, exist_ok=True)
            for src, dst in [(local_a, nas_a), (local_t, nas_t)]:
                if not dst.exists():
                    shutil.copy2(str(src), str(dst))
            continue
        todo.append(s)

    if not todo:
        return len(all_serials)

    ts_path = nas_ep / "raw" / "timestamps" / "timestamp.npy"
    fid_path = nas_ep / "raw" / "timestamps" / "frame_id.npy"
    if not ts_path.exists() or not fid_path.exists():
        print(f"  [skip] no timestamps: {exp_dir.name}", flush=True)
        return 0
    video_times = np.load(ts_path)
    total_frames = len(video_times)

    try:
        actual_qpos, target_qpos = build_qpos_sequences(
            nas_ep, video_times, hand_type, arm_time_offset=arm_time_offset)
    except Exception as e:
        print(f"  [qpos error] {exp_dir.name}: {e}", flush=True)
        return 0

    urdf_base = Path.home() / "AutoDex" / "autodex" / "planner" / "src" / "curobo" / "content" / "assets" / "robot"
    urdf_path = str(urdf_base / f"{hand_type}_description" / f"xarm_{hand_type}.urdf")
    robot = RobotModule(urdf_path)
    robot_dof = robot.get_num_joints()

    # Camera extrinsics in robot frame: cam_from_robot = cam_from_world @ c2r
    render_extrinsics = {}
    for s in todo:
        cam_from_world = np.eye(4)
        cam_from_world[:3, :] = extrinsic_from_camparam[s]
        cam_from_robot = cam_from_world @ c2r
        render_extrinsics[s] = cam_from_robot[:3, :]

    # Pre-collect scene info (once per episode)
    robot.update_cfg(actual_qpos[0, :robot_dof])
    scene = robot.scene
    link_names_ordered = list(scene.geometry.keys())
    scene_meshes = [scene.geometry[ln] for ln in link_names_ordered]
    link_labels = {ln: _label_for_link(ln) for ln in link_names_ordered}

    caps = {}
    writers_a, writers_t = {}, {}
    tmp_a, tmp_t = {}, {}
    W, H, fps = None, None, None
    out_base.mkdir(parents=True, exist_ok=True)

    for s in todo:
        cap = cv2.VideoCapture(str(exp_dir / "videos" / f"{s}.avi"))
        if W is None:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        caps[s] = cap

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        tmp_a[s] = str(out_base / f"_tmp_actual_{s}.avi")
        tmp_t[s] = str(out_base / f"_tmp_target_{s}.avi")
        writers_a[s] = cv2.VideoWriter(tmp_a[s], fourcc, fps, (W, H))
        writers_t[s] = cv2.VideoWriter(tmp_t[s], fourcc, fps, (W, H))

    intrinsic_subset = {s: intrinsic[s] for s in todo}
    gpu_renderer = RobotOverlayRenderer(
        scene_meshes, link_names_ordered, link_labels,
        intrinsic_subset, render_extrinsics, H, W)
    ordered_serials = gpu_renderer.serials

    n_frames = min(min(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps.values()), total_frames)

    t_read = t_fk = t_render = t_write = 0.0

    for fidx in range(n_frames):
        fr = fk = rn = wr = und = 0.0

        t0 = time.time()
        frames_bgr_list = []
        for s in ordered_serials:
            ret, frame = caps[s].read()
            frames_bgr_list.append(frame if ret else np.zeros((H, W, 3), dtype=np.uint8))
        fr = time.time() - t0
        t_read += fr

        und = 0.0

        for seq, writers in [(actual_qpos, writers_a), (target_qpos, writers_t)]:
            t0 = time.time()
            robot.update_cfg(seq[fidx, :robot_dof])
            scene = robot.scene
            link_poses = [scene.graph.get(ln)[0] for ln in link_names_ordered]
            dt = time.time() - t0
            fk += dt
            t_fk += dt

            t0 = time.time()
            overlays = gpu_renderer.render(link_poses, frames_bgr_list)
            dt = time.time() - t0
            rn += dt
            t_render += dt

            t0 = time.time()
            for i, s in enumerate(ordered_serials):
                writers[s].write(overlays[i])
            dt = time.time() - t0
            wr += dt
            t_write += dt

        if per_frame_timing:
            tot = fr + und + fk + rn + wr
            print(f"  f{fidx}/{n_frames} read={fr:.2f} und={und:.2f} fk={fk:.2f} "
                  f"render={rn:.2f} write={wr:.2f} tot={tot:.2f}s", flush=True)
        if frame_pbar:
            frame_pbar.update(1)

    for cap in caps.values():
        cap.release()
    for writers in [writers_a, writers_t]:
        for w in writers.values():
            w.release()

    # Convert mp4v temp -> H264 mp4, limit concurrent ffmpeg to avoid system OOM
    t0 = time.time()
    MAX_PARALLEL = 8
    jobs = []
    for s in todo:
        for tmp_path, final_name in [
            (tmp_a[s], f"overlay_actual_{s}.mp4"),
            (tmp_t[s], f"overlay_target_{s}.mp4"),
        ]:
            jobs.append((tmp_path, str(out_base / final_name)))

    running = []
    for tmp_path, final_path in jobs:
        while len(running) >= MAX_PARALLEL:
            for i, (rp, rt) in enumerate(running):
                if rp.poll() is not None:
                    try: os.remove(rt)
                    except OSError: pass
                    running.pop(i)
                    break
            else:
                time.sleep(0.05)
        p = subprocess.Popen(
            ["ffmpeg", "-y", "-i", tmp_path,
             "-c:v", "libx264", "-preset", "fast", "-crf", "23",
             "-pix_fmt", "yuv420p", "-loglevel", "quiet", final_path])
        running.append((p, tmp_path))
    for p, tmp_path in running:
        p.wait()
        try: os.remove(tmp_path)
        except OSError: pass
    t_convert = time.time() - t0

    print(f"  {exp_dir.name}: {len(todo)}cams  read={t_read:.1f}s fk={t_fk:.1f}s "
          f"render={t_render:.1f}s write={t_write:.1f}s convert={t_convert:.1f}s", flush=True)
    return len(all_serials)


# ── Main ─────────────────────────────────────────────────────────────────────

def discover_work(hand, objects, ep_filter, serial_filter):
    hand_dir = EXP_BASE / hand
    if not hand_dir.exists():
        return []
    if objects is None:
        objects = sorted(d.name for d in hand_dir.iterdir() if d.is_dir())

    work = []
    for obj in objects:
        obj_dir = hand_dir / obj
        if not obj_dir.is_dir():
            continue
        exps = []
        for ep in sorted(obj_dir.iterdir()):
            if not ep.is_dir():
                continue
            if ep_filter and ep.name not in ep_filter:
                continue
            videos_dir = ep / "videos"
            if not videos_dir.is_dir():
                continue
            serials = sorted(p.stem for p in videos_dir.glob("*.avi"))
            if serial_filter:
                serials = [s for s in serials if s in serial_filter]
            if serials:
                exps.append((ep.name, serials))
        if exps:
            work.append((obj, exps))
    return work


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default="xarm")
    p.add_argument("--hand", required=True, choices=["allegro", "inspire"])
    p.add_argument("--obj", nargs="+", default=None)
    p.add_argument("--ep", nargs="+", default=None)
    p.add_argument("--serial", nargs="+", default=None)
    p.add_argument("--arm-time-offset", type=float, default=0.03)
    p.add_argument("--per-frame-timing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    work = discover_work(args.hand, args.obj, args.ep, args.serial)
    if not work:
        print("No work to do.")
        return

    total_eps = sum(len(exps) for _, exps in work)
    total_cams = sum(len(serials) for _, exps in work for _, serials in exps)
    print(f"=== {len(work)} objects, {total_eps} episodes, {total_cams} cameras ===", flush=True)

    if args.dry_run:
        for obj, exps in work:
            print(f"  {obj}: {len(exps)} eps, {sum(len(s) for _,s in exps)} cams")
        return

    ep_pbar = tqdm(total=total_eps, desc="episodes", unit="ep", position=0, dynamic_ncols=True)
    frame_pbar = tqdm(desc="frames", unit="f", position=1, dynamic_ncols=True)

    # Flat episode list: [(obj, ep_name, serials), ...]
    flat_eps = []
    for obj, exps in work:
        for ep_name, serials in exps:
            flat_eps.append((obj, ep_name, serials))

    def ep_done_on_nas(obj, ep_name, serials):
        return nas_has_all(OUTPUT_BASE / args.hand / obj / ep_name, serials)

    upload_thread = None
    prefetch_thread = None

    # Prefetch first non-done episode (blocking)
    first_idx = None
    for i, (obj, ep_name, serials) in enumerate(flat_eps):
        if not ep_done_on_nas(obj, ep_name, serials):
            nas_ep = EXP_BASE / args.hand / obj / ep_name
            local_ep = LOCAL_CACHE / args.hand / obj / ep_name
            print(f"Downloading {obj}/{ep_name}...", flush=True)
            download_episode(nas_ep, local_ep)
            first_idx = i
            break

    current_upload_obj = None  # track upload boundary per object

    for ei, (obj, ep_name, serials) in enumerate(flat_eps):
        if ep_done_on_nas(obj, ep_name, serials):
            ep_pbar.update(1)
            ep_pbar.set_postfix_str(f"{obj}/{ep_name} skip")
            continue

        nas_ep = EXP_BASE / args.hand / obj / ep_name
        local_ep = LOCAL_CACHE / args.hand / obj / ep_name
        out_base = LOCAL_CACHE / "output" / args.hand / obj / ep_name
        nas_out_base = OUTPUT_BASE / args.hand / obj / ep_name

        # Wait for prefetch of this episode if in flight
        if prefetch_thread is not None:
            prefetch_thread.join()
            prefetch_thread = None
        if not _is_videos_cached(local_ep / "videos"):
            print(f"Downloading {obj}/{ep_name}...", flush=True)
            download_episode(nas_ep, local_ep)

        # Start prefetching next non-done episode in background
        for ni in range(ei + 1, len(flat_eps)):
            n_obj, n_ep, n_ser = flat_eps[ni]
            if not ep_done_on_nas(n_obj, n_ep, n_ser):
                n_nas = EXP_BASE / args.hand / n_obj / n_ep
                n_local = LOCAL_CACHE / args.hand / n_obj / n_ep
                if not _is_videos_cached(n_local / "videos"):
                    prefetch_thread = threading.Thread(
                        target=download_episode, args=(n_nas, n_local), daemon=True)
                    prefetch_thread.start()
                break

        n_ts = 400
        ts_path = nas_ep / "raw" / "timestamps" / "timestamp.npy"
        if ts_path.exists():
            n_ts = len(np.load(ts_path))
        frame_pbar.reset()
        frame_pbar.total = n_ts
        frame_pbar.set_postfix_str(f"{obj}/{ep_name}")

        process_experiment(local_ep, nas_ep, args.hand, args.arm, out_base, nas_out_base,
                           serial_filter=args.serial,
                           arm_time_offset=args.arm_time_offset,
                           frame_pbar=frame_pbar,
                           per_frame_timing=args.per_frame_timing)
        ep_pbar.update(1)

        # Delete this episode's input cache (videos only — small dir)
        if local_ep.exists():
            shutil.rmtree(local_ep)

        # Upload this episode's output to NAS in background (per-episode)
        if out_base.exists() and any(out_base.iterdir()):
            if upload_thread is not None:
                upload_thread.join()
            upload_thread = threading.Thread(
                target=upload_episode,
                args=(out_base, nas_out_base),
                daemon=True)
            upload_thread.start()

    if upload_thread is not None:
        upload_thread.join()

    ep_pbar.close()
    frame_pbar.close()
    print(f"Output: {OUTPUT_BASE / args.hand}", flush=True)


if __name__ == "__main__":
    main()