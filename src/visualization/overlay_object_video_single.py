"""Render mesh-overlay videos for ONE episode from GoTrack world poses.

Stateless single-episode worker. Used by src/process/batch_object_overlay.py.
Prints `[overlay_progress] N/TOTAL` lines for parent-process tqdm parsing.

Args take explicit paths instead of NAS layout assumptions.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr

PARADEX_ROOT = Path.home() / "paradex"
sys.path.insert(0, str(PARADEX_ROOT))
from paradex.image.projection import intr_opencv_to_opengl_proj


OVERLAY_BGR = (0, 255, 0)
ALPHA = 0.5

_GLCAM_IN_CVCAM = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1],
], dtype=np.float32)


class ObjectOverlayRenderer:
    def __init__(self, mesh, intrinsics, extrinsics_cw, H, W, color_bgr=OVERLAY_BGR, alpha=ALPHA, device="cuda"):
        self.device = device
        self.serials = sorted(intrinsics.keys())
        self.N = len(self.serials)
        self.H, self.W = H, W
        self.alpha = float(alpha)
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

        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        self.base_verts = torch.from_numpy(verts).to(device)
        self.faces = torch.from_numpy(faces).to(device)
        self.V = verts.shape[0]
        self.color = torch.tensor([color_bgr[0], color_bgr[1], color_bgr[2]], dtype=torch.float32, device=device)

    def render(self, pose_world, frames_bgr_list):
        device = self.device
        pose_w = torch.as_tensor(pose_world, dtype=torch.float32, device=device)
        v_h = torch.cat([self.base_verts, torch.ones(self.V, 1, device=device)], dim=1)
        verts_world = (v_h @ pose_w.T)[:, :3]
        v_homo = torch.cat([verts_world, torch.ones(self.V, 1, device=device)], dim=1)
        pos_clip = torch.einsum("nij,vj->nvi", self.mtx, v_homo).contiguous()
        rast_out, _ = dr.rasterize(self.glctx, pos_clip, self.faces, resolution=(self.H, self.W))
        mask = (rast_out[..., 3] > 0).float()
        mask = torch.flip(mask, dims=[1])
        mask3 = mask.unsqueeze(-1)

        frames_np = np.stack(frames_bgr_list)
        frames_gpu = torch.from_numpy(frames_np).to(device).float()
        color = self.color[None, None, None, :]
        overlay = frames_gpu * (1.0 - mask3 * self.alpha) + color * (mask3 * self.alpha)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", required=True)
    ap.add_argument("--cam_param_dir", required=True)
    ap.add_argument("--gotrack_records", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    args = ap.parse_args()

    videos_dir = Path(args.videos_dir)
    cam_param_dir = Path(args.cam_param_dir)
    rec_path = Path(args.gotrack_records)
    mesh_path = Path(args.mesh)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    intrinsics, extrinsics = load_cam_param(cam_param_dir)
    available = {p.stem for p in videos_dir.glob("*.avi")}
    serials = sorted(s for s in intrinsics if s in available)
    intrinsics = {s: intrinsics[s] for s in serials}
    extrinsics = {s: extrinsics[s] for s in serials}

    records = json.load(open(rec_path))
    poses = {}
    for r in records:
        if r.get("pose_world") is None:
            continue
        poses[int(r["frame_index"])] = np.array(r["pose_world"], dtype=np.float64)
    if not poses:
        print("[error] no poses in records", flush=True)
        sys.exit(1)

    cap_tmp = cv2.VideoCapture(str(videos_dir / f"{serials[0]}.avi"))
    W = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_tmp.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_tmp.release()

    mesh = trimesh.load(str(mesh_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

    renderer = ObjectOverlayRenderer(mesh, intrinsics, extrinsics, H, W, alpha=args.alpha)
    ordered = renderer.serials

    caps, writers, tmps = {}, {}, {}
    for s in ordered:
        caps[s] = cv2.VideoCapture(str(videos_dir / f"{s}.avi"))
        tmps[s] = str(out_dir / f"_tmp_{s}.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writers[s] = cv2.VideoWriter(tmps[s], fourcc, fps, (W, H))

    n_to_render = min(n_frames, max(poses.keys()) + 1)
    print(f"[overlay] {n_to_render} frames x {len(ordered)} cams at {W}x{H}@{fps:.1f}", flush=True)

    last_pose = None
    t_read = t_render = t_write = 0.0
    for fidx in range(n_to_render):
        t0 = time.time()
        frames = []
        for s in ordered:
            ret, fr = caps[s].read()
            frames.append(fr if ret else np.zeros((H, W, 3), dtype=np.uint8))
        t_read += time.time() - t0

        pose_w = poses.get(fidx, last_pose)
        if pose_w is None:
            t0 = time.time()
            for i, s in enumerate(ordered):
                writers[s].write(frames[i])
            t_write += time.time() - t0
            print(f"[overlay_progress] {fidx+1}/{n_to_render}", flush=True)
            continue
        last_pose = pose_w

        t0 = time.time()
        overlays = renderer.render(pose_w, frames)
        t_render += time.time() - t0

        t0 = time.time()
        for i, s in enumerate(ordered):
            writers[s].write(overlays[i])
        t_write += time.time() - t0
        print(f"[overlay_progress] {fidx+1}/{n_to_render}", flush=True)

    for cap in caps.values():
        cap.release()
    for w in writers.values():
        w.release()

    print("[overlay] encoding h264...", flush=True)
    t0 = time.time()
    MAX_PARALLEL = 8
    jobs = [(tmps[s], str(out_dir / f"overlay_{s}.mp4")) for s in ordered]
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

    print(f"[overlay] done. read={t_read:.1f}s render={t_render:.1f}s write={t_write:.1f}s convert={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()