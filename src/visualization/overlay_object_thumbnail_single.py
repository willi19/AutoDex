"""Render pregrasp-moment thumbnails for ONE episode.

For each camera, seek to the video frame closest to the pregrasp timestamp
(from result.json execution_states, offset by +0.03s per robot-video sync),
overlay the initial object pose (pose_world.npy) mesh on the frame, and save
24 per-camera PNGs plus one combined grid PNG.

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


OVERLAY_BGR = (0, 255, 0)
ALPHA = 0.5
ROBOT_VIDEO_OFFSET_S = 0.03

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
    return idx, dt, pregrasp_iso


def read_frame(video_path, frame_idx, H, W):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return np.zeros((H, W, 3), dtype=np.uint8)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", required=True)
    ap.add_argument("--cam_param_dir", required=True)
    ap.add_argument("--pose_world", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--result_json", required=True)
    ap.add_argument("--timestamps", required=True, help="raw/timestamps/timestamp.npy")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--alpha", type=float, default=ALPHA)
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

    frame_idx, dt, pregrasp_iso = find_pregrasp_frame(args.result_json, args.timestamps)
    print(f"[thumb] pregrasp={pregrasp_iso} frame_idx={frame_idx} dt={dt*1000:.1f}ms", flush=True)

    pose_world = np.load(args.pose_world)

    cap0 = cv2.VideoCapture(str(videos_dir / f"{serials[0]}.avi"))
    W = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap0.release()

    mesh = trimesh.load(str(args.mesh), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

    renderer = ObjectOverlayRenderer(mesh, intrinsics, extrinsics, H, W, alpha=args.alpha)
    ordered = renderer.serials

    total = len(ordered) + 1  # per-cam + grid
    frames = []
    for i, s in enumerate(ordered):
        frames.append(read_frame(videos_dir / f"{s}.avi", frame_idx, H, W))
        print(f"[thumb_progress] {i+1}/{total}", flush=True)

    overlays = renderer.render(pose_world, frames)
    for s, img in zip(ordered, overlays):
        cv2.imwrite(str(out_dir / f"thumb_{s}.png"), img)

    grid = make_image_grid([cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in overlays])
    grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_dir / "thumb_grid.png"), grid_bgr)
    print(f"[thumb_progress] {total}/{total}", flush=True)
    print(f"[thumb] wrote {len(ordered)} thumbs + grid to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
