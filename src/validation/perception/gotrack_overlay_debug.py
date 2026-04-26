#!/usr/bin/env python3
"""Debug overlay for GoTrack: render mesh on selected frames × cameras.

Loads {exp}/object_tracking/gotrack_output/world_pose_records.json, picks
specified frames, projects the mesh into specified camera views. Outputs
per-(frame,serial) PNG and a frames×serials grid.

Usage (foundationpose env):
    python src/validation/perception/gotrack_overlay_debug.py \
        --exp ~/shared_data/AutoDex/experiment/selected_100/inspire/attached_container/20260405_235218 \
        --serials 25305462 25305463 25322639 25322648 \
        --frames 0 10 25 49
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh

AUTODEX_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(AUTODEX_ROOT))
sys.path.insert(0, str(AUTODEX_ROOT / "autodex/perception/thirdparty/FoundationPose"))

from Utils import nvdiffrast_render, make_mesh_tensors
import nvdiffrast.torch as dr

OBJ_BASE = Path.home() / "shared_data" / "AutoDex" / "object" / "paradex"


def load_cam(exp, serial):
    intr = json.load(open(exp / "cam_param" / "intrinsics.json"))[serial]
    extr = np.array(json.load(open(exp / "cam_param" / "extrinsics.json"))[serial], dtype=np.float64)
    K = np.array(intr["intrinsics_undistort"], dtype=np.float32).reshape(3, 3)
    T = np.vstack([extr.reshape(3, 4), [0, 0, 0, 1]]) if extr.size == 12 else extr.reshape(4, 4)
    return K, T


def read_frame(exp, serial, fidx):
    cap = cv2.VideoCapture(str(exp / "videos" / f"{serial}.avi"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    ok, bgr = cap.read()
    cap.release()
    assert ok, f"failed to read frame {fidx} from {serial}"
    return bgr


OVERLAY_BGR = np.array([0, 255, 0], dtype=np.uint8)  # bright green


def overlay(bgr, K, T, pose_w, mt, glctx, alpha):
    H, W = bgr.shape[:2]
    pose_cam = (T @ pose_w).astype(np.float32)
    pose_t = torch.from_numpy(pose_cam).unsqueeze(0).cuda()
    _, depth_r, _ = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=pose_t, glctx=glctx, mesh_tensors=mt, use_light=False)
    mask = (depth_r[0].detach().cpu().numpy() > 0).squeeze()
    out = bgr.copy()
    if mask.any():
        out[mask] = (out[mask].astype(np.float32) * (1 - alpha) + OVERLAY_BGR.astype(np.float32) * alpha).astype(np.uint8)
        # Outline for crisper boundary
        edges = cv2.Canny(mask.astype(np.uint8) * 255, 50, 150)
        out[edges > 0] = OVERLAY_BGR
    return out, int(mask.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--serials", nargs="+", required=True)
    ap.add_argument("--frames", nargs="+", type=int, required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--scale", type=float, default=0.4, help="downscale for grid cells")
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()

    exp = Path(args.exp).expanduser()
    obj = exp.parent.name
    mesh_path = OBJ_BASE / obj / "raw_mesh" / f"{obj}.obj"
    assert mesh_path.exists(), mesh_path

    recs = json.load(open(exp / "object_tracking" / "gotrack_output" / "world_pose_records.json"))
    by_idx = {int(r["frame_index"]): r for r in recs if r.get("pose_world") is not None}
    missing = [f for f in args.frames if f not in by_idx]
    assert not missing, f"frames not in records: {missing} (available: {sorted(by_idx)[:5]}...)"

    glctx = dr.RasterizeCudaContext()
    mesh = trimesh.load(str(mesh_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    mt = make_mesh_tensors(mesh)

    out_dir = Path(args.output_dir) if args.output_dir else AUTODEX_ROOT / "outputs" / "gotrack_debug" / obj
    out_dir.mkdir(parents=True, exist_ok=True)

    cams = {s: load_cam(exp, s) for s in args.serials}
    cells = {}  # (frame, serial) -> bgr image
    for f in args.frames:
        pose_w = np.array(by_idx[f]["pose_world"], dtype=np.float64)
        for s in args.serials:
            K, T = cams[s]
            bgr = read_frame(exp, s, f)
            img, area = overlay(bgr, K, T, pose_w, mt, glctx, args.alpha)
            cv2.putText(img, f"f{f} {s} m={area}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            p = out_dir / f"f{f:04d}_{s}.png"
            cv2.imwrite(str(p), img)
            cells[(f, s)] = img
            print(f"  saved {p.name}  area={area}")

    # Build one grid per frame (rows × cols ≈ sqrt(N))
    H, W = next(iter(cells.values())).shape[:2]
    h, w = int(H * args.scale), int(W * args.scale)
    n = len(args.serials)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    if n == 24:
        cols, rows = 6, 4
    for f in args.frames:
        grid = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
        for k, s in enumerate(args.serials):
            r, c = k // cols, k % cols
            cell = cv2.resize(cells[(f, s)], (w, h))
            grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = cell
        grid_path = out_dir / f"grid_f{f:04d}.png"
        cv2.imwrite(str(grid_path), grid)
        print(f"grid: {grid_path}  ({rows}x{cols} = {n} cams)")


if __name__ == "__main__":
    main()