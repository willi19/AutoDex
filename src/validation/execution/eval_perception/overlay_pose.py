#!/usr/bin/env python3
"""Overlay object mesh pose on all 24 camera views as a 4x6 grid.

Usage:
    python src/validation/execution/eval_perception/overlay_pose.py \
        --data_root ~/shared_data/mingi_object_test \
        --obj attached_container --episode 20260317_172712
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

AUTODEX_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(AUTODEX_ROOT))

_FP_ROOT = AUTODEX_ROOT / "autodex/perception/thirdparty/FoundationPose"
sys.path.insert(0, str(_FP_ROOT))

MESH_ROOT = Path.home() / "shared_data/object_6d/data/mesh"


def find_mesh(obj_name):
    for name in [f"{obj_name}.obj", "simplified.obj", "coacd.obj"]:
        p = MESH_ROOT / obj_name / name
        if p.exists():
            return str(p)
    objs = list((MESH_ROOT / obj_name).glob("*.obj"))
    if objs:
        return str(objs[0])
    raise FileNotFoundError(f"No mesh for {obj_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--obj", type=str, default=None)
    parser.add_argument("--episode", type=str, default=None)
    parser.add_argument("--output", type=str, default=None, help="Output path (default: {capture_dir}/pose_overlay.png)")
    args = parser.parse_args()

    data_root = Path(args.data_root)

    # Collect episodes
    episodes = []
    if args.obj and args.episode:
        episodes.append((args.obj, data_root / args.obj / args.episode))
    elif args.obj:
        for ep_dir in sorted((data_root / args.obj).iterdir()):
            if ep_dir.is_dir():
                episodes.append((args.obj, ep_dir))
    else:
        for obj_dir in sorted(data_root.iterdir()):
            if not obj_dir.is_dir() or obj_dir.name in ("cam_param", "simulate"):
                continue
            for ep_dir in sorted(obj_dir.iterdir()):
                if ep_dir.is_dir():
                    episodes.append((obj_dir.name, ep_dir))

    import trimesh
    import nvdiffrast.torch as dr
    try:
        from Utils import make_mesh_tensors, nvdiffrast_render
    except ImportError:
        from render_utils import make_mesh_tensors, nvdiffrast_render

    purple = np.array([128, 0, 128], dtype=np.uint8)
    green = np.array([0, 200, 0], dtype=np.uint8)

    current_obj = None
    mt = None
    glctx = None

    for obj, capture_dir in episodes:
        pose_path = capture_dir / "pose_world.npy"
        if not pose_path.exists():
            print(f"  {obj}/{capture_dir.name}: no pose_world.npy, skip")
            continue

        # Load mesh (cache per object)
        if obj != current_obj:
            current_obj = obj
            mesh_path = find_mesh(obj)
            mesh = trimesh.load(mesh_path, process=False)
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate([g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])
            mesh_vis = mesh.copy()
            vc = np.tile(np.append(purple, 255).reshape(1, 4), (len(mesh_vis.vertices), 1))
            mesh_vis.visual = trimesh.visual.ColorVisuals(vertex_colors=vc)
            if mt is not None:
                del mt
            mt = make_mesh_tensors(mesh_vis)
            if glctx is None:
                glctx = dr.RasterizeCudaContext()

        pose_world = np.load(str(pose_path))

        with open(capture_dir / "cam_param" / "intrinsics.json") as f:
            intr_raw = json.load(f)
        with open(capture_dir / "cam_param" / "extrinsics.json") as f:
            extr_raw = json.load(f)

        img_dir = capture_dir / "images"
        if not img_dir.exists():
            img_dir = capture_dir / "raw" / "images"
        serials = sorted(p.stem for p in img_dir.glob("*.png"))

        masks_dir = capture_dir / "masks"

        overlays = []
        for s in serials:
            K = np.array(intr_raw[s]["intrinsics_undistort"], dtype=np.float32)
            T = np.array(extr_raw[s], dtype=np.float64)
            if T.shape == (3, 4):
                T = np.vstack([T, [0, 0, 0, 1]])

            img_bgr = cv2.imread(str(img_dir / f"{s}.png"))
            H, W = img_bgr.shape[:2]

            pose_cam = T @ pose_world
            pose_t = torch.as_tensor(pose_cam, device="cuda", dtype=torch.float32).reshape(1, 4, 4)
            rc, _, _ = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=pose_t, glctx=glctx, mesh_tensors=mt, use_light=False)
            rm = rc[0].detach().cpu().numpy().sum(axis=2) > 0

            ov = img_bgr.copy()

            # SAM3 mask overlay (green)
            mask_path = masks_dir / f"{s}.png"
            if mask_path.exists():
                sam = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if sam is not None:
                    if sam.shape[0] != H or sam.shape[1] != W:
                        sam = cv2.resize(sam, (W, H), interpolation=cv2.INTER_NEAREST)
                    m = sam > 127
                    ov[m] = (ov[m].astype(np.float32) * 0.6 + green[[2,1,0]].astype(np.float32) * 0.4).astype(np.uint8)

            # Mesh silhouette overlay (purple)
            ov[rm] = (ov[rm].astype(np.float32) * 0.4 + purple[[2,1,0]].astype(np.float32) * 0.6).astype(np.uint8)

            cv2.putText(ov, s, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            overlays.append(ov)

        if not overlays:
            continue

        cols = 4
        rows = (len(overlays) + cols - 1) // cols
        scale = 0.25
        oh, ow = overlays[0].shape[:2]
        th, tw = int(oh * scale), int(ow * scale)
        grid = np.ones((rows * th, cols * tw, 3), dtype=np.uint8) * 40
        for idx, img in enumerate(overlays):
            r, c = divmod(idx, cols)
            small = cv2.resize(img, (tw, th))
            grid[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = small

        out_path = args.output or str(capture_dir / "pose_overlay.png")
        cv2.imwrite(out_path, grid)
        print(f"{obj}/{capture_dir.name}: saved {out_path}")

    if glctx:
        del mt, glctx


if __name__ == "__main__":
    main()