#!/usr/bin/env python3
"""Step 5: Run FPose with stereo depth + compute mask IoU for both depth methods + select best 5 views.

For each view, computes IoU between FPose silhouette and SAM3 mask.
Ranks views by IoU. Selects best 5 across all episodes per object.

conda: foundationpose
"""
import argparse
import json
import sys
import os
from pathlib import Path

import cv2
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
AUTODEX_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(AUTODEX_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

_FP_ROOT = AUTODEX_ROOT / "autodex/perception/thirdparty/_object_6d_tracking/FoundationPose"
sys.path.insert(0, str(_FP_ROOT))

from Utils import make_mesh_tensors, nvdiffrast_render
import trimesh
import nvdiffrast.torch as dr

MESH_ROOT = Path("/home/mingi/shared_data/object_6d/data/mesh")


def find_mesh(obj_name):
    for name in ["simplified.obj", f"{obj_name}.obj", "coacd.obj", "raw.obj"]:
        p = MESH_ROOT / obj_name / name
        if p.exists():
            return str(p)
    objs = list((MESH_ROOT / obj_name).glob("*.obj"))
    if objs:
        return str(objs[0])
    raise FileNotFoundError(f"No mesh for {obj_name}")


def compute_iou(mask1, mask2):
    inter = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()
    return float(inter / union) if union > 0 else 0.0


def run_fpose_and_iou(capture_dir, depth_dir_name, mesh_path, tracker, serials, intrinsics, extrinsics, img_dir, downscale=0.5):
    """Run FPose with given depth, compute per-view IoU against SAM3 mask.

    Returns dict {serial: {"pose_cam": 4x4, "pose_world": 4x4, "iou": float}}
    """
    masks_dir = capture_dir / "masks"
    depth_dir = capture_dir / depth_dir_name
    results = {}

    for i, s in enumerate(serials):
        mask_path = masks_dir / f"{s}.png"
        depth_path = depth_dir / f"{s}.png"
        if not mask_path.exists() or not depth_path.exists():
            continue

        rgb = cv2.cvtColor(cv2.imread(str(img_dir / f"{s}.png")), cv2.COLOR_BGR2RGB)
        mask = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        depth[(depth < 0.001) | (depth >= 100)] = 0
        K = intrinsics[i].copy()

        ds = downscale
        if ds != 1.0:
            h, w = rgb.shape[:2]
            nw, nh = int(w * ds), int(h * ds)
            rgb = cv2.resize(rgb, (nw, nh))
            depth = cv2.resize(depth, (nw, nh), interpolation=cv2.INTER_NEAREST)
            mask_ds = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
            K[0, :] *= ds
            K[1, :] *= ds
        else:
            mask_ds = mask

        if mask_ds.sum() < 100 or (depth[mask_ds > 0] > 0.001).sum() < 50:
            continue

        tracker.reset()
        try:
            pose_cam = tracker.init(rgb, depth, mask_ds, K, iteration=5)
        except Exception as e:
            print(f"    {s}: FAILED ({e})")
            continue

        T_world_cam = np.linalg.inv(extrinsics[i])
        pose_world = T_world_cam @ pose_cam

        # Compute IoU: render silhouette from pose_cam, compare with original mask
        sam_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127
        results[s] = {
            "pose_cam": pose_cam,
            "pose_world": pose_world,
            "sam_mask": sam_mask,
        }

    return results


def compute_silhouette_ious(results, mesh_tensors, glctx, intrinsics_full, extrinsics_full, serials, masks_dir, H, W):
    """For each source view's pose_world, render into ALL 24 views and compute mean IoU."""
    ious = {}
    for src_s, data in results.items():
        pose_world = data["pose_world"]
        view_ious = []
        for i, tgt_s in enumerate(serials):
            mask_path = masks_dir / f"{tgt_s}.png"
            if not mask_path.exists():
                continue
            sam_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127
            K = intrinsics_full[i].astype(np.float32)
            pose_cam = extrinsics_full[i] @ pose_world
            pose_t = torch.as_tensor(pose_cam, device="cuda", dtype=torch.float32).reshape(1, 4, 4)
            rc, _, _ = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=pose_t, glctx=glctx, mesh_tensors=mesh_tensors, use_light=False)
            sil = rc[0].detach().cpu().numpy().sum(axis=2) > 0
            view_ious.append(compute_iou(sil, sam_mask))
        ious[src_s] = np.mean(view_ious) if view_ious else 0.0
    return ious


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--downscale", type=float, default=0.5)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    obj_dir = data_root / args.obj
    episodes = sorted([d.name for d in obj_dir.iterdir() if d.is_dir() and d.name != "cam_param"])

    mesh_path = find_mesh(args.obj)
    print(f"Object: {args.obj}, Mesh: {mesh_path}")
    print(f"Episodes: {len(episodes)}")

    # Load FPose once
    from autodex.perception import PoseTracker
    tracker = PoseTracker(mesh_path, device_id=0)

    # Load mesh for rendering
    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])
    mesh_tensors = make_mesh_tensors(mesh)
    glctx = dr.RasterizeCudaContext()

    # Per-serial IoU accumulator across episodes
    serial_ious_da3 = {}
    serial_ious_stereo = {}

    for ep in episodes:
        capture_dir = obj_dir / ep
        print(f"\n--- {args.obj}/{ep} ---")

        with open(capture_dir / "cam_param" / "intrinsics.json") as f:
            intr_raw = json.load(f)
        with open(capture_dir / "cam_param" / "extrinsics.json") as f:
            extr_raw = json.load(f)

        img_dir = capture_dir / "images"
        if not img_dir.exists():
            img_dir = capture_dir / "raw" / "images"
        serials = sorted(p.stem for p in img_dir.glob("*.png"))

        intrinsics = np.array([intr_raw[s]["intrinsics_undistort"] for s in serials], dtype=np.float32)
        extrinsics = []
        for s in serials:
            T = np.array(extr_raw[s], dtype=np.float64)
            if T.shape == (3, 4):
                T = np.vstack([T, [0, 0, 0, 1]])
            extrinsics.append(T)
        extrinsics = np.array(extrinsics)

        img = cv2.imread(str(img_dir / f"{serials[0]}.png"))
        H, W = img.shape[:2]

        # DA3 depth — use existing poses if available
        da3_pose_dir = capture_dir / "pose"
        da3_results = {}
        masks_dir = capture_dir / "masks"
        for s in serials:
            cam_path = da3_pose_dir / f"{s}_cam.npy"
            mask_path = masks_dir / f"{s}.png"
            if cam_path.exists() and mask_path.exists():
                da3_results[s] = {
                    "pose_cam": np.load(str(cam_path)),
                    "pose_world": np.load(str(da3_pose_dir / f"{s}.npy")),
                    "sam_mask": cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127,
                }

        if da3_results:
            da3_ious = compute_silhouette_ious(da3_results, mesh_tensors, glctx, intrinsics, extrinsics, serials, masks_dir, H, W)
            print(f"  DA3: {len(da3_ious)} views, mean IoU={np.mean(list(da3_ious.values())):.3f}")
            for s, iou in da3_ious.items():
                serial_ious_da3.setdefault(s, []).append(iou)

        # Stereo depth — run FPose
        stereo_dir = capture_dir / "depth_stereo"
        if stereo_dir.exists() and len(list(stereo_dir.glob("*.png"))) > 0:
            print(f"  Running FPose with stereo depth...")
            stereo_results = run_fpose_and_iou(
                capture_dir, "depth_stereo", mesh_path, tracker,
                serials, intrinsics, extrinsics, img_dir, args.downscale,
            )
            if stereo_results:
                stereo_ious = compute_silhouette_ious(stereo_results, mesh_tensors, glctx, intrinsics, extrinsics, serials, masks_dir, H, W)
                print(f"  Stereo: {len(stereo_ious)} views, mean IoU={np.mean(list(stereo_ious.values())):.3f}")
                for s, iou in stereo_ious.items():
                    serial_ious_stereo.setdefault(s, []).append(iou)

                # Save stereo poses
                stereo_pose_dir = capture_dir / "pose_stereo"
                stereo_pose_dir.mkdir(exist_ok=True)
                for s, data in stereo_results.items():
                    np.save(str(stereo_pose_dir / f"{s}.npy"), data["pose_world"])
                    np.save(str(stereo_pose_dir / f"{s}_cam.npy"), data["pose_cam"])
        else:
            print(f"  No stereo depth, skipping")

    # Aggregate: mean IoU per serial across episodes
    print(f"\n{'='*60}")
    print(f"Best views for {args.obj} (averaged over {len(episodes)} episodes)")
    print(f"{'='*60}")

    print(f"\n--- DA3 depth ---")
    da3_mean = {s: np.mean(ious) for s, ious in serial_ious_da3.items()}
    da3_ranked = sorted(da3_mean.items(), key=lambda x: -x[1])
    for rank, (s, iou) in enumerate(da3_ranked, 1):
        n = len(serial_ious_da3[s])
        marker = " ***" if rank <= args.top_k else ""
        print(f"  {rank:2d}. {s}: IoU={iou:.3f} (n={n}){marker}")

    if serial_ious_stereo:
        print(f"\n--- Stereo depth ---")
        stereo_mean = {s: np.mean(ious) for s, ious in serial_ious_stereo.items()}
        stereo_ranked = sorted(stereo_mean.items(), key=lambda x: -x[1])
        for rank, (s, iou) in enumerate(stereo_ranked, 1):
            n = len(serial_ious_stereo[s])
            marker = " ***" if rank <= args.top_k else ""
            print(f"  {rank:2d}. {s}: IoU={iou:.3f} (n={n}){marker}")

    # Save results
    output = {
        "obj": args.obj,
        "episodes": episodes,
        "top_k": args.top_k,
        "da3": {
            "per_serial_mean_iou": da3_mean,
            "ranking": [s for s, _ in da3_ranked],
            "best_views": [s for s, _ in da3_ranked[:args.top_k]],
        },
    }
    if serial_ious_stereo:
        output["stereo"] = {
            "per_serial_mean_iou": stereo_mean,
            "ranking": [s for s, _ in stereo_ranked],
            "best_views": [s for s, _ in stereo_ranked[:args.top_k]],
        }

    out_path = obj_dir / "best_views.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()