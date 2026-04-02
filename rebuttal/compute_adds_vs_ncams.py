#!/usr/bin/env python3
"""Measure 6D pose accuracy (ADD-S) vs number of cameras.

For each capture, reuses existing SAM3 masks + FPose initial pose,
then runs silhouette matching with camera subsets of size 1..24.
Compares across 3 repetitions at the same pose via ADD-S.

Usage:
    conda activate foundationpose
    python rebuttal/compute_adds_vs_ncams.py \
        --data_dir /path/to/captures \
        --mesh /path/to/mesh.obj \
        --output rebuttal/adds_results.json

Expected data_dir layout (from run_auto.py captures):
    {data_dir}/{obj_name}/{pose_idx}/{rep_idx}/
        images/          # undistorted images (24 cameras)
        cam_param/       # intrinsics.json, extrinsics.json
        _pipeline_tmp/
            masks/       # SAM3 masks per camera
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_cam_data(capture_dir):
    """Load intrinsics, extrinsics, serials, image size."""
    capture_dir = Path(capture_dir)
    with open(capture_dir / "cam_param" / "intrinsics.json") as f:
        intr_raw = json.load(f)
    with open(capture_dir / "cam_param" / "extrinsics.json") as f:
        extr_raw = json.load(f)

    img_dir = capture_dir / "images"
    serials = sorted(p.stem for p in img_dir.glob("*.png"))

    intrinsics = {}
    extrinsics = {}
    for s in serials:
        intrinsics[s] = np.array(intr_raw[s]["intrinsics_undistort"], dtype=np.float32)
        T = np.array(extr_raw[s], dtype=np.float64)
        if T.shape == (3, 4):
            T = np.vstack([T, [0, 0, 0, 1]])
        extrinsics[s] = T

    img0 = cv2.imread(str(img_dir / f"{serials[0]}.png"))
    H, W = img0.shape[:2]

    return intrinsics, extrinsics, serials, H, W


def load_masks(capture_dir, serials):
    """Load SAM3 masks. Returns dict {serial: mask_uint8}."""
    mask_dir = Path(capture_dir) / "_pipeline_tmp" / "masks"
    masks = {}
    for s in serials:
        mp = mask_dir / f"{s}.png"
        if mp.exists():
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m is not None and m.sum() > 100:
                masks[s] = m
    return masks


def build_views(serials_subset, masks, intrinsics, extrinsics):
    """Build views list for SilhouetteOptimizer.optimize()."""
    views = []
    for s in serials_subset:
        if s not in masks:
            continue
        views.append({
            "mask": masks[s],
            "K": intrinsics[s].astype(np.float32),
            "extrinsic": extrinsics[s].astype(np.float64),
        })
    return views


def run_full_pipeline_once(capture_dir, pipeline):
    """Run full perception pipeline (SAM3+DA3+FPose+Sil) once.

    Returns:
        pose_world: 4x4, or None
        timing: dict
    """
    pose_world, timing = pipeline.run(capture_dir=capture_dir)
    return pose_world, timing


def run_sil_only(sil_optimizer, initial_pose, serials_subset, masks, intrinsics, extrinsics,
                 sil_iters=100, sil_lr=0.002):
    """Run silhouette matching with a camera subset.

    Returns:
        pose_world: 4x4 numpy array
        sil_loss: float
    """
    views = build_views(serials_subset, masks, intrinsics, extrinsics)
    if not views:
        return None, float("inf")

    pose_world, sil_loss = sil_optimizer.optimize(
        initial_pose, views, iters=sil_iters, lr=sil_lr, antialias=True,
    )
    return pose_world, sil_loss


def adds_error(pose_a, pose_b, model_points):
    """Compute ADD-S error between two poses."""
    from scipy.spatial import cKDTree

    pts_a = (pose_a[:3, :3] @ model_points.T).T + pose_a[:3, 3]
    pts_b = (pose_b[:3, :3] @ model_points.T).T + pose_b[:3, 3]
    tree = cKDTree(pts_a)
    dists, _ = tree.query(pts_b, k=1)
    return float(np.mean(dists))


def process_capture_group(
    capture_dirs,  # list of dirs for same object+pose (3 reps)
    sil_optimizer,
    initial_poses,  # list of initial poses (one per rep, from full pipeline)
    model_points,
    n_repeats=10,
    sil_iters=100,
    sil_lr=0.002,
):
    """For each camera count k=1..24, run sil matching on random subsets.

    Returns:
        results_per_k: list of {n_cams, adds_mean, adds_max, adds_min, details: [...]}
    """
    # Load cam data + masks for each rep
    reps_data = []
    for cap_dir in capture_dirs:
        intr, extr, serials, H, W = load_cam_data(cap_dir)
        masks = load_masks(cap_dir, serials)
        valid_serials = [s for s in serials if s in masks]
        reps_data.append({
            "capture_dir": cap_dir,
            "intrinsics": intr,
            "extrinsics": extr,
            "serials": serials,
            "valid_serials": valid_serials,
            "masks": masks,
        })

    n_cams_total = len(reps_data[0]["valid_serials"])
    from math import comb

    results_per_k = []
    for k in range(1, n_cams_total + 1):
        total_combos = comb(n_cams_total, k)
        n_samples = min(n_repeats, total_combos)

        # Generate random subsets
        subsets = set()
        for _ in range(n_samples * 10):
            if len(subsets) >= n_samples:
                break
            valid = reps_data[0]["valid_serials"]
            subset = tuple(sorted(np.random.choice(len(valid), k, replace=False)))
            subsets.add(subset)

        subset_results = []
        for subset_idx in subsets:
            cam_subset = [reps_data[0]["valid_serials"][i] for i in subset_idx]

            # Run sil matching for each rep
            poses = []
            losses = []
            for rep_i, (rep, init_pose) in enumerate(zip(reps_data, initial_poses)):
                pose, loss = run_sil_only(
                    sil_optimizer, init_pose, cam_subset,
                    rep["masks"], rep["intrinsics"], rep["extrinsics"],
                    sil_iters=sil_iters, sil_lr=sil_lr,
                )
                if pose is not None:
                    poses.append(pose)
                    losses.append(loss)

            if len(poses) < 2:
                continue

            # Compute pairwise ADD-S across repetitions
            adds_list = []
            for i in range(len(poses)):
                for j in range(i + 1, len(poses)):
                    adds_list.append(adds_error(poses[i], poses[j], model_points))

            subset_results.append({
                "cameras": cam_subset,
                "adds_mean": float(np.mean(adds_list)),
                "adds_max": float(np.max(adds_list)),
                "sil_losses": losses,
                "n_valid_reps": len(poses),
            })

        if not subset_results:
            results_per_k.append({"n_cams": k, "adds_mean": None, "adds_max": None})
            continue

        all_adds = [r["adds_mean"] for r in subset_results]
        results_per_k.append({
            "n_cams": k,
            "adds_mean": float(np.mean(all_adds)),
            "adds_max": float(np.max(all_adds)),
            "adds_min": float(np.min(all_adds)),
            "details": subset_results,
        })

    return results_per_k


def main():
    parser = argparse.ArgumentParser(
        description="Measure ADD-S vs number of cameras (sil matching only)")
    parser.add_argument("--data_dir", required=True,
                        help="Root dir with {obj}/{pose_idx}/{rep_idx}/ captures")
    parser.add_argument("--mesh", required=True, help="Object mesh .obj path")
    parser.add_argument("--n_points", type=int, default=1000,
                        help="Mesh points for ADD-S")
    parser.add_argument("--n_repeats", type=int, default=10,
                        help="Random camera subsets per k")
    parser.add_argument("--sil_iters", type=int, default=100)
    parser.add_argument("--sil_lr", type=float, default=0.002)
    parser.add_argument("--output", default="rebuttal/adds_vs_ncams.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    import trimesh
    mesh = trimesh.load(args.mesh, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    pts, _ = trimesh.sample.sample_surface(mesh, args.n_points)
    model_points = np.asarray(pts, dtype=np.float64)

    # Load silhouette optimizer
    from autodex.perception.silhouette import SilhouetteOptimizer
    sil_optimizer = SilhouetteOptimizer(args.mesh, device="cuda")

    # Discover capture groups: {obj}/{pose_idx}/ -> list of rep dirs
    from tqdm import tqdm

    data_dir = Path(args.data_dir)
    groups = {}  # (obj, pose_idx) -> [rep_dirs]
    for obj_dir in sorted(data_dir.iterdir()):
        if not obj_dir.is_dir():
            continue
        obj_name = obj_dir.name
        for pose_dir in sorted(obj_dir.iterdir()):
            if not pose_dir.is_dir():
                continue
            pose_idx = pose_dir.name
            rep_dirs = sorted([d for d in pose_dir.iterdir() if d.is_dir()])
            if rep_dirs:
                groups[(obj_name, pose_idx)] = rep_dirs

    print(f"Found {len(groups)} capture groups")

    # For each group, we need initial poses (from full pipeline or stored)
    # Check if pose_world.npy already exists in each rep dir
    all_results = []

    # Resume support
    if os.path.isfile(args.output):
        with open(args.output) as f:
            all_results = json.load(f)
        done_keys = {(r["obj_name"], r["pose_idx"]) for r in all_results}
        print(f"Resuming: {len(all_results)} groups done")
    else:
        done_keys = set()

    pbar = tqdm(sorted(groups.items()), desc="ADD-S", unit="group")
    for (obj_name, pose_idx), rep_dirs in pbar:
        if (obj_name, pose_idx) in done_keys:
            pbar.set_postfix_str(f"{obj_name}/{pose_idx}: cached")
            continue

        pbar.set_postfix_str(f"{obj_name}/{pose_idx}")

        # Load initial poses
        initial_poses = []
        for rd in rep_dirs:
            pose_path = rd / "pose_world.npy"
            if not pose_path.exists():
                break
            initial_poses.append(np.load(str(pose_path)))

        if len(initial_poses) != len(rep_dirs):
            pbar.set_postfix_str(f"{obj_name}/{pose_idx}: missing poses, skipped")
            continue

        capture_dirs = [str(rd) for rd in rep_dirs]
        results_per_k = process_capture_group(
            capture_dirs, sil_optimizer, initial_poses, model_points,
            n_repeats=args.n_repeats, sil_iters=args.sil_iters, sil_lr=args.sil_lr,
        )

        result = {
            "obj_name": obj_name,
            "pose_idx": pose_idx,
            "n_reps": len(rep_dirs),
            "results": results_per_k,
        }
        all_results.append(result)

        # Incremental save
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\nTotal: {len(all_results)} groups processed")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
