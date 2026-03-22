#!/usr/bin/env python3
"""Sequential local evaluation: perception + planning on all episodes.

Runs everything locally (no daemons). Times each step.

Usage:
    python src/validation/execution/eval_perception/run_eval_pipeline.py \
        --data_root ~/shared_data/mingi_object_test

    # Single object
    python src/validation/execution/eval_perception/run_eval_pipeline.py \
        --data_root ~/shared_data/mingi_object_test --obj attached_container

    # Force re-run (ignore existing results)
    python src/validation/execution/eval_perception/run_eval_pipeline.py \
        --data_root ~/shared_data/mingi_object_test --force
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

AUTODEX_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(AUTODEX_ROOT))

logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')
logger = logging.getLogger("eval_pipeline")

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


def load_cameras(capture_dir):
    """Load intrinsics, extrinsics, serials from capture dir."""
    capture_dir = Path(capture_dir)
    with open(capture_dir / "cam_param" / "intrinsics.json") as f:
        intr_raw = json.load(f)
    with open(capture_dir / "cam_param" / "extrinsics.json") as f:
        extr_raw = json.load(f)

    img_dir = capture_dir / "images"
    if not img_dir.exists():
        img_dir = capture_dir / "raw" / "images"
    serials = sorted(p.stem for p in img_dir.glob("*.png"))

    intrinsics = {}
    extrinsics = {}
    for s in serials:
        intrinsics[s] = np.array(intr_raw[s]["intrinsics_undistort"], dtype=np.float32)
        T = np.array(extr_raw[s], dtype=np.float64)
        if T.shape == (3, 4):
            T = np.vstack([T, [0, 0, 0, 1]])
        extrinsics[s] = T

    return serials, intrinsics, extrinsics, img_dir


def run_perception_local(capture_dir, obj_name, depth_method, prompt, sil_iters, sil_lr):
    """Run perception pipeline locally (no daemons). Returns (pose_world, timing)."""
    capture_dir = Path(capture_dir)
    serials, intrinsics, extrinsics, img_dir = load_cameras(capture_dir)
    mesh_path = find_mesh(obj_name)

    img0 = cv2.imread(str(img_dir / f"{serials[0]}.png"))
    H, W = img0.shape[:2]

    timing = {}

    # ── Step 1: SAM3 mask ──
    t0 = time.perf_counter()
    from autodex.perception import Sam3ImageSegmentor
    seg = Sam3ImageSegmentor(gpu=0)
    masks_dir = capture_dir / "masks"
    masks_dir.mkdir(exist_ok=True)
    for s in serials:
        rgb = cv2.cvtColor(cv2.imread(str(img_dir / f"{s}.png")), cv2.COLOR_BGR2RGB)
        mask = seg.segment(rgb, prompt)
        if mask is not None:
            cv2.imwrite(str(masks_dir / f"{s}.png"), mask)
    del seg
    import torch; torch.cuda.empty_cache()
    timing["sam3"] = time.perf_counter() - t0
    logger.info(f"  SAM3: {timing['sam3']:.2f}s")

    # ── Step 2: DA3 depth ──
    t0 = time.perf_counter()
    from autodex.perception.depth import get_depth_da3
    images = [cv2.cvtColor(cv2.imread(str(img_dir / f"{s}.png")), cv2.COLOR_BGR2RGB) for s in serials]
    K_arr = np.array([intrinsics[s] for s in serials], dtype=np.float32)
    T_arr = np.array([extrinsics[s] for s in serials], dtype=np.float32)
    depths = get_depth_da3(images, intrinsics=K_arr, extrinsics=T_arr)
    depth_dir = capture_dir / "depth_da3"
    depth_dir.mkdir(exist_ok=True)
    for i, s in enumerate(serials):
        d = depths[i]
        if hasattr(d, 'cpu'):
            d = d.cpu().numpy()
        if d.shape[0] != H or d.shape[1] != W:
            d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
        d_mm = (d * 1000).clip(0, 65535).astype(np.uint16)
        cv2.imwrite(str(depth_dir / f"{s}.png"), d_mm)
    timing["depth"] = time.perf_counter() - t0
    logger.info(f"  Depth: {timing['depth']:.2f}s")

    # ── Step 3: FPose register ──
    t0 = time.perf_counter()
    from autodex.perception import PoseTracker
    tracker = PoseTracker(mesh_path, device_id=0)
    poses_cam = {}
    for s in serials:
        mask_path = masks_dir / f"{s}.png"
        dep_path = depth_dir / f"{s}.png"
        if not mask_path.exists() or not dep_path.exists():
            continue
        rgb = cv2.cvtColor(cv2.imread(str(img_dir / f"{s}.png")), cv2.COLOR_BGR2RGB)
        mask = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        depth = cv2.imread(str(dep_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        depth[(depth < 0.001) | (depth >= 100)] = 0
        K = intrinsics[s].copy()
        # Downscale 0.5
        ds = 0.5
        nw, nh = int(W * ds), int(H * ds)
        rgb_ds = cv2.resize(rgb, (nw, nh))
        depth_ds = cv2.resize(depth, (nw, nh), interpolation=cv2.INTER_NEAREST)
        mask_ds = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        K_ds = K.copy()
        K_ds[0, :] *= ds
        K_ds[1, :] *= ds
        if mask_ds.sum() < 100 or (depth_ds[mask_ds > 0] > 0.001).sum() < 50:
            continue
        tracker.reset()
        try:
            pose_cam = tracker.init(rgb_ds, depth_ds, mask_ds, K_ds, iteration=5)
            poses_cam[s] = pose_cam
        except Exception:
            continue
    del tracker
    torch.cuda.empty_cache()
    timing["fpose"] = time.perf_counter() - t0
    logger.info(f"  FPose: {len(poses_cam)} poses in {timing['fpose']:.2f}s")

    if not poses_cam:
        return None, timing

    # ── Step 4: Best IoU selection ──
    t0 = time.perf_counter()
    _fp_root = AUTODEX_ROOT / "autodex/perception/thirdparty/FoundationPose"
    if str(_fp_root) not in sys.path:
        sys.path.insert(0, str(_fp_root))
    from Utils import make_mesh_tensors, nvdiffrast_render
    import trimesh
    import nvdiffrast.torch as dr

    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])
    mt = make_mesh_tensors(mesh)
    glctx = dr.RasterizeCudaContext()

    best_serial, best_iou, best_pose_world = None, -1, None
    for src_s, pc in poses_cam.items():
        pw = np.linalg.inv(extrinsics[src_s]) @ pc
        ious = []
        for tgt_s in serials:
            mp = masks_dir / f"{tgt_s}.png"
            if not mp.exists():
                continue
            sam_mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 127
            K = intrinsics[tgt_s].astype(np.float32)
            pc_tgt = extrinsics[tgt_s] @ pw
            pt = torch.as_tensor(pc_tgt, device="cuda", dtype=torch.float32).reshape(1, 4, 4)
            rc, _, _ = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=pt, glctx=glctx, mesh_tensors=mt, use_light=False)
            sil = rc[0].detach().cpu().numpy().sum(axis=2) > 0
            inter = (sil & sam_mask).sum()
            union = (sil | sam_mask).sum()
            ious.append(float(inter / union) if union > 0 else 0.0)
        mean_iou = np.mean(ious) if ious else 0.0
        if mean_iou > best_iou:
            best_iou = mean_iou
            best_serial = src_s
            best_pose_world = pw
    del mt, glctx
    timing["select"] = time.perf_counter() - t0
    logger.info(f"  Best IoU: {best_serial} ({best_iou:.3f}) in {timing['select']:.2f}s")

    # ── Step 5: Silhouette matching ──
    t0 = time.perf_counter()
    from autodex.perception.silhouette import SilhouetteOptimizer
    sil_optimizer = SilhouetteOptimizer(mesh_path)
    views = []
    for s in serials:
        mp = masks_dir / f"{s}.png"
        if not mp.exists():
            continue
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None or m.sum() < 100:
            continue
        views.append({
            "mask": m,
            "K": intrinsics[s].astype(np.float32),
            "extrinsic": extrinsics[s].astype(np.float64),
        })
    pose_world = sil_optimizer.optimize(best_pose_world, views, iters=sil_iters, lr=sil_lr, antialias=True)
    del sil_optimizer
    torch.cuda.empty_cache()
    timing["sil"] = time.perf_counter() - t0
    timing["total"] = sum(timing.get(k, 0) for k in ["sam3", "depth", "fpose", "select", "sil"])
    logger.info(f"  Sil: {timing['sil']:.2f}s, Total: {timing['total']:.2f}s")

    return pose_world, timing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--obj", type=str, default=None)
    parser.add_argument("--depth", type=str, default="da3", choices=["da3", "stereo"])
    parser.add_argument("--prompt", type=str, default="object on the checkerboard")
    parser.add_argument("--sil_iters", type=int, default=100)
    parser.add_argument("--sil_lr", type=float, default=0.002)
    parser.add_argument("--force", action="store_true", help="Re-run even if results exist")
    parser.add_argument("--skip_perception", action="store_true")
    parser.add_argument("--grasp_version", type=str, default="selected_100")
    args = parser.parse_args()

    data_root = Path(args.data_root)

    # Build episode list
    episodes = []
    if args.obj:
        obj_dir = data_root / args.obj
        for ep in sorted(obj_dir.iterdir()):
            if ep.is_dir():
                episodes.append((args.obj, str(ep)))
    else:
        for obj_dir in sorted(data_root.iterdir()):
            if not obj_dir.is_dir() or obj_dir.name in ("cam_param", "simulate", "eval_output"):
                continue
            for ep in sorted(obj_dir.iterdir()):
                if ep.is_dir():
                    episodes.append((obj_dir.name, str(ep)))

    print(f"Episodes: {len(episodes)}, Depth: {args.depth}, Force: {args.force}")

    all_timings = []

    for i, (obj, capture_dir) in enumerate(episodes):
        capture_dir = Path(capture_dir)
        print(f"\n[{i+1}/{len(episodes)}] {obj}/{capture_dir.name}")

        timing = {"obj": obj, "episode": capture_dir.name}

        # ── Perception ──
        pose_path = capture_dir / "pose_world.npy"
        if not args.force and pose_path.exists() and not args.skip_perception:
            pose_world = np.load(str(pose_path))
            print(f"  Perception: cached")
        elif args.skip_perception:
            if not pose_path.exists():
                print(f"  SKIP: no pose_world.npy")
                continue
            pose_world = np.load(str(pose_path))
        else:
            pose_world, perc_timing = run_perception_local(
                capture_dir, obj, args.depth, args.prompt, args.sil_iters, args.sil_lr,
            )
            if pose_world is None:
                print(f"  Perception FAILED")
                continue
            timing.update(perc_timing)
            np.save(str(pose_path), pose_world)

        # ── Planning ──
        c2r_path = capture_dir / "cam_param" / "C2R.npy"
        if not c2r_path.exists():
            c2r_path = capture_dir / "C2R.npy"
        if not c2r_path.exists():
            print(f"  SKIP planning: no C2R.npy")
            all_timings.append(timing)
            continue

        from autodex.planner.planner import GraspPlanner
        from autodex.utils.conversion import se32cart

        c2r = np.load(str(c2r_path))
        pose_robot = np.linalg.inv(c2r) @ pose_world
        pose_7d = se32cart(pose_robot).tolist()
        mesh_path = find_mesh(obj)

        scene_cfg = {
            "mesh": {"target": {"pose": pose_7d, "file_path": mesh_path}},
            "cuboid": {"table": {"dims": [2, 3, 0.2], "pose": [1.1, 0, -0.1 + 0.037, 0, 0, 0, 1]}},
        }

        t0 = time.perf_counter()
        try:
            planner = GraspPlanner()
            result = planner.plan(scene_cfg, obj, args.grasp_version)
            t_plan = time.perf_counter() - t0
            timing["planning"] = t_plan
            timing["planning_success"] = result.success
            if result.success:
                np.save(str(capture_dir / "traj.npy"), result.traj)
                print(f"  Planning: OK in {t_plan:.2f}s")
            else:
                print(f"  Planning: FAILED in {t_plan:.2f}s")
        except Exception as e:
            import traceback
            traceback.print_exc()
            t_plan = time.perf_counter() - t0
            timing["planning"] = t_plan
            timing["planning_success"] = False
            print(f"  Planning ERROR: {e}")

        with open(capture_dir / "timing.json", "w") as f:
            json.dump(timing, f, indent=2)
        all_timings.append(timing)

    # Summary
    if all_timings:
        print(f"\n{'='*60}")
        print(f"Summary ({len(all_timings)}/{len(episodes)} episodes)")
        for key in ["sam3", "depth", "fpose", "select", "sil", "total", "planning"]:
            vals = [t[key] for t in all_timings if key in t]
            if vals:
                print(f"  {key:>12}: mean={np.mean(vals):.2f}s min={np.min(vals):.2f}s max={np.max(vals):.2f}s")
        n_success = sum(1 for t in all_timings if t.get("planning_success"))
        n_total = sum(1 for t in all_timings if "planning" in t)
        print(f"  Planning success: {n_success}/{n_total}")

        with open(data_root / "timing_summary.json", "w") as f:
            json.dump(all_timings, f, indent=2)


if __name__ == "__main__":
    main()
