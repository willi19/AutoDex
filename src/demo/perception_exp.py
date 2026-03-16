#!/usr/bin/env python3
"""Generate perception visualizations for all captures in selected_100.

Steps (each needs specific conda env):
  mask_sam3     SAM3 masks               (env: sam3)
  mask_yoloe    YOLO-E masks             (env: foundationpose)
  depth_da3     Depth-Anything-3         (env: sam3)
  depth_stereo  FoundationStereo         (env: foundationpose)
  pose_viz      NMS + silhouette overlay (env: foundationpose)

Output: exp/perception/{obj_name}/{idx}/
  mask_sam3_checkerboard.png      mask_yoloe_checkerboard.png
  mask_sam3_objname.png           mask_yoloe_objname.png
  depth_da3.png                   depth_da3_reproj.png
  depth_stereo.png                depth_stereo_reproj.png
  pose_nms.png                    pose_silhouette.png

Usage:
  python src/demo/perception_exp.py --step mask_sam3
  python src/demo/perception_exp.py --step pose_viz --obj attached_container
  bash src/demo/run_perception_exp.sh [obj_name] [idx]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

AUTODEX_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AUTODEX_ROOT))

DATA_ROOT = Path.home() / "shared_data/RSS2026_Mingi/experiment/selected_100"
OUTPUT_ROOT = AUTODEX_ROOT / "exp" / "perception"
MESH_ROOT = Path.home() / "shared_data/RSS2026_Mingi/object/paradex"
FS_CKPT = AUTODEX_ROOT / "autodex/perception/thirdparty/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth"
FP_DIR = AUTODEX_ROOT / "autodex/perception/thirdparty/_object_6d_tracking/FoundationPose"

logging.basicConfig(level=logging.INFO, format="[percep] %(message)s")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Common helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def discover(obj=None, idx=None):
    """Find all (obj_name, idx_name, capture_path) tuples."""
    captures = []
    for obj_dir in sorted(DATA_ROOT.iterdir()):
        if not obj_dir.is_dir() or (obj and obj_dir.name != obj):
            continue
        for idx_dir in sorted(obj_dir.iterdir()):
            if not idx_dir.is_dir() or (idx and idx_dir.name != idx):
                continue
            if (idx_dir / "images").is_dir() and (idx_dir / "cam_param").is_dir():
                captures.append((obj_dir.name, idx_dir.name, idx_dir))
    return captures


def load_cameras(capture_dir):
    """Load undistorted images, intrinsics, extrinsics."""
    with open(capture_dir / "cam_param" / "intrinsics.json") as f:
        intr = json.load(f)
    with open(capture_dir / "cam_param" / "extrinsics.json") as f:
        extr = json.load(f)

    serials = sorted(intr.keys())
    images, Ks, Ts = {}, {}, {}
    for s in serials:
        p = capture_dir / "images" / f"{s}.png"
        if not p.exists():
            continue
        images[s] = cv2.imread(str(p))
        Ks[s] = np.array(intr[s]["intrinsics_undistort"]).reshape(3, 3)
        ext = np.array(extr[s])
        Ts[s] = np.vstack([ext, [0, 0, 0, 1]]) if ext.shape == (3, 4) else ext

    serials = [s for s in serials if s in images]
    return images, Ks, Ts, serials


def make_grid(imgs, serials, title="", ncols=5, scale=0.25):
    """Render a labeled camera grid image (default 5×5 for 24 cameras + 1 blank)."""
    items = [(s, imgs[s]) for s in serials if s in imgs]
    if not items:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    h0, w0 = items[0][1].shape[:2]
    h, w = int(h0 * scale), int(w0 * scale)
    nrows = (len(items) + ncols - 1) // ncols

    title_h = 40 if title else 0
    grid = np.zeros((title_h + nrows * h, ncols * w, 3), dtype=np.uint8)
    if title:
        cv2.putText(grid, title, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    for i, (s, img) in enumerate(items):
        r, c = divmod(i, ncols)
        cell = cv2.resize(img, (w, h))
        cv2.putText(cell, s[-8:], (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.putText(cell, s[-8:], (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y0 = title_h + r * h
        grid[y0:y0 + h, c * w:(c + 1) * w] = cell
    return grid


def overlay_mask(img, mask, color=(0, 255, 0), alpha=0.5):
    out = img.copy()
    m = (mask > 0) if mask.ndim == 2 else (mask[..., 0] > 0)
    if m.any():
        c = np.array(color, np.float32)
        out[m] = (out[m].astype(np.float32) * (1 - alpha) + c * alpha).astype(np.uint8)
    return out


def depth_colormap(img, depth, alpha=0.7):
    valid = depth > 0.001
    if not valid.any():
        return img.copy()
    d_min, d_max = depth[valid].min(), depth[valid].max()
    norm = np.zeros_like(depth)
    norm[valid] = (depth[valid] - d_min) / (d_max - d_min + 1e-6)
    cmap = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    cmap[~valid] = 0
    mask3 = np.stack([valid] * 3, -1)
    blend = (img.astype(np.float32) * (1 - alpha) + cmap.astype(np.float32) * alpha).astype(np.uint8)
    return np.where(mask3, blend, img)


def save_grid(path, grid):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), grid)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step: mask_sam3
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def step_mask_sam3(args):
    import torch
    from PIL import Image

    sam3_path = str(AUTODEX_ROOT / "autodex/perception/thirdparty/_object_6d_tracking/sam3")
    if sam3_path not in sys.path:
        sys.path.insert(0, sam3_path)

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(device="cuda")
    processor = Sam3Processor(model, device="cuda", confidence_threshold=0.5)

    captures = discover(args.obj, args.idx)
    logging.info(f"mask_sam3: {len(captures)} captures")

    for ci, (obj_name, idx, cap_dir) in enumerate(captures):
        obj_prompt = obj_name.replace("_", " ")
        prompts = {
            "checkerboard": "object on the checkerboard",
            "objname": obj_prompt,
        }
        if args.prompt:
            tag = args.prompt.replace(" ", "_")
            prompts = {tag: args.prompt}
        out_dir = OUTPUT_ROOT / obj_name / idx
        if not getattr(args, 'force', False) and all((out_dir / f"mask_sam3_{tag}.png").exists() for tag in prompts):
            continue

        images, Ks, Ts, serials = load_cameras(cap_dir)
        if not serials:
            continue

        for tag, prompt in prompts.items():
            out_path = out_dir / f"mask_sam3_{tag}.png"
            if not getattr(args, 'force', False) and out_path.exists():
                continue

            overlays = {}
            for s in serials:
                img_rgb = cv2.cvtColor(images[s], cv2.COLOR_BGR2RGB)
                state = processor.set_image(Image.fromarray(img_rgb))
                processor.reset_all_prompts(state)
                output = processor.set_text_prompt(state=state, prompt=prompt)

                masks_t = output.get("masks")
                scores_t = output.get("scores")
                H, W = images[s].shape[:2]

                if masks_t is None or masks_t.numel() == 0:
                    mask_np = np.zeros((H, W), dtype=np.uint8)
                else:
                    best = int(torch.argmax(scores_t).item()) if scores_t is not None and scores_t.numel() > 0 else 0
                    m = masks_t[best]
                    if m.ndim == 3:
                        m = m[0]
                    mask_np = (m.to(torch.uint8) * 255).cpu().numpy()

                overlays[s] = overlay_mask(images[s], mask_np)

            title = f"SAM3 | \"{prompt}\" | {obj_name}/{idx}"
            save_grid(out_path, make_grid(overlays, serials, title=title))

        logging.info(f"  [{ci + 1}/{len(captures)}] {obj_name}/{idx}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step: mask_yoloe
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def step_mask_yoloe(args):
    from ultralytics import YOLO
    from autodex.perception import get_mask_yoloe

    from autodex.perception.mask import YOLOE_WEIGHTS
    model = YOLO(YOLOE_WEIGHTS)

    captures = discover(args.obj, args.idx)
    logging.info(f"mask_yoloe: {len(captures)} captures")

    for ci, (obj_name, idx, cap_dir) in enumerate(captures):
        obj_prompt = obj_name.replace("_", " ")
        prompts = {
            "checkerboard": "object on the checkerboard",
            "objname": obj_prompt,
        }
        if args.prompt:
            tag = args.prompt.replace(" ", "_")
            prompts = {tag: args.prompt}
        out_dir = OUTPUT_ROOT / obj_name / idx
        if not getattr(args, 'force', False) and all((out_dir / f"mask_yoloe_{tag}.png").exists() for tag in prompts):
            continue

        images, Ks, Ts, serials = load_cameras(cap_dir)
        if not serials:
            continue

        for tag, prompt in prompts.items():
            out_path = out_dir / f"mask_yoloe_{tag}.png"
            if not getattr(args, 'force', False) and out_path.exists():
                continue

            model.set_classes([prompt], model.get_text_pe([prompt]))
            overlays = {}
            for s in serials:
                img_rgb = cv2.cvtColor(images[s], cv2.COLOR_BGR2RGB)
                mask_np = get_mask_yoloe(img_rgb, prompt, model=model, conf_thr=0.2)
                if mask_np is None:
                    mask_np = np.zeros(images[s].shape[:2], dtype=np.uint8)
                overlays[s] = overlay_mask(images[s], mask_np)

            title = f"YOLOE | \"{prompt}\" | {obj_name}/{idx}"
            save_grid(out_path, make_grid(overlays, serials, title=title))

        logging.info(f"  [{ci + 1}/{len(captures)}] {obj_name}/{idx}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step: depth_da3
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def step_depth_da3(args):
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE")
    model = model.to(device="cuda")
    model.eval()

    captures = discover(args.obj, args.idx)
    logging.info(f"depth_da3: {len(captures)} captures")

    for ci, (obj_name, idx, cap_dir) in enumerate(captures):
        out_dir = OUTPUT_ROOT / obj_name / idx
        cmap_path = out_dir / "depth_da3.png"
        reproj_path = out_dir / "depth_da3_reproj.png"
        if cmap_path.exists() and reproj_path.exists():
            continue

        images, Ks, Ts, serials = load_cameras(cap_dir)
        if not serials:
            continue

        # DA3 inference
        image_paths = [str(cap_dir / "images" / f"{s}.png") for s in serials]
        K_arr = np.array([Ks[s] for s in serials])
        T_arr = np.array([Ts[s] for s in serials])

        prediction = model.inference(image=image_paths, intrinsics=K_arr, extrinsics=T_arr)

        depths = {}
        for i, s in enumerate(serials):
            d = prediction.depth[i]
            if hasattr(d, "cpu"):
                d = d.cpu().numpy()
            depths[s] = d

        # Depth colormap grid (per-camera)
        if not cmap_path.exists():
            overlays = {}
            for s in serials:
                H, W = images[s].shape[:2]
                d = cv2.resize(depths[s], (W, H)) if depths[s].shape[:2] != (H, W) else depths[s]
                overlays[s] = depth_colormap(images[s], d)
            save_grid(cmap_path, make_grid(overlays, serials,
                                           title=f"DA3 depth | {obj_name}/{idx}"))

        # Fixed-view reprojection grid: fix one target, overlay depth from each source
        if not reproj_path.exists():
            grid = _fixed_view_depth_grid(images, depths, Ks, Ts, serials,
                                          title=f"DA3 reproj | {obj_name}/{idx}")
            save_grid(reproj_path, grid)

        logging.info(f"  [{ci + 1}/{len(captures)}] {obj_name}/{idx}")


def _pick_target_camera(serials, Ts):
    """Pick the camera closest to the centroid of all cameras (most central view)."""
    positions = {}
    for s in serials:
        R, t = Ts[s][:3, :3], Ts[s][:3, 3]
        positions[s] = -R.T @ t
    centroid = np.mean([positions[s] for s in serials], axis=0)
    return min(serials, key=lambda s: np.linalg.norm(positions[s] - centroid))


def _backproject(depth, K, subsample=4):
    """Backproject depth map to camera-frame 3D points (subsampled)."""
    H, W = depth.shape[:2]
    u, v = np.meshgrid(np.arange(0, W, subsample), np.arange(0, H, subsample))
    d_sub = depth[::subsample, ::subsample]
    valid = d_sub > 0.001
    if not valid.any():
        return np.zeros((0, 3), dtype=np.float32), valid
    z = d_sub[valid]
    uu, vv = u[valid].astype(np.float32), v[valid].astype(np.float32)
    pts = np.stack([(uu - K[0, 2]) * z / K[0, 0],
                    (vv - K[1, 2]) * z / K[1, 1], z], axis=1)
    return pts, valid


def _project_depth_onto_target(pts_world, tgt_img, K_tgt, T_tgt):
    """Project world points onto target camera, return depth colormap overlay."""
    H, W = tgt_img.shape[:2]
    R, t = T_tgt[:3, :3], T_tgt[:3, 3]
    pts_cam = (R @ pts_world.T).T + t
    in_front = pts_cam[:, 2] > 0.01
    pts_v = pts_cam[in_front]
    px = (K_tgt[0, 0] * pts_v[:, 0] / pts_v[:, 2] + K_tgt[0, 2]).astype(int)
    py = (K_tgt[1, 1] * pts_v[:, 1] / pts_v[:, 2] + K_tgt[1, 2]).astype(int)
    in_img = (px >= 0) & (px < W) & (py >= 0) & (py < H)

    depth_map = np.zeros((H, W), dtype=np.float32)
    if in_img.any():
        z = pts_v[in_img, 2]
        order = np.argsort(z)[::-1]
        depth_map[py[in_img][order], px[in_img][order]] = z[order]
    return depth_colormap(tgt_img, depth_map)


def _fixed_view_depth_grid(images, depths, Ks, Ts, serials,
                           title="", ncols=5, scale=0.25, subsample=4):
    """Fix one target view; each cell = target image + depth from one source camera.

    Grid is ncols×nrows (5×5 for 24 cameras + 1 blank). Shows cross-view consistency:
    if DA3 is accurate, all cells should show similar depth patterns on the target.
    """
    H, W = next(iter(images.values())).shape[:2]
    tgt = _pick_target_camera(serials, Ts)
    tgt_img = images[tgt]
    K_tgt, T_tgt = Ks[tgt], Ts[tgt]

    # Backproject each source camera's depth to world
    cells = []
    labels = []
    for src in serials:
        d = depths[src]
        if d.shape[:2] != (H, W):
            d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
        pts_cam, valid = _backproject(d, Ks[src], subsample=subsample)
        if len(pts_cam) == 0:
            cells.append(tgt_img.copy())
            labels.append(f"{src[-4:]}→{tgt[-4:]}: empty")
            continue

        R, t = Ts[src][:3, :3], Ts[src][:3, 3]
        pts_world = (R.T @ (pts_cam - t).T).T
        overlay = _project_depth_onto_target(pts_world, tgt_img, K_tgt, T_tgt)
        cells.append(overlay)
        tag = "self" if src == tgt else f"{src[-4:]}→{tgt[-4:]}"
        labels.append(tag)

    # Build grid
    h, w = int(H * scale), int(W * scale)
    nrows = (len(cells) + ncols - 1) // ncols
    title_h = 40 if title else 0
    grid = np.zeros((title_h + nrows * h, ncols * w, 3), dtype=np.uint8)
    if title:
        cv2.putText(grid, title, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    for i, (cell, label) in enumerate(zip(cells, labels)):
        r, c = divmod(i, ncols)
        resized = cv2.resize(cell, (w, h))
        cv2.putText(resized, label, (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.putText(resized, label, (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y0 = title_h + r * h
        grid[y0:y0 + h, c * w:(c + 1) * w] = resized
    return grid


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step: depth_stereo
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _find_best_stereo_pair(serials, Ks, Ts):
    """Select best stereo pair by baseline + view alignment."""
    positions, fwd_dirs, focals = {}, {}, {}
    for s in serials:
        R, t = Ts[s][:3, :3], Ts[s][:3, 3]
        positions[s] = -R.T @ t
        fwd = R[2, :]
        fwd_dirs[s] = fwd / (np.linalg.norm(fwd) + 1e-9)
        focals[s] = float(Ks[s][0, 0])

    def _search(min_b, max_b, min_cos, min_perp=0.30, max_f_ratio=2.0):
        best, best_score = None, -1.0
        for i, s1 in enumerate(serials):
            for j, s2 in enumerate(serials):
                if i >= j:
                    continue
                b = float(np.linalg.norm(positions[s1] - positions[s2]))
                if not (min_b <= b <= max_b):
                    continue
                cs = float(np.dot(fwd_dirs[s1], fwd_dirs[s2]))
                if cs < min_cos:
                    continue
                bd = (positions[s2] - positions[s1])
                bd /= np.linalg.norm(bd) + 1e-9
                mf = fwd_dirs[s1] + fwd_dirs[s2]
                mf /= np.linalg.norm(mf) + 1e-9
                perp = 1.0 - abs(float(np.dot(bd, mf)))
                if perp < min_perp:
                    continue
                f1, f2 = focals[s1], focals[s2]
                if max(f1, f2) / (min(f1, f2) + 1e-9) > max_f_ratio:
                    continue
                if cs > best_score:
                    best_score = cs
                    best = (s1, s2, b)
        return best

    return (_search(0.03, 0.50, 0.77)
            or _search(0.03, 2.00, 0.50, min_perp=0.20, max_f_ratio=3.0))


def step_depth_stereo(args):
    import torch
    from autodex.perception.depth import _setup_foundation_stereo_path
    _setup_foundation_stereo_path()
    from omegaconf import OmegaConf
    from core.foundation_stereo import FoundationStereo
    from core.utils.utils import InputPadder

    if not FS_CKPT.exists():
        logging.error(f"FoundationStereo checkpoint not found: {FS_CKPT}")
        return

    cfg = OmegaConf.load(str(FS_CKPT.parent / "cfg.yaml"))
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    cfg["valid_iters"] = 32
    cfg["hiera"] = 0
    fs_model = FoundationStereo(cfg)
    ckpt = torch.load(str(FS_CKPT), map_location="cuda")
    fs_model.load_state_dict(ckpt["model"])
    fs_model.cuda().eval()

    captures = discover(args.obj, args.idx)
    logging.info(f"depth_stereo: {len(captures)} captures")

    for ci, (obj_name, idx, cap_dir) in enumerate(captures):
        out_dir = OUTPUT_ROOT / obj_name / idx
        cmap_path = out_dir / "depth_stereo.png"
        reproj_path = out_dir / "depth_stereo_reproj.png"
        if cmap_path.exists() and reproj_path.exists():
            continue

        images, Ks, Ts, serials = load_cameras(cap_dir)
        if not serials:
            continue

        pair = _find_best_stereo_pair(serials, Ks, Ts)
        if pair is None:
            logging.warning(f"  {obj_name}/{idx}: no stereo pair found")
            continue

        left_s, right_s, _ = pair
        K_left = Ks[left_s].astype(np.float64)
        K_right = Ks[right_s].astype(np.float64)
        T_left, T_right = Ts[left_s], Ts[right_s]
        left_rgb = cv2.cvtColor(images[left_s], cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(images[right_s], cv2.COLOR_BGR2RGB)
        H, W = left_rgb.shape[:2]

        # Stereo rectify
        T_rel = T_right @ np.linalg.inv(T_left)
        R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
            K_left, None, K_right, None, (W, H),
            T_rel[:3, :3].astype(np.float64), T_rel[:3, 3].astype(np.float64),
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        f_rect = float(P1[0, 0])
        cx_rect, cy_rect = float(P1[0, 2]), float(P1[1, 2])
        baseline_rect = abs(float(P2[0, 3])) / (f_rect + 1e-9)

        if baseline_rect < 0.01 or f_rect > 20000:
            logging.warning(f"  {obj_name}/{idx}: degenerate rectification")
            continue

        map1_l, map2_l = cv2.initUndistortRectifyMap(K_left, None, R1, P1, (W, H), cv2.CV_32FC1)
        map1_r, map2_r = cv2.initUndistortRectifyMap(K_right, None, R2, P2, (W, H), cv2.CV_32FC1)
        left_rect = cv2.remap(left_rgb, map1_l, map2_l, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_rgb, map1_r, map2_r, cv2.INTER_LINEAR)

        # FoundationStereo inference
        img0 = torch.as_tensor(left_rect.copy()).cuda().float()[None].permute(0, 3, 1, 2)
        img1 = torch.as_tensor(right_rect.copy()).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)
        with torch.no_grad(), torch.cuda.amp.autocast(True):
            disp = fs_model.forward(img0, img1, iters=32, test_mode=True)
        disp = padder.unpad(disp.float()).data.cpu().numpy().reshape(H, W)

        # Disparity → 3D world points
        valid = disp >= 0.5
        disp_v = np.maximum(disp[valid], 0.1)
        depth_v = f_rect * baseline_rect / disp_v

        u_grid, v_grid = np.meshgrid(np.arange(W, dtype=np.float32),
                                      np.arange(H, dtype=np.float32))
        pts_rect = np.stack([
            (u_grid[valid] - cx_rect) * depth_v / f_rect,
            (v_grid[valid] - cy_rect) * depth_v / f_rect,
            depth_v], axis=1)
        pts_left = (R1.T @ pts_rect.T).T
        T_left_inv = np.linalg.inv(T_left)
        pts_world = (T_left_inv[:3, :3] @ pts_left.T).T + T_left_inv[:3, 3]

        # Source pixel colors from rectified left image
        left_rect_bgr = cv2.cvtColor(left_rect, cv2.COLOR_RGB2BGR)
        src_colors = left_rect_bgr[valid]

        # Reproject to all cameras
        depths_map, reproj_overlays = {}, {}
        for s in serials:
            R_c, t_c, K_c = Ts[s][:3, :3], Ts[s][:3, 3], Ks[s]
            pts_cam = (R_c @ pts_world.T).T + t_c
            in_front = pts_cam[:, 2] > 0.01
            pts_v = pts_cam[in_front]
            cols_v = src_colors[in_front]
            px = (K_c[0, 0] * pts_v[:, 0] / pts_v[:, 2] + K_c[0, 2]).astype(int)
            py = (K_c[1, 1] * pts_v[:, 1] / pts_v[:, 2] + K_c[1, 2]).astype(int)
            in_img = (px >= 0) & (px < W) & (py >= 0) & (py < H)

            depth_cam = np.zeros((H, W), dtype=np.float32)
            canvas = images[s].copy()
            if in_img.any():
                z = pts_v[in_img, 2]
                order = np.argsort(z)[::-1]
                depth_cam[py[in_img][order], px[in_img][order]] = z[order]
                canvas[py[in_img][order], px[in_img][order]] = cols_v[in_img][order]
            depths_map[s] = depth_cam
            reproj_overlays[s] = canvas

        # Colormap grid (per-camera depth overlay)
        if not cmap_path.exists():
            cmap_overlays = {s: depth_colormap(images[s], depths_map[s]) for s in serials}
            save_grid(cmap_path, make_grid(
                cmap_overlays, serials,
                title=f"FS depth | pair={left_s[-4:]}+{right_s[-4:]} | {obj_name}/{idx}"))

        # Source-color reprojection grid (per-camera)
        if not reproj_path.exists():
            save_grid(reproj_path, make_grid(
                reproj_overlays, serials,
                title=f"FS reproj | {left_s[-4:]}→all | {obj_name}/{idx}"))

        logging.info(f"  [{ci + 1}/{len(captures)}] {obj_name}/{idx}: pair={left_s}+{right_s}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step: pose_viz
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def step_pose_viz(args):
    """Render mesh overlay from existing NMS and silhouette-refined poses.

    Generates:
      pose_nms.png         - NMS pose on all cameras (5×5 grid)
      pose_silhouette.png  - Silhouette-refined pose on all cameras (5×5 grid)
      pose_initial.png     - Fixed target view, each cell = one camera's individual
                             pose estimate rendered on the target (5×5 grid).
                             Shows per-camera agreement before NMS.
    """
    import torch
    import nvdiffrast.torch as dr

    sys.path.insert(0, str(FP_DIR))
    from step5_silhouette_refine import _load_mesh_tensors, _render_silhouette

    glctx = dr.RasterizeCudaContext()

    captures = discover(args.obj, args.idx)
    logging.info(f"pose_viz: {len(captures)} captures")

    # Group by object to reuse mesh tensors
    from collections import defaultdict
    by_obj = defaultdict(list)
    for obj_name, idx, cap_dir in captures:
        by_obj[obj_name].append((idx, cap_dir))

    for obj_name, idx_list in sorted(by_obj.items()):
        # Find mesh
        mesh_path = MESH_ROOT / obj_name / "raw_mesh" / f"{obj_name}.obj"
        if not mesh_path.exists():
            mesh_path = MESH_ROOT / obj_name / "processed_data" / "mesh" / "simplified.obj"
        if not mesh_path.exists():
            logging.warning(f"  No mesh for {obj_name}")
            continue

        mesh, mesh_tensors = _load_mesh_tensors(str(mesh_path), device="cuda")

        for idx, cap_dir in idx_list:
            out_dir = OUTPUT_ROOT / obj_name / idx
            nms_path = out_dir / "pose_nms.png"
            sil_path = out_dir / "pose_silhouette.png"
            init_path = out_dir / "pose_initial.png"
            init_each_path = out_dir / "pose_initial_each.png"

            if nms_path.exists() and sil_path.exists() and init_path.exists() and init_each_path.exists():
                continue

            pose_dir = cap_dir / "outputs" / f"{obj_name}_pose"
            nms_pose_file = pose_dir / "selected_pose_world.txt"
            sil_pose_file = pose_dir / "optimized_pose_world.txt"

            if not nms_pose_file.exists():
                logging.warning(f"  No NMS pose: {obj_name}/{idx}")
                continue

            images, Ks, Ts, serials = load_cameras(cap_dir)
            if not serials:
                continue

            # NMS pose overlay (all cameras)
            if not nms_path.exists():
                pose_world = np.loadtxt(str(nms_pose_file)).reshape(4, 4)
                overlays = _render_pose_overlays(
                    images, Ks, Ts, serials, pose_world,
                    mesh_tensors, glctx, _render_silhouette, color=(128, 0, 128))
                save_grid(nms_path, make_grid(overlays, serials,
                                              title=f"NMS pose | {obj_name}/{idx}"))

            # Silhouette-refined pose overlay (all cameras)
            if not sil_path.exists() and sil_pose_file.exists():
                pose_world = np.loadtxt(str(sil_pose_file)).reshape(4, 4)
                overlays = _render_pose_overlays(
                    images, Ks, Ts, serials, pose_world,
                    mesh_tensors, glctx, _render_silhouette, color=(0, 128, 0))
                save_grid(sil_path, make_grid(overlays, serials,
                                              title=f"Silhouette pose | {obj_name}/{idx}"))

            # Per-camera initial poses on fixed target view
            if not init_path.exists():
                grid = _fixed_view_pose_grid(
                    images, Ks, Ts, serials, pose_dir,
                    mesh_tensors, glctx, _render_silhouette,
                    title=f"Initial poses → fixed view | {obj_name}/{idx}")
                if grid is not None:
                    save_grid(init_path, grid)

            # Per-camera initial poses on each camera's own view
            if not init_each_path.exists():
                overlays = _each_view_pose_grid(
                    images, Ks, Ts, serials, pose_dir,
                    mesh_tensors, glctx, _render_silhouette)
                if overlays:
                    save_grid(init_each_path, make_grid(
                        overlays, serials,
                        title=f"Initial poses (own view) | {obj_name}/{idx}"))

            logging.info(f"  {obj_name}/{idx}")


def _render_pose_overlays(images, Ks, Ts, serials, pose_world,
                          mesh_tensors, glctx, render_fn, color=(128, 0, 128)):
    import torch
    overlays = {}
    for s in serials:
        H, W = images[s].shape[:2]
        pose_cam = Ts[s] @ pose_world
        pose_cam_t = torch.as_tensor(pose_cam, device="cuda", dtype=torch.float32).reshape(1, 4, 4)
        alpha = render_fn(Ks[s], H, W, pose_cam_t, glctx, mesh_tensors)
        mask = alpha[0, :, :, 0].detach().cpu().numpy() > 0.5
        overlays[s] = overlay_mask(images[s], mask.astype(np.uint8) * 255, color=color, alpha=0.5)
    return overlays


def _fixed_view_pose_grid(images, Ks, Ts, serials, pose_dir,
                          mesh_tensors, glctx, render_fn,
                          title="", ncols=5, scale=0.25):
    """Fix one target view; each cell = mesh rendered using one camera's individual
    world-frame pose estimate, projected onto the fixed target camera.

    Shows per-camera pose agreement before NMS — consistent cameras produce
    overlapping meshes, outliers are visually obvious.
    """
    import torch

    tgt = _pick_target_camera(serials, Ts)
    tgt_img = images[tgt]
    H, W = tgt_img.shape[:2]
    K_tgt, T_tgt = Ks[tgt], Ts[tgt]

    cells = []
    labels = []
    for src in serials:
        pose_file = pose_dir / "ob_in_world" / f"{src}.txt"
        if not pose_file.exists():
            cells.append(tgt_img.copy())
            labels.append(f"{src[-4:]}: no pose")
            continue

        pose_world = np.loadtxt(str(pose_file)).reshape(4, 4)
        pose_cam = T_tgt @ pose_world
        pose_cam_t = torch.as_tensor(pose_cam, device="cuda", dtype=torch.float32).reshape(1, 4, 4)

        alpha = render_fn(K_tgt, H, W, pose_cam_t, glctx, mesh_tensors)
        mask = alpha[0, :, :, 0].detach().cpu().numpy() > 0.5
        overlay = overlay_mask(tgt_img, mask.astype(np.uint8) * 255,
                               color=(128, 0, 128), alpha=0.5)
        cells.append(overlay)
        tag = f"{src[-4:]} (self)" if src == tgt else src[-4:]
        labels.append(tag)

    if not cells:
        return None

    # Build grid
    h, w = int(H * scale), int(W * scale)
    nrows = (len(cells) + ncols - 1) // ncols
    title_h = 40 if title else 0
    grid = np.zeros((title_h + nrows * h, ncols * w, 3), dtype=np.uint8)
    if title:
        cv2.putText(grid, title, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    for i, (cell, label) in enumerate(zip(cells, labels)):
        r, c = divmod(i, ncols)
        resized = cv2.resize(cell, (w, h))
        cv2.putText(resized, label, (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.putText(resized, label, (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y0 = title_h + r * h
        grid[y0:y0 + h, c * w:(c + 1) * w] = resized
    return grid


def _each_view_pose_grid(images, Ks, Ts, serials, pose_dir,
                          mesh_tensors, glctx, render_fn):
    """Each cell = camera's own pose rendered on its own image."""
    import torch

    overlays = {}
    for s in serials:
        pose_file = pose_dir / "ob_in_world" / f"{s}.txt"
        if not pose_file.exists():
            overlays[s] = images[s].copy()
            continue

        H, W = images[s].shape[:2]
        pose_world = np.loadtxt(str(pose_file)).reshape(4, 4)
        pose_cam = Ts[s] @ pose_world
        pose_cam_t = torch.as_tensor(pose_cam, device="cuda", dtype=torch.float32).reshape(1, 4, 4)

        alpha = render_fn(Ks[s], H, W, pose_cam_t, glctx, mesh_tensors)
        mask = alpha[0, :, :, 0].detach().cpu().numpy() > 0.5
        overlays[s] = overlay_mask(images[s], mask.astype(np.uint8) * 255,
                                   color=(128, 0, 128), alpha=0.5)
    return overlays


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step: pose_run  (actually run FoundationPose + NMS, then visualize)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_nms(poses_world, mesh_path, iou_threshold=0.5):
    """Consensus-based NMS on world-frame poses using 3D AABB IoU."""
    import trimesh

    mesh = trimesh.load(str(mesh_path), force="mesh")
    mins, maxs = mesh.bounds
    corners = np.array([
        [mins[0], mins[1], mins[2]], [mins[0], mins[1], maxs[2]],
        [mins[0], maxs[1], mins[2]], [mins[0], maxs[1], maxs[2]],
        [maxs[0], mins[1], mins[2]], [maxs[0], mins[1], maxs[2]],
        [maxs[0], maxs[1], mins[2]], [maxs[0], maxs[1], maxs[2]],
    ], dtype=np.float32)

    homo = np.hstack([corners, np.ones((8, 1))])
    aabbs = []
    for p in poses_world:
        pts = (p @ homo.T).T[:, :3]
        aabbs.append((pts.min(axis=0), pts.max(axis=0)))

    n = len(aabbs)
    iou_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            a_min, a_max = aabbs[i]
            b_min, b_max = aabbs[j]
            inter = np.maximum(np.minimum(a_max, b_max) - np.maximum(a_min, b_min), 0)
            inter_vol = inter[0] * inter[1] * inter[2]
            union = np.prod(a_max - a_min) + np.prod(b_max - b_min) - inter_vol
            v = float(inter_vol / union) if union > 0 else 0.0
            iou_mat[i, j] = iou_mat[j, i] = v

    overlap = np.where(iou_mat >= iou_threshold, iou_mat, 0).sum(axis=1)
    return int(np.argmax(overlap))


def step_pose_run(args):
    """Run FoundationPose on all cameras, NMS, then generate overlays.

    Uses masks & depth from {cap_dir}/outputs/{obj}_pose/masks/ and depth/.
    Generates:
      pose_run_nms.png           - NMS pose on all cameras
      pose_run_initial.png       - fixed target view, per-camera poses
      pose_run_initial_each.png  - each camera's own pose on its own view
    """
    import torch
    import nvdiffrast.torch as dr

    sys.path.insert(0, str(FP_DIR))
    from step5_silhouette_refine import _load_mesh_tensors, _render_silhouette
    from autodex.perception import PoseTracker

    glctx = dr.RasterizeCudaContext()

    captures = discover(args.obj, args.idx)
    logging.info(f"pose_run: {len(captures)} captures")

    from collections import defaultdict
    by_obj = defaultdict(list)
    for obj_name, idx, cap_dir in captures:
        by_obj[obj_name].append((idx, cap_dir))

    for obj_name, idx_list in sorted(by_obj.items()):
        mesh_path = MESH_ROOT / obj_name / "raw_mesh" / f"{obj_name}.obj"
        if not mesh_path.exists():
            mesh_path = MESH_ROOT / obj_name / "processed_data" / "mesh" / "simplified.obj"
        if not mesh_path.exists():
            logging.warning(f"  No mesh for {obj_name}")
            continue

        mesh, mesh_tensors = _load_mesh_tensors(str(mesh_path), device="cuda")
        tracker = PoseTracker(str(mesh_path), device_id=0)

        for idx, cap_dir in idx_list:
            out_dir = OUTPUT_ROOT / obj_name / idx
            nms_path = out_dir / "pose_run_nms.png"
            init_path = out_dir / "pose_run_initial.png"
            init_each_path = out_dir / "pose_run_initial_each.png"

            if not getattr(args, 'force', False) and nms_path.exists() and init_path.exists() and init_each_path.exists():
                continue

            pose_out = cap_dir / "outputs" / f"{obj_name}_pose"
            masks_dir = pose_out / "masks"
            depth_dir = pose_out / "depth"
            if not masks_dir.exists() or not depth_dir.exists():
                logging.warning(f"  No masks/depth: {obj_name}/{idx}")
                continue

            images, Ks, Ts, serials = load_cameras(cap_dir)
            if not serials:
                continue

            # Run FoundationPose per camera
            per_cam_world = {}
            for s in serials:
                mask_file = masks_dir / f"{s}.png"
                depth_file = depth_dir / f"{s}.png"
                if not mask_file.exists() or not depth_file.exists():
                    continue

                mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
                if mask is None or mask.sum() == 0:
                    continue
                if mask.ndim == 3:
                    mask = mask[..., 0]

                depth_raw = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
                if depth_raw is None:
                    continue
                depth = depth_raw.astype(np.float32) / 1000.0
                depth[(depth < 0.001) | (depth > 100.0)] = 0

                img_rgb = cv2.cvtColor(images[s], cv2.COLOR_BGR2RGB)
                K = Ks[s].copy()

                # Downscale to 0.5 for speed
                H, W = img_rgb.shape[:2]
                nH, nW = H // 2, W // 2
                img_ds = cv2.resize(img_rgb, (nW, nH))
                depth_ds = cv2.resize(depth, (nW, nH), interpolation=cv2.INTER_NEAREST)
                mask_ds = cv2.resize(mask, (nW, nH), interpolation=cv2.INTER_NEAREST)
                K_ds = K.copy()
                K_ds[0] *= 0.5
                K_ds[1] *= 0.5

                tracker.reset()
                try:
                    pose_cam = tracker.init(img_ds, depth_ds, mask_ds, K_ds, iteration=5)
                    pose_world = np.linalg.inv(Ts[s]) @ pose_cam
                    per_cam_world[s] = pose_world
                except Exception as e:
                    logging.warning(f"  {s}: FP failed: {e}")

            if not per_cam_world:
                logging.warning(f"  No poses: {obj_name}/{idx}")
                continue

            # NMS
            cam_ids = list(per_cam_world.keys())
            poses_list = [per_cam_world[c] for c in cam_ids]
            best_idx = _run_nms(poses_list, mesh_path)
            best_pose = poses_list[best_idx]
            logging.info(f"  NMS selected {cam_ids[best_idx]} ({len(cam_ids)}/{len(serials)} cameras)")

            # Save pose files for viz reuse
            run_pose_dir = out_dir / "pose_run_data" / "ob_in_world"
            run_pose_dir.mkdir(parents=True, exist_ok=True)
            for s, pw in per_cam_world.items():
                np.savetxt(str(run_pose_dir / f"{s}.txt"), pw.reshape(4, 4))
            np.savetxt(str(out_dir / "pose_run_data" / "selected_pose_world.txt"), best_pose.reshape(4, 4))

            # NMS overlay on all cameras
            overlays = _render_pose_overlays(
                images, Ks, Ts, serials, best_pose,
                mesh_tensors, glctx, _render_silhouette, color=(128, 0, 128))
            save_grid(nms_path, make_grid(overlays, serials,
                                          title=f"Pose run NMS | {obj_name}/{idx}"))

            # Per-camera initial poses on fixed target view
            grid = _fixed_view_pose_grid(
                images, Ks, Ts, serials, out_dir / "pose_run_data",
                mesh_tensors, glctx, _render_silhouette,
                title=f"Pose run initial → fixed view | {obj_name}/{idx}")
            if grid is not None:
                save_grid(init_path, grid)

            # Per-camera initial poses on each camera's own view
            overlays_each = _each_view_pose_grid(
                images, Ks, Ts, serials, out_dir / "pose_run_data",
                mesh_tensors, glctx, _render_silhouette)
            if overlays_each:
                save_grid(init_each_path, make_grid(
                    overlays_each, serials,
                    title=f"Pose run initial (own view) | {obj_name}/{idx}"))

            logging.info(f"  {obj_name}/{idx} done")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEPS = {
    "mask_sam3": step_mask_sam3,
    "mask_yoloe": step_mask_yoloe,
    "depth_da3": step_depth_da3,
    "depth_stereo": step_depth_stereo,
    "pose_viz": step_pose_viz,
    "pose_run": step_pose_run,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", required=True, choices=list(STEPS.keys()))
    parser.add_argument("--obj", default=None, help="Filter by object name")
    parser.add_argument("--idx", default=None, help="Filter by capture index")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--prompt", default=None, help="Extra text prompt for mask steps")
    args = parser.parse_args()

    t0 = time.perf_counter()
    STEPS[args.step](args)
    logging.info(f"Done: {args.step} ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()