import argparse
import copy
import glob
import json
import os
import sys
import time

import cv2
import imageio
import numpy as np
import torch
import trimesh

import nvdiffrast.torch as dr
from Utils import nvdiffrast_render


def _ensure_eloftr_on_path(repo_path: str):
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def _load_eloftr(weights_path: str, model_type: str, precision: str, device: str):
    from src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter

    if model_type == "full":
        cfg = copy.deepcopy(full_default_cfg)
    else:
        cfg = copy.deepcopy(opt_default_cfg)

    if precision == "mp":
        cfg["mp"] = True
    elif precision == "fp16":
        cfg["half"] = True

    matcher = LoFTR(config=cfg)
    state = torch.load(weights_path, map_location="cpu")
    matcher.load_state_dict(state["state_dict"])
    matcher = reparameter(matcher)
    if precision == "fp16":
        matcher = matcher.half()
    matcher = matcher.eval().to(device)
    return matcher


def _load_mesh(mesh_path: str):
    loaded = trimesh.load(mesh_path, process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)
        ]
        if not meshes:
            raise RuntimeError("Scene contains no valid Trimesh objects")
        textured = [
            m for m in meshes if isinstance(m.visual, trimesh.visual.texture.TextureVisuals)
        ]
        if textured:
            textured = sorted(textured, key=lambda m: len(m.faces), reverse=True)
            return textured[0]
        return trimesh.util.concatenate(meshes)
    return loaded


def _load_mesh_tensors(mesh_path: str, device: str):
    mesh = _load_mesh(mesh_path)
    mesh_tensors = _make_mesh_tensors(mesh, device=device)
    return mesh, mesh_tensors


def _resize_to_divisible(img: np.ndarray, div: int = 32):
    h, w = img.shape[:2]
    new_w = max(div, (w // div) * div)
    new_h = max(div, (h // div) * div)
    if new_w == w and new_h == h:
        return img, 1.0, 1.0
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    sx = w / float(new_w)
    sy = h / float(new_h)
    return resized, sx, sy


def _match_pair(matcher, img0_gray, img1_gray, device: str, conf_thr: float, mask0=None):
    img0_resized, sx0, sy0 = _resize_to_divisible(img0_gray)
    img1_resized, sx1, sy1 = _resize_to_divisible(img1_gray)

    img0_t = torch.from_numpy(img0_resized)[None][None].to(device) / 255.0
    img1_t = torch.from_numpy(img1_resized)[None][None].to(device) / 255.0
    batch = {"image0": img0_t, "image1": img1_t}

    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch["mkpts0_f"].cpu().numpy()
        mkpts1 = batch["mkpts1_f"].cpu().numpy()
        mconf = batch["mconf"].cpu().numpy()

    if mkpts0.size == 0:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))

    keep = mconf >= conf_thr
    mkpts0 = mkpts0[keep]
    mkpts1 = mkpts1[keep]
    mconf = mconf[keep]

    mkpts0[:, 0] *= sx0
    mkpts0[:, 1] *= sy0
    mkpts1[:, 0] *= sx1
    mkpts1[:, 1] *= sy1

    if mask0 is not None and mkpts0.size > 0:
        h, w = mask0.shape[:2]
        xs = np.clip(np.round(mkpts0[:, 0]).astype(int), 0, w - 1)
        ys = np.clip(np.round(mkpts0[:, 1]).astype(int), 0, h - 1)
        keep_mask = mask0[ys, xs] > 0
        mkpts0 = mkpts0[keep_mask]
        mkpts1 = mkpts1[keep_mask]
        mconf = mconf[keep_mask]
    return mkpts0, mkpts1, mconf


def _axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(axis_angle)
    if theta < 1e-8:
        return torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    axis = axis_angle / theta
    ax, ay, az = axis
    K = torch.tensor(
        [[0, -az, ay], [az, 0, -ax], [-ay, ax, 0]],
        device=axis_angle.device,
        dtype=axis_angle.dtype,
    )
    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    R = eye + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
    return R


def _build_delta_pose(params: torch.Tensor, rot_scale: float, trans_scale: float) -> torch.Tensor:
    rot = params[:3] * rot_scale
    trans = params[3:] * trans_scale
    R = _axis_angle_to_matrix(rot)
    T = torch.eye(4, device=params.device, dtype=params.dtype)
    T[:3, :3] = R
    T[:3, 3] = trans
    return T


def _sample_depth(depth_map: torch.Tensor, mkpts: torch.Tensor):
    h, w = depth_map.shape[-2:]
    x = mkpts[:, 0]
    y = mkpts[:, 1]
    x_norm = (x / (w - 1)) * 2 - 1
    y_norm = (y / (h - 1)) * 2 - 1
    grid = torch.stack([x_norm, y_norm], dim=1).view(1, -1, 1, 2)
    depth = torch.nn.functional.grid_sample(
        depth_map.view(1, 1, h, w),
        grid,
        align_corners=True,
    ).view(-1)
    return depth


def _make_grid_image(imgs, nrow, padding=5, pad_value=255):
    if not imgs:
        return None
    heights = [img.shape[0] for img in imgs]
    widths = [img.shape[1] for img in imgs]
    h = max(heights)
    w = max(widths)
    ncols = nrow
    nrows = int(np.ceil(len(imgs) / float(ncols)))
    grid_h = nrows * h + (nrows - 1) * padding
    grid_w = ncols * w + (ncols - 1) * padding
    grid = np.full((grid_h, grid_w, 3), pad_value, dtype=np.uint8)
    for idx, img in enumerate(imgs):
        r = idx // ncols
        c = idx % ncols
        y0 = r * (h + padding)
        x0 = c * (w + padding)
        ih, iw = img.shape[:2]
        grid[y0 : y0 + ih, x0 : x0 + iw] = img
    return grid


def _draw_matches_single_image(image, kpts0, kpts1, max_draw=100):
    canvas = image.copy()
    if kpts0.shape[0] == 0:
        return canvas
    n = min(kpts0.shape[0], max_draw)
    for i in range(n):
        x0, y0 = kpts0[i]
        x1, y1 = kpts1[i]
        color = (0, 255, 0)
        pt0 = (int(round(x0)), int(round(y0)))
        pt1 = (int(round(x1)), int(round(y1)))
        cv2.circle(canvas, pt0, 2, color, -1)
        cv2.circle(canvas, pt1, 2, color, -1)
        cv2.line(canvas, pt0, pt1, color, 1, lineType=cv2.LINE_AA)
    return canvas


def _overlay_mesh_on_image(image, render_depth, color=(128, 0, 128), alpha=0.6):
    mask = render_depth > 0
    if mask.ndim != 2:
        mask = mask.squeeze()
    overlay = image.copy()
    overlay[mask] = (
        overlay[mask].astype(np.float32) * (1.0 - alpha)
        + np.array(color, dtype=np.float32) * alpha
    ).astype(np.uint8)
    return overlay


def _overlay_binary_mask(image, mask, overlay_color, alpha=0.5):
    overlay = image.copy()
    mask = mask > 0
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.any():
        overlay[mask] = (
            overlay[mask].astype(np.float32) * (1.0 - alpha)
            + np.array(overlay_color, dtype=np.float32) * alpha
        ).astype(np.uint8)
    return overlay


def _collect_images(images_dir: str):
    image_paths = []
    for ext in ("*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG"):
        image_paths.extend(sorted(glob.glob(os.path.join(images_dir, ext))))
    return image_paths


def _load_camera_params(data_dir: str):
    intrinsics_path = os.path.join(data_dir, "intrinsics.json")
    extrinsics_path = os.path.join(data_dir, "extrinsics.json")
    with open(intrinsics_path, "r") as f:
        intrinsics = json.load(f)
    with open(extrinsics_path, "r") as f:
        extrinsics = json.load(f)
    return intrinsics, extrinsics


def _get_mask(mask_dir: str, image_key: str):
    for ext in (".png", ".jpg", ".jpeg"):
        path = os.path.join(mask_dir, f"{image_key}{ext}")
        if os.path.exists(path):
            mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                return None
            if mask.ndim == 3:
                for c in range(3):
                    if mask[..., c].sum() > 0:
                        mask = mask[..., c]
                        break
            return (mask > 0).astype(np.uint8)
    return None


def _make_mesh_tensors(mesh, device="cuda"):
    mesh_tensors = {}
    if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        img = np.array(mesh.visual.material.image.convert("RGB"))[..., :3]
        mesh_tensors["tex"] = torch.as_tensor(img, device=device, dtype=torch.float32)[
            None
        ] / 255.0
        mesh_tensors["uv_idx"] = torch.as_tensor(mesh.faces, device=device, dtype=torch.int32)
        uv = torch.as_tensor(mesh.visual.uv, device=device, dtype=torch.float32)
        uv[:, 1] = 1 - uv[:, 1]
        mesh_tensors["uv"] = uv
    else:
        if mesh.visual.vertex_colors is None:
            mesh.visual.vertex_colors = np.tile(
                np.array([128, 128, 128]).reshape(1, 3), (len(mesh.vertices), 1)
            )
        mesh_tensors["vertex_color"] = torch.as_tensor(
            mesh.visual.vertex_colors[..., :3], device=device, dtype=torch.float32
        ) / 255.0

    mesh_tensors.update(
        {
            "pos": torch.tensor(mesh.vertices, device=device, dtype=torch.float32),
            "faces": torch.tensor(mesh.faces, device=device, dtype=torch.int32),
            "vnormals": torch.tensor(mesh.vertex_normals, device=device, dtype=torch.float32),
        }
    )
    return mesh_tensors


def _projection_matrix_from_intrinsics(K, height, width, znear=0.001, zfar=100.0):
    x0 = 0.0
    y0 = 0.0
    w = float(width)
    h = float(height)
    nc = float(znear)
    fc = float(zfar)
    depth = fc - nc
    q = -(fc + nc) / depth
    qn = -2 * (fc * nc) / depth
    proj = np.array(
        [
            [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
            [0, 2 * K[1, 1] / h, (2 * K[1, 2] - h + 2 * y0) / h, 0],
            [0, 0, q, qn],
            [0, 0, -1, 0],
        ],
        dtype=np.float32,
    )
    return proj


def _to_homo_torch(pts):
    ones = torch.ones((*pts.shape[:-1], 1), dtype=torch.float32, device=pts.device)
    return torch.cat((pts, ones), dim=-1)


def _transform_pts(pts, tf):
    if len(tf.shape) >= 3 and tf.shape[-3] != pts.shape[-2]:
        tf = tf[..., None, :, :]
    return (tf[..., :-1, :-1] @ pts[..., None] + tf[..., :-1, -1:])[..., 0]


def _nvdiffrast_render(K, H, W, ob_in_cams, glctx, mesh_tensors):
    color, depth, _ = nvdiffrast_render(
        K=K,
        H=H,
        W=W,
        ob_in_cams=ob_in_cams,
        glctx=glctx,
        mesh_tensors=mesh_tensors,
        use_light=True,
        light_dir=np.array([0, 0, 1]),
        w_ambient=0.6,
        w_diffuse=0.6,
    )
    alpha = (depth > 0).unsqueeze(-1).float()
    return color, depth, alpha


def _make_soft_mask(mask: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    mask_f = mask.astype(np.float32)
    if ksize <= 1:
        return mask_f
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(mask_f, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    return np.clip(blurred, 0.0, 1.0)


def _tone_map_render(render: np.ndarray, exposure: float) -> np.ndarray:
    render = np.clip(render, 0.0, 1.0)
    render = np.clip(render * exposure, 0.0, 1.0)
    return render




def _compute_mask_bbox(mask: np.ndarray):
    mask = mask.astype(bool)
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    x0 = xs.min()
    y0 = ys.min()
    x1 = xs.max() + 1
    y1 = ys.max() + 1
    return x0, y0, x1, y1


def _crop_and_resize(image: np.ndarray, bbox, out_size: int):
    x0, y0, x1, y1 = bbox
    h, w = image.shape[:2]
    x0 = max(0, min(w - 1, x0))
    x1 = max(x0 + 1, min(w, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(y0 + 1, min(h, y1))
    crop = image[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    scale = float(out_size) / float(max(ch, cw))
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, (x0, y0), scale


def build_argparser():
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser(description="Step 5: EfficientLoFTR pose refinement")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=f"{code_dir}/demo_data/clock_demo",
        help="Path to demo data directory",
    )
    parser.add_argument(
        "--object-name",
        type=str,
        default="clock",
        help="Object name used to resolve demo defaults",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=f"{code_dir}/outputs/clock_pose",
        help="Output directory from previous steps",
    )
    parser.add_argument(
        "--mesh-file",
        type=str,
        default=f"{code_dir}/demo_data/clock_demo/mesh/clock.obj",
        help="Path to object mesh",
    )
    parser.add_argument(
        "--pose-world",
        type=str,
        default="selected_pose_world.txt",
        help="World-space pose filename in output-dir",
    )
    parser.add_argument(
        "--optimized-pose-name",
        type=str,
        default="optimized_pose_world.txt",
        help="Filename to save optimized world-space pose",
    )
    parser.add_argument(
        "--eloftr-repo",
        type=str,
        default="/media/gunhee/DATA/robothome/EfficientLoFTR",
        help="Path to EfficientLoFTR repo",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="EfficientLoFTR checkpoint path (default: <repo>/weights/eloftr_outdoor.ckpt)",
    )
    parser.add_argument("--model-type", type=str, default="full", choices=["full", "opt"])
    parser.add_argument(
        "--precision", type=str, default="fp32", choices=["fp32", "mp", "fp16"]
    )
    parser.add_argument("--match-conf-thr", type=float, default=0.2)
    parser.add_argument("--max-matches-per-view", type=int, default=500)
    parser.add_argument("--max-draw-matches", type=int, default=100)
    parser.add_argument("--max-viz-views", type=int, default=24)
    parser.add_argument(
        "--debug-mask-render-first",
        action="store_true",
        help="Save initial mask/render overlap debug images per view",
    )
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rot-scale", type=float, default=1.0)
    parser.add_argument("--trans-scale", type=float, default=0.01)
    parser.add_argument("--use-ransac", action="store_true", help="Filter correspondences with PnP RANSAC.")
    parser.add_argument("--ransac-reproj-thr", type=float, default=5.0)
    parser.add_argument("--ransac-min-inliers", type=int, default=30)
    parser.add_argument("--lm-lambda-init", type=float, default=1e-3, help="Initial LM damping.")
    parser.add_argument("--lm-lambda-up", type=float, default=10.0, help="LM damping increase factor.")
    parser.add_argument("--lm-lambda-down", type=float, default=10.0, help="LM damping decrease factor.")
    parser.add_argument(
        "--crop-match",
        action="store_true",
        help="Crop both real/render images by union mask and resize for matching.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=480,
        help="Target size (long side) for cropped matching images.",
    )
    parser.add_argument(
        "--rematch-each-iter",
        action="store_true",
        help="Recompute matches and correspondences each optimization iteration.",
    )
    parser.add_argument(
        "--no-rematch-each-iter",
        action="store_true",
        help="Disable re-matching each optimization iteration.",
    )
    parser.add_argument(
        "--rematch-interval",
        type=int,
        default=10,
        help="Rematch every N iterations (only used when rematch is enabled).",
    )
    parser.add_argument(
        "--min-matches-per-view",
        type=int,
        default=20,
        help="Drop views with fewer correspondences than this.",
    )
    parser.add_argument(
        "--overlay-grid-path",
        type=str,
        default="",
        help="Optional path to save overlay grid image (default: <output-dir>/step5_overlay_grid.png)",
    )
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    default_data_dir = parser.get_default("data_dir")
    default_output_dir = parser.get_default("output_dir")
    default_mesh_file = parser.get_default("mesh_file")
    if args.object_name and args.object_name != "clock":
        if args.data_dir == default_data_dir:
            args.data_dir = os.path.join(code_dir, "demo_data", f"{args.object_name}_demo")
        if args.output_dir == default_output_dir:
            args.output_dir = os.path.join(code_dir, "outputs", f"{args.object_name}_pose")
        if args.mesh_file == default_mesh_file:
            args.mesh_file = os.path.join(
                args.data_dir, "mesh", f"{args.object_name}.obj"
            )
    if not os.path.isfile(args.mesh_file):
        candidate_obj = os.path.join(args.data_dir, "mesh", f"{args.object_name}.obj")
        candidate_ply = os.path.join(args.data_dir, "mesh", f"{args.object_name}.ply")
        if os.path.isfile(candidate_obj):
            args.mesh_file = candidate_obj
        elif os.path.isfile(candidate_ply):
            args.mesh_file = candidate_ply
    if args.no_rematch_each_iter:
        args.rematch_each_iter = False
    else:
        args.rematch_each_iter = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for EfficientLoFTR refinement.")

    _ensure_eloftr_on_path(args.eloftr_repo)
    weights_path = args.weights or os.path.join(args.eloftr_repo, "weights", "eloftr_outdoor.ckpt")
    matcher = _load_eloftr(weights_path, args.model_type, args.precision, device="cuda")

    pose_world_path = os.path.join(args.output_dir, args.pose_world)
    pose_world_init = np.loadtxt(pose_world_path).reshape(4, 4)
    pose_world_init_t = torch.tensor(pose_world_init, device="cuda", dtype=torch.float32)

    mesh, mesh_tensors = _load_mesh_tensors(args.mesh_file, device="cuda")
    glctx = dr.RasterizeCudaContext()
    images_dir = os.path.join(args.data_dir, "images")
    masks_dir = os.path.join(args.data_dir, "masks")
    masked_dir = os.path.join(args.data_dir, "masked_image")
    os.makedirs(masked_dir, exist_ok=True)
    image_paths = _collect_images(images_dir)
    intrinsics_dict, extrinsics_dict = _load_camera_params(args.data_dir)

    def build_observations(pose_world_np, save_viz: bool):
        obs_start = time.perf_counter()
        observations = []
        crop_match_viz = []
        initial_match_viz = []
        for image_path in image_paths:
            frame_start = time.perf_counter()
            image_key = os.path.splitext(os.path.basename(image_path))[0]
            if image_key not in intrinsics_dict or image_key not in extrinsics_dict:
                continue
            color = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if color is None:
                continue
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            mask_full = _get_mask(masks_dir, image_key)
            if mask_full is None:
                continue
            if save_viz:
                masked_color = color.copy()
                masked_color[mask_full == 0] = 0
                cv2.imwrite(
                    os.path.join(masked_dir, f"{image_key}.png"),
                    cv2.cvtColor(masked_color, cv2.COLOR_RGB2BGR),
                )
            K = np.array(intrinsics_dict[image_key]["intrinsics"]).reshape(3, 3)
            ext_3x4 = np.array(extrinsics_dict[image_key])
            extrinsic = np.vstack([ext_3x4, np.array([0, 0, 0, 1])])

            pose_cam = extrinsic @ pose_world_np
            pose_cam_t = torch.tensor(pose_cam, device="cuda", dtype=torch.float32).reshape(1, 4, 4)
            render_color_t, render_depth_t, alpha_t = _nvdiffrast_render(
                K, color.shape[0], color.shape[1], pose_cam_t, glctx, mesh_tensors
            )
            render_color = (render_color_t[0].detach().cpu().numpy() * 255.0).astype(np.uint8)
            render_depth = render_depth_t[0].detach().cpu().numpy()
            render_gray = cv2.cvtColor(render_color, cv2.COLOR_RGB2GRAY)

            real_gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
            mask_for_match = mask_full
            if args.crop_match:
                render_mask = render_depth > 0
                bbox_real = _compute_mask_bbox(mask_full > 0)
                bbox_render = _compute_mask_bbox(render_mask)
                if bbox_real is not None and bbox_render is not None:
                    real_crop, offset0, scale0 = _crop_and_resize(real_gray, bbox_real, args.crop_size)
                    render_crop, offset1, scale1 = _crop_and_resize(render_gray, bbox_render, args.crop_size)
                    mask_crop, _, _ = _crop_and_resize(mask_full.astype(np.uint8), bbox_real, args.crop_size)
                    render_mask_crop, _, _ = _crop_and_resize(
                        render_mask.astype(np.uint8), bbox_render, args.crop_size
                    )
                    real_gray = real_crop
                    render_gray = render_crop * (render_mask_crop > 0)
                    mask_for_match = (mask_crop > 0).astype(np.uint8)
                else:
                    offset0 = (0, 0)
                    offset1 = (0, 0)
                    scale0 = 1.0
                    scale1 = 1.0
            else:
                offset0 = (0, 0)
                offset1 = (0, 0)
                scale0 = 1.0
                scale1 = 1.0
            if save_viz and args.debug_mask_render_first:
                alpha_np = alpha_t[0].detach().cpu().numpy()
                mask_overlay = _overlay_binary_mask(color, mask_full, (0, 255, 0), alpha=0.45)
                render_overlay = _overlay_binary_mask(color, alpha_np > 0.5, (255, 0, 0), alpha=0.45)
                debug_vis = np.concatenate([mask_overlay, render_overlay], axis=1)
                cv2.imwrite(
                    os.path.join(args.output_dir, f"step5_debug_mask_render_{image_key}.png"),
                    cv2.cvtColor(debug_vis, cv2.COLOR_RGB2BGR),
                )

            mkpts0, mkpts1, mconf = _match_pair(
                matcher,
                real_gray,
                render_gray,
                device="cuda",
                conf_thr=args.match_conf_thr,
                mask0=mask_for_match,
            )
            if save_viz and args.crop_match and len(crop_match_viz) < args.max_viz_views:
                real_vis = cv2.cvtColor(real_gray, cv2.COLOR_GRAY2RGB)
                render_vis = cv2.cvtColor(render_gray, cv2.COLOR_GRAY2RGB)
                h0, w0 = real_vis.shape[:2]
                h1, w1 = render_vis.shape[:2]
                if h0 != h1:
                    if h0 < h1:
                        pad = np.zeros((h1 - h0, w0, 3), dtype=real_vis.dtype)
                        real_vis = np.vstack([real_vis, pad])
                    else:
                        pad = np.zeros((h0 - h1, w1, 3), dtype=render_vis.dtype)
                        render_vis = np.vstack([render_vis, pad])
                crop_match_viz.append(np.concatenate([real_vis, render_vis], axis=1))
            if args.crop_match and (
                scale0 != 1.0
                or scale1 != 1.0
                or offset0 != (0, 0)
                or offset1 != (0, 0)
            ):
                mkpts0 = mkpts0 / scale0
                mkpts1 = mkpts1 / scale1
                mkpts0[:, 0] += offset0[0]
                mkpts0[:, 1] += offset0[1]
                mkpts1[:, 0] += offset1[0]
                mkpts1[:, 1] += offset1[1]
            if mkpts0.shape[0] == 0:
                continue
            if mkpts0.shape[0] < args.min_matches_per_view:
                continue
            if mkpts0.shape[0] > args.max_matches_per_view:
                sel = np.random.choice(mkpts0.shape[0], args.max_matches_per_view, replace=False)
                mkpts0 = mkpts0[sel]
                mkpts1 = mkpts1[sel]
                mconf = mconf[sel]

            if save_viz and len(initial_match_viz) < args.max_viz_views:
                overlay = _overlay_mesh_on_image(color, render_depth)
                viz = _draw_matches_single_image(
                    overlay, mkpts0, mkpts1, max_draw=args.max_draw_matches
                )
                initial_match_viz.append(viz)

            mkpts1_t = torch.tensor(mkpts1, device="cuda", dtype=torch.float32)
            depth_map = render_depth_t[0].detach()
            depth = _sample_depth(depth_map, mkpts1_t)
            valid = depth > 1e-6
            if valid.sum() == 0:
                continue
            mkpts0_t = torch.tensor(mkpts0, device="cuda", dtype=torch.float32)[valid]
            mkpts1_t = mkpts1_t[valid]
            depth = depth[valid]

            invK = torch.tensor(np.linalg.inv(K), device="cuda", dtype=torch.float32)
            uv1 = torch.cat([mkpts1_t, torch.ones((mkpts1_t.shape[0], 1), device="cuda")], dim=1)
            rays = (invK @ uv1.t()).t()
            xyz_cam = rays * depth[:, None]

            pose_cam_t = torch.tensor(pose_cam, device="cuda", dtype=torch.float32)
            inv_pose_cam = torch.linalg.inv(pose_cam_t)
            xyz_cam_h = torch.cat([xyz_cam, torch.ones((xyz_cam.shape[0], 1), device="cuda")], dim=1)
            xyz_obj = (inv_pose_cam @ xyz_cam_h.t()).t()[:, :3]

            if args.use_ransac and xyz_obj.shape[0] >= 6:
                xyz_np = xyz_obj.detach().cpu().numpy()
                uv_np = mkpts0_t.detach().cpu().numpy()
                ok, _, _, inliers = cv2.solvePnPRansac(
                    xyz_np,
                    uv_np,
                    K,
                    None,
                    reprojectionError=args.ransac_reproj_thr,
                    iterationsCount=100,
                )
                if not ok or inliers is None or len(inliers) < args.ransac_min_inliers:
                    continue
                inliers = inliers.reshape(-1)
                xyz_obj = xyz_obj[inliers]
                mkpts0_t = mkpts0_t[inliers]
                print(f"{image_key}: ransac_inliers={len(inliers)}")

            observations.append(
                {
                    "image_key": image_key,
                    "xyz_obj": xyz_obj,
                    "uv": mkpts0_t,
                    "K": torch.tensor(K, device="cuda", dtype=torch.float32),
                    "extrinsic": torch.tensor(extrinsic, device="cuda", dtype=torch.float32),
                    "H": color.shape[0],
                    "W": color.shape[1],
                }
            )
            elapsed = time.perf_counter() - frame_start
            print(f"[step5_eloftr] obs {image_key}: {elapsed:.3f}s")
        obs_elapsed = time.perf_counter() - obs_start
        print(f"[step5_eloftr] build_observations: {obs_elapsed:.3f}s")
        return observations, crop_match_viz, initial_match_viz

    observations, crop_match_viz, initial_match_viz = build_observations(
        pose_world_init, save_viz=True
    )
    if crop_match_viz:
        crop_grid_path = os.path.join(args.output_dir, "step5_crop_match_grid.png")
        grid = _make_grid_image(crop_match_viz, nrow=2, padding=10, pad_value=255)
        os.makedirs(os.path.dirname(crop_grid_path), exist_ok=True)
        imageio.imwrite(crop_grid_path, grid)
        print(f"Saved crop match grid to {crop_grid_path}")

    if not observations:
        print("Warning: No valid matches found for refinement. Saving initial pose.")
        optimized_path = os.path.join(args.output_dir, args.optimized_pose_name)
        np.savetxt(optimized_path, pose_world_init.reshape(4, 4))
        print(f"Saved optimized pose to {optimized_path}")

        overlay_images = []
        purple = np.array([128, 0, 128], dtype=np.uint8)
        mesh_overlay = mesh.copy()
        vertex_colors = np.tile(
            np.append(purple, 255).reshape(1, 4), (len(mesh_overlay.vertices), 1)
        )
        mesh_overlay.visual = trimesh.visual.ColorVisuals(vertex_colors=vertex_colors)
        mesh_overlay_tensors = _make_mesh_tensors(mesh_overlay, device="cuda")

        for image_path in image_paths:
            image_key = os.path.splitext(os.path.basename(image_path))[0]
            if image_key not in intrinsics_dict or image_key not in extrinsics_dict:
                continue
            color = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if color is None:
                continue
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            K = np.array(intrinsics_dict[image_key]["intrinsics"]).reshape(3, 3)
            ext_3x4 = np.array(extrinsics_dict[image_key])
            extrinsic = np.vstack([ext_3x4, np.array([0, 0, 0, 1])])
            pose_cam = extrinsic @ pose_world_init
            pose_cam_t = torch.tensor(pose_cam, device="cuda", dtype=torch.float32).reshape(
                1, 4, 4
            )
            render_color_t, render_depth_t, _ = _nvdiffrast_render(
                K, color.shape[0], color.shape[1], pose_cam_t, glctx, mesh_overlay_tensors
            )
            render_depth = render_depth_t[0].detach().cpu().numpy()
            render_mask = render_depth > 0
            overlay = color.copy()
            alpha = 0.6
            overlay[render_mask] = (
                overlay[render_mask].astype(np.float32) * (1.0 - alpha)
                + purple.astype(np.float32) * alpha
            ).astype(np.uint8)
            overlay_images.append(overlay)

        if overlay_images:
            grid_path = args.overlay_grid_path or os.path.join(
                args.output_dir, "step5_overlay_grid.png"
            )
            grid = _make_grid_image(overlay_images, nrow=4, padding=5, pad_value=255)
            os.makedirs(os.path.dirname(grid_path), exist_ok=True)
            imageio.imwrite(grid_path, grid)
            print(f"Saved overlay grid to {grid_path}")
        if initial_match_viz:
            match_grid = _make_grid_image(
                initial_match_viz, nrow=2, padding=10, pad_value=255
            )
            match_path = os.path.join(args.output_dir, "step5_matches_initial_grid.png")
            imageio.imwrite(match_path, match_grid)
            print(f"Saved initial match grid to {match_path}")
        return

    overlay_images_init = []
    for image_path in image_paths:
        image_key = os.path.splitext(os.path.basename(image_path))[0]
        if image_key not in intrinsics_dict or image_key not in extrinsics_dict:
            continue
        color = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if color is None:
            continue
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        K = np.array(intrinsics_dict[image_key]["intrinsics"]).reshape(3, 3)
        ext_3x4 = np.array(extrinsics_dict[image_key])
        extrinsic = np.vstack([ext_3x4, np.array([0, 0, 0, 1])])
        pose_cam = extrinsic @ pose_world_init
        pose_cam_t = torch.tensor(pose_cam, device="cuda", dtype=torch.float32).reshape(
            1, 4, 4
        )
        render_color_t, _, _ = _nvdiffrast_render(
            K, color.shape[0], color.shape[1], pose_cam_t, glctx, mesh_tensors
        )
        render_color = (render_color_t[0].detach().cpu().numpy() * 255.0).astype(np.uint8)
        overlay_images_init.append(render_color)

    if overlay_images_init:
        init_grid_path = os.path.join(args.output_dir, "step5_overlay_grid_init.png")
        grid = _make_grid_image(overlay_images_init, nrow=4, padding=5, pad_value=255)
        os.makedirs(os.path.dirname(init_grid_path), exist_ok=True)
        imageio.imwrite(init_grid_path, grid)
        print(f"Saved initial overlay grid to {init_grid_path}")

    params = torch.nn.Parameter(torch.zeros(6, device="cuda", dtype=torch.float32))

    def compute_residuals(p):
        delta_pose = _build_delta_pose(params, args.rot_scale, args.trans_scale)
        pose_world = delta_pose @ pose_world_init_t
        Rw = pose_world[:3, :3]
        tw = pose_world[:3, 3]
        if not observations:
            return torch.empty((0,), device="cuda")

        total_count = sum([obs["uv"].shape[0] for obs in observations])
        if total_count <= 0:
            return torch.empty((0,), device="cuda")
        residuals = []
        view_weights = []
        for obs in observations:
            xyz_obj = obs["xyz_obj"]
            uv = obs["uv"]
            K = obs["K"]
            ext = obs["extrinsic"]
            Xw = (Rw @ xyz_obj.t()).t() + tw[None, :]
            Xc = (ext[:3, :3] @ Xw.t()).t() + ext[:3, 3][None, :]
            x = Xc[:, 0] / Xc[:, 2]
            y = Xc[:, 1] / Xc[:, 2]
            u = K[0, 0] * x + K[0, 2]
            v = K[1, 1] * y + K[1, 2]
            residual = torch.stack([u - uv[:, 0], v - uv[:, 1]], dim=1).reshape(-1)
            residuals.append(residual)
            view_weights.append(float(uv.shape[0]))
        weights = torch.tensor(view_weights, device="cuda", dtype=torch.float32)
        weights = torch.sqrt(weights / (weights.sum() + 1e-9))
        weighted = []
        for res, w in zip(residuals, weights):
            weighted.append(res * w)
        return torch.cat(weighted, dim=0)

    lm_lambda = args.lm_lambda_init
    for it in range(args.iters):
        iter_start = time.perf_counter()
        if args.rematch_each_iter and (it % max(args.rematch_interval, 1) == 0):
            delta_pose = _build_delta_pose(params, args.rot_scale, args.trans_scale).detach()
            pose_world_current = (delta_pose @ pose_world_init_t).detach().cpu().numpy()
            observations, _, _ = build_observations(pose_world_current, save_viz=False)

        residual = compute_residuals(params)
        if residual.numel() == 0:
            print("No residuals available for LM optimization.")
            break
        loss = (residual * residual).mean()

        J = torch.autograd.functional.jacobian(compute_residuals, params, create_graph=False)
        if J.dim() == 1:
            J = J.unsqueeze(0)
        JtJ = J.T @ J
        Jtr = J.T @ residual
        delta = torch.linalg.solve(JtJ + lm_lambda * torch.eye(6, device="cuda"), -Jtr)

        with torch.no_grad():
            new_params = params + delta
        new_residual = compute_residuals(new_params)
        if new_residual.numel() == 0:
            lm_lambda *= args.lm_lambda_up
        else:
            new_loss = (new_residual * new_residual).mean()
            if new_loss < loss:
                with torch.no_grad():
                    params.copy_(new_params)
                lm_lambda = max(lm_lambda / args.lm_lambda_down, 1e-9)
                loss = new_loss
            else:
                lm_lambda *= args.lm_lambda_up

        if it == 0 or (it + 1) % 10 == 0 or it + 1 == args.iters:
            print(
                f"Iter {it + 1}/{args.iters} - loss: {loss.item():.6f} (lm_lambda: {lm_lambda:.3e})"
            )
        iter_elapsed = time.perf_counter() - iter_start
        print(f"[step5_eloftr] iter {it + 1}: {iter_elapsed:.3f}s")

    delta_pose = _build_delta_pose(params, args.rot_scale, args.trans_scale).detach()
    pose_world_opt = (delta_pose @ pose_world_init_t).detach().cpu().numpy()
    optimized_path = os.path.join(args.output_dir, args.optimized_pose_name)
    np.savetxt(optimized_path, pose_world_opt.reshape(4, 4))
    print(f"Saved optimized pose to {optimized_path}")

    overlay_images = []
    purple = np.array([128, 0, 128], dtype=np.uint8)
    mesh_overlay = mesh.copy()
    vertex_colors = np.tile(
        np.append(purple, 255).reshape(1, 4), (len(mesh_overlay.vertices), 1)
    )
    mesh_overlay.visual = trimesh.visual.ColorVisuals(vertex_colors=vertex_colors)
    mesh_overlay_tensors = _make_mesh_tensors(mesh_overlay, device="cuda")

    final_match_viz = []
    for image_path in image_paths:
        image_key = os.path.splitext(os.path.basename(image_path))[0]
        if image_key not in intrinsics_dict or image_key not in extrinsics_dict:
            continue
        color = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if color is None:
            continue
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        K = np.array(intrinsics_dict[image_key]["intrinsics"]).reshape(3, 3)
        ext_3x4 = np.array(extrinsics_dict[image_key])
        extrinsic = np.vstack([ext_3x4, np.array([0, 0, 0, 1])])
        pose_cam = extrinsic @ pose_world_opt
        pose_cam_t = torch.tensor(pose_cam, device="cuda", dtype=torch.float32).reshape(
            1, 4, 4
        )
        render_color_overlay_t, render_depth_t, _ = _nvdiffrast_render(
            K, color.shape[0], color.shape[1], pose_cam_t, glctx, mesh_overlay_tensors
        )
        render_depth = render_depth_t[0].detach().cpu().numpy()
        render_mask = render_depth > 0
        overlay = color.copy()
        alpha = 0.6
        overlay[render_mask] = (
            overlay[render_mask].astype(np.float32) * (1.0 - alpha)
            + purple.astype(np.float32) * alpha
        ).astype(np.uint8)
        overlay_images.append(overlay)

        if len(final_match_viz) < args.max_viz_views:
            render_color_base_t, _, _ = _nvdiffrast_render(
                K, color.shape[0], color.shape[1], pose_cam_t, glctx, mesh_tensors
            )
            render_color_base = (render_color_base_t[0].detach().cpu().numpy() * 255.0).astype(
                np.uint8
            )
            render_gray = cv2.cvtColor(render_color_base, cv2.COLOR_RGB2GRAY)
            mask = _get_mask(masks_dir, image_key)
            real_gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
            if args.crop_match and mask is not None:
                render_mask = render_depth_t[0].detach().cpu().numpy() > 0
                bbox_real = _compute_mask_bbox(mask > 0)
                bbox_render = _compute_mask_bbox(render_mask)
                if bbox_real is not None and bbox_render is not None:
                    real_crop, offset0, scale0 = _crop_and_resize(real_gray, bbox_real, args.crop_size)
                    render_crop, offset1, scale1 = _crop_and_resize(render_gray, bbox_render, args.crop_size)
                    mask_crop, _, _ = _crop_and_resize(mask.astype(np.uint8), bbox_real, args.crop_size)
                    render_mask_crop, _, _ = _crop_and_resize(
                        render_mask.astype(np.uint8), bbox_render, args.crop_size
                    )
                    real_gray_use = real_crop
                    render_gray_use = render_crop * (render_mask_crop > 0)
                    mask_use = (mask_crop > 0).astype(np.uint8)
                else:
                    real_gray_use = real_gray
                    render_gray_use = render_gray
                    mask_use = mask
                    offset0 = (0, 0)
                    offset1 = (0, 0)
                    scale0 = 1.0
                    scale1 = 1.0
            else:
                real_gray_use = real_gray
                render_gray_use = render_gray
                mask_use = mask
                offset0 = (0, 0)
                offset1 = (0, 0)
                scale0 = 1.0
                scale1 = 1.0
            mkpts0, mkpts1, _ = _match_pair(
                matcher,
                real_gray_use,
                render_gray_use,
                device="cuda",
                conf_thr=args.match_conf_thr,
                mask0=mask_use,
            )
            if args.crop_match and (
                scale0 != 1.0
                or scale1 != 1.0
                or offset0 != (0, 0)
                or offset1 != (0, 0)
            ):
                mkpts0 = mkpts0 / scale0
                mkpts1 = mkpts1 / scale1
                mkpts0[:, 0] += offset0[0]
                mkpts0[:, 1] += offset0[1]
                mkpts1[:, 0] += offset1[0]
                mkpts1[:, 1] += offset1[1]
            overlay = _overlay_mesh_on_image(color, render_depth)
            viz = _draw_matches_single_image(
                overlay, mkpts0, mkpts1, max_draw=args.max_draw_matches
            )
            final_match_viz.append(viz)

    if overlay_images:
        grid_path = args.overlay_grid_path or os.path.join(
            args.output_dir, "step5_overlay_grid.png"
        )
        grid = _make_grid_image(overlay_images, nrow=4, padding=5, pad_value=255)
        os.makedirs(os.path.dirname(grid_path), exist_ok=True)
        imageio.imwrite(grid_path, grid)
        print(f"Saved overlay grid to {grid_path}")

    if initial_match_viz:
        match_grid = _make_grid_image(
            initial_match_viz, nrow=2, padding=10, pad_value=255
        )
        match_path = os.path.join(args.output_dir, "step5_matches_initial_grid.png")
        imageio.imwrite(match_path, match_grid)
        print(f"Saved initial match grid to {match_path}")

    if final_match_viz:
        match_grid = _make_grid_image(
            final_match_viz, nrow=2, padding=10, pad_value=255
        )
        match_path = os.path.join(args.output_dir, "step5_matches_final_grid.png")
        imageio.imwrite(match_path, match_grid)
        print(f"Saved final match grid to {match_path}")


if __name__ == "__main__":
    main()
