import argparse
import glob
import json
import os
import time

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import trimesh

import nvdiffrast.torch as dr

glcam_in_cvcam = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=float
)


def projection_matrix_from_intrinsics(K, height, width, znear, zfar, window_coords="y_down"):
    x0 = 0
    y0 = 0
    w = width
    h = height
    nc = znear
    fc = zfar
    depth = float(fc - nc)
    q = -(fc + nc) / depth
    qn = -2 * (fc * nc) / depth

    if window_coords == "y_up":
        proj = np.array(
            [
                [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
                [0, -2 * K[1, 1] / h, (-2 * K[1, 2] + h + 2 * y0) / h, 0],
                [0, 0, q, qn],
                [0, 0, -1, 0],
            ]
        )
    elif window_coords == "y_down":
        proj = np.array(
            [
                [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
                [0, 2 * K[1, 1] / h, (2 * K[1, 2] - h + 2 * y0) / h, 0],
                [0, 0, q, qn],
                [0, 0, -1, 0],
            ]
        )
    else:
        raise NotImplementedError
    return proj


def to_homo_torch(pts):
    ones = torch.ones((*pts.shape[:-1], 1), dtype=pts.dtype, device=pts.device)
    return torch.cat((pts, ones), dim=-1)


def make_mesh_tensors(mesh, device="cuda", max_tex_size=None):
    mesh_tensors = {}
    if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        img = np.array(mesh.visual.material.image.convert("RGB"))[..., :3]
        if max_tex_size is not None:
            max_size = max(img.shape[0], img.shape[1])
            if max_size > max_tex_size:
                scale = 1 / max_size * max_tex_size
                img = cv2.resize(img, fx=scale, fy=scale, dsize=None)
        mesh_tensors["tex"] = (
            torch.as_tensor(img, device=device, dtype=torch.float)[None] / 255.0
        )
        mesh_tensors["uv_idx"] = torch.as_tensor(mesh.faces, device=device, dtype=torch.int)
        uv = torch.as_tensor(mesh.visual.uv, device=device, dtype=torch.float)
        uv[:, 1] = 1 - uv[:, 1]
        mesh_tensors["uv"] = uv
    else:
        if mesh.visual.vertex_colors is None:
            mesh.visual.vertex_colors = np.tile(
                np.array([128, 128, 128]).reshape(1, 3), (len(mesh.vertices), 1)
            )
        mesh_tensors["vertex_color"] = (
            torch.as_tensor(mesh.visual.vertex_colors[..., :3], device=device, dtype=torch.float)
            / 255.0
        )

    mesh_tensors.update(
        {
            "pos": torch.tensor(mesh.vertices, device=device, dtype=torch.float),
            "faces": torch.tensor(mesh.faces, device=device, dtype=torch.int),
            "vnormals": torch.tensor(mesh.vertex_normals, device=device, dtype=torch.float),
        }
    )
    return mesh_tensors


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
    mesh_tensors = make_mesh_tensors(mesh, device=device)
    return mesh, mesh_tensors


def _render_silhouette(K, H, W, ob_in_cams, glctx, mesh_tensors):
    pos = mesh_tensors["pos"]
    faces = mesh_tensors["faces"]
    glcam = torch.tensor(glcam_in_cvcam, device=ob_in_cams.device, dtype=ob_in_cams.dtype)
    ob_in_glcams = glcam[None] @ ob_in_cams
    proj = projection_matrix_from_intrinsics(K, height=H, width=W, znear=0.001, zfar=100)
    proj = torch.as_tensor(proj.reshape(-1, 4, 4), device=ob_in_cams.device, dtype=ob_in_cams.dtype)
    pos_homo = to_homo_torch(pos)
    pos_clip = (proj @ ob_in_glcams)[:, None] @ pos_homo[None, ..., None]
    pos_clip = pos_clip[..., 0]
    rast_out, _ = dr.rasterize(glctx, pos_clip, faces, resolution=np.asarray([H, W]))
    alpha = torch.clamp(rast_out[..., -1:], 0, 1)
    alpha = dr.antialias(alpha, rast_out, pos_clip, faces)
    alpha = torch.flip(alpha, dims=[1])
    return alpha


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def _build_pose_from_r6d_t(r6d: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    R = rotation_6d_to_matrix(r6d)
    T = torch.eye(4, device=r6d.device, dtype=r6d.dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def silhouette_iou_loss(pred_silhouette, target_silhouette):
    intersection = (pred_silhouette * target_silhouette).sum()
    union = (pred_silhouette + target_silhouette).clamp(0, 1).sum()
    return 1 - (intersection / (union + 1e-9))


def _collect_images(images_dir: str):
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        paths.extend(glob.glob(os.path.join(images_dir, ext)))
    return sorted(paths)


def _load_camera_params(data_dir: str):
    intrinsics_path = os.path.join(data_dir, "intrinsics.json")
    extrinsics_path = os.path.join(data_dir, "extrinsics.json")
    with open(intrinsics_path, "r", encoding="utf-8") as f:
        intrinsics = json.load(f)
    with open(extrinsics_path, "r", encoding="utf-8") as f:
        extrinsics = json.load(f)
    return intrinsics, extrinsics


def _get_mask(mask_dir: str, image_key: str):
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        path = os.path.join(mask_dir, f"{image_key}{ext}")
        if os.path.exists(path):
            mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue
            if mask.ndim == 3:
                for c in range(mask.shape[2]):
                    if mask[..., c].sum() > 0:
                        mask = mask[..., c]
                        break
            return (mask > 0).astype(np.uint8)
    return None


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


def _make_soft_mask(mask: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    mask_f = mask.astype(np.float32)
    if ksize <= 1:
        return mask_f
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(mask_f, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    return np.clip(blurred, 0.0, 1.0)


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


def build_argparser():
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser(description="Step 5: silhouette-only pose refinement")
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
        default="optimized_pose_world_sil.txt",
        help="Filename to save optimized world-space pose",
    )
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use-iou", action="store_true", help="Add silhouette IoU loss term")
    parser.add_argument("--iou-weight", type=float, default=1.0)
    parser.add_argument(
        "--debug-grad",
        action="store_true",
        help="Print gradient norms for mse/iou terms",
    )
    parser.add_argument("--mask-blur-ksize", type=int, default=9)
    parser.add_argument("--mask-blur-sigma", type=float, default=3.0)
    parser.add_argument(
        "--debug-grid-dir",
        type=str,
        default="",
        help="Directory to save per-iter overlay grids (default: <output-dir>/step5_sil_debug)",
    )
    parser.add_argument(
        "--save-iter-grids",
        action="store_true",
        help="Save per-iteration overlay grid images",
    )
    parser.add_argument("--grid-nrow", type=int, default=4)
    parser.add_argument("--grid-pad", type=int, default=5)
    parser.add_argument("--overlay-alpha", type=float, default=0.5)
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
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for silhouette refinement.")

    pose_world_path = os.path.join(args.output_dir, args.pose_world)
    pose_world_init = np.loadtxt(pose_world_path).reshape(4, 4)
    pose_world_init_t = torch.tensor(pose_world_init, device="cuda", dtype=torch.float32)

    mesh, mesh_tensors = _load_mesh_tensors(args.mesh_file, device="cuda")
    glctx = dr.RasterizeCudaContext()
    images_dir = os.path.join(args.data_dir, "images")
    masks_dir = os.path.join(args.data_dir, "masks")
    image_paths = _collect_images(images_dir)
    intrinsics_dict, extrinsics_dict = _load_camera_params(args.data_dir)

    views = []
    for image_path in image_paths:
        image_key = os.path.splitext(os.path.basename(image_path))[0]
        if image_key not in intrinsics_dict or image_key not in extrinsics_dict:
            continue
        color = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if color is None:
            continue
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        mask = _get_mask(masks_dir, image_key)
        if mask is None:
            continue
        mask_soft = _make_soft_mask(mask, args.mask_blur_ksize, args.mask_blur_sigma)
        K = np.array(intrinsics_dict[image_key]["intrinsics"]).reshape(3, 3)
        ext_3x4 = np.array(extrinsics_dict[image_key])
        extrinsic = np.vstack([ext_3x4, np.array([0, 0, 0, 1])])
        mask_t = torch.tensor(mask_soft, device="cuda", dtype=torch.float32)
        views.append(
            {
                "image_key": image_key,
                "color": color,
                "mask": mask,
                "mask_t": mask_t,
                "K": K,
                "extrinsic": torch.tensor(extrinsic, device="cuda", dtype=torch.float32),
            }
        )

    if not views:
        raise RuntimeError("No valid views found for silhouette refinement.")

    debug_dir = args.debug_grid_dir or os.path.join(args.output_dir, "step5_sil_debug")
    if args.save_iter_grids:
        os.makedirs(debug_dir, exist_ok=True)

    r6d_init = matrix_to_rotation_6d(pose_world_init_t[:3, :3].unsqueeze(0))[0]
    t_init = pose_world_init_t[:3, 3]
    optim_r6d = torch.nn.Parameter(r6d_init.clone())
    optim_t = torch.nn.Parameter(t_init.clone())
    optimizer = torch.optim.Adam([optim_r6d, optim_t], lr=args.lr)

    for it in range(args.iters):
        iter_start = time.perf_counter()
        optimizer.zero_grad()
        pose_world = _build_pose_from_r6d_t(optim_r6d, optim_t)

        total_loss = 0.0
        mse_total = 0.0
        iou_total = 0.0
        overlay_images = [] if args.save_iter_grids else None
        for view in views:
            pose_cam_t = (view["extrinsic"] @ pose_world).reshape(1, 4, 4)
            alpha_t = _render_silhouette(
                K=view["K"],
                H=view["color"].shape[0],
                W=view["color"].shape[1],
                ob_in_cams=pose_cam_t,
                glctx=glctx,
                mesh_tensors=mesh_tensors,
            )
            render_mask = alpha_t[0, :, :, 0]
            sil_mse = F.mse_loss(render_mask, view["mask_t"], reduction="mean")
            mse_total = mse_total + sil_mse
            total_loss = total_loss + sil_mse
            if args.use_iou:
                sil_iou = silhouette_iou_loss(render_mask, view["mask_t"])
                iou_total = iou_total + sil_iou
                total_loss = total_loss + args.iou_weight * sil_iou

            if args.save_iter_grids:
                render_depth = alpha_t[0, :, :, 0].detach().cpu().numpy()
                overlay = _overlay_binary_mask(
                    view["color"],
                    render_depth > 0.5,
                    (128, 0, 128),
                    alpha=args.overlay_alpha,
                )
                overlay_images.append(overlay)

        loss = total_loss / float(len(views))
        if args.debug_grad and args.use_iou:
            mse_loss = mse_total / float(len(views))
            iou_loss = (iou_total / float(len(views))) * args.iou_weight
            iou_grads = torch.autograd.grad(
                iou_loss, [optim_r6d, optim_t], retain_graph=True, allow_unused=True
            )
            mse_grads = torch.autograd.grad(
                mse_loss, [optim_r6d, optim_t], retain_graph=True, allow_unused=True
            )
            iou_norm = sum([g.norm().item() for g in iou_grads if g is not None])
            mse_norm = sum([g.norm().item() for g in mse_grads if g is not None])
            print(f"Grad norms - mse: {mse_norm:.6e}, iou: {iou_norm:.6e}")
        loss.backward()
        optimizer.step()

        if it == 0 or (it + 1) % 10 == 0 or it + 1 == args.iters:
            if args.use_iou:
                print(f"Iter {it + 1}/{args.iters} - sil_loss: {loss.item():.6f} (mse+iou)")
            else:
                print(f"Iter {it + 1}/{args.iters} - sil_loss: {loss.item():.6f}")
        iter_elapsed = time.perf_counter() - iter_start
        print(f"[step5_sil] iter {it + 1}: {iter_elapsed:.3f}s")

        if args.save_iter_grids and overlay_images:
            grid = _make_grid_image(
                overlay_images, nrow=args.grid_nrow, padding=args.grid_pad, pad_value=255
            )
            if grid is not None:
                grid_path = os.path.join(debug_dir, f"iter_{it + 1:04d}.png")
                imageio.imwrite(grid_path, grid)

    pose_world_opt = _build_pose_from_r6d_t(optim_r6d, optim_t).detach().cpu().numpy()
    optimized_path = os.path.join(args.output_dir, args.optimized_pose_name)
    np.savetxt(optimized_path, pose_world_opt.reshape(4, 4))
    print(f"Saved optimized pose to {optimized_path}")

    overlay_images = []
    pose_world = _build_pose_from_r6d_t(optim_r6d, optim_t).detach()
    for view in views:
        pose_cam_t = (view["extrinsic"] @ pose_world).reshape(1, 4, 4)
        alpha_t = _render_silhouette(
            K=view["K"],
            H=view["color"].shape[0],
            W=view["color"].shape[1],
            ob_in_cams=pose_cam_t,
            glctx=glctx,
            mesh_tensors=mesh_tensors,
        )
        render_mask = alpha_t[0, :, :, 0].detach().cpu().numpy()
        overlay = _overlay_binary_mask(
            view["color"],
            render_mask > 0.5,
            (128, 0, 128),
            alpha=args.overlay_alpha,
        )
        overlay_images.append(overlay)
    grid = _make_grid_image(
        overlay_images, nrow=args.grid_nrow, padding=args.grid_pad, pad_value=255
    )
    if grid is not None:
        final_grid_path = os.path.join(args.output_dir, "step5_sil_overlay_grid_final.png")
        imageio.imwrite(final_grid_path, grid)
        print(f"Saved final overlay grid to {final_grid_path}")


if __name__ == "__main__":
    main()
