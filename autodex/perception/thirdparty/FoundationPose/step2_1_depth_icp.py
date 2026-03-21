import argparse
import glob
import os
import time

import cv2
import imageio
import numpy as np
import open3d as o3d
import torch
import trimesh

import nvdiffrast.torch as dr

from datareader import ParadexReader
from Utils import make_grid_image, make_mesh_tensors, nvdiffrast_render


def _load_mesh(mesh_path: str):
    loaded = trimesh.load(mesh_path)
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)
        ]
        if not meshes:
            raise RuntimeError("Scene contains no valid Trimesh objects")
        return trimesh.util.concatenate(meshes)
    return loaded


def _collect_depth_paths(depth_dir: str):
    depth_paths = []
    for ext in ("*.png", "*.tiff", "*.tif", "*.exr"):
        depth_paths.extend(sorted(glob.glob(os.path.join(depth_dir, ext))))
    return depth_paths


def _backproject_depth(depth_m: np.ndarray, K: np.ndarray, mask: np.ndarray, max_points: int):
    ys, xs = np.where(mask & (depth_m > 1e-6))
    if ys.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if max_points > 0 and ys.size > max_points:
        sel = np.random.choice(ys.size, max_points, replace=False)
        ys = ys[sel]
        xs = xs[sel]
    z = depth_m[ys, xs]
    x = (xs - K[0, 2]) * z / K[0, 0]
    y = (ys - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    return pts


def _build_point_cloud(points: np.ndarray, voxel_size: float):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    if len(pcd.points) >= 3:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=max(voxel_size * 2.0, 0.01), max_nn=30
            )
        )
        pcd.normalize_normals()
    return pcd


def _run_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    max_corr: float,
    max_iter: int,
    init: np.ndarray,
):
    if len(source.points) == 0 or len(target.points) == 0:
        return None
    if not target.has_normals():
        return None
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_corr,
        init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria,
    )
    return result


def build_argparser():
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser(description="Step 2.1: Depth ICP pose refinement")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=f"{code_dir}/demo_data/baby_beaker_demo",
        help="Path to demo data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=f"{code_dir}/outputs/baby_beaker_pose",
        help="Output directory for ICP outputs",
    )
    parser.add_argument(
        "--mesh-file",
        type=str,
        default=f"{code_dir}/demo_data/baby_beaker_demo/mesh/baby_beaker.obj",
        help="Path to object mesh",
    )
    parser.add_argument(
        "--save-name",
        type=str,
        default="icp_pose_world.txt",
        help="Filename to save refined world-space pose",
    )
    parser.add_argument("--voxel-size", type=float, default=0.002, help="Voxel size for ICP.")
    parser.add_argument("--max-corr", type=float, default=0.01, help="Max correspondence distance.")
    parser.add_argument("--icp-iter", type=int, default=100, help="ICP iterations per view.")
    parser.add_argument("--mesh-samples", type=int, default=50000, help="Mesh samples for ICP.")
    parser.add_argument(
        "--max-depth-points",
        type=int,
        default=80000,
        help="Maximum number of depth points to sample from the mask (0 = all).",
    )
    parser.add_argument(
        "--mask-erode",
        type=int,
        default=2,
        help="Erode mask by this many pixels before back-projection (0 = no erosion).",
    )
    parser.add_argument(
        "--overlay-grid-path",
        type=str,
        default="",
        help="Optional path to save overlay grid image (default: <output-dir>/step2_1_overlay_grid.png)",
    )
    return parser


def main():
    args = build_argparser().parse_args()

    mesh = _load_mesh(args.mesh_file)
    mesh_samples = trimesh.sample.sample_surface(mesh, args.mesh_samples)[0].astype(np.float32)

    reader = ParadexReader(video_dir=args.data_dir, shorter_side=None, zfar=np.inf)
    depth_dir = os.path.join(args.data_dir, "depth")
    depth_paths = _collect_depth_paths(depth_dir)
    if not depth_paths:
        raise FileNotFoundError(f"No depth images found in {depth_dir}")

    merged_world_points = []
    collect_start = time.perf_counter()
    for i in range(len(reader.color_files)):
        frame_start = time.perf_counter()
        frame_id = reader.id_strs[i]
        depth_path = os.path.join(depth_dir, f"{frame_id}.png")
        if not os.path.exists(depth_path):
            continue

        extrinsic = reader.get_extrinsic(i)
        if extrinsic is None:
            continue
        K = reader.get_K(i)
        mask = reader.get_mask(i)
        if mask is None:
            continue

        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            continue
        depth_m = depth_raw.astype(np.float32) / 1000.0
        mask_bool = mask > 0
        if args.mask_erode > 0:
            kernel = np.ones((args.mask_erode, args.mask_erode), np.uint8)
            mask_bool = cv2.erode(mask_bool.astype(np.uint8), kernel, iterations=1) > 0
        mask_bool = mask_bool & (depth_m > 1e-6)
        pts_cam = _backproject_depth(depth_m, K, mask_bool, args.max_depth_points)
        if pts_cam.shape[0] < 100:
            continue
        pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1), dtype=np.float32)], axis=1)
        pts_world = (np.linalg.inv(extrinsic) @ pts_cam_h.T).T[:, :3]
        merged_world_points.append(pts_world)
        elapsed = time.perf_counter() - frame_start
        print(f"[step2_1] {frame_id}: {elapsed:.3f}s")
    collect_elapsed = time.perf_counter() - collect_start
    print(f"[step2_1] collected depth points: {collect_elapsed:.3f}s")

    if not merged_world_points:
        raise RuntimeError("No valid depth points found to build merged point cloud.")

    merged_world_points = np.concatenate(merged_world_points, axis=0)
    target_pcd = _build_point_cloud(merged_world_points, args.voxel_size)
    source_pcd = _build_point_cloud(mesh_samples, args.voxel_size)
    src_center = np.asarray(source_pcd.get_center())
    tgt_center = np.asarray(target_pcd.get_center())
    init = np.eye(4, dtype=np.float64)
    init[:3, 3] = tgt_center - src_center
    icp_start = time.perf_counter()
    icp_result = _run_icp(
        source=source_pcd,
        target=target_pcd,
        max_corr=args.max_corr,
        max_iter=args.icp_iter,
        init=init,
    )
    icp_elapsed = time.perf_counter() - icp_start
    print(f"[step2_1] ICP: {icp_elapsed:.3f}s")
    if icp_result is None:
        raise RuntimeError("ICP failed to produce a valid result.")

    best_pose_world = icp_result.transformation.astype(np.float32)
    save_path = os.path.join(args.output_dir, args.save_name)
    np.savetxt(save_path, best_pose_world.reshape(4, 4))
    print(f"Saved ICP-refined pose -> {save_path}")
    results_path = os.path.join(args.output_dir, "step2_1_icp_metrics.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(f"fitness={icp_result.fitness:.6f} rmse={icp_result.inlier_rmse:.6f}\n")
    print(f"Saved ICP metrics -> {results_path}")

    pcd_path = os.path.join(args.output_dir, "step2_1_depth_world.ply")
    o3d.io.write_point_cloud(pcd_path, target_pcd)
    print(f"Saved merged depth point cloud -> {pcd_path}")

    glctx = dr.RasterizeCudaContext()
    purple = np.array([128, 0, 128], dtype=np.uint8)
    mesh_for_overlay = mesh.copy()
    vertex_colors = np.tile(
        np.append(purple, 255).reshape(1, 4), (len(mesh_for_overlay.vertices), 1)
    )
    mesh_for_overlay.visual = trimesh.visual.ColorVisuals(vertex_colors=vertex_colors)
    mesh_tensors_overlay = make_mesh_tensors(mesh_for_overlay, device="cuda")

    overlay_images = []
    overlay_start = time.perf_counter()
    for i in range(len(reader.color_files)):
        frame_start = time.perf_counter()
        extrinsic = reader.get_extrinsic(i)
        if extrinsic is None:
            continue
        color = reader.get_color(i)
        K = reader.get_K(i)
        pose_cam = extrinsic @ best_pose_world
        pose_tensor = torch.as_tensor(pose_cam, device="cuda", dtype=torch.float32).reshape(
            1, 4, 4
        )
        render_color, _, _ = nvdiffrast_render(
            K=K,
            H=color.shape[0],
            W=color.shape[1],
            ob_in_cams=pose_tensor,
            glctx=glctx,
            mesh_tensors=mesh_tensors_overlay,
            use_light=False,
        )
        render_color = render_color[0].detach().cpu().numpy()
        render_mask = render_color.sum(axis=2) > 0
        overlay = color.copy()
        alpha = 0.6
        overlay[render_mask] = (
            overlay[render_mask].astype(np.float32) * (1.0 - alpha)
            + purple.astype(np.float32) * alpha
        ).astype(np.uint8)
        overlay_images.append(overlay)
        elapsed = time.perf_counter() - frame_start
        print(f"[step2_1] overlay {reader.id_strs[i]}: {elapsed:.3f}s")

    if overlay_images:
        grid_path = args.overlay_grid_path or os.path.join(
            args.output_dir, "step2_1_overlay_grid.png"
        )
        grid = make_grid_image(overlay_images, nrow=4, padding=5, pad_value=255)
        os.makedirs(os.path.dirname(grid_path), exist_ok=True)
        imageio.imwrite(grid_path, grid)
        print(f"Saved overlay grid to {grid_path}")
    overlay_elapsed = time.perf_counter() - overlay_start
    print(f"[step2_1] overlay total: {overlay_elapsed:.3f}s")


if __name__ == "__main__":
    main()
