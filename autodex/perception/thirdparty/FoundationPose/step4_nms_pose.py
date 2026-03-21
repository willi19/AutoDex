import argparse
import glob
import os
import time

import imageio
import numpy as np
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


def _bbox_corners_from_bounds(bounds):
    mins, maxs = bounds
    corners = np.array(
        [
            [mins[0], mins[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [mins[0], maxs[1], mins[2]],
            [mins[0], maxs[1], maxs[2]],
            [maxs[0], mins[1], mins[2]],
            [maxs[0], mins[1], maxs[2]],
            [maxs[0], maxs[1], mins[2]],
            [maxs[0], maxs[1], maxs[2]],
        ],
        dtype=np.float32,
    )
    return corners


def _transform_points(points, pose):
    homog = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    transformed = (pose @ homog.T).T[:, :3]
    return transformed


def _aabb_from_points(points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return mins, maxs


def _iou_aabb(a, b):
    a_min, a_max = a
    b_min, b_max = b
    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    inter = np.maximum(inter_max - inter_min, 0.0)
    inter_vol = inter[0] * inter[1] * inter[2]
    a_vol = np.prod(a_max - a_min)
    b_vol = np.prod(b_max - b_min)
    union = a_vol + b_vol - inter_vol
    if union <= 0:
        return 0.0
    return float(inter_vol / union)


def select_pose_by_nms(pose_files, mesh, iou_threshold: float):
    poses = [np.loadtxt(p).reshape(4, 4) for p in pose_files]
    corners = _bbox_corners_from_bounds(mesh.bounds)

    aabbs = []
    for pose in poses:
        pts_world = _transform_points(corners, pose)
        aabbs.append(_aabb_from_points(pts_world))

    n = len(aabbs)
    iou_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i, n):
            iou = _iou_aabb(aabbs[i], aabbs[j])
            iou_matrix[i, j] = iou
            iou_matrix[j, i] = iou

    overlap_counts = (iou_matrix >= iou_threshold).sum(axis=1)
    best_idx = int(np.argmax(overlap_counts))
    return best_idx, poses[best_idx], iou_matrix[best_idx]


def build_argparser():
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser(description="Step 4: NMS over world-space poses")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=f"{code_dir}/outputs/clock_pose",
        help="Output directory containing ob_in_world",
    )
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
        "--mesh-file",
        type=str,
        default=f"{code_dir}/demo_data/clock_demo/mesh/clock.obj",
        help="Path to object mesh",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="3D AABB IoU threshold for overlap counting",
    )
    parser.add_argument(
        "--save-name",
        type=str,
        default="selected_pose_world.txt",
        help="Filename to save selected world-space pose",
    )
    parser.add_argument(
        "--overlay-grid-path",
        type=str,
        default="",
        help="Optional path to save overlay grid image (default: <output-dir>/step4_overlay_grid.png)",
    )
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    code_dir = os.path.dirname(os.path.realpath(__file__))
    default_output_dir = parser.get_default("output_dir")
    default_data_dir = parser.get_default("data_dir")
    default_mesh_file = parser.get_default("mesh_file")
    if args.object_name and args.object_name != "clock":
        if args.output_dir == default_output_dir:
            args.output_dir = os.path.join(code_dir, "outputs", f"{args.object_name}_pose")
        if args.data_dir == default_data_dir:
            args.data_dir = os.path.join(code_dir, "demo_data", f"{args.object_name}_demo")
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

    pose_dir = os.path.join(args.output_dir, "ob_in_world")
    pose_files = sorted(glob.glob(os.path.join(pose_dir, "*.txt")))
    if not pose_files:
        raise FileNotFoundError(f"No pose files found in {pose_dir}")

    mesh = _load_mesh(args.mesh_file)
    nms_start = time.perf_counter()
    best_idx, best_pose, _ = select_pose_by_nms(
        pose_files=pose_files,
        mesh=mesh,
        iou_threshold=args.iou_threshold,
    )
    nms_elapsed = time.perf_counter() - nms_start
    print(f"[step4] NMS selection: {nms_elapsed:.3f}s")

    save_path = os.path.join(args.output_dir, args.save_name)
    np.savetxt(save_path, best_pose.reshape(4, 4))
    print(f"Selected pose index {best_idx} -> {save_path}")

    reader = ParadexReader(video_dir=args.data_dir, shorter_side=None, zfar=np.inf)
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
        pose_cam = extrinsic @ best_pose
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
        print(f"[step4] overlay {reader.id_strs[i]}: {elapsed:.3f}s")

    if overlay_images:
        grid_path = args.overlay_grid_path or os.path.join(
            args.output_dir, "step4_overlay_grid.png"
        )
        grid = make_grid_image(overlay_images, nrow=4, padding=5, pad_value=255)
        os.makedirs(os.path.dirname(grid_path), exist_ok=True)
        imageio.imwrite(grid_path, grid)
        print(f"Saved overlay grid to {grid_path}")
    overlay_elapsed = time.perf_counter() - overlay_start
    print(f"[step4] overlay total: {overlay_elapsed:.3f}s")



if __name__ == "__main__":
    main()
