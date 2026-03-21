import argparse
import os
import time

import cv2
import imageio
import numpy as np
import torch
import trimesh

from estimater import *
from datareader import *
from Utils import make_grid_image, make_mesh_tensors, nvdiffrast_render


def _load_mesh(mesh_path: str):
    loaded = trimesh.load(mesh_path, process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)
        ]
        if not meshes:
            raise RuntimeError("Scene contains no valid Trimesh objects")
        mesh = trimesh.util.concatenate(meshes)
        logging.info(f"Loaded Scene with {len(meshes)} meshes, combined into single mesh")
        return mesh
    return loaded


def build_argparser():
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser(description="Step 3: FoundationPose per-view estimation")
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
    parser.add_argument("--est-refine-iter", type=int, default=5)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=f"{code_dir}/outputs/clock_pose",
        help="Directory to save poses and posed meshes",
    )
    parser.add_argument(
        "--overlay-grid-path",
        type=str,
        default="",
        help="Optional path to save overlay grid image (default: <output-dir>/step3_overlay_grid.png)",
    )
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    code_dir = os.path.dirname(os.path.realpath(__file__))
    default_data_dir = parser.get_default("data_dir")
    default_mesh_file = parser.get_default("mesh_file")
    default_output_dir = parser.get_default("output_dir")
    if args.object_name and args.object_name != "clock":
        if args.data_dir == default_data_dir:
            args.data_dir = os.path.join(code_dir, "demo_data", f"{args.object_name}_demo")
        if args.mesh_file == default_mesh_file:
            args.mesh_file = os.path.join(
                args.data_dir, "mesh", f"{args.object_name}.obj"
            )
        if args.output_dir == default_output_dir:
            args.output_dir = os.path.join(code_dir, "outputs", f"{args.object_name}_pose")

    if not os.path.isfile(args.mesh_file):
        candidate_obj = os.path.join(args.data_dir, "mesh", f"{args.object_name}.obj")
        candidate_ply = os.path.join(args.data_dir, "mesh", f"{args.object_name}.ply")
        if os.path.isfile(candidate_obj):
            args.mesh_file = candidate_obj
        elif os.path.isfile(candidate_ply):
            args.mesh_file = candidate_ply

    set_logging_format()
    set_seed(0)

    mesh = _load_mesh(args.mesh_file)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    overlay_images = []

    purple = np.array([128, 0, 128], dtype=np.uint8)
    mesh_for_overlay = mesh.copy()
    vertex_colors = np.tile(
        np.append(purple, 255).reshape(1, 4), (len(mesh_for_overlay.vertices), 1)
    )
    mesh_for_overlay.visual = trimesh.visual.ColorVisuals(vertex_colors=vertex_colors)
    mesh_tensors_overlay = make_mesh_tensors(mesh_for_overlay, device="cuda")

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=args.output_dir,
        debug=args.debug,
        glctx=glctx,
    )
    logging.info("estimator initialization done")

    reader = ParadexReader(video_dir=args.data_dir, shorter_side=None, zfar=np.inf)

    cam_dir = os.path.join(args.output_dir, "ob_in_cam")
    world_dir = os.path.join(args.output_dir, "ob_in_world")
    mesh_cam_dir = os.path.join(args.output_dir, "meshes_cam")
    mesh_world_dir = os.path.join(args.output_dir, "meshes_world")
    os.makedirs(cam_dir, exist_ok=True)
    os.makedirs(world_dir, exist_ok=True)
    os.makedirs(mesh_cam_dir, exist_ok=True)
    os.makedirs(mesh_world_dir, exist_ok=True)

    total_start = time.perf_counter()
    for i in range(len(reader.color_files)):
        frame_start = time.perf_counter()
        logging.info(f"i:{i}")
        color = reader.get_color(i)
        depth = reader.get_depth(i)
        K = reader.get_K(i)

        mask = reader.get_mask(i)
        if mask is None:
            logging.warning(f"No mask found for frame {i}, skipping...")
            continue

        pose_cam = est.register(
            K=K,
            rgb=color,
            depth=depth,
            ob_mask=mask.astype(bool),
            iteration=args.est_refine_iter,
        )

        extrinsic = reader.get_extrinsic(i)
        if extrinsic is not None:
            pose_world = np.linalg.inv(extrinsic) @ pose_cam
        else:
            pose_world = None
            logging.warning(f"No extrinsics found for frame {i}, skipping world-space pose")

        frame_id = reader.id_strs[i]
        np.savetxt(os.path.join(cam_dir, f"{frame_id}.txt"), pose_cam.reshape(4, 4))

        mesh_cam = mesh.copy()
        mesh_cam.apply_transform(pose_cam)
        mesh_cam.export(os.path.join(mesh_cam_dir, f"{frame_id}.obj"))

        if pose_world is not None:
            np.savetxt(os.path.join(world_dir, f"{frame_id}.txt"), pose_world.reshape(4, 4))
            mesh_world = mesh.copy()
            mesh_world.apply_transform(pose_world)
            mesh_world.export(os.path.join(mesh_world_dir, f"{frame_id}.obj"))

        if args.debug >= 1:
            center_pose = pose_cam @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(
                color,
                ob_in_cam=center_pose,
                scale=0.1,
                K=K,
                thickness=3,
                transparency=0,
                is_input_rgb=True,
            )
            cv2.imshow("foundationpose", vis[..., ::-1])
            cv2.waitKey(1)

        try:
            pose_tensor = torch.as_tensor(
                pose_cam, device="cuda", dtype=torch.float32
            ).reshape(1, 4, 4)
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
        except Exception as exc:
            logging.warning(f"Overlay render failed for frame {frame_id}: {exc}")
        elapsed = time.perf_counter() - frame_start
        logging.info(f"[step3] frame {i}: {elapsed:.3f}s")
        if args.debug >= 2:
            track_vis_dir = os.path.join(args.output_dir, "step3_track_vis")
            os.makedirs(track_vis_dir, exist_ok=True)
            imageio.imwrite(os.path.join(track_vis_dir, f"{frame_id}.png"), vis)

    total_elapsed = time.perf_counter() - total_start
    logging.info(f"[step3] total: {total_elapsed:.3f}s")

    if overlay_images:
        grid_path = args.overlay_grid_path or os.path.join(
            args.output_dir, "step3_overlay_grid.png"
        )
        grid = make_grid_image(overlay_images, nrow=4, padding=5, pad_value=255)
        os.makedirs(os.path.dirname(grid_path), exist_ok=True)
        imageio.imwrite(grid_path, grid)
        logging.info(f"Saved overlay grid to {grid_path}")


if __name__ == "__main__":
    main()
