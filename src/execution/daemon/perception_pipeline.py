#!/usr/bin/env python3
"""Real-world perception + planning pipeline.

Orchestrates:
  SAM3 (remote daemons) → Depth (local) → FPose (remote daemons) → Best IoU → Sil matching → Planning

Communication with remote daemons uses NAS file paths (no image serialization).

Usage:
    from autodex.perception.pipeline import PerceptionPipeline

    pipeline = PerceptionPipeline(
        sam3_hosts=[("192.168.0.102", 5001), ("192.168.0.103", 5001), ("192.168.0.104", 5001)],
        fpose_hosts=[("192.168.0.104", 5003), ("192.168.0.105", 5003), ("192.168.0.106", 5003)],
        mesh_path="/path/to/mesh.obj",
    )
    pose_world = pipeline.run(capture_dir="/path/to/episode")
"""
import json
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


class ZMQClient:
    """ZMQ REQ client. Sends/receives JSON strings (NAS paths, not images)."""

    def __init__(self, host: str, port: int, timeout_ms: int = 30000):
        import zmq
        self.host = host
        self.port = port
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.connect(f"tcp://{host}:{port}")

    def request(self, data: dict) -> dict:
        self.sock.send_string(json.dumps(data))
        return json.loads(self.sock.recv_string())

    def ping(self) -> bool:
        try:
            r = self.request({"command": "ping"})
            return r.get("status") == "ok"
        except:
            return False

    def close(self):
        self.sock.close()
        self.ctx.term()


class PerceptionPipeline:
    """Full perception pipeline: mask → depth → pose → sil matching.

    Remote daemons (ZMQ, NAS paths): SAM3, FPose
    Local models (direct call): DA3 depth, SilhouetteOptimizer
    """

    # Best 5 camera serials (cross-object, from eval pipeline)
    BEST_5_DA3 = ["25322638", "25322645", "24080331", "25322639", "25322643"]
    BEST_5_STEREO = ["25305461", "25305463", "25322651", "25322639", "24122734"]

    def __init__(
        self,
        sam3_hosts: List[Tuple[str, int]],
        fpose_hosts: List[Tuple[str, int]],
        mesh_path: str,
        depth_method: str = "da3",  # "da3" or "stereo"
        device: str = "cuda",
    ):
        self.sam3_clients = [ZMQClient(h, p) for h, p in sam3_hosts]
        self.fpose_clients = [ZMQClient(h, p) for h, p in fpose_hosts]
        self.mesh_path = mesh_path
        self.depth_method = depth_method
        self.device = device

        # Local models (loaded lazily)
        self._da3_model = None
        self._stereo_trt = None
        self._sil_optimizer = None

    def _load_da3(self):
        if self._da3_model is not None:
            return
        logger.info("Loading DA3 model...")
        t0 = time.perf_counter()
        _da3_src = str(Path(__file__).resolve().parents[3] / "autodex/perception/thirdparty/Depth-Anything-3/src")
        if _da3_src not in sys.path:
            sys.path.insert(0, _da3_src)
        from depth_anything_3.api import DepthAnything3
        self._da3_model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE")
        self._da3_model = self._da3_model.to(self.device)
        self._da3_model.eval()
        logger.info(f"DA3 loaded in {time.perf_counter() - t0:.2f}s")

    def _load_stereo_trt(self):
        if self._stereo_trt is not None:
            return
        logger.info("Loading stereo TRT...")
        from autodex.perception.depth import StereoDepthTRT
        self._stereo_trt = StereoDepthTRT()
        logger.info("Stereo TRT loaded")

    def _load_sil_optimizer(self):
        if self._sil_optimizer is not None:
            return
        logger.info("Loading silhouette optimizer...")
        from autodex.perception.silhouette import SilhouetteOptimizer
        self._sil_optimizer = SilhouetteOptimizer(self.mesh_path, device=self.device)
        logger.info("Silhouette optimizer loaded")

    def run(
        self,
        capture_dir: str,
        prompt: str = "object on the checkerboard",
        sil_iters: int = 100,
        sil_lr: float = 0.002,
    ) -> Optional[np.ndarray]:
        """Run full perception pipeline on a capture directory.

        Args:
            capture_dir: path with images/, cam_param/, etc.
            prompt: SAM3 text prompt
            sil_iters: silhouette optimization iterations
            sil_lr: silhouette optimization learning rate

        Returns:
            pose_world: 4x4 object pose in world frame, or None on failure
        """
        capture_dir = Path(capture_dir)
        t_start = time.perf_counter()

        # Load camera data
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

        # Get image size
        img0 = cv2.imread(str(img_dir / f"{serials[0]}.png"))
        H, W = img0.shape[:2]

        # Working dir for intermediate results
        work_dir = capture_dir / "_pipeline_tmp"
        work_dir.mkdir(exist_ok=True)
        mask_dir = work_dir / "masks"
        depth_dir = work_dir / "depth"
        mask_dir.mkdir(exist_ok=True)
        depth_dir.mkdir(exist_ok=True)

        # ── Step 1: SAM3 masks (remote, parallel) ──
        t0 = time.perf_counter()
        masks = self._run_sam3_parallel(serials, img_dir, mask_dir, prompt)
        t_sam3 = time.perf_counter() - t0
        n_masks = sum(1 for v in masks.values() if v)
        logger.info(f"SAM3: {n_masks}/{len(serials)} masks in {t_sam3:.2f}s")

        # ── Step 1.5: Select best 5 views (pre-determined) ──
        best5 = self.BEST_5_DA3 if self.depth_method == "da3" else self.BEST_5_STEREO
        fpose_serials = [s for s in best5 if masks.get(s) and s in serials]
        if not fpose_serials:
            # Fallback: top 5 by mask size
            mask_sizes = {s: masks[s] for s in serials if masks.get(s)}
            fpose_serials = sorted(mask_sizes, key=mask_sizes.get, reverse=True)[:5]
        logger.info(f"Selected {len(fpose_serials)} views for depth+FPose: {fpose_serials}")

        # ── Step 2: Depth (local, best 5 only) ──
        t0 = time.perf_counter()
        if self.depth_method == "stereo":
            depth_serials = self._run_stereo_depth(capture_dir, fpose_serials, intrinsics, extrinsics, depth_dir)
        else:
            depth_serials = self._run_da3_depth(fpose_serials, img_dir, intrinsics, extrinsics, depth_dir)
        t_depth = time.perf_counter() - t0
        logger.info(f"Depth ({self.depth_method}): {len(depth_serials)} views in {t_depth:.2f}s")

        # Filter to views with both mask and depth
        valid_serials = [s for s in depth_serials if masks.get(s)]
        if not valid_serials:
            logger.error("No views with both mask and depth")
            return None, None, None

        # ── Step 3: FPose register (remote, parallel, best 5 only) ──
        t0 = time.perf_counter()
        poses_cam = self._run_fpose_parallel(
            valid_serials, img_dir, depth_dir, mask_dir, intrinsics,
        )
        t_fpose = time.perf_counter() - t0
        logger.info(f"FPose: {len(poses_cam)} poses in {t_fpose:.2f}s")

        if not poses_cam:
            logger.error("No valid poses")
            return None, None

        # ── Step 4: Select best pose by mask IoU ──
        t0 = time.perf_counter()
        best_serial, best_pose_world = self._select_best_pose(
            poses_cam, mask_dir, intrinsics, extrinsics, serials, H, W,
        )
        t_select = time.perf_counter() - t0
        logger.info(f"Best pose: {best_serial} (IoU selection in {t_select:.2f}s)")

        # ── Step 5: Silhouette matching (local) ──
        t0 = time.perf_counter()
        views = []
        for s in serials:
            mp = mask_dir / f"{s}.png"
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

        self._load_sil_optimizer()
        pose_world = self._sil_optimizer.optimize(
            best_pose_world, views, iters=sil_iters, lr=sil_lr, antialias=True,
        )
        t_sil = time.perf_counter() - t0
        logger.info(f"Sil matching: {t_sil:.2f}s ({len(views)} views, {sil_iters} iters)")

        t_total = time.perf_counter() - t_start
        logger.info(f"Total: {t_total:.2f}s "
                     f"(SAM3={t_sam3:.1f} Depth={t_depth:.1f} FPose={t_fpose:.1f} Select={t_select:.1f} Sil={t_sil:.1f})")

        timing = {
            "total": t_total,
            "sam3": t_sam3,
            "depth": t_depth,
            "fpose": t_fpose,
            "select": t_select,
            "sil": t_sil,
            "n_masks": n_masks,
            "n_depth": len(depth_serials),
            "n_poses": len(poses_cam),
            "n_views_sil": len(views),
            "best_serial": best_serial,
            "depth_method": self.depth_method,
        }

        return pose_world, timing

    # ── Private methods ──

    def _run_sam3_parallel(self, serials, img_dir, mask_dir, prompt):
        """Distribute SAM3 across remote daemons via NAS paths.

        Each client processes its chunk sequentially (ZMQ REQ/REP),
        but different clients run in parallel.
        """
        n = len(self.sam3_clients)
        chunks = [[] for _ in range(n)]
        for i, s in enumerate(serials):
            chunks[i % n].append(s)

        masks = {}

        def process_chunk(client_idx):
            results = {}
            for s in chunks[client_idx]:
                try:
                    r = self.sam3_clients[client_idx].request({
                        "image_path": str(img_dir / f"{s}.png"),
                        "prompt": prompt,
                        "output_path": str(mask_dir / f"{s}.png"),
                    })
                    results[s] = r.get("found", False)
                except Exception as e:
                    logger.warning(f"SAM3 {s}: {e}")
                    results[s] = False
            return results

        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = [pool.submit(process_chunk, i) for i in range(n)]
            for f in as_completed(futures):
                masks.update(f.result())

        return masks

    def _run_da3_depth(self, serials, img_dir, intrinsics, extrinsics, depth_dir):
        """Run DA3 depth locally (batch)."""
        self._load_da3()
        images = [cv2.cvtColor(cv2.imread(str(img_dir / f"{s}.png")), cv2.COLOR_BGR2RGB) for s in serials]
        K_arr = np.array([intrinsics[s] for s in serials], dtype=np.float32)
        T_arr = np.array([extrinsics[s] for s in serials], dtype=np.float32)

        prediction = self._da3_model.inference(image=images, intrinsics=K_arr, extrinsics=T_arr)

        H, W = images[0].shape[:2]
        for i, s in enumerate(serials):
            d = prediction.depth[i]
            d_np = d.cpu().numpy() if hasattr(d, 'cpu') else np.asarray(d)
            if d_np.shape[0] != H or d_np.shape[1] != W:
                d_np = cv2.resize(d_np, (W, H), interpolation=cv2.INTER_NEAREST)
            d_mm = (d_np * 1000).clip(0, 65535).astype(np.uint16)
            cv2.imwrite(str(depth_dir / f"{s}.png"), d_mm)

        return serials

    def _run_stereo_depth(self, capture_dir, serials, intrinsics, extrinsics, depth_dir):
        """Run stereo TRT depth locally (all valid pairs, left views only)."""
        from autodex.perception.depth import find_all_stereo_pairs, build_rectify_maps
        self._load_stereo_trt()

        C2R_path = capture_dir / "C2R.npy"
        if not C2R_path.exists():
            C2R_path = capture_dir / "cam_param" / "C2R.npy"
        C2R = np.load(str(C2R_path)) if C2R_path.exists() else None

        pairs = find_all_stereo_pairs(serials, intrinsics, extrinsics, C2R)
        logger.info(f"Found {len(pairs)} stereo pairs")

        depth_serials = []
        for left_s, right_s in pairs:
            img_dir = capture_dir / "images"
            if not img_dir.exists():
                img_dir = capture_dir / "raw" / "images"

            left_rgb = cv2.cvtColor(cv2.imread(str(img_dir / f"{left_s}.png")), cv2.COLOR_BGR2RGB)
            right_rgb = cv2.cvtColor(cv2.imread(str(img_dir / f"{right_s}.png")), cv2.COLOR_BGR2RGB)

            K_l, K_r = intrinsics[left_s], intrinsics[right_s]
            T_l, T_r = extrinsics[left_s], extrinsics[right_s]

            maps = build_rectify_maps(K_l, K_r, T_l, T_r, left_rgb.shape[:2])
            if maps is None:
                continue

            # Rectify
            left_rect = cv2.remap(left_rgb, maps["map_left"][0], maps["map_left"][1], cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_rgb, maps["map_right"][0], maps["map_right"][1], cv2.INTER_LINEAR)

            # TRT disparity
            disp = self._stereo_trt._run_trt(left_rect, right_rect)

            # Disparity → depth (left view)
            disp_full = cv2.resize(disp, (maps["W_out"], maps["H_out"]), interpolation=cv2.INTER_LINEAR)
            disp_full *= maps["W_out"] / disp.shape[1]
            depth_rect = maps["f_rect"] * maps["baseline"] / (disp_full + 1e-10)

            # Un-rectify to original left view
            H_orig, W_orig = left_rgb.shape[:2]
            inv_map = maps.get("inv_map_left")
            if inv_map is not None:
                depth_orig = cv2.remap(depth_rect, inv_map[0], inv_map[1], cv2.INTER_NEAREST)
            else:
                depth_orig = cv2.resize(depth_rect, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)

            # rz correction
            rz = maps.get("rz_left")
            if rz is not None:
                depth_orig = depth_orig / rz.astype(np.float32)

            depth_orig[(depth_orig < 0.01) | (depth_orig > 10)] = 0
            d_mm = (depth_orig * 1000).clip(0, 65535).astype(np.uint16)
            cv2.imwrite(str(depth_dir / f"{left_s}.png"), d_mm)
            depth_serials.append(left_s)

        return depth_serials

    def _run_fpose_parallel(self, serials, img_dir, depth_dir, mask_dir, intrinsics):
        """Distribute FPose across remote daemons via NAS paths.

        Each client processes its chunk sequentially (ZMQ REQ/REP),
        but different clients run in parallel.
        """
        n = len(self.fpose_clients)
        chunks = [[] for _ in range(n)]
        for i, s in enumerate(serials):
            chunks[i % n].append(s)

        poses_cam = {}

        def process_chunk(client_idx):
            results = {}
            for s in chunks[client_idx]:
                try:
                    r = self.fpose_clients[client_idx].request({
                        "image_path": str(img_dir / f"{s}.png"),
                        "depth_path": str(depth_dir / f"{s}.png"),
                        "mask_path": str(mask_dir / f"{s}.png"),
                        "K": intrinsics[s].tolist(),
                        "mode": "register",
                        "iteration": 5,
                        "downscale": 0.5,
                    })
                    if "pose" in r:
                        results[s] = np.array(r["pose"]).reshape(4, 4)
                except Exception as e:
                    logger.warning(f"FPose {s}: {e}")
            return results

        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = [pool.submit(process_chunk, i) for i in range(n)]
            for f in as_completed(futures):
                poses_cam.update(f.result())

        return poses_cam

    def _select_best_pose(self, poses_cam, mask_dir, intrinsics, extrinsics, all_serials, H, W):
        """Select best pose by mean mask IoU across all views.

        Uses cached mesh tensors + glctx from silhouette optimizer if available.
        """
        self._load_sil_optimizer()
        mt = self._sil_optimizer.mesh_tensors
        glctx = self._sil_optimizer.glctx
        from autodex.perception.silhouette import nvdiffrast_render

        # Pre-load all masks
        sam_masks = {}
        for s in all_serials:
            mp = mask_dir / f"{s}.png"
            if mp.exists():
                m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    sam_masks[s] = m > 127

        best_serial, best_iou, best_pose = None, -1, None

        for src_s, pose_cam in poses_cam.items():
            pose_world = np.linalg.inv(extrinsics[src_s]) @ pose_cam
            ious = []

            for tgt_s in all_serials:
                if tgt_s not in sam_masks:
                    continue
                K = intrinsics[tgt_s].astype(np.float32)
                pc = extrinsics[tgt_s] @ pose_world
                pt = torch.as_tensor(pc, device="cuda", dtype=torch.float32).reshape(1, 4, 4)
                rc, _, _ = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=pt, glctx=glctx,
                                              mesh_tensors=mt, use_light=False)
                sil = rc[0].detach().cpu().numpy().sum(axis=2) > 0
                inter = (sil & sam_masks[tgt_s]).sum()
                union = (sil | sam_masks[tgt_s]).sum()
                ious.append(float(inter / union) if union > 0 else 0.0)

            mean_iou = np.mean(ious) if ious else 0.0
            if mean_iou > best_iou:
                best_iou = mean_iou
                best_serial = src_s
                best_pose = pose_world

        logger.info(f"Best: {best_serial}, mean IoU={best_iou:.3f}")
        return best_serial, best_pose

    def change_object(self, mesh_path: str):
        """Change target object."""
        self.mesh_path = mesh_path
        self._sil_optimizer = None
        for client in self.fpose_clients:
            try:
                client.request({"command": "reset_mesh", "mesh_path": mesh_path})
            except Exception as e:
                logger.warning(f"Failed to reset mesh: {e}")

    def close(self):
        for c in self.sam3_clients:
            c.close()
        for c in self.fpose_clients:
            c.close()