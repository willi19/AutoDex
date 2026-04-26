"""In-process FoundPose first-frame init wrapper.

Replaces the disk/subprocess-based MV-GoTrack
``scripts/run_foundpose_first_frame_init.py`` with a callable class:

    init = FoundPoseInit(mesh_path, assets_root, obj_name, device='cuda:0')
    per_view = init.estimate_per_view(images_rgb, masks_bool, intrinsics, extrinsics)
    # per_view[serial] = {pose_world (4x4 m), quality, inliers, template_id, timings}

Stage A (onboarding) is run on first instantiation if no cached repre exists.

Must run in the ``gotrack`` conda env (depends on FoundPose's torch / dinov2
/ xformers stack).
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_GOTRACK_ROOT = Path(__file__).resolve().parent / "thirdparty/MV-GoTrack"
if str(_GOTRACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_GOTRACK_ROOT))


def _foundpose_repre_path(assets_root: Path, dataset_name: str, object_id: int = 1) -> Path:
    return assets_root / "object_repre" / "v1" / dataset_name / str(object_id) / "repre.pth"


def _bbox_xyxy_from_mask(mask_bool: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(mask_bool)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return np.array([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                    dtype=np.float32)


class FoundPoseInit:
    """Per-view FoundPose pose estimation, in-process.

    Args:
        mesh_path: source mesh (.obj/.ply/.glb).
        assets_root: where to cache Stage-A repre (e.g. outputs/foundpose_assets/{obj}).
        obj_name: dataset_name tag used in onboarding (must match across cache + init).
        object_id: integer id (default 1, matches single-object usage).
        device: cuda device string.
        crop_size, crop_rel_pad, extractor_name, ...: FoundPoseOpts knobs.
        translation_scale: FoundPose returns t in mesh units (mm by default with
            mesh_scale=1000); we multiply by this to get meters. Default 1e-3.
    """

    DEFAULT_OPTS = dict(
        crop_size=(280, 280),
        crop_rel_pad=0.05,
        extractor_name="dinov2_vits14-reg",
        match_template_type="tfidf",
        match_feat_matching_type="cyclic_buddies",
        match_top_n_templates=5,
        match_top_k_buddies=300,
        pnp_type="opencv",
        pnp_ransac_iter=400,
        pnp_inlier_thresh=10.0,
        translation_scale=1e-3,
        min_mask_pixels=200,
    )

    def __init__(
        self,
        mesh_path: str,
        assets_root: str,
        obj_name: str,
        object_id: int = 1,
        device: str = "cuda:0",
        reference_intrinsics_json: Optional[str] = None,
        reference_camera_id: Optional[str] = None,
        opts: Optional[Mapping[str, Any]] = None,
        force_onboard: bool = False,
    ):
        self.mesh_path = Path(mesh_path).resolve()
        self.assets_root = Path(assets_root).resolve()
        self.obj_name = obj_name
        self.object_id = int(object_id)
        self.device = device
        self.opts = dict(self.DEFAULT_OPTS)
        if opts:
            self.opts.update(opts)

        repre_path = _foundpose_repre_path(self.assets_root, obj_name, self.object_id)
        if not repre_path.is_file() or force_onboard:
            if reference_intrinsics_json is None or reference_camera_id is None:
                raise FileNotFoundError(
                    f"FoundPose repre missing at {repre_path}. "
                    "Pass reference_intrinsics_json + reference_camera_id to onboard."
                )
            # If repre is missing but partial assets exist (e.g. previous failed
            # onboard left template dirs behind), force overwrite so the onboard
            # script doesn't bail on "Output directory already exists".
            need_overwrite = force_onboard or self.assets_root.exists()
            self._onboard(reference_intrinsics_json, reference_camera_id, need_overwrite)
            if not repre_path.is_file():
                raise RuntimeError(f"Onboarding finished but repre still missing: {repre_path}")

        self._build_model(repre_path.parent.parent)  # repre_dir = .../object_repre/v1/{dataset_name}/

    def _onboard(
        self,
        reference_intrinsics_json: str,
        reference_camera_id: str,
        force: bool,
    ) -> None:
        """Run Stage A onboarding in-process by importing the onboard script's main."""
        # The onboard script is structured as a CLI (argparse-based main()), but its core
        # work is bop_toolkit + pyrender. Easier to import its module-level helpers; for
        # now, exec it as a child process within the same env. This is the only subprocess
        # call we keep, and only on first run per object.
        import subprocess
        cmd = [
            sys.executable,
            str(_GOTRACK_ROOT / "scripts/onboard_custom_mesh_for_foundpose.py"),
            "--mesh-path", str(self.mesh_path),
            "--object-id", str(self.object_id),
            "--dataset-name", self.obj_name,
            "--output-root", str(self.assets_root),
            "--reference-intrinsics-json", str(reference_intrinsics_json),
            "--reference-camera-id", str(reference_camera_id),
            "--reference-image-scale", "1.0",
            "--mesh-scale", "1000.0",
            "--min-num-viewpoints", "57",
            "--num-inplane-rotations", "14",
            "--ssaa-factor", "4.0",
            "--pca-components", "256",
            "--cluster-num", "2048",
        ]
        if force:
            cmd.append("--overwrite")
        env = os.environ.copy()
        env["PYOPENGL_PLATFORM"] = env.get("PYOPENGL_PLATFORM", "egl")
        env["EGL_PLATFORM"] = env.get("EGL_PLATFORM", "surfaceless")
        logger.info(f"[FoundPoseInit] Onboarding {self.obj_name} (mesh={self.mesh_path.name})")
        t0 = time.perf_counter()
        res = subprocess.run(cmd, env=env)
        if res.returncode != 0:
            raise RuntimeError(f"FoundPose onboarding failed for {self.obj_name}")
        logger.info(f"[FoundPoseInit] Onboarded {self.obj_name} in {time.perf_counter() - t0:.1f}s")

    def _build_model(self, repre_dir: Path) -> None:
        import torch
        from model import config as model_config
        from model.foundpose import FoundPose

        if self.device.startswith("cuda"):
            cuda_idx = int(self.device.split(":")[-1]) if ":" in self.device else 0
            torch.cuda.set_device(cuda_idx)

        opts = model_config.FoundPoseOpts(
            crop_size=tuple(self.opts["crop_size"]),
            crop_rel_pad=float(self.opts["crop_rel_pad"]),
            extractor_name=str(self.opts["extractor_name"]),
            grid_cell_size=14.0,
            match_template_type=str(self.opts["match_template_type"]),
            match_top_n_templates=int(self.opts["match_top_n_templates"]),
            match_feat_matching_type=str(self.opts["match_feat_matching_type"]),
            match_top_k_buddies=int(self.opts["match_top_k_buddies"]),
            pnp_type=str(self.opts["pnp_type"]),
            pnp_ransac_iter=int(self.opts["pnp_ransac_iter"]),
            pnp_inlier_thresh=float(self.opts["pnp_inlier_thresh"]),
            debug=False,
        )
        t0 = time.perf_counter()
        model = FoundPose(opts=opts)
        model.to(self.device)
        model.eval()
        model.onboarding(repre_dir=repre_dir)
        model.post_onboarding_processing()
        self.model = model
        self.model_load_sec = time.perf_counter() - t0
        logger.info(f"[FoundPoseInit] Model loaded in {self.model_load_sec:.1f}s")

    def _build_camera_model(self, K: np.ndarray, ext_cw: np.ndarray, width: int, height: int):
        from utils import structs
        T_world_from_eye = np.linalg.inv(np.asarray(ext_cw, dtype=np.float64))
        return structs.PinholePlaneCameraModel(
            width=int(width), height=int(height),
            f=(float(K[0, 0]), float(K[1, 1])),
            c=(float(K[0, 2]), float(K[1, 2])),
            T_world_from_eye=T_world_from_eye,
        )

    def estimate_one_view(
        self,
        image_rgb: np.ndarray,
        mask_bool: np.ndarray,
        K: np.ndarray,
        ext_cw: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        """Run FoundPose on one view. Returns None if mask too small or PnP failed.

        Returns:
            {
                "pose_world":   4x4 in meters,
                "pose_camera":  4x4 in meters,
                "quality":      float (PnP quality / inlier ratio),
                "inliers":      int,
                "template_id":  int,
                "mask_pixels":  int,
                "timings":      {"preprocess_sec", "backbone_sec",
                                 "estimate_poses_sec", "template_retrieval_sec",
                                 "pnp_sec"},
            }
        """
        import torch
        from utils import data_util, structs, transform3d

        h, w = image_rgb.shape[:2]
        n_pixels = int(mask_bool.sum())
        if n_pixels < int(self.opts["min_mask_pixels"]):
            return None

        bbox = _bbox_xyxy_from_mask(mask_bool)
        if bbox is None:
            return None

        camera_model = self._build_camera_model(K, ext_cw, w, h)
        obj_anno = structs.ObjectAnnotation(
            dataset=self.obj_name,
            lid=self.object_id,
            masks_modal=mask_bool.astype(np.bool_),
            boxes_modal=bbox.astype(np.float32),
            score=1.0,
        )
        scene_obs = structs.SceneObservation(
            scene_id=0, im_id=0, image=image_rgb, camera=camera_model,
            objects_anno=[obj_anno], time=0.0,
        )

        timings: Dict[str, float] = {}
        t0 = time.perf_counter()
        inputs = data_util.convert_default_detections_to_foundpose_inputs(
            scene_observation=scene_obs,
            crop_size=tuple(self.opts["crop_size"]),
            crop_rel_pad=float(self.opts["crop_rel_pad"]),
        )
        timings["preprocess_sec"] = time.perf_counter() - t0
        detections = inputs["detections"]
        if len(detections) == 0:
            return None
        crop_detections = inputs["crop_detections"]

        obj_id = int(detections.labels[0].item())
        repre = self.model.objects_repre[obj_id]

        with torch.no_grad():
            t1 = time.perf_counter()
            features = self.model.backbone(crop_detections.rgbs.to(self.device))
            feature_maps = features["feature_maps"]
            torch.cuda.synchronize()
            timings["backbone_sec"] = time.perf_counter() - t1

            t2 = time.perf_counter()
            estimate = self.model.estimate_poses(
                obj_id=obj_id,
                repre=repre,
                feature_map_chw=feature_maps[0],
                crop_mask=crop_detections.masks[0],
                crop_camera=crop_detections.cameras[0],
            )
            torch.cuda.synchronize()
            timings["estimate_poses_sec"] = time.perf_counter() - t2

        if estimate is None or not estimate.get("final_poses"):
            return None
        inner = estimate.get("times", {}) or {}
        timings["template_retrieval_sec"] = float(inner.get("establish_correspondences_sec", 0.0))
        timings["pnp_sec"] = float(inner.get("pnp_sec", 0.0))

        final = estimate["final_poses"][0]
        pose_cam_native = transform3d.Rt_to_4x4_numpy(
            R=np.asarray(final["R_m2c"], dtype=np.float64),
            t=np.asarray(final["t_m2c"], dtype=np.float64).reshape(1, 3),
        )
        scale = float(self.opts["translation_scale"])
        pose_camera_m = pose_cam_native.copy()
        pose_camera_m[:3, 3] *= scale
        T_world_from_crop_cam = np.asarray(
            crop_detections.cameras[0].T_world_from_eye, dtype=np.float64
        )
        pose_world_m = T_world_from_crop_cam @ pose_camera_m

        inliers = final.get("inliers", 0)
        if hasattr(inliers, "shape"):
            inliers_count = int(np.asarray(inliers).reshape(-1).shape[0])
        else:
            try:
                inliers_count = int(inliers)
            except Exception:
                inliers_count = 0

        return {
            "pose_world": pose_world_m,
            "pose_camera": pose_camera_m,
            "quality": float(final.get("quality", 0.0)),
            "inliers": inliers_count,
            "template_id": int(final.get("template_id", -1)),
            "mask_pixels": n_pixels,
            "timings": timings,
        }

    def estimate_per_view(
        self,
        images_rgb: Dict[str, np.ndarray],
        masks_bool: Dict[str, np.ndarray],
        intrinsics: Dict[str, np.ndarray],
        extrinsics: Dict[str, np.ndarray],
    ) -> Dict[str, Dict[str, Any]]:
        """Run FoundPose on every camera. Returns {serial: result_or_None}."""
        out: Dict[str, Dict[str, Any]] = {}
        for s, img in images_rgb.items():
            if s not in masks_bool or s not in intrinsics or s not in extrinsics:
                out[s] = None
                continue
            try:
                out[s] = self.estimate_one_view(
                    img, masks_bool[s], intrinsics[s], extrinsics[s],
                )
            except Exception as exc:
                logger.warning(f"[FoundPoseInit] {s} failed: {exc}")
                out[s] = None
        return out
