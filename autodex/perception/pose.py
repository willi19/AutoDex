import sys
from pathlib import Path
from typing import Optional

import numpy as np

_GUNHEE_FP = Path(__file__).parent / "thirdparty/_object_6d_tracking/FoundationPose"


def _setup_foundation_pose_path():
    path = str(_GUNHEE_FP)
    if path not in sys.path:
        sys.path.insert(0, path)
    # mycpp is a C++ extension; Utils.py uses the wrong import path so it falls back to None.
    # Pre-add the build dir so we can patch it in after import.
    mycpp_build = str(_GUNHEE_FP / "mycpp/build")
    if mycpp_build not in sys.path:
        sys.path.insert(0, mycpp_build)


class PoseTracker:
    """6D pose tracker using FoundationPose.

    Usage:
        tracker = PoseTracker(mesh_path, device_id=0)
        pose = tracker.init(rgb, depth, mask, K)    # first frame
        pose = tracker.track(rgb, depth, K)          # subsequent frames
        tracker.reset()                              # restart tracking
    """

    def __init__(self, mesh_path: str, device_id: int = 0):
        """
        Args:
            mesh_path: path to object mesh file (.obj, .ply)
            device_id: CUDA device ID
        """
        _setup_foundation_pose_path()

        import trimesh
        import torch
        from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
        import nvdiffrast.torch as dr

        import estimater as _est
        if _est.mycpp is None:
            try:
                import mycpp as _mycpp
                _est.mycpp = _mycpp
            except ImportError:
                pass

        self.device_id = device_id
        self.mesh = trimesh.load(mesh_path, force="mesh")
        # Cast to float32 — FoundationPose internally computes mesh_diameter
        # from vertices, and numpy.float64 propagates into torch tensors causing
        # dtype mismatches (float64 @ float32) in crop window computation.
        self.mesh.vertices = self.mesh.vertices.astype(np.float32)

        with torch.cuda.device(device_id):
            scorer = ScorePredictor()
            refiner = PoseRefinePredictor()
            # Reduce internal render resolution to prevent OOM
            # (matches ~/shared_data/_object_6d_tracking/run/run_object_6d_pipeline.py)
            refiner.cfg['input_resize'] = (80, 80)
            glctx = dr.RasterizeCudaContext()
            self.estimator = FoundationPose(
                model_pts=self.mesh.vertices,
                model_normals=self.mesh.vertex_normals,
                mesh=self.mesh,
                scorer=scorer,
                refiner=refiner,
                glctx=glctx,
                debug=0,
                debug_dir="/tmp/foundationpose_debug",
            )

    def init(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        K: np.ndarray,
        iteration: int = 5,
    ) -> np.ndarray:
        """Register first frame pose.

        Args:
            rgb: RGB image (H, W, 3)
            depth: depth map in meters (H, W)
            mask: binary mask (H, W)
            K: camera intrinsics (3, 3)
            iteration: number of refinement iterations

        Returns:
            pose: 4x4 transformation matrix (object in camera frame)
        """
        import torch

        K_f32 = K.astype(np.float32) if K.dtype != np.float32 else K
        depth_f32 = depth.astype(np.float32) if depth.dtype != np.float32 else depth
        with torch.cuda.device(self.device_id):
            pose = self.estimator.register(
                K=K_f32, rgb=rgb, depth=depth_f32, ob_mask=mask.astype(bool), iteration=iteration
            )
        return pose

    def track(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        iteration: int = 2,
    ) -> np.ndarray:
        """Track object in subsequent frames.

        Args:
            rgb: RGB image (H, W, 3)
            depth: depth map in meters (H, W)
            K: camera intrinsics (3, 3)
            iteration: number of refinement iterations

        Returns:
            pose: 4x4 transformation matrix (object in camera frame)
        """
        import torch

        K_f32 = K.astype(np.float32) if K.dtype != np.float32 else K
        depth_f32 = depth.astype(np.float32) if depth.dtype != np.float32 else depth
        with torch.cuda.device(self.device_id):
            pose = self.estimator.track_one(rgb=rgb, depth=depth_f32, K=K_f32, iteration=iteration)
        return pose

    def reset(self):
        """Reset tracker to initial state (re-registration required)."""
        self.estimator.pose_last = None
