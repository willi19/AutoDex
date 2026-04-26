"""Cross-view pose selection utilities.

Given multiple candidate object poses (each from a different source view) and
ground-truth masks across all cameras, pick the candidate whose rendered
silhouette best matches the masks (mean IoU across views).

Used by:
  - PerceptionPipeline._select_best_pose (FoundationPose register candidates)
  - FoundPose / PicoPose first-frame init scripts (per-view PnP candidates)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


_FP_ROOT = Path(__file__).resolve().parent / "thirdparty/FoundationPose"
if str(_FP_ROOT) not in sys.path:
    sys.path.insert(0, str(_FP_ROOT))


def _render_silhouette(
    pose_world: np.ndarray,
    K: np.ndarray,
    extrinsic: np.ndarray,
    H: int,
    W: int,
    glctx,
    mesh_tensors,
) -> np.ndarray:
    """Render mesh silhouette in target camera given world pose. Returns bool (H, W)."""
    from Utils import nvdiffrast_render

    pose_cam = extrinsic @ pose_world
    pt = torch.as_tensor(pose_cam, device="cuda", dtype=torch.float32).reshape(1, 4, 4)
    K = np.asarray(K, dtype=np.float32)
    rc, _, _ = nvdiffrast_render(
        K=K, H=H, W=W, ob_in_cams=pt, glctx=glctx,
        mesh_tensors=mesh_tensors, use_light=False,
    )
    return rc[0].detach().cpu().numpy().sum(axis=2) > 0


def compute_cross_view_iou(
    pose_world: np.ndarray,
    masks: Dict[str, np.ndarray],
    intrinsics: Dict[str, np.ndarray],
    extrinsics: Dict[str, np.ndarray],
    H: int,
    W: int,
    glctx,
    mesh_tensors,
) -> Tuple[float, Dict[str, float]]:
    """Mean IoU of rendered mesh vs SAM mask across all views.

    Args:
        pose_world: 4x4 object pose in world frame.
        masks: serial -> bool (H, W) mask. Skip serials missing here.
        intrinsics: serial -> 3x3 K.
        extrinsics: serial -> 4x4 world->cam.
        H, W: image size.
        glctx, mesh_tensors: nvdiffrast context + mesh tensors (pre-built).

    Returns:
        mean_iou, per_view_iou (serial -> float).
    """
    per_view = {}
    for s, mask in masks.items():
        if s not in intrinsics or s not in extrinsics:
            continue
        sil = _render_silhouette(
            pose_world, intrinsics[s], extrinsics[s], H, W, glctx, mesh_tensors,
        )
        inter = (sil & mask).sum()
        union = (sil | mask).sum()
        per_view[s] = float(inter / union) if union > 0 else 0.0
    mean_iou = float(np.mean(list(per_view.values()))) if per_view else 0.0
    return mean_iou, per_view


def select_best_pose_by_iou(
    candidates: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    intrinsics: Dict[str, np.ndarray],
    extrinsics: Dict[str, np.ndarray],
    H: int,
    W: int,
    glctx,
    mesh_tensors,
) -> Tuple[Optional[str], Optional[np.ndarray], float, Dict[str, float]]:
    """Pick candidate pose with highest mean cross-view mask IoU.

    Args:
        candidates: source_serial -> 4x4 pose_world. Each entry is a candidate.
        masks: serial -> bool (H, W) mask (used as ground truth for IoU).
        intrinsics, extrinsics, H, W, glctx, mesh_tensors: see compute_cross_view_iou.

    Returns:
        best_serial, best_pose_world, best_mean_iou, per_candidate_mean_iou (serial -> float).
        Returns (None, None, -1, {}) if no candidate produced any IoU.
    """
    best_serial: Optional[str] = None
    best_pose: Optional[np.ndarray] = None
    best_iou: float = -1.0
    per_cand: Dict[str, float] = {}

    for src_s, pose_world in candidates.items():
        mean_iou, _ = compute_cross_view_iou(
            pose_world, masks, intrinsics, extrinsics, H, W, glctx, mesh_tensors,
        )
        per_cand[src_s] = mean_iou
        if mean_iou > best_iou:
            best_iou = mean_iou
            best_serial = src_s
            best_pose = pose_world

    return best_serial, best_pose, best_iou, per_cand


def load_masks_bool(
    mask_dir: Path,
    serials: List[str],
    threshold: int = 127,
) -> Dict[str, np.ndarray]:
    """Load uint8 masks from {mask_dir}/{serial}.png as bool (H, W) dict.

    Skips files that don't exist or fail to load.
    """
    out: Dict[str, np.ndarray] = {}
    for s in serials:
        mp = Path(mask_dir) / f"{s}.png"
        if not mp.exists():
            continue
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        out[s] = m > threshold
    return out
