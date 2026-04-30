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


def _build_view_batch(
    masks: Dict[str, np.ndarray],
    intrinsics: Dict[str, np.ndarray],
    extrinsics: Dict[str, np.ndarray],
    H: int,
    W: int,
    device: str = "cuda",
):
    """Stack per-view tensors used by batched IoU rendering.

    Returns (serials, mask_batch (N,H,W) bool, extrinsic_batch (N,4,4),
             proj_batch (N,4,4), glcam_t (4,4)) or None if no valid views.
    """
    from Utils import projection_matrix_from_intrinsics, glcam_in_cvcam

    serials: list = []
    mask_list: list = []
    extr_list: list = []
    proj_list: list = []
    for s, mask in masks.items():
        if s not in intrinsics or s not in extrinsics:
            continue
        K = np.asarray(intrinsics[s], dtype=np.float32)
        proj = projection_matrix_from_intrinsics(K, height=H, width=W,
                                                  znear=0.001, zfar=100)
        proj_list.append(torch.as_tensor(proj.reshape(4, 4), device=device,
                                          dtype=torch.float32))
        extr_list.append(torch.as_tensor(np.asarray(extrinsics[s]),
                                          device=device, dtype=torch.float32))
        mask_list.append(torch.as_tensor(mask, device=device, dtype=torch.bool))
        serials.append(s)

    if not serials:
        return None

    mask_batch = torch.stack(mask_list, dim=0)
    extrinsic_batch = torch.stack(extr_list, dim=0)
    proj_batch = torch.stack(proj_list, dim=0)
    glcam_t = torch.as_tensor(glcam_in_cvcam, device=device, dtype=torch.float32)
    return serials, mask_batch, extrinsic_batch, proj_batch, glcam_t


def _render_silhouette_bool_batched(
    H: int, W: int,
    ob_in_cams: torch.Tensor,   # (N, 4, 4)
    proj_t: torch.Tensor,       # (N, 4, 4)
    glcam_t: torch.Tensor,      # (4, 4)
    glctx,
    mesh_tensors,
) -> torch.Tensor:
    """Batched silhouette mask render. Returns bool (N, H, W)."""
    import nvdiffrast.torch as dr
    from Utils import to_homo_torch

    pos = mesh_tensors["pos"]
    faces = mesh_tensors["faces"]
    pos_homo = to_homo_torch(pos)
    ob_in_glcams = glcam_t[None] @ ob_in_cams                     # (N, 4, 4)
    proj_pose = proj_t @ ob_in_glcams                              # (N, 4, 4)
    pos_clip = proj_pose[:, None] @ pos_homo[None, ..., None]      # (N, V, 4, 1)
    pos_clip = pos_clip[..., 0]                                    # (N, V, 4)
    rast_out, _ = dr.rasterize(glctx, pos_clip, faces, resolution=np.asarray([H, W]))
    sil = rast_out[..., -1] > 0                                    # (N, H, W)
    sil = torch.flip(sil, dims=[1])
    return sil


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
    """Mean IoU of rendered mesh vs SAM mask across all views (batched).

    Stacks all valid views into a single rasterize call instead of looping.
    Returns mean IoU and per-view IoU (serial -> float).
    """
    batch = _build_view_batch(masks, intrinsics, extrinsics, H, W)
    if batch is None:
        return 0.0, {}
    serials, mask_batch, extrinsic_batch, proj_batch, glcam_t = batch

    pose_world_t = torch.as_tensor(pose_world, device="cuda", dtype=torch.float32)
    pose_cam_batch = extrinsic_batch @ pose_world_t                # (N, 4, 4)
    sil = _render_silhouette_bool_batched(
        H=H, W=W, ob_in_cams=pose_cam_batch, proj_t=proj_batch,
        glcam_t=glcam_t, glctx=glctx, mesh_tensors=mesh_tensors,
    )                                                              # (N, H, W) bool

    inter = (sil & mask_batch).sum(dim=(1, 2)).float()
    union = (sil | mask_batch).sum(dim=(1, 2)).float()
    per_view_iou = torch.where(union > 0, inter / union, torch.zeros_like(inter))

    iou_arr = per_view_iou.cpu().tolist()
    per_view = dict(zip(serials, iou_arr))
    return float(np.mean(iou_arr)) if iou_arr else 0.0, per_view


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
