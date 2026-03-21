"""Render utilities — thin wrapper around FoundationPose's Utils.py."""
import sys
from pathlib import Path

_FP_ROOT = Path(__file__).resolve().parents[4] / "autodex/perception/thirdparty/_object_6d_tracking/FoundationPose"
if str(_FP_ROOT) not in sys.path:
    sys.path.insert(0, str(_FP_ROOT))

from Utils import make_mesh_tensors, nvdiffrast_render, glcam_in_cvcam, to_homo_torch, projection_matrix_from_intrinsics
