import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

_JUNC0NG = Path(__file__).parent / "thirdparty/object-6d-tracking"
_GUNHEE = Path(__file__).parent / "thirdparty/_object_6d_tracking"
_FS_ROOT = Path(__file__).parent / "thirdparty/FoundationStereo"


def _setup_foundation_stereo_path():
    path = str(_FS_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


def get_depth_stereo_pytorch(
    left_img: np.ndarray,
    right_img: np.ndarray,
    K: np.ndarray,
    baseline: float,
    model=None,
    ckpt_dir: Optional[str] = None,
    valid_iters: int = 32,
) -> np.ndarray:
    """Estimate metric depth from stereo pair using FoundationStereo (PyTorch).

    Args:
        left_img: left RGB image (H, W, 3)
        right_img: right RGB image (H, W, 3)
        K: camera intrinsic matrix (3, 3) for left camera
        baseline: stereo baseline in meters
        model: pre-loaded FoundationStereo model (loads from ckpt_dir if None)
        ckpt_dir: path to model_best_bp2.pth (default: thirdparty/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth)
        valid_iters: number of flow-field updates

    Returns:
        depth: metric depth map in meters (H, W), same resolution as input
    """
    _setup_foundation_stereo_path()
    from omegaconf import OmegaConf
    from core.foundation_stereo import FoundationStereo
    from core.utils.utils import InputPadder

    if ckpt_dir is None:
        ckpt_dir = str(_FS_ROOT / "pretrained_models/23-51-11/model_best_bp2.pth")

    if model is None:
        cfg = OmegaConf.load(str(Path(ckpt_dir).parent / "cfg.yaml"))
        if "vit_size" not in cfg:
            cfg["vit_size"] = "vitl"
        cfg["valid_iters"] = valid_iters
        cfg["hiera"] = 0
        model = FoundationStereo(cfg)
        ckpt = torch.load(ckpt_dir, map_location="cuda")
        model.load_state_dict(ckpt["model"])
        model.cuda()
        model.eval()

    orig_h, orig_w = left_img.shape[:2]

    img0 = torch.as_tensor(left_img.copy()).cuda().float()[None].permute(0, 3, 1, 2)
    img1 = torch.as_tensor(right_img.copy()).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)

    with torch.no_grad(), torch.cuda.amp.autocast(True):
        disp = model.forward(img0, img1, iters=valid_iters, test_mode=True)
    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(orig_h, orig_w)

    depth = K[0, 0] * baseline / (disp + 1e-10)
    return depth


def get_depth_stereo(
    left_img: np.ndarray,
    right_img: np.ndarray,
    model,
    K: np.ndarray,
    baseline: float,
    height: int = None,
    width: int = None,
) -> np.ndarray:
    """Estimate depth from stereo pair using FoundationStereo (TensorRT/ONNX).

    Args:
        left_img: left RGB image (H, W, 3)
        right_img: right RGB image (H, W, 3)
        model: pre-loaded TensorRT or ONNX model
        K: camera intrinsic matrix (3, 3)
        baseline: stereo baseline in meters
        height: resize height for model input (None = keep original)
        width: resize width for model input (None = keep original)

    Returns:
        depth: depth map in meters (H, W)
    """
    foundation_stereo_path = str(_JUNC0NG / "thirdparty/FoundationStereo")
    if foundation_stereo_path not in sys.path:
        sys.path.insert(0, foundation_stereo_path)

    orig_h, orig_w = left_img.shape[:2]

    if height and width:
        left_resized = cv2.resize(left_img, (width, height))
        right_resized = cv2.resize(right_img, (width, height))
        scale_x = width / orig_w
        scale_y = height / orig_h
    else:
        left_resized = left_img
        right_resized = right_img
        scale_x = scale_y = 1.0

    H, W = left_resized.shape[:2]

    left_tensor = torch.as_tensor(left_resized.copy()).float()[None].permute(0, 3, 1, 2)
    right_tensor = torch.as_tensor(right_resized.copy()).float()[None].permute(0, 3, 1, 2)

    is_onnx = not hasattr(model, "run") or hasattr(model, "get_inputs")
    if is_onnx:
        disp = model.run(None, {"left": left_tensor.numpy(), "right": right_tensor.numpy()})[0]
    else:
        disp = model.run([left_tensor.numpy(), right_tensor.numpy()])[0]

    disp = disp.squeeze()
    if disp.ndim == 1:
        disp = disp.reshape(H, W)

    fx_scaled = K[0, 0] * scale_x
    depth = fx_scaled * baseline / (disp + 1e-10)

    if height and width:
        depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    return depth


def get_depth_da3(
    images: List[np.ndarray],
    intrinsics: Optional[np.ndarray] = None,
    model=None,
    process_res: int = 504,
) -> List[np.ndarray]:
    """Estimate depth from monocular images using Depth-Anything-3.

    Args:
        images: list of RGB images (H, W, 3)
        intrinsics: camera intrinsics (N, 3, 3) or None
        model: pre-loaded DepthAnything3 model (loads da3-large if None)
        process_res: processing resolution

    Returns:
        depth_maps: list of depth maps (H, W) in meters
    """
    da3_src = str(_GUNHEE / "Depth-Anything-3/src")
    if da3_src not in sys.path:
        sys.path.insert(0, da3_src)

    if model is None:
        from depth_anything_3.api import DepthAnything3
        model = DepthAnything3(model_name="da3-large")

    prediction = model.inference(
        image=images,
        intrinsics=intrinsics,
        process_res=process_res,
    )

    return [d.cpu().numpy() for d in prediction.depth]
