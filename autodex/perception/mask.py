import sys
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

_JUNC0NG = Path(__file__).parent / "thirdparty/object-6d-tracking"
_GUNHEE = Path(__file__).parent / "thirdparty/_object_6d_tracking"
_WEIGHTS = Path(__file__).parent / "thirdparty/weights"
YOLOE_WEIGHTS = str(_WEIGHTS / "yoloe-26x-seg.pt")


def get_mask_yoloe(
    rgb: np.ndarray,
    target_class: str,
    model=None,
    conf_thr: float = 0.2,
) -> Optional[np.ndarray]:
    """Generate binary mask using YOLO-E.

    Args:
        rgb: RGB image (H, W, 3)
        target_class: text class name (e.g. "banana")
        model: pre-loaded YOLO model (loads yoloe-26x-seg.pt if None)
        conf_thr: confidence threshold

    Returns:
        mask: uint8 (H, W) with values 0/255, or None if not detected
    """
    from ultralytics import YOLO

    if model is None:
        model = YOLO("yoloe-26x-seg.pt")
        model.set_classes([target_class], model.get_text_pe([target_class]))

    results = model.predict(rgb, conf=conf_thr, verbose=False, device="cuda", retina_masks=True)

    if not results or results[0].masks is None or not len(results[0].boxes):
        return None

    result = results[0]
    best_idx = result.boxes.conf.cpu().numpy().argmax()

    orig_h, orig_w = rgb.shape[:2]
    raw_mask = result.masks.data[best_idx].cpu().numpy()
    mask = cv2.resize(raw_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    mask = (mask > 0.5).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def get_mask_sam3(
    images_folder: str,
    text_prompt: str,
    predictor=None,
    gpu: int = 0,
) -> Dict[int, np.ndarray]:
    """Generate masks for all frames using SAM3 video predictor.

    Args:
        images_folder: path to folder containing image files
        text_prompt: text prompt for segmentation (e.g. "object")
        predictor: pre-loaded SAM3 predictor (builds one if None)
        gpu: GPU index to use

    Returns:
        dict mapping frame_index -> binary mask (H, W) bool
    """
    sam3_path = str(_GUNHEE / "sam3")
    if sam3_path not in sys.path:
        sys.path.insert(0, sam3_path)

    if predictor is None:
        from sam3.model_builder import build_sam3_video_predictor
        predictor = build_sam3_video_predictor(gpus_to_use=[gpu])

    response = predictor.handle_request(
        dict(type="start_session", resource_path=str(images_folder))
    )
    session_id = response["session_id"]

    predictor.handle_request(
        dict(type="add_prompt", session_id=session_id, frame_index=0, text=text_prompt)
    )

    masks = {}
    for response in predictor.handle_stream_request(
        dict(type="propagate_in_video", session_id=session_id)
    ):
        frame_idx = response["frame_index"]
        logits = response["outputs"].get("out_mask_logits", [])
        if logits:
            combined = np.zeros_like(logits[0][0].cpu().numpy() > 0, dtype=bool)
            for logit in logits:
                combined |= logit[0].cpu().numpy() > 0
            masks[frame_idx] = combined

    return masks
