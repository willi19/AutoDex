"""Segmentation module — class-based wrappers for YOLOE and SAM3.

Each class loads the model once in __init__ and exposes:
    segment(rgb, prompt)       → single image mask
    segment_batch(rgbs, prompt) → list of masks
    segment_video(video_path, prompt) → dict[frame_idx, mask]
"""

import gc
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

_THIS_DIR = Path(__file__).parent
_SAM3_ROOT = _THIS_DIR / "thirdparty/sam3"
_WEIGHTS = _THIS_DIR / "thirdparty/weights"
YOLOE_WEIGHTS = str(_WEIGHTS / "yoloe-26x-seg.pt")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _all_masks_from_yoloe(result, h: int, w: int) -> Optional[List]:
    """Extract all masks with confidence from a single YOLOE result.

    Returns: [(uint8_mask, conf), ...] sorted by conf descending, or None.
    """
    if result.masks is None or not len(result.boxes):
        return None
    confs = result.boxes.conf.cpu().numpy()
    out = []
    for i in range(len(confs)):
        raw = result.masks.data[i].cpu().numpy()
        mask = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_u8 = (mask > 0.5).astype(np.uint8) * 255
        out.append((mask_u8, float(confs[i])))
    out.sort(key=lambda x: x[1], reverse=True)
    return out if out else None


# Colors for multi-mask debug overlay (BGR)
_DEBUG_COLORS = [
    (0, 255, 0),    # green
    (0, 0, 255),    # red
    (255, 0, 0),    # blue
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
    (128, 0, 255),  # purple
    (0, 128, 255),  # orange
]


def _to_mask_u8(m) -> np.ndarray:
    """Convert bool/uint8 mask to uint8 0/255."""
    m = np.asarray(m).squeeze()
    u8 = m.astype(np.uint8)
    if u8.max() == 1:
        u8 = u8 * 255
    return u8


def best_mask(frame_masks: List) -> np.ndarray:
    """Pick highest-prob mask from [(mask, prob), ...] list."""
    if len(frame_masks) == 1:
        return frame_masks[0][0]
    best_idx = int(np.argmax([p for _, p in frame_masks]))
    return frame_masks[best_idx][0]


def save_mask_video(
    masks: Dict[int, List],
    video_path: str,
    out_dir: str,
    serial: str,
    fps: float,
    save_debug: bool = True,
):
    """Write mask + optional debug overlay videos.

    Args:
        masks: {frame_idx: [(mask, prob), ...]} — individual masks per frame.
            Each mask is bool/uint8 (H,W). Uses best_mask() for the saved mask.
        video_path: source video for debug overlay
        out_dir: capture directory (writes obj_mask/ and obj_mask_debug/ under it)
        serial: camera serial (filename stem)
        fps: output video fps
        save_debug: if True, also write debug video with multi-color overlay
    """
    if not masks:
        return

    sample_frame = next(iter(masks.values()))
    sample_m = np.asarray(sample_frame[0][0]).squeeze()
    h, w = sample_m.shape[:2]

    mask_dir = Path(out_dir) / "obj_mask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    mask_writer = cv2.VideoWriter(str(mask_dir / f"{serial}.avi"), fourcc, fps, (w, h), False)

    debug_writer = None
    if save_debug:
        debug_dir = Path(out_dir) / "obj_mask_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_writer = cv2.VideoWriter(str(debug_dir / f"{serial}.avi"), fourcc, fps, (w, h), True)

    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        frame_masks = masks.get(idx)
        if frame_masks:
            mask_u8 = _to_mask_u8(best_mask(frame_masks))
        else:
            mask_u8 = np.zeros((h, w), dtype=np.uint8)
        mask_writer.write(mask_u8)
        if debug_writer is not None:
            if frame_masks and len(frame_masks) > 1:
                # Multi-mask: each mask gets a different color
                vis = bgr.copy()
                for mi, (mk, _prob) in enumerate(frame_masks):
                    mk_bool = np.asarray(mk).squeeze().astype(bool)
                    color = _DEBUG_COLORS[mi % len(_DEBUG_COLORS)]
                    overlay = np.zeros_like(vis)
                    overlay[mk_bool] = color
                    vis[mk_bool] = cv2.addWeighted(
                        vis, 0.5, overlay, 0.5, 0
                    )[mk_bool]
                debug_writer.write(vis)
            else:
                green = np.zeros_like(bgr)
                green[:, :, 1] = mask_u8
                debug_writer.write(cv2.addWeighted(bgr, 1.0, green, 0.5, 0))
        idx += 1

    cap.release()
    mask_writer.release()
    if debug_writer is not None:
        debug_writer.release()


# ── YOLOE ────────────────────────────────────────────────────────────────────

class YoloeSegmentor:
    """YOLOE-based segmentation. Model loaded once, prompt set lazily."""

    def __init__(self, weights: str = YOLOE_WEIGHTS, gpu: int = 0, conf_thr: float = 0.2):
        import torch
        torch.cuda.set_device(gpu)
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf_thr = conf_thr
        self._current_prompt = None

    def _ensure_prompt(self, prompt: str):
        if prompt != self._current_prompt:
            self.model.set_classes([prompt], self.model.get_text_pe([prompt]))
            self._current_prompt = prompt

    def segment(self, rgb: np.ndarray, prompt: str) -> Optional[np.ndarray]:
        """Single image → uint8 mask (H,W) 0/255, or None."""
        self._ensure_prompt(prompt)
        results = self.model.predict(
            rgb, conf=self.conf_thr, verbose=False, device="cuda", retina_masks=True,
        )
        if not results:
            return None
        h, w = rgb.shape[:2]
        all_masks = _all_masks_from_yoloe(results[0], h, w)
        if all_masks is None:
            return None
        return best_mask(all_masks)

    def segment_batch(
        self, rgbs: List[np.ndarray], prompt: str, batch_size: int = 64,
    ) -> List[Optional[List]]:
        """Batch of images → list of [(mask, conf), ...] per image (same length as input)."""
        self._ensure_prompt(prompt)
        if not rgbs:
            return []
        h, w = rgbs[0].shape[:2]
        out = [None] * len(rgbs)
        for start in range(0, len(rgbs), batch_size):
            batch = rgbs[start : start + batch_size]
            results = self.model.predict(
                batch, conf=self.conf_thr, verbose=False, device="cuda", retina_masks=True,
            )
            for i, result in enumerate(results):
                out[start + i] = _all_masks_from_yoloe(result, h, w)
        return out

    def segment_video(
        self,
        video_path: str,
        prompt: str,
        batch_size: int = 64,
        skip: int = 1,
        probe_frames: int = 5,
    ) -> Optional[Dict[int, List]]:
        """Full video → {frame_idx: [(mask, conf), ...]}. Returns None if probe fails.

        Args:
            skip: process every Nth frame, reuse mask for skipped frames.
            probe_frames: abort if no mask in first N frames.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            return None

        # Probe
        probe = self.segment_batch(frames[:probe_frames], prompt)
        if not any(m is not None for m in probe):
            return None

        # Keyframe detection
        if skip <= 1:
            masks_list = self.segment_batch(frames, prompt, batch_size=batch_size)
            masks = {i: m for i, m in enumerate(masks_list) if m is not None}
        else:
            key_indices = list(range(0, len(frames), skip))
            key_frames = [frames[i] for i in key_indices]
            key_masks_list = self.segment_batch(key_frames, prompt, batch_size=batch_size)
            key_masks = {}
            for ki, m in zip(key_indices, key_masks_list):
                if m is not None:
                    key_masks[ki] = m
            # Fill skipped frames with nearest keyframe mask
            masks = {}
            last_mask = None
            for idx in range(len(frames)):
                if idx in key_masks:
                    last_mask = key_masks[idx]
                if last_mask is not None:
                    masks[idx] = last_mask

        return masks


# ── SAM3 ─────────────────────────────────────────────────────────────────────

class Sam3Segmentor:
    """SAM3 video predictor. Model loaded once, sessions managed per video."""

    def __init__(self, gpu: int = 0):
        import os
        import torch
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        torch.cuda.set_device(gpu)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        torch.inference_mode().__enter__()

        sam3_path = str(_SAM3_ROOT)
        if sam3_path not in sys.path:
            sys.path.insert(0, sam3_path)
        from sam3.model_builder import build_sam3_video_predictor

        self.predictor = build_sam3_video_predictor(gpus_to_use=[gpu])
        # Prevent OOM on long videos by trimming past non-conditioning frame outputs.
        # NOTE: do NOT set offload_output_to_cpu_for_eval=True — it causes KeyError
        # on 'maskmem_features' when add_new_mask calls _run_single_frame_inference
        # with run_mem_encoder=False (the trimmed output omits that key).
        self.predictor.model.tracker.trim_past_non_cond_mem_for_eval = True
        self.gpu = gpu

    def _cleanup(self, session_id):
        import torch
        try:
            self.predictor.handle_request(dict(type="close_session", session_id=session_id))
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()

    def segment_video(
        self,
        video_path: str,
        prompt: str,
        fallback_prompts: Optional[List[str]] = None,
        probe_frames: int = 5,
    ) -> Optional[Dict[int, List]]:
        """Run SAM3 video propagation.

        Returns:
            {frame_idx: [(bool_mask, prob), ...]} or None.
            Each frame has a list of (mask, probability) tuples — one per
            detected object. Caller decides how to combine (union, best, etc.).

        Args:
            prompt: primary text prompt
            fallback_prompts: additional prompts to try if primary fails
            probe_frames: abort prompt if no mask in first N frames
        """
        prompts = [prompt] + (fallback_prompts or [])
        sid = None
        try:
            resp = self.predictor.handle_request(
                dict(type="start_session", resource_path=video_path)
            )
            sid = resp["session_id"]

            for p in prompts:
                # Reset state (clears previous prompt, keeps frames loaded)
                self.predictor.handle_request(dict(type="reset_session", session_id=sid))
                import torch
                torch.cuda.empty_cache()

                self.predictor.handle_request(
                    dict(type="add_prompt", session_id=sid, frame_index=0, text=p)
                )

                masks = {}
                stream = self.predictor.handle_stream_request(
                    dict(type="propagate_in_video", session_id=sid,
                         propagation_direction="forward")
                )
                aborted = False
                for resp in stream:
                    fidx = resp["frame_index"]
                    out = resp["outputs"]
                    binary_masks = out.get("out_binary_masks")
                    if binary_masks is not None and len(binary_masks) > 0:
                        probs = out.get("out_probs")
                        frame_masks = []
                        for mi in range(len(binary_masks)):
                            m = binary_masks[mi]
                            # Detach from GPU to avoid OOM when storing all masks
                            if hasattr(m, 'cpu'):
                                m = m.cpu().numpy()
                            prob = float(probs[mi]) if probs is not None and mi < len(probs) else 1.0
                            frame_masks.append((m, prob))
                        masks[fidx] = frame_masks

                    if fidx >= probe_frames - 1 and not masks:
                        stream.close()
                        aborted = True
                        break

                if not aborted and masks:
                    return masks

            return None
        finally:
            if sid is not None:
                self._cleanup(sid)

    def segment(self, rgb: np.ndarray, prompt: str) -> Optional[np.ndarray]:
        """Single image segmentation via SAM3 (writes temp image, runs 1-frame session)."""
        import tempfile
        tmp = tempfile.mkdtemp()
        img_path = Path(tmp) / "000000.jpg"
        cv2.imwrite(str(img_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        masks = self.segment_video(str(tmp), prompt)
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
        if masks and 0 in masks:
            return _to_mask_u8(best_mask(masks[0]))
        return None