#!/usr/bin/env python3
"""One-off: run SAM3 mask on attached_container/20260121_163623 with prompt 'wall mounted dispenser'."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

CAP_DIR = Path.home() / "shared_data/RSS2026_Mingi/experiment/selected_100/attached_container/20260121_163623"
OUT_DIR = Path(__file__).resolve().parents[2] / "exp/perception/attached_container/20260121_163623"
PROMPT = "wall mounted dispenser"


def overlay_mask(img, mask, color=(0, 255, 0), alpha=0.5):
    out = img.copy()
    m = (mask > 0) if mask.ndim == 2 else (mask[..., 0] > 0)
    if m.any():
        c = np.array(color, np.float32)
        out[m] = (out[m].astype(np.float32) * (1 - alpha) + c * alpha).astype(np.uint8)
    return out


def make_grid(imgs, serials, title="", ncols=5, scale=0.25):
    items = [(s, imgs[s]) for s in serials if s in imgs]
    if not items:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    h0, w0 = items[0][1].shape[:2]
    h, w = int(h0 * scale), int(w0 * scale)
    nrows = (len(items) + ncols - 1) // ncols
    title_h = 40 if title else 0
    grid = np.zeros((title_h + nrows * h, ncols * w, 3), dtype=np.uint8)
    if title:
        cv2.putText(grid, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    for i, (s, img) in enumerate(items):
        r, c = divmod(i, ncols)
        cell = cv2.resize(img, (w, h))
        cv2.putText(cell, s[-8:], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.putText(cell, s[-8:], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y0 = title_h + r * h
        grid[y0:y0 + h, c * w:(c + 1) * w] = cell
    return grid


# Load cameras
with open(CAP_DIR / "cam_param" / "intrinsics.json") as f:
    intr = json.load(f)
serials = sorted(intr.keys())
images = {}
for s in serials:
    p = CAP_DIR / "images" / f"{s}.png"
    if p.exists():
        images[s] = cv2.imread(str(p))
serials = [s for s in serials if s in images]

# Run SAM3
model = build_sam3_image_model(device="cuda")
processor = Sam3Processor(model, device="cuda", confidence_threshold=0.5)

overlays = {}
for s in serials:
    img_rgb = cv2.cvtColor(images[s], cv2.COLOR_BGR2RGB)
    state = processor.set_image(Image.fromarray(img_rgb))
    processor.reset_all_prompts(state)
    output = processor.set_text_prompt(state=state, prompt=PROMPT)

    masks_t = output.get("masks")
    scores_t = output.get("scores")
    H, W = images[s].shape[:2]

    if masks_t is None or masks_t.numel() == 0:
        mask_np = np.zeros((H, W), dtype=np.uint8)
    else:
        best = int(torch.argmax(scores_t).item()) if scores_t is not None and scores_t.numel() > 0 else 0
        m = masks_t[best]
        if m.ndim == 3:
            m = m[0]
        mask_np = (m.to(torch.uint8) * 255).cpu().numpy()

    overlays[s] = overlay_mask(images[s], mask_np)

title = f'SAM3 | "{PROMPT}" | attached_container/20260121_163623'
grid = make_grid(overlays, serials, title=title)

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "mask_sam3_wall_mounted_dispenser.png"
cv2.imwrite(str(out_path), grid)
print(f"Saved: {out_path}")