#!/usr/bin/env python3
"""First-frame mask generation using YOLOE for pose initialization.

FoundationPose tracking only needs a mask on the first frame (init),
so this script generates a single mask image per video instead of
processing all frames.

Saves: obj_mask_first/{serial}.png  (binary mask, 255=object)

Pre-download videos:
    python src/perception/download_videos.py --base ... --serials 22684755 23263780

Run:
    python -u src/perception/batch_mask_first.py \
        --base /home/mingi/paradex1/capture/eccv2026/inspire_f1 \
        --serials 22684755 23263780

Then run pose (no full mask video needed):
    python -u src/perception/batch_pose.py \
        --base /home/mingi/paradex1/capture/eccv2026/inspire_f1 \
        --serials 22684755 23263780 --mesh_dir /home/mingi/mesh
"""

import os
import argparse
from pathlib import Path

import cv2
import numpy as np

_WEIGHTS = Path(__file__).resolve().parents[2] / "autodex/perception/thirdparty/weights"
YOLOE_WEIGHTS = str(_WEIGHTS / "yoloe-26x-seg.pt")

PROBE_FRAMES = 5  # try first N frames to find a valid mask
CACHE_ROOT = os.path.expanduser("~/video_cache")
NETWORK_PREFIX = "/home/mingi/paradex1/capture"


def _get_cache_base(base_dir):
    base = str(Path(base_dir).resolve())
    if base.startswith(NETWORK_PREFIX):
        rel = base[len(NETWORK_PREFIX):].lstrip("/")
    else:
        rel = Path(base_dir).name
    return os.path.join(CACHE_ROOT, rel)


def collect_tasks(base_dir, serial_filter=None):
    """Collect tasks — videos without first-frame mask and without full mask video."""
    cache_base = Path(_get_cache_base(base_dir))
    if not cache_base.is_dir():
        return []
    tasks = []
    for obj_dir in sorted(cache_base.iterdir()):
        if not obj_dir.is_dir():
            continue
        prompt = obj_dir.name
        for idx_dir in sorted(obj_dir.iterdir()):
            if not idx_dir.is_dir():
                continue
            video_dir = idx_dir / "videos"
            if not video_dir.is_dir():
                continue
            for vp in sorted(video_dir.glob("*.avi")):
                serial = vp.stem
                if serial_filter and serial not in serial_filter:
                    continue
                # Skip if first-frame mask already exists
                if (idx_dir / "obj_mask_first" / f"{serial}.png").exists():
                    continue
                # Skip if full mask video already exists (already have all frames)
                if (idx_dir / "obj_mask" / f"{serial}.avi").exists():
                    continue
                tasks.append((str(vp), serial, str(idx_dir), prompt))
    return tasks


def get_first_frame_mask(model, video_path, conf_thr, n_probe):
    """Run YOLOE on first N frames, return (mask, frame_index) for the first valid one."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, -1

    for idx in range(n_probe):
        ret, bgr = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        results = model.predict([rgb], conf=conf_thr, verbose=False, device="cuda", retina_masks=True)
        result = results[0]
        if result.masks is not None and len(result.boxes):
            h, w = bgr.shape[:2]
            best_idx = result.boxes.conf.cpu().numpy().argmax()
            raw_mask = result.masks.data[best_idx].cpu().numpy()
            mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_LINEAR)
            cap.release()
            return (mask > 0.5).astype(np.uint8) * 255, idx

    cap.release()
    return None, -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base dir (network FS path)")
    parser.add_argument("--serials", nargs="+", default=None)
    parser.add_argument("--prompt", default="object", help="YOLOE text prompt (default: 'object')")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--probe_frames", type=int, default=PROBE_FRAMES,
                        help="Number of frames to try (default: 5)")
    args = parser.parse_args()
    print("Starting first-frame mask generation (YOLOE)...", flush=True)

    import torch
    torch.cuda.set_device(args.gpu)

    serial_filter = set(args.serials) if args.serials else None
    tasks = collect_tasks(args.base, serial_filter)
    if not tasks:
        print("Nothing to do (all videos already have masks).")
        return

    print(f"{len(tasks)} videos to process | GPU: {args.gpu}", flush=True)

    from ultralytics import YOLO
    print("Loading YOLOE...", flush=True)
    model = YOLO(YOLOE_WEIGHTS)
    model.set_classes([args.prompt], model.get_text_pe([args.prompt]))
    print(f"YOLOE ready. Prompt: '{args.prompt}'", flush=True)

    done = 0
    skipped = 0
    total = len(tasks)

    for local_path, serial, cache_dir, obj_name in tasks:
        idx_name = Path(cache_dir).name
        print(f"[{done + skipped + 1}/{total}] {obj_name}/{idx_name}/{serial}", flush=True)

        try:
            mask, frame_idx = get_first_frame_mask(model, local_path, args.conf, args.probe_frames)
            if mask is not None:
                out_dir = os.path.join(cache_dir, "obj_mask_first")
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{serial}.png")
                cv2.imwrite(out_path, mask)
                done += 1
                print(f"  Saved (frame {frame_idx}): {out_path}", flush=True)
            else:
                skipped += 1
                print(f"  No mask in first {args.probe_frames} frames, skipped", flush=True)
        except Exception as e:
            import traceback
            print(f"  Error: {e}", flush=True)
            traceback.print_exc()
            skipped += 1

    print(f"\nAll done! {done}/{total} saved, {skipped} skipped.", flush=True)


if __name__ == "__main__":
    main()
