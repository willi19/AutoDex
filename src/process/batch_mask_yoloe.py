#!/usr/bin/env python3
"""YOLOE mask generation — reads/saves from local cache. No network I/O.

Pre-download videos:
    python src/perception/download_videos.py --base ... --serials 22684755 23263780

Run YOLOE:
    python -u src/perception/batch_mask_yoloe.py --base ... --serials 22684755 23263780

Upload results:
    python src/perception/upload_results.py --base ...

If a prompt fails probe (first 5 frames), skips all videos with that prompt.
"""

import os
import time
import argparse
from pathlib import Path

import cv2
import numpy as np

_WEIGHTS = Path(__file__).resolve().parents[2] / "autodex/perception/thirdparty/weights"
YOLOE_WEIGHTS = str(_WEIGHTS / "yoloe-26x-seg.pt")

PROBE_FRAMES = 5
CACHE_ROOT = os.path.expanduser("~/video_cache")
NETWORK_PREFIX = "/home/mingi/paradex1/capture"


def _get_cache_base(base_dir):
    base = str(Path(base_dir).resolve())
    if base.startswith(NETWORK_PREFIX):
        rel = base[len(NETWORK_PREFIX):].lstrip("/")
    else:
        rel = Path(base_dir).name
    return os.path.join(CACHE_ROOT, rel)


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 30.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        frames.append(bgr)
    cap.release()
    return frames, fps




def run_yoloe_full(frames, model, conf_thr=0.2, batch_size=64):
    """Run YOLOE on full frames (used for probe)."""
    if not frames:
        return {}
    h, w = frames[0].shape[:2]
    masks = {}
    for batch_start in range(0, len(frames), batch_size):
        batch_bgr = frames[batch_start:batch_start + batch_size]
        batch_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in batch_bgr]
        results = model.predict(batch_rgb, conf=conf_thr, verbose=False, device="cuda", retina_masks=True)
        for i, result in enumerate(results):
            if result.masks is not None and len(result.boxes):
                best_idx = result.boxes.conf.cpu().numpy().argmax()
                raw_mask = result.masks.data[best_idx].cpu().numpy()
                mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_LINEAR)
                masks[batch_start + i] = (mask > 0.5).astype(bool)
    return masks


def run_yoloe_skip(frames, model, conf_thr=0.2, batch_size=64, skip=3):
    """Run YOLOE on every Nth frame, reuse mask for skipped frames."""
    if not frames:
        return {}
    h, w = frames[0].shape[:2]

    # Pick keyframes
    key_indices = list(range(0, len(frames), skip))
    key_frames = [frames[i] for i in key_indices]

    # Batch detect on keyframes only
    key_masks = {}
    for batch_start in range(0, len(key_frames), batch_size):
        batch_bgr = key_frames[batch_start:batch_start + batch_size]
        batch_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in batch_bgr]
        results = model.predict(batch_rgb, conf=conf_thr, verbose=False, device="cuda", retina_masks=True)
        for i, result in enumerate(results):
            if result.masks is not None and len(result.boxes):
                best_idx = result.boxes.conf.cpu().numpy().argmax()
                raw_mask = result.masks.data[best_idx].cpu().numpy()
                mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_LINEAR)
                key_masks[key_indices[batch_start + i]] = (mask > 0.5).astype(bool)

    # Fill in skipped frames with nearest keyframe mask
    masks = {}
    last_mask = None
    for idx in range(len(frames)):
        if idx in key_masks:
            last_mask = key_masks[idx]
        if last_mask is not None:
            masks[idx] = last_mask

    return masks


def save_mask_videos(masks, bgr_frames, save_dir, serial, fps):
    """Save mask + debug videos to local cache dir."""
    t0 = time.time()
    h, w = bgr_frames[0].shape[:2]

    mask_dir = os.path.join(save_dir, "obj_mask_yoloe")
    debug_dir = os.path.join(save_dir, "obj_mask_yoloe_debug")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    mask_writer = cv2.VideoWriter(os.path.join(mask_dir, f"{serial}.avi"), fourcc, fps, (w, h), False)
    debug_writer = cv2.VideoWriter(os.path.join(debug_dir, f"{serial}.avi"), fourcc, fps, (w, h), True)

    for idx, bgr in enumerate(bgr_frames):
        mask = masks.get(idx)
        mask_u8 = mask.astype(np.uint8) * 255 if mask is not None else np.zeros((h, w), dtype=np.uint8)
        mask_writer.write(mask_u8)
        green = np.zeros_like(bgr)
        green[:, :, 1] = mask_u8
        overlay = cv2.addWeighted(bgr, 1.0, green, 0.5, 0)
        debug_writer.write(overlay)

    mask_writer.release()
    debug_writer.release()
    print(f"  [save] {len(bgr_frames)} frames in {time.time()-t0:.2f}s", flush=True)


def collect_tasks(base_dir, serial_filter=None):
    """Collect tasks from local cache. Returns list of (local_path, serial, cache_capture_dir, prompt)."""
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
                # Skip if mask already exists in cache
                if (idx_dir / "obj_mask_yoloe" / f"{serial}.avi").exists():
                    continue
                tasks.append((str(vp), serial, str(idx_dir), prompt))
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base dir (network FS path, used for cache mapping)")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--serials", nargs="+", default=None)
    args = parser.parse_args()
    print("Starting YOLOE batch mask generation...", flush=True)

    import torch
    torch.cuda.set_device(args.gpu)

    serial_filter = set(args.serials) if args.serials else None
    all_tasks = collect_tasks(args.base, serial_filter)

    if not all_tasks:
        print("Nothing to do. Run download_videos.py first?")
        return

    prompt_text = args.prompt if args.prompt else "object"
    all_tasks = [(lp, s, cd, prompt_text) for lp, s, cd, _ in all_tasks]

    print(f"{len(all_tasks)} videos to process | GPU: {args.gpu}", flush=True)

    print("Loading YOLOE weights...", flush=True)
    from ultralytics import YOLO
    current_prompt = None
    model = YOLO(YOLOE_WEIGHTS)
    print("YOLOE model loaded.", flush=True)

    done = 0
    vi = 0
    skipped = 0

    for local_path, serial, cache_dir, prompt in all_tasks:
        if prompt != current_prompt:
            print(f"  Setting prompt: '{prompt}'...", flush=True)
            model.set_classes([prompt], model.get_text_pe([prompt]))
            current_prompt = prompt

        vi += 1
        obj_name = Path(cache_dir).parent.name
        idx_name = Path(cache_dir).name
        print(f"  [{vi}/{len(all_tasks)}] {obj_name}/{idx_name}/{serial} | {cache_dir}", flush=True)
        try:
            t0 = time.time()
            frames, fps = read_video(local_path)
            if not frames:
                print(f"  [{vi}] Empty video: {serial}", flush=True)
                continue
            print(f"  [{vi}] Read {serial}: {len(frames)} frames in {time.time()-t0:.2f}s", flush=True)

            # Probe first 5 frames (full frame detection)
            probe_masks = run_yoloe_full(frames[:PROBE_FRAMES], model, conf_thr=args.conf, batch_size=PROBE_FRAMES)
            if not probe_masks:
                print(f"  [{vi}] Probe failed ({serial}), skipping", flush=True)
                skipped += 1
                continue

            # Full processing with frame skipping
            t0 = time.time()
            masks = run_yoloe_skip(frames, model, conf_thr=args.conf, batch_size=args.batch_size)
            print(f"  [{vi}] {len(masks)}/{len(frames)} frames in {time.time()-t0:.2f}s", flush=True)

            if masks:
                save_mask_videos(masks, frames, cache_dir, serial, fps)
                done += 1
            else:
                print(f"  [{vi}] No masks: {serial}", flush=True)
                skipped += 1
        except Exception as e:
            print(f"  [{vi}] Error {serial}: {e}", flush=True)

    print(f"All done! {done}/{vi} saved, {skipped} skipped.", flush=True)


if __name__ == "__main__":
    main()
