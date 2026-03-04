#!/usr/bin/env python3
"""SAM3 mask generation — fallback for videos YOLOE missed.

Reads/saves from local cache. No network I/O.
Tries object name first, then "object on white table" as fallback.
Skips video if first 5 frames produce no masks.

Pre-download videos:
    python src/perception/download_videos.py --base ... --serials 22684755 23263780

Run YOLOE first:
    python -u src/perception/batch_mask_yoloe.py --base ... --serials 22684755 23263780

Run SAM3 fallback:
    python -u src/perception/batch_mask.py --base ... --serials 22684755 23263780

Upload results:
    python src/perception/upload_results.py --base ...
"""

import os
import sys
import time
import argparse
from pathlib import Path

import gc

import cv2
import numpy as np
import torch

_SAM3_ROOT = Path(__file__).resolve().parents[2] / "autodex/perception/thirdparty/sam3"

PROBE_FRAMES = 5
MAX_FRAMES = 550
CACHE_ROOT = os.path.expanduser("~/video_cache")
NETWORK_PREFIX = "/home/mingi/paradex1/capture"
FALLBACK_PROMPT = "object"


def _get_cache_base(base_dir):
    base = str(Path(base_dir).resolve())
    if base.startswith(NETWORK_PREFIX):
        rel = base[len(NETWORK_PREFIX):].lstrip("/")
    else:
        rel = Path(base_dir).name
    return os.path.join(CACHE_ROOT, rel)



def save_mask_videos(masks, video_path, save_dir, serial, fps):
    """Save mask + debug videos to cache dir."""
    t0 = time.time()
    h, w = next(iter(masks.values())).shape

    mask_dir = os.path.join(save_dir, "obj_mask")
    debug_dir = os.path.join(save_dir, "obj_mask_debug")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    mask_writer = cv2.VideoWriter(os.path.join(mask_dir, f"{serial}.avi"), fourcc, fps, (w, h), False)
    debug_writer = cv2.VideoWriter(os.path.join(debug_dir, f"{serial}.avi"), fourcc, fps, (w, h), True)

    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        mask = masks.get(idx)
        mask_u8 = mask.astype(np.uint8) * 255 if mask is not None else np.zeros((h, w), dtype=np.uint8)
        mask_writer.write(mask_u8)
        green = np.zeros_like(bgr)
        green[:, :, 1] = mask_u8
        overlay = cv2.addWeighted(bgr, 1.0, green, 0.5, 0)
        debug_writer.write(overlay)
        idx += 1

    cap.release()
    mask_writer.release()
    debug_writer.release()
    print(f"  [save] {idx} frames in {time.time()-t0:.2f}s", flush=True)


def collect_tasks(base_dir, serial_filter=None):
    """Collect tasks from local cache — only videos without masks (YOLOE missed)."""
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
                # Skip if mask already exists
                if (idx_dir / "obj_mask" / f"{serial}.avi").exists():
                    continue
                tasks.append((str(vp), serial, str(idx_dir), prompt))
    return tasks


def _cleanup_gpu(predictor, sid):
    """Close session and free GPU memory."""
    try:
        predictor.handle_request(dict(type="close_session", session_id=sid))
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()


def run_sam3_video(predictor, video_path, prompts, n_probe=5):
    """Run SAM3 directly on video file. Loads frames once, tries multiple prompts.

    Args:
        video_path: path to .avi video file
        prompts: list of prompts to try in order
        n_probe: abort after this many frames if no masks found

    Returns (masks_dict, winning_prompt) or (None, None).
    """
    sid = None
    try:
        resp = predictor.handle_request(
            dict(type="start_session", resource_path=video_path)
        )
        sid = resp["session_id"]

        for p in prompts:
            print(f"  Trying '{p}'...", flush=True)
            t0 = time.time()

            # Reset state before each prompt (clears previous prompt/masks, keeps frames loaded)
            predictor.handle_request(dict(type="reset_session", session_id=sid))
            torch.cuda.empty_cache()  # release feature_cache/cached_frame_outputs from allocator

            predictor.handle_request(
                dict(type="add_prompt", session_id=sid, frame_index=0, text=p)
            )

            masks = {}
            stream = predictor.handle_stream_request(
                dict(type="propagate_in_video", session_id=sid,
                     propagation_direction="forward")
            )
            aborted = False
            for resp in stream:
                fidx = resp["frame_index"]
                out = resp["outputs"]
                binary_masks = out.get("out_binary_masks")
                if binary_masks is not None and len(binary_masks) > 0:
                    combined = binary_masks[0]
                    for m in binary_masks[1:]:
                        combined |= m
                    masks[fidx] = combined

                # After probe frames: abort early if no masks found
                if fidx >= n_probe - 1 and not masks:
                    stream.close()
                    aborted = True
                    break

            dt = time.time() - t0
            if aborted or not masks:
                print(f"  No masks in first {n_probe} frames with '{p}' ({dt:.1f}s)", flush=True)
                continue
            else:
                print(f"  {len(masks)} frames in {dt:.1f}s", flush=True)
                return masks, p

        return None, None
    finally:
        if sid is not None:
            _cleanup_gpu(predictor, sid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base dir (network FS path, used for cache mapping)")
    parser.add_argument("--prompt", default=None, help="Override prompt (skips obj_name -> fallback logic)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--serials", nargs="+", default=None)
    args = parser.parse_args()
    print("Starting SAM3 fallback mask generation...", flush=True)

    # Must be set before the CUDA allocator initializes.
    # Prevents OOM from fragmented reserved-but-unallocated cache across videos.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    torch.cuda.set_device(args.gpu)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    torch.inference_mode().__enter__()

    serial_filter = set(args.serials) if args.serials else None
    all_tasks = collect_tasks(args.base, serial_filter)

    if not all_tasks:
        print("Nothing to do (all videos already have masks).")
        return

    print(f"{len(all_tasks)} videos to process (YOLOE missed) | GPU: {args.gpu}", flush=True)

    # Load SAM3
    sam3_path = str(_SAM3_ROOT)
    if sam3_path not in sys.path:
        sys.path.insert(0, sam3_path)
    from sam3.model_builder import build_sam3_video_predictor

    print("Loading SAM3...", flush=True)
    predictor = build_sam3_video_predictor(gpus_to_use=[args.gpu])
    print("SAM3 ready.", flush=True)

    prompt_override = args.prompt
    done = 0
    vi = 0

    total = len(all_tasks)
    for local_path, serial, cache_dir, obj_name in all_tasks:
        vi += 1

        try:
            print(f"  [{vi}/{total}] {obj_name}/{Path(cache_dir).name}/{serial} | {cache_dir}", flush=True)

            # Get fps and frame count
            cap = cv2.VideoCapture(local_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if n_frames > MAX_FRAMES:
                print(f"  [{vi}/{total}] Skipping {serial}: {n_frames} frames > {MAX_FRAMES}", flush=True)
                continue

            # Try "object" first, then obj_name (unless --prompt overrides)
            if prompt_override:
                prompts_to_try = [prompt_override]
            else:
                prompts_to_try = [FALLBACK_PROMPT, obj_name]

            masks, winning_prompt = run_sam3_video(predictor, local_path, prompts_to_try, n_probe=PROBE_FRAMES)

            if masks:
                print(f"  === SAVING {obj_name}/{Path(cache_dir).name}/{serial} ({len(masks)} masks) ===", flush=True)
                save_mask_videos(masks, local_path, cache_dir, serial, fps)
                done += 1
                print(f"  === SAVED [{done}/{total}] ===", flush=True)
            else:
                print(f"  [{vi}] Failed all prompts: {serial}", flush=True)

        except Exception as e:
            import traceback
            print(f"  [{vi}] Error {serial}: {e}", flush=True)
            traceback.print_exc()

    print(f"All done! {done}/{vi} videos saved.", flush=True)


if __name__ == "__main__":
    main()
