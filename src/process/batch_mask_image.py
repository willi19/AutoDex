#!/usr/bin/env python3
"""SAM3 mask generation using BATCHED IMAGE mode.

Processes N frames per forward pass instead of 1-by-1.
Faster GPU utilization but no temporal tracking (each frame is independent).

Input:  {path}/videos/{serial}.avi
Output: {path}/obj_mask/{serial}.avi        (binary mask)
        {path}/obj_mask_debug/{serial}.avi  (overlay)

Usage:
    python src/perception/batch_mask_image.py --path .../inspire_f1/baseball/0 --batch-size 8
"""

import os
import sys
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_SAM3_ROOT = Path(__file__).resolve().parents[2] / "autodex/perception/thirdparty/sam3"


def load_video(video_path: str):
    """Read all frames from .avi. Returns (PIL list, BGR list, fps)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], [], 30.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pil_frames, bgr_frames = [], []
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        bgr_frames.append(bgr)
        pil_frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    cap.release()
    return pil_frames, bgr_frames, fps


def save_mask_videos(masks, bgr_frames, save_dir, serial, fps):
    """Write mask + debug overlay videos."""
    t0 = time.time()
    h, w = bgr_frames[0].shape[:2]

    mask_dir = os.path.join(save_dir, "obj_mask")
    debug_dir = os.path.join(save_dir, "obj_mask_debug")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    mask_writer = cv2.VideoWriter(os.path.join(mask_dir, f"{serial}.avi"), fourcc, fps, (w, h), False)
    debug_writer = cv2.VideoWriter(os.path.join(debug_dir, f"{serial}.avi"), fourcc, fps, (w, h), True)

    for i, bgr in enumerate(bgr_frames):
        mask = masks.get(i)
        mask_u8 = mask.astype(np.uint8) * 255 if mask is not None else np.zeros((h, w), dtype=np.uint8)
        mask_writer.write(mask_u8)

        green = np.zeros_like(bgr)
        green[:, :, 1] = mask_u8
        overlay = cv2.addWeighted(bgr, 1.0, green, 0.5, 0)
        debug_writer.write(overlay)

    mask_writer.release()
    debug_writer.release()
    print(f"  [upload] {len(bgr_frames)} frames in {time.time()-t0:.2f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    import torch
    torch.cuda.set_device(args.gpu)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    torch.inference_mode().__enter__()

    capture_dir = Path(args.path)
    video_dir = capture_dir / "videos"
    if not video_dir.is_dir():
        print(f"No videos/ directory in {capture_dir}")
        return

    all_videos = sorted(video_dir.glob("*.avi"))
    if not all_videos:
        print(f"No .avi files in {video_dir}")
        return

    prompt = args.prompt or capture_dir.parent.name

    # Filter already done
    tasks = [(str(vp), vp.stem) for vp in all_videos
             if not (capture_dir / "obj_mask" / f"{vp.stem}.avi").exists()]
    if not tasks:
        print("Nothing to do.")
        return

    print(f"{len(tasks)} videos to process | prompt: '{prompt}' | batch_size: {args.batch_size} | GPU: {args.gpu}")

    # Load SAM3 image model
    sam3_path = str(_SAM3_ROOT)
    if sam3_path not in sys.path:
        sys.path.insert(0, sam3_path)

    from sam3 import build_sam3_image_model
    from sam3.train.data.sam3_image_dataset import InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint
    from sam3.train.data.collator import collate_fn_api as collate
    from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
    from sam3.eval.postprocessors import PostProcessImage
    from sam3.model.utils.misc import copy_data_to_device

    model = build_sam3_image_model(compile=False)
    print("SAM3 image model ready")

    transform = ComposeAPI(transforms=[
        RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
        ToTensorAPI(),
        NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    postprocessor = PostProcessImage(
        max_dets_per_img=-1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        detection_threshold=0.5,
        to_cpu=False,
    )

    global_counter = 0

    def make_datapoint(pil_img, text):
        nonlocal global_counter
        w, h = pil_img.size
        dp = Datapoint(find_queries=[], images=[SAMImage(data=pil_img, objects=[], size=[h, w])])
        dp.find_queries.append(FindQueryLoaded(
            query_text=text,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=global_counter,
                original_image_id=global_counter,
                original_category_id=1,
                original_size=[w, h],
                object_id=0,
                frame_index=0,
            )
        ))
        qid = global_counter
        global_counter += 1
        return dp, qid

    for vi, (video_path, serial) in enumerate(tasks):
        t0 = time.time()
        pil_frames, bgr_frames, fps = load_video(video_path)
        if not pil_frames:
            print(f"[{vi+1}/{len(tasks)}] Skip (empty): {serial}")
            continue
        print(f"  [download] {len(pil_frames)} frames in {time.time()-t0:.2f}s")

        masks = {}
        bs = args.batch_size
        t_proc = time.time()

        for batch_start in range(0, len(pil_frames), bs):
            batch_end = min(batch_start + bs, len(pil_frames))
            datapoints = []
            qids = []
            for fi in range(batch_start, batch_end):
                dp, qid = make_datapoint(pil_frames[fi], prompt)
                dp = transform(dp)
                datapoints.append(dp)
                qids.append((fi, qid))

            batch = collate(datapoints, dict_key="dummy")["dummy"]
            batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)
            output = model(batch)
            results = postprocessor.process_results(output, batch.find_metadatas)

            for fi, qid in qids:
                res = results.get(qid)
                if res is not None and "masks" in res and len(res["masks"]) > 0:
                    # Take highest confidence mask
                    best = res["scores"].argmax()
                    mask = res["masks"][best].squeeze().cpu().numpy()
                    masks[fi] = mask

        print(f"  [process] {len(masks)}/{len(pil_frames)} frames in {time.time()-t_proc:.2f}s")

        if masks:
            save_mask_videos(masks, bgr_frames, str(capture_dir), serial, fps)
            print(f"[{vi+1}/{len(tasks)}] Saved: {capture_dir}/obj_mask/{serial}.avi ({len(masks)}/{len(pil_frames)} frames)")
        else:
            print(f"[{vi+1}/{len(tasks)}] No masks: {capture_dir}/obj_mask/{serial}.avi")

    print("All done!")


if __name__ == "__main__":
    main()
