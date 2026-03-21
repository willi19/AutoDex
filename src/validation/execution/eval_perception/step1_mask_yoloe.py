#!/usr/bin/env python3
"""Run YOLOE segmentation on all views, save masks + grid overlay. (conda: foundationpose)"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

AUTODEX_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(AUTODEX_ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="object on the checkerboard")
    args = parser.parse_args()

    data_root = Path(args.data_root)

    from autodex.perception import YoloeSegmentor
    seg = YoloeSegmentor(gpu=0)

    for obj_dir in sorted(data_root.iterdir()):
        if not obj_dir.is_dir() or obj_dir.name in ("cam_param", "simulate"):
            continue
        for ep_dir in sorted(obj_dir.iterdir()):
            if not ep_dir.is_dir():
                continue

            img_dir = ep_dir / "images"
            if not img_dir.exists():
                img_dir = ep_dir / "raw" / "images"
            prompt = args.prompt if args.prompt != "obj_name" else obj_dir.name.replace("_", " ")
            suffix = "yoloe" if args.prompt != "obj_name" else "yoloe_objname"
            mask_dir = ep_dir / f"masks_{suffix}"

            # Skip if already done
            serials = sorted(p.stem for p in img_dir.glob("*.png"))
            existing = [f for f in mask_dir.glob("*.png") if f.name != "grid.png"] if mask_dir.exists() else []
            if len(existing) >= len(serials):
                print(f"  {obj_dir.name}/{ep_dir.name}: skip (already done)")
                continue

            mask_dir.mkdir(exist_ok=True)
            print(f"  {obj_dir.name}/{ep_dir.name}: {len(serials)} views, prompt='{prompt}'")

            overlays = []
            for s in serials:
                rgb = cv2.cvtColor(cv2.imread(str(img_dir / f"{s}.png")), cv2.COLOR_BGR2RGB)
                mask = seg.segment(rgb, prompt)

                if mask is not None:
                    cv2.imwrite(str(mask_dir / f"{s}.png"), mask)
                else:
                    cv2.imwrite(str(mask_dir / f"{s}.png"), np.zeros(rgb.shape[:2], dtype=np.uint8))

                img_bgr = cv2.imread(str(img_dir / f"{s}.png"))
                if mask is not None:
                    m = mask > 127
                    img_bgr[m] = (img_bgr[m].astype(float) * 0.5 + np.array([0, 0, 255], dtype=float) * 0.5).astype(np.uint8)
                cv2.putText(img_bgr, s, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                overlays.append(img_bgr)

            # Grid
            cols = 4
            rows = (len(overlays) + cols - 1) // cols
            scale = 0.25
            oh, ow = overlays[0].shape[:2]
            th, tw = int(oh * scale), int(ow * scale)
            grid = np.ones((rows * th, cols * tw, 3), dtype=np.uint8) * 40
            for idx, img in enumerate(overlays):
                r, c = divmod(idx, cols)
                small = cv2.resize(img, (tw, th))
                grid[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = small
            cv2.imwrite(str(mask_dir / "grid.png"), grid)

    print("Done")


if __name__ == "__main__":
    main()