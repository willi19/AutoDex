#!/usr/bin/env python3
"""Check obj_mask status for captures.

Reports per-object: X/N episodes complete, and per-episode: M/S serials done.

Usage:
    python src/perception/check_mask.py \
        --base /home/mingi/paradex1/capture/eccv2026/inspire_f1
    python src/perception/check_mask.py \
        --base /home/mingi/paradex1/capture/eccv2026/inspire_f1 \
        --serials 22684755 23263780
"""

import argparse
import os
from pathlib import Path

CACHE_ROOT = os.path.expanduser("~/video_cache")
NETWORK_PREFIX = "/home/mingi/paradex1/capture"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--serials", nargs="+", default=None)
    args = parser.parse_args()

    base = str(Path(args.base).resolve())
    if base.startswith(NETWORK_PREFIX):
        rel = base[len(NETWORK_PREFIX):].lstrip("/")
    else:
        rel = Path(args.base).name
    cache_base = Path(CACHE_ROOT) / rel
    net_base = Path(args.base)

    if args.serials:
        serials = args.serials
    else:
        serials = sorted({
            p.stem
            for p in cache_base.rglob("videos/*.avi")
        })

    n_serials = len(serials)

    # Collect episode-level data: {obj: [(idx_name, n_ok, missing_serials, status)]}
    # status: "ok" | "partial" | "no_video" | "no_cam_param"
    obj_data = {}  # obj_name -> list of (idx_name, n_ok, missing_serials, status)
    total_captures = 0
    total_ok = 0
    total_partial = 0
    total_no_video = 0
    total_no_cam_param = 0

    for net_obj_dir in sorted(net_base.iterdir()):
        if not net_obj_dir.is_dir():
            continue
        episodes = []
        for net_idx_dir in sorted(net_obj_dir.iterdir()):
            if not net_idx_dir.is_dir():
                continue
            total_captures += 1
            has_cam_param = (net_idx_dir / "cam_param").is_dir()
            idx_dir = cache_base / net_obj_dir.name / net_idx_dir.name

            if not has_cam_param:
                total_no_cam_param += 1
                episodes.append((net_idx_dir.name, 0, [], "no_cam_param"))
                continue

            # Check how many serials have video
            serials_with_video = [
                s for s in serials
                if (idx_dir / "videos" / f"{s}.avi").exists()
            ]
            if not serials_with_video:
                total_no_video += 1
                episodes.append((net_idx_dir.name, 0, [], "no_video"))
                continue

            # Check masks
            missing = []
            for serial in serials_with_video:
                mask_in_cache = (idx_dir / "obj_mask" / f"{serial}.avi").exists()
                mask_on_net = (net_idx_dir / "obj_mask" / f"{serial}.avi").exists()
                if not mask_in_cache and not mask_on_net:
                    missing.append(serial)

            n_ok = len(serials_with_video) - len(missing)
            n_total = len(serials_with_video)
            if not missing:
                total_ok += 1
                episodes.append((net_idx_dir.name, n_ok, [], "ok"))
            else:
                total_partial += 1
                episodes.append((net_idx_dir.name, n_ok, missing, "partial"))

        if episodes:
            obj_data[net_obj_dir.name] = episodes

    print(f"\n=== Mask Status ({cache_base}) ===")
    print(f"Serials: {' '.join(serials)}")
    print(f"Total captures: {total_captures}  |  OK: {total_ok}  Partial/Missing: {total_partial}  No video: {total_no_video}  No cam_param: {total_no_cam_param}")

    for obj_name, episodes in obj_data.items():
        n_ep = len(episodes)
        n_ep_ok = sum(1 for _, _, _, st in episodes if st == "ok")
        # Show object summary line
        print(f"\n  {obj_name}: {n_ep_ok}/{n_ep} episodes")
        for idx_name, n_ok, missing, status in episodes:
            if status == "ok":
                continue
            elif status == "no_cam_param":
                print(f"    {idx_name}: no cam_param")
            elif status == "no_video":
                print(f"    {idx_name}: no video")
            else:
                print(f"    {idx_name}: {n_ok}/{n_ok + len(missing)}  missing: {' '.join(missing)}")


if __name__ == "__main__":
    main()
