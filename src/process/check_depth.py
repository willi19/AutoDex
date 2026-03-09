#!/usr/bin/env python3
"""Check depth status for inspire_f1 captures.

Reports:
  - Missing depth files
  - Depth files with wrong frame count (vs min(left, right))

Usage:
    python src/perception/check_depth.py \
        --base /home/mingi/paradex1/capture/eccv2026/inspire_f1 \
        --left_serial 22684755 --right_serial 23263780
"""

import argparse
import os
from pathlib import Path

import cv2

CACHE_ROOT = os.path.expanduser("~/video_cache")
NETWORK_PREFIX = "/home/mingi/paradex1/capture"


def get_frame_count(path):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--left_serial", required=True)
    parser.add_argument("--right_serial", required=True)
    args = parser.parse_args()

    base = str(Path(args.base).resolve())
    if base.startswith(NETWORK_PREFIX):
        rel = base[len(NETWORK_PREFIX):].lstrip("/")
    else:
        rel = Path(args.base).name
    cache_base = Path(CACHE_ROOT) / rel
    net_base = Path(args.base)

    missing = []  # (global_idx, label)
    invalid = []  # (global_idx, label, n_depth, n_expected, n_left, n_right)
    no_video = []  # label
    no_cam_param = []  # label
    ok = []
    global_idx = 0

    for net_obj_dir in sorted(net_base.iterdir()):
        if not net_obj_dir.is_dir():
            continue
        for net_idx_dir in sorted(net_obj_dir.iterdir()):
            if not net_idx_dir.is_dir():
                continue
            label = f"{net_obj_dir.name}/{net_idx_dir.name}"
            if not (net_idx_dir / "cam_param").is_dir():
                no_cam_param.append(label)
                continue
            idx_dir = cache_base / net_obj_dir.name / net_idx_dir.name
            video_dir = idx_dir / "videos"
            left_vid = video_dir / f"{args.left_serial}.avi"
            right_vid = video_dir / f"{args.right_serial}.avi"
            if not left_vid.exists() or not right_vid.exists():
                no_video.append(label)
                continue

            global_idx += 1
            depth_dir = idx_dir / "depth"
            depth_candidates = [
                depth_dir / f"{args.left_serial}.avi",
                depth_dir / f"{args.right_serial}.avi",
            ]
            depth_file = next((f for f in depth_candidates if f.exists() and f.stat().st_size > 0), None)

            if depth_file is None:
                missing.append((global_idx, label))
                continue

            n_left = get_frame_count(left_vid)
            n_right = get_frame_count(right_vid)
            n_expected = min(n_left, n_right)
            n_depth = get_frame_count(depth_file)

            if n_depth != n_expected:
                invalid.append((global_idx, label, n_depth, n_expected, n_left, n_right))
            else:
                ok.append(label)

    total = global_idx
    print(f"\n=== Depth Status ({cache_base}) ===")
    print(f"Total captures with videos: {total}")
    print(f"  OK:        {len(ok)}")
    print(f"  Missing:   {len(missing)}")
    print(f"  Invalid:   {len(invalid)}")
    print(f"  No video:      {len(no_video)}")
    print(f"  No cam_param:  {len(no_cam_param)}")

    if missing:
        print(f"\n--- Missing ({len(missing)}) ---")
        for gidx, label in missing:
            print(f"  #{gidx} {label}  (--last_episodes {total - gidx + 1})")

    if invalid:
        print(f"\n--- Invalid frame count ({len(invalid)}) ---")
        for gidx, label, nd, ne, nl, nr in invalid:
            print(f"  #{gidx} {label}: depth={nd}, expected={ne} (left={nl}, right={nr})  (--last_episodes {total - gidx + 1})")

    if no_video:
        print(f"\n--- No video ({len(no_video)}) ---")
        for label in no_video:
            print(f"  {label}")

    if no_cam_param:
        print(f"\n--- No cam_param ({len(no_cam_param)}) ---")
        for label in no_cam_param:
            print(f"  {label}")

    need = missing + [(g, l, 0, 0, 0, 0) for g, l, *_ in invalid]
    if need:
        min_gidx = min(g for g, *_ in need)
        print(f"\nTo cover all missing+invalid: --last_episodes {total - min_gidx + 1}")


if __name__ == "__main__":
    main()
