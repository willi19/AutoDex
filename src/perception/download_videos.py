#!/usr/bin/env python3
"""Download stereo videos from network FS to local cache.

Mirrors path structure:
  /home/mingi/paradex1/capture/eccv2026/inspire_f1/{obj}/{idx}/videos/{serial}.avi
  -> ~/video_cache/eccv2026/inspire_f1/{obj}/{idx}/{serial}.avi

Only downloads videos that don't already have obj_mask results.

Usage:
    python src/perception/download_videos.py --base /home/mingi/paradex1/capture/eccv2026/inspire_f1 --serials 22684755 23263780
"""

import os
import shutil
import time
import argparse
from pathlib import Path

CACHE_ROOT = os.path.expanduser("~/video_cache")
# Strip this prefix from the base path to get relative cache path
NETWORK_PREFIX = "/home/mingi/paradex1/capture"


def get_cache_base(base_dir, cache_root=CACHE_ROOT):
    """Map network path to local cache path, preserving eccv2026/... structure."""
    base = str(Path(base_dir).resolve())
    if base.startswith(NETWORK_PREFIX):
        rel = base[len(NETWORK_PREFIX):].lstrip("/")
    else:
        rel = Path(base_dir).name
    return os.path.join(cache_root, rel)


def download_videos(base_dir, serial_filter=None, cache_root=CACHE_ROOT):
    base = Path(base_dir)
    cache_base = get_cache_base(base_dir, cache_root)
    total_size = 0
    count = 0
    skipped = 0

    print(f"Scanning {base}...", flush=True)
    print(f"Cache:   {cache_base}", flush=True)
    t0 = time.time()

    for obj_dir in sorted(base.iterdir()):
        if not obj_dir.is_dir():
            continue
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
                if (idx_dir / "obj_mask" / f"{serial}.avi").exists():
                    skipped += 1
                    continue

                local_dir = os.path.join(cache_base, obj_dir.name, idx_dir.name, "videos")
                local_path = os.path.join(local_dir, f"{serial}.avi")

                if os.path.exists(local_path):
                    count += 1
                    continue

                os.makedirs(local_dir, exist_ok=True)
                t1 = time.time()
                tmp_path = local_path + ".tmp"
                with open(str(vp), 'rb') as src, open(tmp_path, 'wb') as dst:
                    shutil.copyfileobj(src, dst)
                os.rename(tmp_path, local_path)
                sz = os.path.getsize(local_path)
                total_size += sz
                count += 1
                print(f"  [{count}] {obj_dir.name}/{idx_dir.name}/{serial}.avi ({sz/1e6:.1f} MB, {time.time()-t1:.1f}s)", flush=True)

    dt = time.time() - t0
    print(f"\nDone! {count} videos ({total_size / 1e6:.0f} MB) in {dt:.1f}s | {skipped} skipped (already have masks)", flush=True)
    print(f"Cache: {cache_base}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base dir e.g. .../inspire_f1")
    parser.add_argument("--serials", nargs="+", default=None, help="Only download these camera serials")
    args = parser.parse_args()

    serial_filter = set(args.serials) if args.serials else None
    download_videos(args.base, serial_filter)


if __name__ == "__main__":
    main()
