#!/usr/bin/env python3
"""Upload mask results from local cache back to network FS.

Copies:
  ~/video_cache/eccv2026/inspire_f1/{obj}/{idx}/obj_mask/{serial}.avi
  -> /home/mingi/paradex1/capture/eccv2026/inspire_f1/{obj}/{idx}/obj_mask/{serial}.avi

Usage:
    python src/perception/upload_results.py --base /home/mingi/paradex1/capture/eccv2026/inspire_f1
"""

import os
import shutil
import time
import argparse
from pathlib import Path

CACHE_ROOT = os.path.expanduser("~/video_cache")
NETWORK_PREFIX = "/home/mingi/paradex1/capture"


def _get_cache_base(base_dir):
    base = str(Path(base_dir).resolve())
    if base.startswith(NETWORK_PREFIX):
        rel = base[len(NETWORK_PREFIX):].lstrip("/")
    else:
        rel = Path(base_dir).name
    return os.path.join(CACHE_ROOT, rel)


def upload_results(base_dir):
    cache_base = Path(_get_cache_base(base_dir))
    base = Path(base_dir)

    if not cache_base.is_dir():
        print(f"Cache not found: {cache_base}")
        return

    print(f"Cache:   {cache_base}", flush=True)
    print(f"Network: {base}", flush=True)

    count = 0
    total_size = 0
    t0 = time.time()

    for obj_dir in sorted(cache_base.iterdir()):
        if not obj_dir.is_dir():
            continue
        for idx_dir in sorted(obj_dir.iterdir()):
            if not idx_dir.is_dir():
                continue

            # Upload obj_mask, obj_mask_debug, depth, and pose
            for subdir in ["obj_mask", "obj_mask_first", "obj_mask_debug", "depth", "pose", "pose_overlay", "pose_overlay_merged"]:
                src_dir = idx_dir / subdir
                if not src_dir.is_dir():
                    continue
                dst_dir = base / obj_dir.name / idx_dir.name / subdir
                os.makedirs(str(dst_dir), exist_ok=True)

                # Recursively copy all files (handles nested dirs like pose/{serial}/)
                for f in sorted(src_dir.rglob("*")):
                    if not f.is_file():
                        continue
                    rel = f.relative_to(src_dir)
                    dst = dst_dir / rel
                    if dst.exists() and dst.stat().st_size == f.stat().st_size:
                        continue
                    os.makedirs(str(dst.parent), exist_ok=True)
                    t1 = time.time()
                    shutil.copy(str(f), str(dst))
                    sz = f.stat().st_size
                    total_size += sz
                    count += 1
                    print(f"  [{count}] {obj_dir.name}/{idx_dir.name}/{subdir}/{rel} ({sz/1e6:.1f} MB, {time.time()-t1:.1f}s)", flush=True)

    dt = time.time() - t0
    print(f"\nDone! {count} files ({total_size / 1e6:.0f} MB) uploaded in {dt:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base dir (network FS)")
    args = parser.parse_args()
    upload_results(args.base)


if __name__ == "__main__":
    main()
