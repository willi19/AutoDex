#!/usr/bin/env python3
"""Video segmentation (SAM3 or YOLOE).

Supports two modes:
  --capture_dir : process a single episode (all cameras)
  --base        : batch all episodes under a directory (with progress/ETA)

Usage:
    # SAM3 — single episode
    conda activate sam3
    python -u src/process/mask.py \
        --capture_dir /home/mingi/shared_data/RSS2026_Mingi/experiment/selected_100/apple/20260206_181110 \
        --prompt "object on the checkerboard"

    # YOLOE — single episode
    conda activate foundationpose
    python -u src/process/mask.py --method yoloe \
        --capture_dir /path/to/episode --conf 0.2

    # SAM3 — batch all episodes
    conda activate sam3
    python -u src/process/mask.py \
        --base /home/mingi/shared_data/RSS2026_Mingi/experiment/selected_100

    # Batch with sharding (multi-GPU)
    python -u src/process/mask.py --base ... --shard 0/3 --gpu 0
    python -u src/process/mask.py --base ... --shard 1/3 --gpu 1

    # Filter to specific objects
    python -u src/process/mask.py --base ... --objects apple banana
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

AUTODEX_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AUTODEX_ROOT))


# ── Episode discovery ────────────────────────────────────────────────────────

def _find_episodes(base, objects=None):
    """Find all {obj}/{idx} dirs with videos/."""
    base = Path(base)
    episodes = []
    for obj_dir in sorted(base.iterdir()):
        if not obj_dir.is_dir():
            continue
        if objects and obj_dir.name not in objects:
            continue
        for idx_dir in sorted(obj_dir.iterdir()):
            if not idx_dir.is_dir():
                continue
            if (idx_dir / "videos").is_dir():
                episodes.append(idx_dir)
    return episodes


def _is_done(capture_dir):
    """Check if all cameras have mask videos."""
    video_dir = capture_dir / "videos"
    mask_dir = capture_dir / "obj_mask"
    if not mask_dir.exists():
        return False
    video_serials = {p.stem for p in video_dir.glob("*.avi")}
    mask_serials = {p.stem for p in mask_dir.glob("*.avi") if p.stat().st_size > 0}
    return video_serials == mask_serials


# ── Process one episode ──────────────────────────────────────────────────────

def process_episode_sam3(seg, capture_dir, prompt, serials=None):
    """Run SAM3 on all cameras in one episode. Returns (done, failed, total)."""
    from autodex.perception import save_mask_video

    capture_dir = Path(capture_dir)
    video_dir = capture_dir / "videos"
    all_serials = sorted(p.stem for p in video_dir.glob("*.avi"))
    if serials:
        all_serials = [s for s in all_serials if s in set(serials)]

    done = 0
    failed = 0
    skipped = 0

    for cam_idx, serial in enumerate(all_serials):
        mask_path = capture_dir / "obj_mask" / f"{serial}.avi"
        if mask_path.exists() and mask_path.stat().st_size > 0:
            skipped += 1
            continue

        print(f"  cam [{cam_idx+1}/{len(all_serials)}] {serial}.avi", flush=True)
        video_path = str(video_dir / f"{serial}.avi")
        t0 = time.perf_counter()

        masks = seg.segment_video(
            video_path, prompt,
            fallback_prompts=["object"],
        )
        dt = time.perf_counter() - t0

        if masks is None:
            print(f"    {serial}: FAILED ({dt:.1f}s)", flush=True)
            failed += 1
            continue

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        save_mask_video(masks, video_path, str(capture_dir), serial, fps, save_debug=True)
        done += 1
        print(f"    {serial}: {len(masks)} frames ({dt:.1f}s)", flush=True)

    return done + skipped, failed, len(all_serials)


def process_episode_yoloe(seg, capture_dir, prompt, batch_size=50, skip=3, serials=None):
    """Run YOLOE on all cameras in one episode. Returns (done, failed, total)."""
    from autodex.perception import save_mask_video

    capture_dir = Path(capture_dir)
    video_dir = capture_dir / "videos"
    all_serials = sorted(p.stem for p in video_dir.glob("*.avi"))
    if serials:
        all_serials = [s for s in all_serials if s in set(serials)]

    done = 0
    failed = 0
    skipped = 0

    for cam_idx, serial in enumerate(all_serials):
        mask_path = capture_dir / "obj_mask" / f"{serial}.avi"
        if mask_path.exists() and mask_path.stat().st_size > 0:
            skipped += 1
            continue

        print(f"  cam [{cam_idx+1}/{len(all_serials)}] {serial}.avi", flush=True)
        video_path = str(video_dir / f"{serial}.avi")
        t0 = time.perf_counter()

        masks = seg.segment_video(
            video_path, prompt, batch_size=batch_size, skip=skip,
        )
        dt = time.perf_counter() - t0

        if masks is None:
            print(f"    {serial}: FAILED ({dt:.1f}s)", flush=True)
            failed += 1
            continue

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        save_mask_video(masks, video_path, str(capture_dir), serial, fps, save_debug=True)
        done += 1
        print(f"    {serial}: {len(masks)} frames ({dt:.1f}s)", flush=True)

    return done + skipped, failed, len(all_serials)


# ── Format time ──────────────────────────────────────────────────────────────

def _format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h{m:02d}m"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Video segmentation (SAM3 or YOLOE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
modes:
  --capture_dir DIR    Process a single episode (all cameras)
  --base DIR           Batch all episodes under DIR (with progress/ETA)

examples:
  %(prog)s --capture_dir /path/to/apple/20260206_181110
  %(prog)s --method yoloe --capture_dir /path/to/episode --conf 0.2
  %(prog)s --base /path/to/selected_100 --shard 0/3 --gpu 0
  %(prog)s --base /path/to/selected_100 --objects apple banana
""",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--capture_dir", type=str, help="Single episode directory")
    group.add_argument("--base", type=str, help="Batch: parent of all episodes")

    parser.add_argument("--method", choices=["sam3", "yoloe"], default="sam3")
    parser.add_argument("--prompt", type=str, default="object on the checkerboard")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--serials", nargs="+", default=None,
                        help="Only process these camera serials")

    # Batch-mode options
    parser.add_argument("--objects", nargs="*", default=None,
                        help="Only process these object names (batch mode)")
    parser.add_argument("--shard", type=str, default=None,
                        help="Shard spec: RANK/TOTAL (e.g. 0/3)")

    # YOLOE-specific
    parser.add_argument("--conf", type=float, default=0.2,
                        help="YOLOE confidence threshold (default: 0.2)")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="YOLOE batch size (default: 50)")
    parser.add_argument("--skip", type=int, default=3,
                        help="YOLOE frame skip (default: 3)")
    args = parser.parse_args()

    # Load segmentor
    if args.method == "sam3":
        from autodex.perception import Sam3Segmentor
        print(f"Loading SAM3 on GPU {args.gpu}...", flush=True)
        seg = Sam3Segmentor(gpu=args.gpu)
    else:
        from autodex.perception import YoloeSegmentor
        print(f"Loading YOLOE on GPU {args.gpu} (conf={args.conf})...", flush=True)
        seg = YoloeSegmentor(gpu=args.gpu, conf_thr=args.conf)
    print("Segmentor ready.", flush=True)

    def _run_episode(capture_dir):
        if args.method == "sam3":
            return process_episode_sam3(seg, capture_dir, args.prompt, serials=args.serials)
        else:
            return process_episode_yoloe(seg, capture_dir, args.prompt,
                                         batch_size=args.batch_size, skip=args.skip,
                                         serials=args.serials)

    if args.capture_dir:
        # Single episode mode
        done, failed, total = _run_episode(args.capture_dir)
        print(f"\nDone! {done}/{total} cameras, {failed} failed.", flush=True)
    else:
        # Batch mode
        episodes = _find_episodes(args.base, args.objects)

        # Sharding
        if args.shard:
            rank, n_shards = map(int, args.shard.split("/"))
            episodes = [e for i, e in enumerate(episodes) if i % n_shards == rank]
            print(f"Shard {rank}/{n_shards}: {len(episodes)} episodes", flush=True)

        todo = [e for e in episodes if not _is_done(e)]
        print(f"Total: {len(episodes)} episodes, {len(episodes) - len(todo)} done, "
              f"{len(todo)} to process", flush=True)

        if not todo:
            print("Nothing to do.")
            return

        total_done = 0
        total_failed = 0
        t_start = time.perf_counter()
        base = Path(args.base)

        for i, capture_dir in enumerate(todo):
            rel = capture_dir.relative_to(base)
            elapsed = time.perf_counter() - t_start

            if i > 0:
                avg = elapsed / i
                remaining = avg * (len(todo) - i)
                eta_str = f"ETA {_format_time(remaining)}"
            else:
                eta_str = "ETA --"

            # Re-check in case another process finished it
            if _is_done(capture_dir):
                print(f"[{i+1}/{len(todo)}] {rel} SKIP (done)", flush=True)
                continue

            print(f"\n[{i+1}/{len(todo)}] {rel}  "
                  f"(elapsed {_format_time(elapsed)}, {eta_str})", flush=True)

            try:
                done, failed, total_cams = _run_episode(capture_dir)
                total_done += done
                total_failed += failed
                print(f"  {done}/{total_cams} cameras, {failed} failed", flush=True)
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
                import traceback
                traceback.print_exc()

        total_elapsed = time.perf_counter() - t_start
        print(f"\nAll done! {total_done} cameras processed, {total_failed} failed, "
              f"{_format_time(total_elapsed)} total", flush=True)


if __name__ == "__main__":
    main()
