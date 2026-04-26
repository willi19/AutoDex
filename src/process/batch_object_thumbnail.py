"""Batch-render pregrasp-moment thumbnails per episode.

Mirrors src/process/batch_object_overlay.py but:
  - No GoTrack tracking (uses pose_world.npy init pose directly)
  - Single pregrasp frame per camera → 24 PNGs + 1 grid PNG per episode

Per episode:
  foundationpose env → overlay_object_thumbnail_single.py

Usage:
    python src/process/batch_object_thumbnail.py --hand inspire
    python src/process/batch_object_thumbnail.py --hand allegro --obj attached_container
    python src/process/batch_object_thumbnail.py --hand inspire --ep 20260405_235218
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from tqdm import tqdm


EXP_BASE = Path.home() / "shared_data" / "AutoDex" / "experiment" / "selected_100"
OUTPUT_BASE = Path.home() / "shared_data" / "AutoDex" / "object_overlay_thumbnail"
LOCAL_CACHE = Path.home() / "cache" / "batch_object_thumbnail"
OBJ_BASE = Path.home() / "shared_data" / "AutoDex" / "object" / "paradex"
FPOSE_PY = Path.home() / "miniconda3" / "envs" / "foundationpose" / "bin" / "python"
THUMB_SCRIPT = Path.home() / "AutoDex" / "src" / "visualization" / "overlay_object_thumbnail_single.py"


# ── Skip / cache ─────────────────────────────────────────────────────────────

def thumbs_done(nas_out_dir, serials):
    if not (nas_out_dir / "thumb_grid.png").exists():
        return False
    for s in serials:
        if not (nas_out_dir / f"thumb_{s}.png").exists():
            return False
    return True


def _is_videos_cached(local_videos_dir):
    if not local_videos_dir.is_dir():
        return False
    return any(local_videos_dir.glob("*.avi"))


def download_episode(nas_ep, local_ep):
    local_videos = local_ep / "videos"
    if _is_videos_cached(local_videos):
        return
    if local_videos.exists():
        shutil.rmtree(local_videos)
    local_ep.mkdir(parents=True, exist_ok=True)
    nas_videos = nas_ep / "videos"
    if nas_videos.is_dir():
        shutil.copytree(str(nas_videos), str(local_videos))


def upload_dir(local_dir, nas_dir):
    if not local_dir.exists():
        return
    nas_dir.mkdir(parents=True, exist_ok=True)
    for item in local_dir.rglob("*"):
        if item.is_dir():
            continue
        rel = item.relative_to(local_dir)
        dst = nas_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(str(item), str(dst))


# ── Thumbnail rendering ──────────────────────────────────────────────────────

THUMB_PROGRESS_RE = re.compile(r"\[thumb_progress\]\s+(\d+)/(\d+)")


def run_thumbnail(local_ep, nas_ep, obj_name, local_out, frame_pbar=None):
    if not (local_ep / "videos").is_dir():
        return False
    if not (nas_ep / "pose_world.npy").exists():
        print(f"  [skip] no pose_world.npy: {nas_ep.name}", flush=True)
        return False
    if not (nas_ep / "result.json").exists():
        print(f"  [skip] no result.json: {nas_ep.name}", flush=True)
        return False
    ts_path = nas_ep / "raw" / "timestamps" / "timestamp.npy"
    if not ts_path.exists():
        print(f"  [skip] no raw/timestamps/timestamp.npy: {nas_ep.name}", flush=True)
        return False

    local_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(FPOSE_PY), str(THUMB_SCRIPT),
        "--videos_dir", str(local_ep / "videos"),
        "--cam_param_dir", str(nas_ep / "cam_param"),
        "--pose_world", str(nas_ep / "pose_world.npy"),
        "--mesh", str(OBJ_BASE / obj_name / "raw_mesh" / f"{obj_name}.obj"),
        "--result_json", str(nas_ep / "result.json"),
        "--timestamps", str(ts_path),
        "--output_dir", str(local_out),
    ]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    t0 = time.time()
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    last_n = 0
    for line in proc.stdout:
        m = THUMB_PROGRESS_RE.search(line)
        if m:
            n_done, total = int(m.group(1)), int(m.group(2))
            if frame_pbar is not None:
                if frame_pbar.total != total:
                    frame_pbar.reset(total=total)
                    last_n = 0
                frame_pbar.update(n_done - last_n)
                last_n = n_done
        elif line.strip():
            print(line.rstrip(), flush=True)
    proc.wait()
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"  [fail] thumbnail rc={proc.returncode}: {nas_ep.name}", flush=True)
        return False
    print(f"  [thumb] {nas_ep.name}: {elapsed:.1f}s", flush=True)
    return True


# ── Discovery ────────────────────────────────────────────────────────────────

def discover_work(hand, objects, ep_filter):
    hand_dir = EXP_BASE / hand
    if not hand_dir.exists():
        return []
    if objects is None:
        objects = sorted(d.name for d in hand_dir.iterdir() if d.is_dir())
    work = []
    for obj in objects:
        obj_dir = hand_dir / obj
        if not obj_dir.is_dir():
            continue
        eps = []
        for ep in sorted(obj_dir.iterdir()):
            if not ep.is_dir():
                continue
            if ep_filter and ep.name not in ep_filter:
                continue
            if not (ep / "videos").is_dir():
                continue
            if not (ep / "pose_world.npy").exists():
                continue
            if not (ep / "result.json").exists():
                continue
            if not (ep / "raw" / "timestamps" / "timestamp.npy").exists():
                continue
            serials = sorted(p.stem for p in (ep / "videos").glob("*.avi"))
            if serials:
                eps.append((ep.name, serials))
        if eps:
            work.append((obj, eps))
    return work


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hand", required=True, choices=["allegro", "inspire"])
    p.add_argument("--obj", nargs="+", default=None)
    p.add_argument("--ep", nargs="+", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    work = discover_work(args.hand, args.obj, args.ep)
    if not work:
        print("No work to do.")
        return
    flat = [(obj, ep, serials) for obj, eps in work for ep, serials in eps]
    print(f"=== {len(work)} objects, {len(flat)} episodes ===", flush=True)
    if args.dry_run:
        for obj, eps in work:
            print(f"  {obj}: {len(eps)} eps")
        return

    ep_pbar = tqdm(total=len(flat), desc="episodes", unit="ep", position=0, dynamic_ncols=True)
    frame_pbar = tqdm(desc="cams", unit="c", position=1, dynamic_ncols=True)

    upload_thread = None
    prefetch_thread = None

    def needs_work(obj, ep, serials):
        nas_out = OUTPUT_BASE / args.hand / obj / ep
        return not thumbs_done(nas_out, serials)

    for obj, ep, serials in flat:
        if needs_work(obj, ep, serials):
            print(f"Downloading {obj}/{ep}...", flush=True)
            download_episode(EXP_BASE / args.hand / obj / ep,
                             LOCAL_CACHE / args.hand / obj / ep)
            break

    for ei, (obj, ep, serials) in enumerate(flat):
        nas_ep = EXP_BASE / args.hand / obj / ep
        nas_out = OUTPUT_BASE / args.hand / obj / ep

        if thumbs_done(nas_out, serials):
            ep_pbar.update(1)
            ep_pbar.set_postfix_str(f"{obj}/{ep} skip")
            continue

        local_ep = LOCAL_CACHE / args.hand / obj / ep
        local_out = LOCAL_CACHE / "thumb_output" / args.hand / obj / ep

        if prefetch_thread is not None:
            prefetch_thread.join()
            prefetch_thread = None
        if not _is_videos_cached(local_ep / "videos"):
            print(f"Downloading {obj}/{ep}...", flush=True)
            download_episode(nas_ep, local_ep)

        for ni in range(ei + 1, len(flat)):
            n_obj, n_ep, n_ser = flat[ni]
            if not needs_work(n_obj, n_ep, n_ser):
                continue
            n_nas = EXP_BASE / args.hand / n_obj / n_ep
            n_local = LOCAL_CACHE / args.hand / n_obj / n_ep
            if not _is_videos_cached(n_local / "videos"):
                prefetch_thread = threading.Thread(
                    target=download_episode, args=(n_nas, n_local), daemon=True)
                prefetch_thread.start()
            break

        ep_pbar.set_postfix_str(f"{obj}/{ep}")
        frame_pbar.reset()
        frame_pbar.set_postfix_str(f"{obj}/{ep}")
        ok = run_thumbnail(local_ep, nas_ep, obj, local_out, frame_pbar=frame_pbar)
        if ok:
            if upload_thread is not None:
                upload_thread.join()
            upload_thread = threading.Thread(
                target=lambda lo=local_out, no=nas_out: (
                    upload_dir(lo, no),
                    shutil.rmtree(lo, ignore_errors=True)),
                daemon=True)
            upload_thread.start()

        ep_pbar.update(1)
        if local_ep.exists():
            shutil.rmtree(local_ep)

    if upload_thread is not None:
        upload_thread.join()
    ep_pbar.close()
    frame_pbar.close()
    print(f"Output: {OUTPUT_BASE / args.hand}", flush=True)


if __name__ == "__main__":
    main()
