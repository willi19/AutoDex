"""End-to-end batch: GoTrack tracking + mesh overlay videos per episode.

Mirrors src/visualization/overlay_robot_video.py:
  - Discover work across hand/obj/episode
  - Skip episodes whose final overlay outputs already exist on NAS
  - Cache videos locally, prefetch next, upload current in background
  - Episode + frame tqdm bars
  - Subprocesses both stages in their own conda envs

Per episode:
  Phase 1 (env: gotrack)         GoTrack tracking → world_pose_records.json
  Phase 2 (env: foundationpose)  Render mesh overlay → overlay_{serial}.mp4

Usage:
    python src/process/batch_object_overlay.py --hand inspire
    python src/process/batch_object_overlay.py --hand inspire --obj attached_container
    python src/process/batch_object_overlay.py --hand allegro --ep 20260405_073417
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm


EXP_BASE = Path.home() / "shared_data" / "AutoDex" / "experiment" / "selected_100"
OUTPUT_BASE = Path.home() / "shared_data" / "AutoDex" / "object_overlay_video"
LOCAL_CACHE = Path.home() / "cache" / "batch_object_overlay"
OBJ_BASE = Path.home() / "shared_data" / "AutoDex" / "object" / "paradex"
GOTRACK_ROOT = Path.home() / "AutoDex" / "autodex" / "perception" / "thirdparty" / "MV-GoTrack"
GOTRACK_PY = Path.home() / "miniconda3" / "envs" / "gotrack" / "bin" / "python"
FPOSE_PY = Path.home() / "miniconda3" / "envs" / "foundationpose" / "bin" / "python"
ANCHOR_DIR = GOTRACK_ROOT / "anchor_banks"
OVERLAY_SCRIPT = Path.home() / "AutoDex" / "src" / "visualization" / "overlay_object_video_single.py"

GOTRACK_REL = "object_tracking/gotrack_output"
DEFAULT_TRACK_CAMS = None  # None → use ALL available cameras in the episode
NUM_ANCHORS = 256


# ── Skip / cache ─────────────────────────────────────────────────────────────

def gotrack_done(nas_ep):
    rec = nas_ep / GOTRACK_REL / "world_pose_records.json"
    if not rec.exists():
        return False
    try:
        recs = json.load(open(rec))
        return any(r.get("pose_world") is not None for r in recs)
    except Exception:
        return False


def overlay_done(nas_out_dir, serials):
    for s in serials:
        if not (nas_out_dir / f"overlay_{s}.mp4").exists():
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


# ── Anchor bank ──────────────────────────────────────────────────────────────

def ensure_anchor_bank(obj_name, num_anchors=NUM_ANCHORS):
    bank = ANCHOR_DIR / f"{obj_name}.npz"
    if bank.exists():
        return bank
    mesh = OBJ_BASE / obj_name / "raw_mesh" / f"{obj_name}.obj"
    assert mesh.exists(), f"mesh not found: {mesh}"
    ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  generating anchor bank for {obj_name}...", flush=True)
    subprocess.run(
        [str(GOTRACK_PY), "scripts/generate_anchor_bank.py",
         "--mesh-path", str(mesh), "--output-path", str(bank),
         "--num-anchors", str(num_anchors)],
        check=True, cwd=str(GOTRACK_ROOT))
    return bank


def write_init_pose_jsons(videos_dir, pose_world_npy, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    pose = np.load(pose_world_npy)
    serials = sorted(p.stem for p in videos_dir.glob("*.avi"))
    for s in serials:
        rec = [{
            "frame_index": 0,
            "pose_world": pose.tolist(),
            "certainty_count_above_threshold": 1000,
            "status": "ok",
        }]
        with open(out_dir / f"{s}.json", "w") as f:
            json.dump(rec, f)
    return serials


# ── Phase 1: GoTrack ─────────────────────────────────────────────────────────

GOTRACK_PROGRESS_RE = re.compile(r"gotrack_anchor_mv:\s*\d+%\|[^|]*\|\s*(\d+)/(\d+)")


def run_gotrack(local_ep, nas_ep, obj_name, local_gt_out, track_cams, frame_pbar=None):
    pose_npy = nas_ep / "pose_world.npy"
    if not pose_npy.exists():
        print(f"  [skip] no pose_world.npy: {nas_ep.name}", flush=True)
        return False
    if not (nas_ep / "cam_param" / "intrinsics.json").exists():
        print(f"  [skip] no cam_param: {nas_ep.name}", flush=True)
        return False
    if not (local_ep / "videos").is_dir():
        print(f"  [skip] no local videos: {nas_ep.name}", flush=True)
        return False

    bank = ensure_anchor_bank(obj_name)

    stage = local_ep / "gotrack_input"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    (stage / "videos").symlink_to((local_ep / "videos").resolve(), target_is_directory=True)
    (stage / "cam_param").symlink_to((nas_ep / "cam_param").resolve(), target_is_directory=True)
    shutil.copy2(str(pose_npy), str(stage / "pose_world.npy"))

    init_dir = stage / "gotrack_init_poses"
    available = write_init_pose_jsons(local_ep / "videos", pose_npy, init_dir)
    if track_cams:
        use_cams = [c for c in track_cams if c in available]
    elif DEFAULT_TRACK_CAMS:
        use_cams = [c for c in DEFAULT_TRACK_CAMS if c in available]
    else:
        use_cams = available
    if not use_cams:
        print(f"  [skip] no track cameras available: {nas_ep.name}", flush=True)
        return False

    out_dir = local_gt_out
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(GOTRACK_PY), "run_multiview_gotrack_anchor_online.py",
        "--gpus", "0",
        "--input-root", str(stage),
        "--output-dir", str(out_dir),
        "--mesh-path", str(OBJ_BASE / obj_name / "raw_mesh" / f"{obj_name}.obj"),
        "--object-id", "1",
        "--checkpoint-path", str(GOTRACK_ROOT / "gotrack_checkpoint.pt"),
        "--init-pose-source", str(init_dir),
        "--camera-ids", *use_cams,
        "--anchor-bank-path", str(bank),
        "--num-anchors", str(NUM_ANCHORS),
        "--mask-free",
        "--unit-scale-mode", "meters_to_mm",
        "--disable-all-view-post-visualization",
    ]
    env = dict(os.environ)
    env["GOTRACK_RENDER_BACKEND"] = "pyglet"
    env["PYTHONUNBUFFERED"] = "1"

    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=str(GOTRACK_ROOT), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    last_n = 0
    for line in proc.stdout:
        m = GOTRACK_PROGRESS_RE.search(line)
        if m:
            n_done, total = int(m.group(1)), int(m.group(2))
            if frame_pbar is not None:
                if frame_pbar.total != total:
                    frame_pbar.reset(total=total)
                    last_n = 0
                frame_pbar.update(n_done - last_n)
                last_n = n_done
        elif line.strip() and not line.lstrip().startswith("gotrack_anchor_mv:"):
            print(line.rstrip(), flush=True)
    proc.wait()
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"  [fail] gotrack rc={proc.returncode}: {nas_ep.name}", flush=True)
        return False
    print(f"  [gotrack] {nas_ep.name}: {len(use_cams)}cams, {elapsed:.1f}s", flush=True)
    return True


# ── Phase 2: overlay rendering ───────────────────────────────────────────────

OVERLAY_PROGRESS_RE = re.compile(r"\[overlay_progress\]\s+(\d+)/(\d+)")


def run_overlay(local_ep, nas_ep, obj_name, local_overlay_out, frame_pbar=None):
    if not (local_ep / "videos").is_dir():
        return False
    if not (nas_ep / GOTRACK_REL / "world_pose_records.json").exists():
        print(f"  [skip overlay] no gotrack output: {nas_ep.name}", flush=True)
        return False

    local_overlay_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(FPOSE_PY), str(OVERLAY_SCRIPT),
        "--videos_dir", str(local_ep / "videos"),
        "--cam_param_dir", str(nas_ep / "cam_param"),
        "--gotrack_records", str(nas_ep / GOTRACK_REL / "world_pose_records.json"),
        "--mesh", str(OBJ_BASE / obj_name / "raw_mesh" / f"{obj_name}.obj"),
        "--output_dir", str(local_overlay_out),
    ]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    t0 = time.time()
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    last_n = 0
    for line in proc.stdout:
        m = OVERLAY_PROGRESS_RE.search(line)
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
        print(f"  [fail] overlay rc={proc.returncode}: {nas_ep.name}", flush=True)
        return False
    print(f"  [overlay] {nas_ep.name}: {elapsed:.1f}s", flush=True)
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
    p.add_argument("--track-cams", nargs="+", default=None)
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
    frame_pbar = tqdm(desc="frames", unit="f", position=1, dynamic_ncols=True)

    upload_thread = None
    prefetch_thread = None

    def needs_work(obj, ep, serials):
        nas_ep = EXP_BASE / args.hand / obj / ep
        nas_overlay_out = OUTPUT_BASE / args.hand / obj / ep
        return not (gotrack_done(nas_ep) and overlay_done(nas_overlay_out, serials))

    # Prefetch first
    for obj, ep, serials in flat:
        if needs_work(obj, ep, serials):
            print(f"Downloading {obj}/{ep}...", flush=True)
            download_episode(EXP_BASE / args.hand / obj / ep,
                             LOCAL_CACHE / args.hand / obj / ep)
            break

    for ei, (obj, ep, serials) in enumerate(flat):
        nas_ep = EXP_BASE / args.hand / obj / ep
        nas_overlay_out = OUTPUT_BASE / args.hand / obj / ep
        nas_gt_out = nas_ep / GOTRACK_REL

        do_gt = not gotrack_done(nas_ep)
        do_ov = not overlay_done(nas_overlay_out, serials)

        if not (do_gt or do_ov):
            ep_pbar.update(1)
            ep_pbar.set_postfix_str(f"{obj}/{ep} skip")
            continue

        local_ep = LOCAL_CACHE / args.hand / obj / ep
        local_gt_out = LOCAL_CACHE / "gt_output" / args.hand / obj / ep / "gotrack_output"
        local_overlay_out = LOCAL_CACHE / "overlay_output" / args.hand / obj / ep

        if prefetch_thread is not None:
            prefetch_thread.join()
            prefetch_thread = None
        if not _is_videos_cached(local_ep / "videos"):
            print(f"Downloading {obj}/{ep}...", flush=True)
            download_episode(nas_ep, local_ep)

        # Prefetch next
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

        # Phase 1: GoTrack
        if do_gt:
            frame_pbar.reset()
            frame_pbar.set_postfix_str(f"{obj}/{ep} gotrack")
            ok = run_gotrack(local_ep, nas_ep, obj, local_gt_out,
                              track_cams=args.track_cams, frame_pbar=frame_pbar)
            if ok:
                upload_dir(local_gt_out, nas_gt_out)
                shutil.rmtree(local_gt_out, ignore_errors=True)

        # Phase 2: overlay
        if do_ov and gotrack_done(nas_ep):
            frame_pbar.reset()
            frame_pbar.set_postfix_str(f"{obj}/{ep} overlay")
            ok = run_overlay(local_ep, nas_ep, obj, local_overlay_out, frame_pbar=frame_pbar)
            if ok:
                if upload_thread is not None:
                    upload_thread.join()
                upload_thread = threading.Thread(
                    target=lambda: (upload_dir(local_overlay_out, nas_overlay_out),
                                    shutil.rmtree(local_overlay_out, ignore_errors=True)),
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