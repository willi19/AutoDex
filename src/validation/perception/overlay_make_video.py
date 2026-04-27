#!/usr/bin/env python3
"""Read fp_pre.npz + ref_pre.npz per ep and assemble 144s video alternating fp/ref per source cam.

Layout per frame:
  - 6 col × 4 row grid of 24 cam overlays (mesh rendered at given pose)
  - source cam cell highlighted with thick blue border (16px)
  - per-cell label: cam serial
  - bottom banner: "src=SERIAL / fp_pre" (or ref_pre)

Order: source cams alphabetical. For each source cam:
  3s fp_pre frame (green mesh) + 3s ref_pre frame (magenta mesh) = 6s
24 source cams × 6s = 144s per video.

Output: outputs/foundpose_overlay/{obj}/{ep}/video.mp4 × 9 (one per ep).

Run inside `gotrack` env.
"""
from __future__ import annotations
import argparse, csv, json, os, subprocess, sys, time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

MESH_ROOT = Path("/home/mingi/shared_data/AutoDex/object/paradex")
EXP_ROOT = Path("/home/mingi/shared_data/AutoDex/experiment/selected_100/allegro")
OUT_ROOT = REPO_ROOT / "outputs/foundpose_overlay"
RESULTS_CSV = REPO_ROOT / "outputs/foundpose_init_compare/selected_100/results.csv"

FPS = 30
SECS_PER_FRAME = 3
GRID_COLS = 6
BORDER_PX = 16
BORDER_BGR = (255, 0, 0)  # blue
FP_COLOR_RGB = (0, 200, 0)        # green
REF_COLOR_RGB = (200, 0, 200)     # magenta


def _resolve_mesh(obj):
    for sub in [MESH_ROOT/obj/"raw_mesh"/f"{obj}.obj",
                MESH_ROOT/obj/"processed_data"/"mesh"/"raw.obj",
                MESH_ROOT/obj/"processed_data"/"mesh"/"simplified.obj"]:
        if sub.exists(): return sub
    raise FileNotFoundError(obj)


def _load_episode(ep):
    img_dir = ep/"images"
    intr = json.load(open(ep/"cam_param"/"intrinsics.json"))
    extr = json.load(open(ep/"cam_param"/"extrinsics.json"))
    serials = sorted(p.stem for p in img_dir.glob("*.png"))
    images, K, T = {}, {}, {}
    for s in serials:
        if s not in intr or s not in extr: continue
        bgr = cv2.imread(str(img_dir/f"{s}.png"))
        if bgr is None: continue
        images[s] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        K[s] = np.asarray(intr[s]["intrinsics_undistort"], np.float64)
        e = np.asarray(extr[s], np.float64)
        if e.shape == (3, 4): e = np.vstack([e, [0, 0, 0, 1]])
        T[s] = e
    return images, K, T


def _render_overlay(image_rgb, pose_world, K, T, glctx, mesh_tensors, color_rgb, alpha=0.5):
    import torch
    sys.path.insert(0, str(REPO_ROOT/"autodex/perception/thirdparty/FoundationPose"))
    from Utils import nvdiffrast_render
    H, W = image_rgb.shape[:2]
    K = np.asarray(K, np.float32)
    pose_cam = T @ pose_world
    pt = torch.as_tensor(pose_cam, device="cuda", dtype=torch.float32).reshape(1, 4, 4)
    rc, _, _ = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=pt, glctx=glctx,
                                  mesh_tensors=mesh_tensors, use_light=False)
    sil_np = (rc[0].sum(dim=2) > 0).detach().cpu().numpy()
    out = image_rgb.copy()
    color_arr = np.array(color_rgb, dtype=np.float32)
    out[sil_np] = (out[sil_np] * (1 - alpha) + color_arr * alpha).astype(np.uint8)
    return out


def _make_grid_frame(images_rgb_dict, src_cam, banner_text, cell_w=384, cell_h=216):
    """Make grid frame (BGR) with 6 col × 4 row, source cam highlighted, bottom banner."""
    keys = sorted(images_rgb_dict.keys())
    rows = (len(keys) + GRID_COLS - 1) // GRID_COLS
    grid = np.zeros((rows * cell_h, GRID_COLS * cell_w, 3), dtype=np.uint8)
    for i, k in enumerate(keys):
        r, c = i // GRID_COLS, i % GRID_COLS
        bgr = cv2.cvtColor(images_rgb_dict[k], cv2.COLOR_RGB2BGR)
        small = cv2.resize(bgr, (cell_w, cell_h))
        if k == src_cam:
            cv2.rectangle(small, (0, 0), (cell_w - 1, cell_h - 1), BORDER_BGR, BORDER_PX)
        cv2.putText(small, k, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        grid[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = small
    # Bottom banner
    banner_h = 60
    canvas = np.zeros((grid.shape[0] + banner_h, grid.shape[1], 3), dtype=np.uint8)
    canvas[:grid.shape[0]] = grid
    cv2.putText(canvas, banner_text, (20, grid.shape[0] + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def _render_24cam_overlay(pose, images, K, T, glctx, mt, color_rgb):
    """Render mesh overlay at pose in all 24 cams. Returns dict serial -> RGB image."""
    out = {}
    for s in sorted(images.keys()):
        out[s] = _render_overlay(images[s], pose, K[s], T[s], glctx, mt, color_rgb)
    return out


def _grey_placeholder(images):
    """For source_cam without candidate: grey overlay (no mesh)."""
    out = {}
    for s in sorted(images.keys()):
        out[s] = (images[s].astype(np.float32) * 0.4).astype(np.uint8)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", type=str, default=str(RESULTS_CSV))
    parser.add_argument("--out-root", type=str, default=str(OUT_ROOT))
    parser.add_argument("--n-eps-per-obj", type=int, default=1)
    args = parser.parse_args()

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(args.results_csv)))
    by_obj: Dict[str, List[str]] = {}
    for r in rows:
        by_obj.setdefault(r["object"], []).append(r["episode"])
    selected = [(o, eps[:args.n_eps_per_obj]) for o, eps in by_obj.items()]

    from autodex.perception.silhouette import SilhouetteOptimizer

    t_start = time.perf_counter()
    flat_eps = [(o, e) for o, eps in selected for e in eps]
    ep_pbar = tqdm(total=len(flat_eps), desc="episodes", position=0)
    for obj, eps in selected:
        try: mesh_path = _resolve_mesh(obj)
        except Exception as e:
            print(f"[skip {obj}] {e}"); continue

        sil_opt = SilhouetteOptimizer(str(mesh_path), device="cuda")
        glctx, mt = sil_opt.glctx, sil_opt.mesh_tensors

        for ep_name in eps:
            ep_out = out_root / obj / ep_name
            fp_npz = ep_out / "fp_pre.npz"
            ref_npz = ep_out / "ref_pre.npz"
            if not fp_npz.exists() or not ref_npz.exists():
                print(f"[{obj}/{ep_name}] missing fp_pre.npz or ref_pre.npz, skip")
                continue

            fp_data = np.load(fp_npz)
            ref_data = np.load(ref_npz)
            fp_poses = {k.replace("cam_", ""): np.asarray(fp_data[k], np.float64) for k in fp_data.files}
            ref_poses = {k.replace("cam_", ""): np.asarray(ref_data[k], np.float64) for k in ref_data.files}

            ep = EXP_ROOT/obj/ep_name
            images, K, T = _load_episode(ep)
            serials = sorted(images.keys())

            # Compute frame size from first cell
            sample = next(iter(images.values()))
            cell_h, cell_w = sample.shape[0] // 2, sample.shape[1] // 2
            rows_in_grid = (len(serials) + GRID_COLS - 1) // GRID_COLS
            frame_h = rows_in_grid * cell_h + 60
            frame_w = GRID_COLS * cell_w

            video_path = ep_out / "video.mp4"
            # Pipe BGR frames to ffmpeg → H.264 encode.
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{frame_w}x{frame_h}",
                "-pix_fmt", "bgr24", "-r", str(FPS),
                "-i", "-",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-preset", "medium", "-crf", "23",
                str(video_path),
            ]
            import subprocess
            ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

            class _W:
                def write(self, frame):
                    ffmpeg_proc.stdin.write(np.ascontiguousarray(frame).tobytes())
                def release(self):
                    ffmpeg_proc.stdin.close()
                    ffmpeg_proc.wait()
            writer = _W()
            print(f"\n[{obj}/{ep_name}] writing video ({frame_w}×{frame_h} @ {FPS}fps)")
            t0 = time.perf_counter()

            for src_cam in tqdm(serials, desc=f"{obj}/{ep_name}", leave=False, position=1):
                # ── fp_pre frame ──
                if src_cam in fp_poses:
                    ovs = _render_24cam_overlay(fp_poses[src_cam], images, K, T, glctx, mt, FP_COLOR_RGB)
                    banner = f"src={src_cam} / fp_pre"
                else:
                    ovs = _grey_placeholder(images)
                    banner = f"src={src_cam} / fp_pre (no candidate)"
                frame = _make_grid_frame(ovs, src_cam, banner, cell_w=cell_w, cell_h=cell_h)
                for _ in range(SECS_PER_FRAME * FPS):
                    writer.write(frame)

                # ── ref_pre frame ──
                if src_cam in ref_poses:
                    ovs = _render_24cam_overlay(ref_poses[src_cam], images, K, T, glctx, mt, REF_COLOR_RGB)
                    banner = f"src={src_cam} / ref_pre"
                else:
                    ovs = _grey_placeholder(images)
                    banner = f"src={src_cam} / ref_pre (no candidate)"
                frame = _make_grid_frame(ovs, src_cam, banner, cell_w=cell_w, cell_h=cell_h)
                for _ in range(SECS_PER_FRAME * FPS):
                    writer.write(frame)

            writer.release()
            ep_pbar.update(1)
            print(f"  done ({time.perf_counter() - t0:.1f}s) → {video_path}")

    ep_pbar.close()
    print(f"\n[all done] {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
