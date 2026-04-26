"""Debug GoTrack on a single episode across multiple camera counts.

For each --cams value:
  1. Stage 0: ensure anchor bank for object
  2. Stage 0.5: select N cameras (FPS sampling on extrinsic positions)
  3. Stage 0.5: write per-cam init pose JSON from {exp}/pose_world.npy
  4. Stage C: run_multiview_gotrack_anchor_online_multi_object.py (PnP off)
  5. Render mesh-overlay video for ALL 24 cameras using fused world poses

Then prints comparison table (timing, anchor inlier ratio, jitter).

Usage:
  python src/validation/perception/gotrack_pipeline_debug.py \
    --exp ~/shared_data/AutoDex/experiment/selected_100/inspire/bamboo_box/20260406_012549 \
    --cams 4 8 12 24

Run from any env with numpy+tqdm; subprocesses gotrack/foundationpose internally.
"""
import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np


GOTRACK_ROOT = Path.home() / "AutoDex" / "autodex" / "perception" / "thirdparty" / "MV-GoTrack"
GOTRACK_PY = Path.home() / "miniconda3" / "envs" / "gotrack" / "bin" / "python"
FPOSE_PY = Path.home() / "miniconda3" / "envs" / "foundationpose" / "bin" / "python"
OBJ_BASE = Path.home() / "shared_data" / "AutoDex" / "object" / "paradex"
ANCHOR_DIR = GOTRACK_ROOT / "anchor_banks"
OVERLAY_SCRIPT = Path.home() / "AutoDex" / "src" / "visualization" / "overlay_object_video_single.py"


# ── Stage helpers ────────────────────────────────────────────────────────────

def ensure_anchor_bank(obj_name, num_anchors=256):
    bank = ANCHOR_DIR / f"{obj_name}.npz"
    if bank.exists():
        return bank
    mesh = OBJ_BASE / obj_name / "raw_mesh" / f"{obj_name}.obj"
    assert mesh.exists(), mesh
    ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[anchor] generating {bank.name}...", flush=True)
    subprocess.run(
        [str(GOTRACK_PY), "scripts/generate_anchor_bank.py",
         "--mesh-path", str(mesh), "--output-path", str(bank),
         "--num-anchors", str(num_anchors)],
        check=True, cwd=str(GOTRACK_ROOT))
    return bank


def select_cameras(exp_dir, n):
    """FPS-sample N cameras from extrinsics. Returns list of serial strings."""
    out = subprocess.check_output(
        [str(GOTRACK_PY), "scripts/select_cameras.py",
         "--extrinsics-json", str(exp_dir / "cam_param" / "extrinsics.json"),
         "--num-cameras", str(n),
         "--mode", "fps"],
        cwd=str(GOTRACK_ROOT), text=True).strip()
    return out.split()


def write_init_pose_jsons(exp_dir, init_dir, cams):
    """0413_mingi-style init: pose_world.npy → frame_poses/{cam}.json per camera."""
    init_dir.mkdir(parents=True, exist_ok=True)
    fp_dir = init_dir / "frame_poses"
    fp_dir.mkdir(parents=True, exist_ok=True)
    pose = np.load(exp_dir / "pose_world.npy")
    if pose.shape == (3, 4):
        pose = np.vstack([pose, [0, 0, 0, 1]])
    record = [{
        "frame_index": 0,
        "pose_world": pose.tolist(),
        "certainty_count_above_threshold": 1000.0,
        "status": "ok",
    }]
    for s in cams:
        with open(fp_dir / f"{s}.json", "w") as f:
            json.dump(record, f)


def stage_input_root(exp_dir, stage_dir):
    """Symlink videos+cam_param so GoTrack input layout matches."""
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)
    (stage_dir / "videos").symlink_to((exp_dir / "videos").resolve(), target_is_directory=True)
    (stage_dir / "cam_param").symlink_to((exp_dir / "cam_param").resolve(), target_is_directory=True)


def run_stage_c(stage_dir, init_dir, output_dir, obj_name, cams,
                input_resize_scale=0.5, first_frame_num_iters=5):
    """Call 0411 Stage C (multi-object tracker, used as single-object)."""
    bank = ANCHOR_DIR / f"{obj_name}.npz"
    mesh = OBJ_BASE / obj_name / "raw_mesh" / f"{obj_name}.obj"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(GOTRACK_PY), "run_multiview_gotrack_anchor_online_multi_object.py",
        "--input-root", str(stage_dir),
        "--output-root", str(output_dir),
        "--checkpoint-path", str(GOTRACK_ROOT / "gotrack_checkpoint.pt"),
        "--gpus", "0",
        "--camera-ids", *cams,
        "--object-names", obj_name,
        "--object-ids", "1",
        "--mesh-paths", str(mesh),
        "--init-pose-sources", str(init_dir),
        "--anchor-bank-paths", str(bank),
        "--num-iters", "1",
        "--first-frame-num-iters", str(first_frame_num_iters),
        "--num-anchors", "256",
        "--mesh-scale", "1.0",
        "--unit-scale-mode", "auto",
        "--mask-free",
        "--skip-pnp",
        "--optimized-input-pipeline-v2",
        "--optim-v2-crop-camera-workers", "4",
        "--optim-v2-warp-grid-workers", "4",
        "--optim-template-update-interval", "2",
        "--template-renderer-backend", "nvdiffrast",
        "--input-resize-scale", str(input_resize_scale),
        "--forward-precision", "fp32",
        "--torch-compile", "off",
        "--max-frames", "-1",
        "--worker-mode", "auto",
        "--tri-fit-worker-mode", "process",
        "--triangulation-worker-mode", "auto",
        "--status-log-every", "50",
        "--debug-level", "0",
    ]
    env = dict(os.environ,
               PYTHONUNBUFFERED="1",
               PYOPENGL_PLATFORM="egl",
               EGL_PLATFORM="surfaceless")
    print(f"[stage_c] {len(cams)}cams → {output_dir.name}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(GOTRACK_ROOT), env=env)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        print(f"[stage_c] FAIL rc={proc.returncode}", flush=True)
        return None
    return elapsed


def render_overlay(exp_dir, gotrack_records_path, mesh_path, output_dir):
    cmd = [
        str(FPOSE_PY), str(OVERLAY_SCRIPT),
        "--videos_dir", str(exp_dir / "videos"),
        "--cam_param_dir", str(exp_dir / "cam_param"),
        "--gotrack_records", str(gotrack_records_path),
        "--mesh", str(mesh_path),
        "--output_dir", str(output_dir),
    ]
    print(f"[overlay] → {output_dir.name}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=dict(os.environ, PYTHONUNBUFFERED="1"))
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        print(f"[overlay] FAIL rc={proc.returncode}", flush=True)
        return None
    return elapsed


# ── Analysis ─────────────────────────────────────────────────────────────────

def analyze_run(out_dir, obj_name):
    """Extract timing + jitter metrics from a Stage C output dir."""
    summary_path = out_dir / obj_name / "summary.json"
    pose_path = out_dir / obj_name / "world_pose_records.json"
    if not summary_path.exists() or not pose_path.exists():
        return None

    summary = json.load(open(summary_path))
    records = json.load(open(pose_path))
    poses = []
    n_inlier = []
    n_tri = []
    fit_residuals = []
    for r in records:
        if r.get("pose_world") is None:
            continue
        poses.append(np.array(r["pose_world"]))
        n_inlier.append(r.get("num_inlier_anchors", 0))
        n_tri.append(r.get("num_triangulated_anchors", 0))
        if r.get("mean_anchor_fit_residual_mm") is not None:
            fit_residuals.append(r["mean_anchor_fit_residual_mm"])

    poses = np.array(poses)
    n = len(poses)
    if n < 2:
        return None

    # Translation jitter: stddev of consecutive frame deltas (mm)
    trans = poses[:, :3, 3] * 1000
    deltas = np.diff(trans, axis=0)
    trans_jitter_mm = float(deltas.std())

    # Rotation jitter (deg): stddev of consecutive rotation angle changes
    from scipy.spatial.transform import Rotation
    rots = Rotation.from_matrix(poses[:, :3, :3])
    rel = rots[:-1].inv() * rots[1:]
    angles_deg = np.degrees(np.linalg.norm(rel.as_rotvec(), axis=1))
    rot_jitter_deg = float(angles_deg.std())

    return {
        "n_frames": n,
        "runtime_sec": summary["runtime_sec"],
        "s_per_frame": summary["runtime_sec"] / summary["processed_frames"],
        "fps": summary["throughput_processed_fps"],
        "mean_inlier": float(np.mean(n_inlier)),
        "mean_tri": float(np.mean(n_tri)),
        "mean_fit_residual_mm": float(np.mean(fit_residuals)) if fit_residuals else None,
        "trans_jitter_mm": trans_jitter_mm,
        "rot_jitter_deg": rot_jitter_deg,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--obj", default=None)
    p.add_argument("--cams", nargs="+", type=int, default=[4, 8, 12, 24])
    p.add_argument("--output", default=None,
                   help="Default: outputs/gotrack_debug/{obj}/{ep}/")
    p.add_argument("--input-resize-scale", type=float, default=0.5)
    p.add_argument("--first-frame-num-iters", type=int, default=5)
    p.add_argument("--skip-overlay", action="store_true")
    args = p.parse_args()

    exp_dir = Path(args.exp).expanduser()
    obj = args.obj or exp_dir.parent.name
    ep_name = exp_dir.name
    out_root = Path(args.output) if args.output else \
        Path.home() / "AutoDex" / "outputs" / "gotrack_debug" / obj / ep_name
    out_root.mkdir(parents=True, exist_ok=True)

    mesh = OBJ_BASE / obj / "raw_mesh" / f"{obj}.obj"
    assert mesh.exists(), f"mesh not found: {mesh}"
    assert (exp_dir / "pose_world.npy").exists(), f"pose_world.npy not in {exp_dir}"

    print(f"=== gotrack pipeline debug: {obj}/{ep_name} ===")
    print(f"   cams: {args.cams}")
    print(f"   resize: {args.input_resize_scale}, first_frame_iters: {args.first_frame_num_iters}")
    print(f"   output: {out_root}")
    print()

    ensure_anchor_bank(obj)

    results = {}
    for n in args.cams:
        print(f"\n────── {n} cam ──────")
        run_dir = out_root / f"{n}cam"
        stage_dir = run_dir / "stage"
        init_dir = run_dir / "init_poses"
        gtrack_dir = run_dir / "gotrack_output"
        overlay_dir = run_dir / "overlay"

        cams = select_cameras(exp_dir, n)
        print(f"[select] {n} cams: {cams}")
        stage_input_root(exp_dir, stage_dir)
        write_init_pose_jsons(exp_dir, init_dir, cams)

        elapsed = run_stage_c(stage_dir, init_dir, gtrack_dir, obj, cams,
                              input_resize_scale=args.input_resize_scale,
                              first_frame_num_iters=args.first_frame_num_iters)
        if elapsed is None:
            results[n] = None
            continue

        metrics = analyze_run(gtrack_dir, obj)
        results[n] = {"wall_sec": elapsed, **(metrics or {})}

        if not args.skip_overlay:
            records = gtrack_dir / obj / "world_pose_records.json"
            render_overlay(exp_dir, records, mesh, overlay_dir)

    # ── Comparison table ──
    print("\n\n================  Comparison  ================")
    fmt = "{:>6} {:>8} {:>8} {:>7} {:>10} {:>10} {:>14} {:>13} {:>13}"
    print(fmt.format("ncam", "wall(s)", "frames", "s/frm", "mean_inl", "mean_tri",
                     "fit_res(mm)", "trans_jit(mm)", "rot_jit(deg)"))
    for n in args.cams:
        r = results.get(n)
        if r is None:
            print(f"{n:>6}  ── failed ──")
            continue
        print(fmt.format(
            n,
            f"{r['wall_sec']:.0f}",
            r.get("n_frames", -1),
            f"{r.get('s_per_frame', 0):.2f}",
            f"{r.get('mean_inlier', 0):.1f}",
            f"{r.get('mean_tri', 0):.1f}",
            f"{r.get('mean_fit_residual_mm') or 0:.2f}",
            f"{r.get('trans_jitter_mm', 0):.3f}",
            f"{r.get('rot_jitter_deg', 0):.4f}",
        ))

    out_json = out_root / "compare.json"
    json.dump(results, open(out_json, "w"), indent=2, default=float)
    print(f"\n[saved] {out_json}")


if __name__ == "__main__":
    main()
