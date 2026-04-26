# Distributed FoundPose-init + GoTrack-tracking

Replaces centralized PerceptionPipeline with FoundPose init (centralized) +
GoTrack tracking (distributed across capture1-6). In-process wrappers + ZMQ
daemons.

Goal: replace `src/execution/daemon/perception_pipeline.py` (SAM3 + DA3 + FPose + IoU + Sil refine, ~37s/init, no tracking) with:
  - **Init** (1×/episode): SAM3 + FoundPose (centralized, robot PC) + IoU select + Sil refine
  - **Tracking** (every frame): GoTrack mask-free, **distributed**: each capture PC runs stage 1-4 on its own 4 cameras, robot PC does stage 5-6 (triangulate + Kabsch)

**Why:** GoTrack is much faster than re-running FoundPose every frame (uses prior pose, no template DB search). Mask-free → no SAM3 needed during tracking. Distributed because sending raw images at 10 Hz from 24 cams = 1.5 GB/s (impossible); sending anchor obs (256 anchors × 4 cam × ~12 bytes) = 12 KB/PC/frame, trivial.

**How to apply:** when working on real-robot perception execution, prefer this path over the old PerceptionPipeline. Tracking uses GoTrack only after FoundPose-based init succeeds.

## Pipeline architecture

| Stage | Where | Code |
|---|---|---|
| 1. Frame capture | capture1-6 (4 cam each, PySpin → SHM) | paradex MultiCameraReader |
| 2. SAM3 mask (init only) | capture1-3 (existing daemon) | `src/execution/daemon/perception_daemon.py --model sam3` |
| 3. FoundPose per-view + Stage A onboard | robot PC (centralized) | `autodex/perception/foundpose_init.py:FoundPoseInit` |
| 4. Cross-view IoU select | robot PC | `autodex/perception/pose_select.py:select_best_pose_by_iou` |
| 5. Sil refine (init) | robot PC | `autodex/perception/silhouette.py:SilhouetteOptimizer.optimize` |
| 6. GoTrack stage 1-4 (per-cam) | capture1-6 (each PC its own 4 cam) | `autodex/perception/gotrack_engine.py:GoTrackEngine` |
| 7. Triangulate + Kabsch (per-frame) | robot PC | `autodex/perception/gotrack_tracker.py:GoTrackTracker` |

## Key files added

- `autodex/perception/pose_select.py` — `select_best_pose_by_iou(candidates, masks, intrinsics, extrinsics, H, W, glctx, mesh_tensors)`. Used by both FoundPose IoU select and (now) `PerceptionPipeline._select_best_pose`.
- `autodex/perception/foundpose_init.py` — `FoundPoseInit` class. Wraps MV-GoTrack's `scripts/run_foundpose_first_frame_init.py` *logic* (not the script itself) for in-process per-view PnP. Stage A onboard cached at `outputs/foundpose_assets/{obj}/object_repre/v1/{obj}/1/repre.pth`.
- `autodex/perception/gotrack_engine.py` — `GoTrackEngine` class. 1 PC, N cam, GoTrack stage 1-4 in-process. Uses `_process_group_for_timestep_anchor` + `_build_anchor_observations_for_frame` from MV-GoTrack. Returns per-cam `{uv_curr, confidence, valid_mask, selected_mask, anchor_ids, positions_o, crop_intrinsic, T_world_from_crop_cam}`.
- `autodex/perception/gotrack_tracker.py` — `GoTrackTracker` class for robot PC. ZMQ SUB to capture PCs, frame_id sync buffer, `triangulate_anchor_observations` + `robust_fit_pose_from_anchors`, prior PUB.
- `src/execution/daemon/gotrack_daemon.py` — capture-PC daemon. Uses paradex `MultiCameraReader` (SHM) + `DataPublisher` (PUB obs) + custom SUB (prior pose) + `CommandReceiver` (init/start/stop).
- `src/validation/perception/foundpose_init_compare.py` — comparison: FoundPose+sil vs PerceptionPipeline reference. 20 obj × 10 ep, resumable CSV.
- `src/validation/perception/gotrack_engine_dryrun.py` — local API check for `GoTrackEngine` without capture PCs.
- `src/process/batch_onboard_foundpose.py` — parallel Stage A onboarder (4 workers default) for N objects at once. Local-only, not committed; useful when you need to pre-onboard many objects before evaluation. ~N×18 min serial → ~N×18/workers min.

## Memory locations

- **GoTrack repo**: `autodex/perception/thirdparty/MV-GoTrack/` (forked, see [Setup] for installation)
- **Conda env**: `gotrack` (`~/miniconda3/envs/gotrack/`). Python 3.10, torch 2.0, xformers 0.0.20, faiss-gpu, dinov2.
- **GoTrack checkpoint**: `autodex/perception/thirdparty/MV-GoTrack/gotrack_checkpoint.pt` (1.5 GB, LFS — restore via `cp .git/lfs/objects/f7/d1/<hash> gotrack_checkpoint.pt` if `git stash` removes it).
- **FoundPose repre cache** (per object): `outputs/foundpose_assets/{obj}/object_repre/v1/{obj}/1/repre.pth`. Generated once via `scripts/onboard_custom_mesh_for_foundpose.py`.
- **GoTrack anchor bank** (per object): `autodex/perception/thirdparty/MV-GoTrack/anchor_banks/{obj}.npz`. Generated once via `scripts/generate_anchor_bank.py --num-anchors 256`.
- **`gotrack` env extras**: needed by FoundationPose `Utils.py` import (used by SilhouetteOptimizer + cross-view IoU): `psutil`, `pandas`, `open3d`, `transformations`, `ruamel.yaml`. Install via `~/miniconda3/envs/gotrack/bin/pip install ...`.

## Setup steps

### 1. MV-GoTrack repo
The fork lives at `autodex/perception/thirdparty/MV-GoTrack/`. Initial install (already done on robot PC):
```
git clone <fork-url> autodex/perception/thirdparty/MV-GoTrack
cd autodex/perception/thirdparty/MV-GoTrack
conda env create -n gotrack -f environment.yml
conda activate gotrack
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
pip install psutil pandas open3d transformations ruamel.yaml  # for FoundationPose Utils import
```

### 2. Per-object onboarding
- FoundPose template DB (used for init):
  ```
  python scripts/onboard_custom_mesh_for_foundpose.py \
      --mesh-path ~/shared_data/AutoDex/object/paradex/{obj}/raw_mesh/{obj}.obj \
      --object-id 1 --dataset-name {obj} \
      --output-root outputs/foundpose_assets/{obj} \
      --reference-intrinsics-json {ep}/cam_param/intrinsics.json \
      --reference-camera-id <serial> \
      --mesh-scale 1000.0 --reference-image-scale 1.0 \
      --min-num-viewpoints 57 --num-inplane-rotations 14 \
      --ssaa-factor 4.0 --pca-components 256 --cluster-num 2048
  ```
  ~수 분/object. Triggered automatically by `FoundPoseInit.__init__` if `repre.pth` missing.

- GoTrack anchor bank (used for tracking):
  ```
  python autodex/perception/thirdparty/MV-GoTrack/scripts/generate_anchor_bank.py \
      --mesh-path ~/shared_data/AutoDex/object/paradex/{obj}/raw_mesh/{obj}.obj \
      --output-path autodex/perception/thirdparty/MV-GoTrack/anchor_banks/{obj}.npz \
      --num-anchors 256
  ```
  ~수 초/object.

### 3. Daemon setup per capture PC
Each capture PC (capture1-6) needs:
- Same repo synced: `git fetch origin && git reset --hard origin/main`
- `gotrack` env (replicate from robot PC: `conda env export -n gotrack > env.yml` then create on capture)
- nvdiffrast installed in env
- GoTrack checkpoint synced: `cp ~/AutoDex/autodex/perception/thirdparty/MV-GoTrack/gotrack_checkpoint.pt /home/mingi/AutoDex/autodex/perception/thirdparty/MV-GoTrack/`
- Anchor bank for the target object synced (or symlinked to NAS): `~/AutoDex/autodex/perception/thirdparty/MV-GoTrack/anchor_banks/{obj}.npz`

Daemon launch (per capture PC):
```
conda activate gotrack
cd ~/AutoDex
python src/execution/daemon/gotrack_daemon.py \
    --port-obs 1235 --port-prior 1236 --port-cmd 6892 \
    --robot-ip 192.168.0.100   # replace with robot PC IP
```

Robot PC tracker:
```
conda activate gotrack
python autodex/perception/gotrack_tracker.py \
    --capture-ips 192.168.0.101 192.168.0.102 ... 192.168.0.106 \
    --port-obs 1235 --port-prior 1236 \
    --min-cams-per-frame 6 \
    --init-pose-npy <ep>/pose_world.npy \
    --max-frames 1000
```

## ZMQ channel layout

| Channel | Direction | Pattern | Port | Payload |
|---|---|---|---|---|
| anchor_obs | capture → robot | PUB/SUB (paradex DataPublisher / DataCollector-style) | 1235 | `{frame_id, serial, uv_curr (M,2), confidence (M), valid_mask (M), selected_mask (M), anchor_ids, positions_o, crop_intrinsic, T_world_from_crop_cam}` |
| prior_pose | robot → capture | PUB/SUB (CONFLATE=1, fire-and-forget, latest only) | 1236 | `{frame_id, pose_world (4×4), ts}` |
| control | robot → capture | REQ/REP (paradex CommandSender / CommandReceiver) | 6892 | `{command: init/start/stop/exit, info: {mesh_path, anchor_bank_path, intrinsics, extrinsics, object_id, ...}}` |

Bandwidth budget: anchor_obs ~12 KB/PC/frame × 6 PC × 10 Hz ≈ 720 KB/s. prior_pose ~128 byte/broadcast. Trivial.

## Validation results so far (2026-04-26 partial, 86/200 rows, 8/20 objects)

`outputs/foundpose_init_compare/selected_100/results.csv`:
- **Translation**: pre-sil median 4.7 mm (p90 13.7), post-sil median 0.09 mm (p90 1.7).
- **Rotation**: pre-sil median 9.8°, post-sil median 3.4° but **p90 180° (~25% of cases flip 180°)** — symmetry/cylinder failure mode. sil refine can't recover.
- **Per-object 180° flip rate**: white_soap_dish 0/10, metal_scoop_small 0/10, icecream_scoop 1/10, organizer_beige 3/10, wood_organizer 6/16, pepsi_light 4/10 (cylinder), beige_brush 5/10 (symmetric).
- **Mask IoU (pre-sil)**: median 0.876, mean 0.856, p90 0.934.
- **Time**: FoundPose 24-cam centralized compute median 5.5 s; sil refine 4.6 s; SAM3 mask 2.5 s. Reference (current PerceptionPipeline) total 31 s = sam3 2.5 + depth 0.9-4.6 + fpose 4.4 + select 6.7 + sil 4.5.
- **→ FoundPose+sil init: ~10s vs reference ~32s = 3× faster on non-symmetric objects, similar accuracy. Symmetric handling (cylinder snap or symmetry-aware error) needed.**

Pending stat additions (computed in CSV header but backfill needed for old rows):
- `iou_pre_mean/max`, `iou_post_mean/max`: cross-view IoU before/after sil refine, mean and best-view max.
- `sil_loss_pre`, `sil_loss_post`: silhouette MSE before/after sil refine (existing `sil_loss` = post).
- `ref_iou_mean/max`, `ref_sil_loss`: same metrics on reference pose for direct comparison.

Resumable: re-running same command appends new (obj, ep) rows; old rows preserved.

## GoTrack tracking performance (offline benchmark, prior to distributed setup)

Measured on robot PC alone (centralized) with `gotrack_pipeline_debug.py` —
not the distributed daemon path. These are **indicative compute costs** for
the GoTrack stage 1-4 pipeline; the distributed setup will differ in two ways:

  1. Per-PC compute: each capture PC handles 4 cams locally, so per-PC
     latency ≈ 4-cam offline number (~0.16 s/frame). Frame rate is bounded
     by the slowest PC.
  2. Network: ~12 KB anchor obs per PC per frame (trivial vs raw image
     streaming).

Bamboo_box, 432 frames, single 3090:

| ncam | wall(s) | s/frame | trans_jit (mm) | rot_jit (deg) |
|---|---:|---:|---:|---:|
| 4  | 68  | 0.158 | 0.683 | 0.355 |
| 8  | 120 | 0.278 | 0.594 | 0.286 |
| **12** | **172** | **0.400** | **0.218** | **0.298** |
| 24 | 329 | 0.763 | 0.304 | 0.464 |

Stage breakdown (s/frame, per cam basis):

| ncam | tmpl_render | crop  | DINOv2 forward | anchor_obs |
|---|---:|---:|---:|---:|
| 4  | 0.0022 | 0.006 | 0.088 | 0.009 |
| 8  | 0.0026 | 0.011 | 0.159 | 0.014 |
| 12 | 0.0029 | 0.019 | 0.228 | 0.022 |
| 24 | 0.0046 | 0.030 | 0.436 | 0.035 |

DINOv2 forward dominates 60-70% of compute. Near-linear in cam count.

**Sweet spot in centralized setup: 12 cam.** Translation jitter lowest
(0.218 mm), rotation also good, half the cost of 24 cam. 24 cam adds noisy
contributors that slightly hurt translation while marginally helping rotation.

**Drift behavior:** GoTrack `pose_world` is a fresh fit each frame from
scratch (anchor triangulation + Kabsch RANSAC). No accumulated drift. Prior
pose only affects template render position and anchor query starting point.
Catastrophic failure mode: if prior is so wrong the rendered bbox doesn't
overlap the live object, anchor confidences collapse, fit fails, tracking
lost (no recovery without re-init via FoundPose).

## Pending validation (after returning to internal network)

1. **GoTrackEngine dry-run** (single PC, no capture daemons): `python src/validation/perception/gotrack_engine_dryrun.py --obj attached_container`. Verifies our guesses about MV-GoTrack internal API (`_process_group_for_timestep_anchor` return shape, `crop_intrinsic` / `T_world_from_crop_cam` fields in `debug_data`).
2. **One-PC daemon dry-run**: launch `gotrack_daemon.py` on one capture PC with real SHM cameras, send dummy prior + init from a stub robot script, verify anchor_obs payload arrives.
3. **6 PC integration**: full pipeline with FoundPose init → GoTrack tracking, capture all frames, log per-frame world pose to CSV, compare to current pipeline baseline.

## Anti-patterns we already corrected (don't repeat)

- Initially I subprocess-called `run_foundpose_first_frame_init.py` (CLI script) instead of importing FoundPose in-process. User pointed out this was unnecessary indirection: that CLI script writes `summary.json` + 1-frame .avi files just to round-trip data through disk. The right pattern is `FoundPoseInit` wrapper.
- Same mistake almost made for GoTrack — script-based wrapper at `gotrack_pipeline_debug.py` does subprocess for batch validation, but production tracker should call `_process_group_for_timestep_anchor` directly in-process (this is what `gotrack_engine.py` does).
- pyrender fallback in MV-GoTrack's `utils/renderer_nvdiffrast.py` was silently making everything 25× slower. Removed (see `project_object_6d_tracking.md`). nvdiffrast must be installed in `gotrack` env or it fails fast.
