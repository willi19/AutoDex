# src/process — Batch Processing Pipelines

CLI scripts for running mask, depth, and pose on capture directories.

Each script supports two modes:
- `--capture_dir DIR` — process a single episode (all cameras)
- `--base DIR` — batch all episodes under DIR (with progress/ETA)

## Quick Reference

| Task | Script | Conda Env | Key Flags |
|------|--------|-----------|-----------|
| Mask (SAM3) | `mask.py` | `sam3` | `--method sam3` (default) |
| Mask (YOLOE) | `mask.py` | `foundationpose` | `--method yoloe --conf 0.2` |
| Depth (auto pair) | `depth.py` | `foundation_stereo` | (default) |
| Depth (fixed pair) | `depth.py` | `foundation_stereo` | `--serials S1 S2` |
| Pose (tracking) | `pose.py` | `foundationpose` | `--mesh FILE` |
| Pose (+ overlay) | `pose.py` | `foundationpose` | `--overlay --mesh FILE` |
| Download videos | `download_videos.py` | any | |
| Upload results | `upload_results.py` | any | |

## Mask

### SAM3 — single episode

```bash
conda activate sam3
python -u src/process/mask.py \
    --capture_dir /home/mingi/shared_data/RSS2026_Mingi/experiment/selected_100/apple/20260206_181110 \
    --prompt "object on the checkerboard"
```

Skips cameras where `obj_mask/{serial}.avi` already exists. Tries prompt, then `"object"` as fallback.

### YOLOE — single episode

```bash
conda activate foundationpose
python -u src/process/mask.py --method yoloe \
    --capture_dir /path/to/episode \
    --conf 0.2 --batch-size 50
```

### Batch all episodes (with sharding)

```bash
conda activate sam3
python -u src/process/mask.py \
    --base /home/mingi/shared_data/RSS2026_Mingi/experiment/selected_100 \
    --prompt "object on the checkerboard"

# Filter to specific objects
python -u src/process/mask.py --base ... --objects apple banana

# Multi-GPU sharding
python -u src/process/mask.py --base ... --shard 0/3 --gpu 0
python -u src/process/mask.py --base ... --shard 1/3 --gpu 1
python -u src/process/mask.py --base ... --shard 2/3 --gpu 2
```

## Depth

### Auto pair — single episode (recommended)

Automatically selects stereo pairs for all cameras.

```bash
conda activate foundation_stereo
python -u src/process/depth.py \
    --capture_dir /home/mingi/shared_data/RSS2026_Mingi/experiment/selected_100/apple/20260206_181110

# Quick test (first frame only)
python -u src/process/depth.py --capture_dir ... --num_frames 1

# Regenerate overlay only (no depth recomputation)
python -u src/process/depth.py --capture_dir ... --overlay_only
```

Skips pairs whose depth AVI already has the correct frame count.

### Fixed pair — manual serials

```bash
conda activate foundation_stereo
python -u src/process/depth.py \
    --capture_dir /path/to/episode \
    --serials 22684755 23263780
```

Auto-detects left/right from extrinsics. Saves depth under the correct left serial.

### Batch all episodes (with ETA)

```bash
conda activate foundation_stereo
python -u src/process/depth.py \
    --base /home/mingi/shared_data/RSS2026_Mingi/experiment/selected_100
```

Shows progress: `[42/1989] apple/20260206 (elapsed 1.2h, ETA 3h15m)`

## Pose

### Tracking — single episode

Runs FoundationPose on all cameras that have video + depth + mask.

```bash
conda activate foundationpose
python -u src/process/pose.py \
    --capture_dir /path/to/episode \
    --mesh /home/mingi/shared_data/object_6d/data/mesh/apple/apple.obj \
    --downscale 0.5
```

Saves `pose/{serial}.npy` (cam frame) and `pose/{serial}_world.npy` (world frame).

### Tracking + overlay video

```bash
conda activate foundationpose
python -u src/process/pose.py --overlay \
    --capture_dir /path/to/episode \
    --mesh /path/to/mesh.obj
```

Saves `pose_overlay/{serial}.avi` with purple mesh overlay.

### Batch all episodes

```bash
conda activate foundationpose
python -u src/process/pose.py \
    --base /home/mingi/shared_data/RSS2026_Mingi/experiment/selected_100 \
    --mesh_dir /home/mingi/shared_data/object_6d/data/mesh

# With overlay
python -u src/process/pose.py --overlay \
    --base /path/to/selected_100 --mesh_dir /path/to/meshes
```

Mesh lookup: `{mesh_dir}/{obj_name}/{obj_name}.obj` → `processed_data/mesh/simplified.obj` → glob fallback.

## Data Transfer

```bash
# Download videos from network FS to local cache
python src/process/download_videos.py \
    --base /home/mingi/paradex1/capture/eccv2026/inspire_f1 \
    --serials 22684755 23263780

# Upload results (masks, depth, pose) from cache to network FS
python src/process/upload_results.py \
    --base /home/mingi/paradex1/capture/eccv2026/inspire_f1
```

## Output Layout

```
{capture_dir}/
├── videos/          # input: .avi per camera (already undistorted)
├── cam_param/       # input: intrinsics.json, extrinsics.json
├── obj_mask/        # mask: .avi per camera (MJPG)
├── depth/           # depth: .avi per camera (FFV1, uint16 mm BGR)
├── depth_debug/     # debug: rectified pairs, TRT input with epipolar lines
├── depth_overlay/   # debug: cross-view reprojection grids
├── pose/            # pose: {serial}.npy (N,4,4 cam), {serial}_world.npy (world)
└── pose_overlay/    # overlay: .avi with mesh overlay per camera
```

## Conda Environments

| Environment | Scripts |
|-------------|---------|
| `foundation_stereo` | `depth.py` |
| `foundationpose` | `mask.py --method yoloe`, `pose.py` |
| `sam3` | `mask.py --method sam3` |
| any | `download_videos.py`, `upload_results.py` |

## Legacy Scripts

The following old scripts are superseded by the consolidated versions above:

| Old Script | Replaced By |
|------------|-------------|
| `batch_mask.py` | `mask.py --method sam3 --capture_dir` |
| `batch_mask_yoloe.py` | `mask.py --method yoloe --capture_dir` |
| `batch_mask_all.py` | `mask.py --base` |
| `batch_depth.py` | `depth.py --serials S1 S2` |
| `batch_depth_auto.py` | `depth.py --capture_dir` |
| `run_depth_all.py` | `depth.py --base` |
| `batch_pose.py` | `pose.py --capture_dir` |
| `batch_pose_overlay.py` | `pose.py --overlay` |
