# Perception Pipeline

Scripts for processing stereo capture data: video download, mask generation, depth estimation, pose tracking, and result upload.

## Directory / Path Conventions

| Path | Description |
|------|-------------|
| `/home/mingi/paradex1/capture/eccv2026/inspire_f1/{obj}/{idx}/` | **Network FS** — source videos, `cam_param/`, destination for uploads |
| `~/video_cache/eccv2026/inspire_f1/{obj}/{idx}/` | **Local cache** — working directory for all processing |

All scripts read `--base` as the network FS path and derive the cache path automatically.

### Cache layout per episode

```
{obj}/{idx}/
  videos/           {serial}.avi            RGB stereo videos (input)
  obj_mask/         {serial}.avi            Binary mask video (MJPG)
  obj_mask_first/   {serial}.png            First-frame mask only
  obj_mask_debug/   {serial}.avi            Mask overlay on RGB (debug)
  depth/            {serial}.avi            Depth video (FFV1, uint16 encoded)
  pose/             pose_world.npy          (N,4,4) world-frame poses
                    depth.png / seg_grid.png / overlay.png  (debug)
  pose_overlay/     {serial}.png + grid.png Per-camera mesh overlay images
```

---

## Typical Pipeline Order

```
1. download_videos     Download AVI videos from network FS to cache
2. batch_mask_yoloe    Generate object masks (fast, YOLOE)
   batch_mask          SAM3 fallback for videos YOLOE missed
   batch_mask_first    First-frame only mask (for pose init, lighter)
3. batch_depth         Stereo depth estimation (FoundationStereo TRT)
4. batch_pose          6D pose tracking (FoundationPose)
5. batch_overlay       Render mesh overlay on all cameras
6. upload_results      Upload results back to network FS
```

---

## Scripts

### `download_videos.py` — Download videos to cache

Copies `{obj}/{idx}/videos/{serial}.avi` from network FS to local cache.

```bash
python src/perception/download_videos.py \
    --base /home/mingi/paradex1/capture/eccv2026/inspire_f1 \
    --serials 22684755 23263780
```

**Skipping:** skips if destination file already exists (size not checked).

---

### `batch_mask_yoloe.py` — YOLOE mask generation (fast)

Runs YOLOE on all cached videos using the object directory name as the text prompt.

```bash
python -u src/perception/batch_mask_yoloe.py \
    --base /home/mingi/paradex1/capture/eccv2026/inspire_f1 \
    --serials 22684755 23263780
```

**Skipping:** skips if `obj_mask/{serial}.avi` already exists in cache or on network FS.
**Probe:** tests first 5 frames — if no mask found, skips **all** videos with that object prompt.
Saves: `obj_mask/{serial}.avi` + `obj_mask_debug/{serial}.avi`

---

### `batch_mask.py` — SAM3 fallback mask generation

Runs SAM3 on videos that YOLOE missed (i.e., no mask exists yet). Tries `"object"` first, then the object name.

```bash
python -u src/perception/batch_mask.py \
    --base /home/mingi/paradex1/capture/eccv2026/inspire_f1 \
    --serials 22684755 23263780 \
    [--prompt "custom prompt"]  # override auto prompt
    [--gpu 0]
```

**Skipping:** skips if `obj_mask/{serial}.avi` exists in cache OR on network FS.
**Probe:** aborts after 5 frames if no mask found with a given prompt, tries next prompt.
**Frame limit:** skips videos with > 1200 frames.
Saves: `obj_mask/{serial}.avi` + `obj_mask_debug/{serial}.avi`

---

### `batch_mask_first.py` — First-frame mask (for pose init)

Generates a single PNG mask per video (only frame 0). Lighter alternative to full mask video — sufficient for FoundationPose initialization.

```bash
python -u src/perception/batch_mask_first.py \
    --base /home/mingi/paradex1/capture/eccv2026/inspire_f1 \
    --serials 22684755 23263780
```

**Skipping:** skips if `obj_mask_first/{serial}.png` already exists, OR if full `obj_mask/{serial}.avi` already exists.
Saves: `obj_mask_first/{serial}.png`

---

### `batch_depth.py` — Stereo depth estimation (FoundationStereo TRT)

Runs FoundationStereo TRT engine on stereo video pairs. Reads `cam_param/` from network FS for stereo rectification.

```bash
conda activate foundation_stereo
python -u src/perception/batch_depth.py \
    --base /home/mingi/paradex1/capture/eccv2026/inspire_f1 \
    --left_serial 22684755 --right_serial 23263780 \
    [--last_episodes 200]   # process last N episodes
    [--max_episodes 100]    # process first N episodes
    [--engine path/to/engine]
```

**Skipping:** checks depth validity per-episode (right before processing), so multiple parallel instances with overlapping ranges safely skip each other's completed work. Valid = depth file exists (either serial) AND frame count == `min(left_frames, right_frames)`.

**Auto-swap:** automatically detects if left/right cameras are physically swapped via stereoRectify and swaps them. Depth is always saved under the physically-correct left serial after swap.

**Depth encoding:** uint16 millimeters packed into B (low byte) + G (high byte) channels of FFV1 AVI.

**Global index `#N`:** each episode has a global index that matches `check_depth.py` output, useful for calculating `--last_episodes`.

Requires: `conda activate foundation_stereo`, videos in cache, `cam_param/` on network FS.
Saves: `depth/{left_serial}.avi`

---

### `batch_pose.py` — 6D pose tracking (FoundationPose)

Tracks object pose across all frames using FoundationPose. Uses depth + mask for initialization, then tracks without mask.

```bash
conda activate foundationpose
python -u src/perception/batch_pose.py \
    --base /home/mingi/paradex1/capture/eccv2026/inspire_f1 \
    --serials 22684755 23263780 \
    --mesh_dir /home/mingi/mesh \
    [--gpu 0] \
    [--downscale 0.5]       # inference downscale (default 0.5)
    [--est_refine_iter 5]   # refinement iterations (default 5)
```

**Task selection:** picks the first serial (from `--serials`) that has video + depth + mask for each episode. Prefers `obj_mask_first/{serial}.png` over full `obj_mask/{serial}.avi`.
**Skipping:** skips episodes where no mesh found, no depth, or no mask.
Groups episodes by mesh to reuse the PoseTracker per object.

Saves: `pose/pose_world.npy` (N,4,4 float32, NaN for failed frames), `pose/depth.png`, `pose/seg_grid.png`, `pose/overlay.png`

---

### `batch_overlay.py` — Mesh overlay visualization

For each episode with `pose_world.npy`, renders the mesh onto frame 1 of all cameras and saves per-camera PNGs + a grid.

```bash
conda activate foundationpose
python -u src/perception/batch_overlay.py \
    --base /home/mingi/paradex1/capture/eccv2026/inspire_f1 \
    --mesh_dir /home/mingi/mesh \
    [--gpu 0] \
    [--render_downscale 0.5]
```

**Skipping:** no explicit skip — rerenders every episode with a valid `pose_world.npy`.
Saves: `pose_overlay/{serial}.png` + `pose_overlay/grid.png`

---

### `upload_results.py` — Upload results to network FS

Copies processed results from cache back to network FS.

```bash
python src/perception/upload_results.py \
    --base /home/mingi/paradex1/capture/eccv2026/inspire_f1
```

Uploads: `obj_mask`, `obj_mask_first`, `obj_mask_debug`, `depth`, `pose`, `pose_overlay`, `pose_overlay_merged`

**Skipping behavior:**
- `obj_mask`, `pose`, etc.: skips if destination exists and **same file size**
- `depth`: **always overwrites** (no size check)

---

### `check_depth.py` — Depth status report

Reports depth status across all captures, iterating from network FS.

```bash
python src/perception/check_depth.py \
    --base /home/mingi/paradex1/capture/eccv2026/inspire_f1 \
    --left_serial 22684755 --right_serial 23263780
```

**Categories:** OK, Missing, Invalid (wrong frame count), No video (not in cache), No cam_param
**Output:** global index `#N` per missing/invalid entry, with `--last_episodes X` hint to cover it in `batch_depth.py`.
**Note:** global indices match `batch_depth.py` (both require `cam_param` + both serials present).

---

### `check_mask.py` — Mask status report

Reports mask status per episode, grouped by object.

```bash
python src/perception/check_mask.py \
    --base /home/mingi/paradex1/capture/eccv2026/inspire_f1
    [--serials 22684755 23263780]   # omit to auto-discover from cache
```

**Output:** per object `X/N episodes complete`, per incomplete episode shows `M/S serials done` and which serials are missing.
**Mask check:** looks in both cache AND network FS.

---

## Conda Environments

| Script | Environment |
|--------|-------------|
| `batch_depth.py` | `foundation_stereo` |
| `batch_pose.py`, `batch_overlay.py` | `foundationpose` |
| All others | base / any |
