# AutoDex

Autonomous dexterous manipulation pipeline: perception, planning, and execution.

## Project Structure

```
autodex/                    # Core library
  perception/               # Mask, depth, pose estimation
    thirdparty/             # SAM3, FoundationStereo, object-6d-tracking (not tracked in git)
  planner/                  # Grasp planning (cuRobo)
  executor/                 # Robot execution
  simulator/                # MuJoCo simulation

src/                        # Scripts
  perception/               # Batch mask generation (SAM3, YOLOE)
  executor/                 # Grasp selection
  validation/               # Validation pipeline (mask, depth, pose, compare)
  analysis/                 # Simulation analysis
  visualization/            # Mesh processing, visualization
```

## Conda Environments

Different components require separate environments due to conflicting dependencies.

### `sam3` — Perception (mask generation, depth, pose)

```bash
conda create -n sam3 python=3.12
conda activate sam3
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install numpy==1.26.4 opencv-python scipy open3d trimesh
pip install -e .  # install autodex package
```

**Third-party setup** (under `autodex/perception/thirdparty/`):

```bash
# SAM3 — video segmentation
cd autodex/perception/thirdparty/sam3
pip install -e .

# YOLOE weights
mkdir -p autodex/perception/thirdparty/weights
# Place yoloe-26x-seg.pt in the weights directory
```

### `foundationpose` — Object 6D pose estimation

```bash
conda create -n foundationpose python=3.9
conda activate foundationpose
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics==8.4.15
```

### `foundation_stereo` — Stereo depth estimation

```bash
conda create -n foundation_stereo python=3.11
conda activate foundation_stereo
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

### `bodex` — Planning (cuRobo + MuJoCo)

```bash
conda create -n bodex python=3.10
conda activate bodex
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
# Install cuRobo from source (see BODex repo)
```

## Usage

### Mask Generation Pipeline

All commands use `--base` to specify the network FS capture directory.

```bash
conda activate sam3
BASE=/home/mingi/paradex1/capture/eccv2026/inspire_f1
SERIALS="22684755 23263780"

# 1. Download videos to local cache (~//video_cache/)
python src/perception/download_videos.py --base $BASE --serials $SERIALS

# 2. Run YOLOE (fast per-frame detection)
python -u src/perception/batch_mask_yoloe.py --base $BASE --serials $SERIALS

# 3. Run SAM3 fallback (video tracking for videos YOLOE missed)
python -u src/perception/batch_mask.py --base $BASE --serials $SERIALS

# 4. Upload results back to network FS
python src/perception/upload_results.py --base $BASE
```

**Notes:**
- SAM3 skips videos >1200 frames to avoid GPU OOM
- SAM3 tries prompt `"object"` first, then the object folder name as fallback
- YOLOE saves to `obj_mask_yoloe/`, SAM3 saves to `obj_mask/`
- Both skip videos that already have masks in the cache
- `upload_results.py` only copies (never deletes), skips existing files

### Validation

```bash
conda activate sam3
bash src/validation/perception/run_all.sh
```
