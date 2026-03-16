# src/validation/perception

Debugging and validation pipeline for the `autodex.perception` module. Runs mask, depth, and pose independently, then compares all combinations visually.

## Structure

```
src/validation/perception/
├── scene.py                     # Single-scene overlay: capture → pose → mesh projection grid
└── multiobject/                 # Multi-object combinatorial validation pipeline
    ├── run_all.sh               # End-to-end runner (switches conda envs per step)
    ├── step1_mask.py            # Mask generation (SAM3 or YOLO-E)
    ├── step2_depth.py           # Depth estimation (DA3 or FoundationStereo)
    ├── step3_pose.py            # FoundationPose + NMS → mesh overlay
    └── step4_compare.py         # Cross-config comparison grids + timing CSV
```

## Pipeline Overview (`multiobject/`)

The pipeline validates perception by running **all combinations** of mask × depth methods through pose estimation:

```
Step 1 (mask)     Step 2 (depth)         Step 3 (pose)              Step 4
─────────────     ──────────────         ─────────────              ──────
sam3        ──┬── da3              ──┬── sam3_da3        ──┐
yoloe       ──┘   foundationstereo ──┘── sam3_fs           ├── compare/
                                     ├── yoloe_da3         │   summary.csv
                                     └── yoloe_fs        ──┘   {serial}_compare.png
                                                               depth_compare.png
```

### Output Directory Layout
```
{data_dir}/validation_output/
├── segmentation/
│   ├── sam3/                    # step1 output
│   │   ├── images/              # undistorted images (shared via symlink)
│   │   ├── camera_data.npz      # serials, intrinsics, extrinsics arrays
│   │   ├── object_info.json     # {obj_name: {text: prompt}} from input
│   │   ├── objects/{name}/masks/        # binary masks per camera
│   │   ├── objects/{name}/masks_debug/  # green overlay debug images
│   │   ├── masks_combined/grid.png      # all objects colored, all cameras
│   │   └── timing.json
│   └── yoloe/                   # same structure, symlinks images from sam3
├── depth/
│   ├── da3/
│   │   ├── depth/               # uint16 PNG (millimeters)
│   │   ├── depth_overlay/       # cross-camera reprojection overlays
│   │   ├── source_info.json     # {method, pairs, source_serial}
│   │   └── timing.json
│   └── foundationstereo/        # same structure
├── pose/{seg}_{depth}/
│   ├── objects/{name}/
│   │   ├── ob_in_cam/           # 4x4 pose per camera (.txt)
│   │   ├── ob_in_world/         # 4x4 pose in world frame (.txt)
│   │   ├── selected_pose_world.txt  # NMS-selected best pose
│   │   └── visualizations/      # purple mesh overlay per camera + grid.png
│   ├── sources.json             # points to seg_dir and depth_dir
│   └── timing.json
└── compare/
    ├── summary.csv              # timing table across all configs
    ├── depth_compare.png        # depth colormap grid (all methods × all cameras)
    └── {serial}_compare.png     # mask + pose overlay per config
```

## Step Details

### Step 1: Mask (`step1_mask.py`)
- **Input**: `--data_dir` with raw images + `object_info.json`, or `--reuse_images_from` existing output
- **Methods**: `sam3` (conda: sam3) | `yoloe` (conda: foundationpose)
- SAM3 uses `Sam3Processor.set_text_prompt()` per image; YOLO-E uses `get_mask_yoloe()` from autodex.perception
- Saves per-object masks, debug overlays, and combined colored grid

### Step 2: Depth (`step2_depth.py`)
- **Input**: `--seg_dir` (reads camera_data.npz and images from step1 output)
- **Methods**: `da3` (conda: sam3) | `stereo` (conda: foundation_stereo or foundationpose)
- DA3: per-camera monocular depth via `DepthAnything3.from_pretrained()`
- Stereo: auto-selects best camera pair (`find_best_stereo_pair`), runs one stereo pass, reprojects 3D points to all cameras
- Stereo pair selection: filters by baseline (0.03-0.5m), view alignment (cos_sim > 0.77), perpendicularity, focal ratio
- Depth saved as uint16 PNG (mm); cross-camera reprojection overlay for consistency check

### Step 3: Pose (`step3_pose.py`)
- **Input**: `--seg_dir`, `--depth_dir`, `--mesh_dir`
- Runs `PoseTracker.init()` independently per camera (reset each time)
- NMS via world-frame AABB IoU to select best single pose across cameras
- Renders purple mesh overlay on all cameras using nvdiffrast
- `--downscale 0.5` by default (half resolution for speed)
- Mesh lookup: `{mesh_dir}/{obj_name}/{obj_name}.obj` → `processed_data/mesh/simplified.obj` → glob fallback

### Step 4: Compare (`step4_compare.py`)
- **Input**: `--output_base` (auto-detects pose/ subdirs)
- Per-camera grid: original | mask overlays | pose overlays, one row per config
- Depth grid: colormap overlays for all depth methods with source labels
- Summary CSV: timing breakdown (mask_s, depth_s, pose_s, total_s)

## scene.py (Single Scene Validation)
```bash
# Live capture + pose
python src/validation/perception/scene.py --obj attached_container --ref_idx 000

# From existing capture
python src/validation/perception/scene.py --obj attached_container --ref_idx 000 \
    --img_dir ~/shared_data/RSS2026_Mingi/experiment/demo/attached_container/20260310_120000
```
- Uses `paradex.image.ImageDict` for undistortion and mesh projection
- `get_scene_image_dict_template()` runs 6D pose estimation via reference template
- Green mesh overlay on all camera views + blue table cuboid, merged into grid

## Conda Environments
- `sam3`: SAM3 mask, DA3 depth
- `foundationpose`: YOLO-E mask, FoundationPose (step3)
- `foundation_stereo`: FoundationStereo depth (PyTorch)

## Key Input Format
`object_info.json` (placed in data_dir):
```json
{
  "object_name": {"text": "text prompt for segmentation"},
  "banana": {"text": "yellow banana"}
}
```

## Dependencies on autodex.perception
- `step1_mask.py`: `get_mask_yoloe`, `get_mask_sam3`
- `step2_depth.py`: `get_depth_da3`, `get_depth_stereo`, `get_depth_stereo_pytorch`
- `step3_pose.py`: `PoseTracker`