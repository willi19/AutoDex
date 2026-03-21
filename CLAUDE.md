# AutoDex

Dexterous manipulation pipeline: perception → planning → execution.

## Repository Structure

```
autodex/                    # Core library (importable package)
├── perception/             # Mask, depth, pose (has its own CLAUDE.md)
│   ├── mask.py             # YOLOE + SAM3 segmentation
│   ├── depth.py            # FoundationStereo + Depth-Anything-3
│   ├── pose.py             # FoundationPose tracking
│   ├── stereo_video_depth.py  # CLI batch stereo depth (TRT)
│   └── thirdparty/         # External model repos + weights
│       └── weights/        # yoloe-26x-seg.pt, mobileclip2_b.ts
├── planner/                # Motion planning (has its own CLAUDE.md)
│   └── planner.py          # GraspPlanner: IK → plan_single_js
├── executor/               # Robot execution
│   └── real.py             # Real robot executor
└── utils/
    └── file_io.py          # (WIP) Cache management, download/upload utilities

src/                        # Scripts & CLI wrappers
├── process/                # Batch processing pipelines
│   ├── batch_mask.py       # SAM3 video segmentation
│   ├── batch_mask_yoloe.py # YOLOE single-frame segmentation
│   ├── batch_mask_all.py   # Run mask on all cameras
│   ├── batch_depth.py      # Stereo depth (fixed camera pair, TRT)
│   ├── batch_depth_auto.py # Stereo depth (auto pair selection, all cameras, TRT)
│   ├── batch_pose.py       # FoundationPose batch tracking
│   ├── batch_pose_overlay.py # Per-camera pose tracking + mesh overlay video
│   ├── download_videos.py  # Network FS → local cache
│   └── upload_results.py   # Local cache → network FS
├── demo/                   # Demo scripts
│   ├── real.py             # Real robot demo
│   ├── perception_exp.py   # Perception experiment runner
│   └── run_perception.py   # Perception pipeline demo
├── visualization/
│   ├── mesh_process/       # Mesh viewers (object, scene, table_top)
│   └── turntable_grasp.py  # Turntable video renderer for grasp candidates
├── grasp_generation/
│   ├── BODex/              # Dexterous grasp generation (has its own CLAUDE.md)
│   │   ├── generate.py     # Main entry point
│   │   ├── run.sh          # Batch runner — edit object list here
│   │   └── src/curobo/     # Forked cuRobo library
│   └── sim_filter/         # MuJoCo validation + set cover selection (has its own CLAUDE.md)
└── validation/             # Validation & comparison scripts
    └── perception/         # Perception pipeline validation (has its own CLAUDE.md)
        ├── scene.py        # Single-scene overlay validation
        ├── stereo_rectify.py  # Stereo rectification visualization
        ├── viz_stereo_pairs.py # Visualize auto-selected stereo pairs
        └── multiobject/    # Multi-object combinatorial validation pipeline

bodex_outputs/              # BODex grasp generation results (gitignored)
logging/                    # Run logs (grasp_generation/generate.log)

Visualization/              # Scene visualization & evaluation
├── scene.py                # Viser-based scene viewer
└── ...                     # Evaluation, paper figures
```

## Key Conventions

- **Local cache**: `~/video_cache/` mirrors network FS structure. Mapping: strip `/home/mingi/paradex1/capture/` prefix.
- **Video format**: `.avi` throughout. Masks use MJPG codec. Depth uses FFV1 (lossless, uint16 mm encoded as BGR).
- **Camera params**: `cam_param/intrinsics.json` + `extrinsics.json` per capture dir. Keyed by serial string.
  - `intrinsics.json` values are dicts with `intrinsics_undistort`, `original_intrinsics`, `dist_params`, `width`, `height`.
- **Capture dir layout**: `{base}/{obj_name}/{idx}/` with `videos/`, `cam_param/`, `depth/`, `obj_mask/`, etc.
- **Model weights**: All in `autodex/perception/thirdparty/weights/`. Use `YOLOE_WEIGHTS` from `autodex.perception.mask` to reference.

## Stereo Depth Pipeline

Two depth scripts exist:

- **`batch_depth.py`**: Fixed camera pair (manual `--left_serial` / `--right_serial`). Proven, simple.
- **`batch_depth_auto.py`**: Auto pair selection for all cameras. Uses rig-based adjacency grouping (focal_group × z_level, angle-sorted, MAX_ANGLE_GAP=40°).

### Stereo Rectification

Both use `cv2.stereoRectify` to get R1/R2 rotation matrices.
`src/process/depth.py` uses the **validation approach** (same as `stereo_rectify.py`):
- Uses `f_orig = max(K_left[0,0], K_right[0,0])` instead of stereoRectify's `f_rect` (which can be degenerate for wide-baseline pairs).
- Oversized canvas → valid region → workspace crop → final P matrix.
- Both left/right use the same P matrix (same f, cx, cy), preserving epipolar alignment.

### Stereo Rectification Cropping Rules (IMPORTANT)

1. **Valid region**: Use UNION (`valid_l | valid_r`) — NEVER intersection, it cuts off the right camera's content.
2. **Workspace crop**: Crop TO the fixed robot-frame bounding box, baked into the P matrix via `initUndistortRectifyMap` (one remap step, no intermediate full-size image).
   - Fixed bounds in robot frame: `ws_min=[0.35, -0.30, 0.0]`, `ws_max=[0.80, 0.21, 0.4]`
   - Same constants for every capture — determined once from charuco triangulation, NOT recomputed.
   - Project 8 bbox corners via `C2R.npy` + extrinsics + R1/R2 into rectified space.
   - Take UNION of both cameras' projections so the full bbox is visible in both views.
3. **Same cx** for both cameras — no per-camera cx offset, no disparity correction needed.
4. **Aspect ratio filter**: Skip pairs with ratio > 2.5:1 (degenerate wide-baseline pairs).
5. **Objects must NEVER be cut off** — the bbox must fully contain the workspace.

### Disparity-to-Depth: Rectified Z vs Original Z

**Critical**: The stereo formula `depth = f * B / disparity` gives Z in the **rectified** camera frame, not the original camera frame. When un-rectifying depth back to original pixel coordinates, you must divide by `rz` — the Z component of `R1 @ K_inv @ [u, v, 1]` for each original pixel `(u, v)`:

```
Z_orig = Z_rect / rz
```

Without this, cameras with large R1 rotation (e.g. 65° for wide-baseline pairs) get ~30-50% depth error, causing massive cross-view reprojection misalignment. Cameras with small R1 rotation (~20°) appear fine because `rz ≈ 1`.

This bug was subtle because:
- Per-camera depth colormaps look visually correct (relative depth ordering is preserved)
- Self-reprojection (pixel → 3D → same pixel) is trivially perfect for any depth value
- Only cross-view reprojection reveals the error, and the magnitude depends on R1 rotation angle

### Depth Debugging Checklist

When stereo depth looks wrong, use **cross-view reprojection** to validate — NOT per-camera colormaps or self-reprojection (both hide errors). Steps:
1. Pick a source camera with depth, backproject to 3D world using `K_src`, `T_src`
2. Reproject 3D points to a different camera using `K_tgt`, `T_tgt`
3. Overlay reprojected points on the target camera's image — features (checkerboard, objects) should align
4. If misaligned: check `rz` correction, stereo pair quality (R1 rotation angle), baseline/focal length

`batch_depth_auto.py --overlay_only` generates cross-view reprojection grids in `depth_overlay/`.

### Depth Encoding

FFV1 codec, uint16 millimeters as BGR: `B = low_byte, G = high_byte, R = 0`.
Use `encode_depth_uint16()` / `decode_depth_uint16()` from `autodex.perception.depth`.

## Conda Environments

- `foundation_stereo`: FoundationStereo TRT depth (`tensorrt` + `pycuda` installed here)
- `foundationpose`: FoundationPose, YOLOE
- `sam3`: SAM3 segmentation, Depth-Anything-3

## Grasp Candidate Visualization

`src/visualization/turntable_grasp.py` renders turntable videos of grasp candidates (object + Allegro hand) using Open3D offscreen renderer (EGL headless).

### Data Sources

- **Candidates**: `{candidate_path}/{version}/{obj_name}/{scene_type}/{scene_id}/{grasp_name}/` — contains `wrist_se3.npy`, `grasp_pose.npy`, `pregrasp_pose.npy`
- **Setcover order**: `{code_path}/order/{version}/{obj_name}/setcover_order.json` — ranked grasp list (greedy set cover)
- **Object meshes**: `{obj_path}/{obj_name}/raw_mesh/{obj_name}.obj`
- **Object pose**: `{obj_path}/{obj_name}/scene/table/4.json`
- **Robot URDF**: `{urdf_path}/allegro_hand_description_right.urdf`

Paths from `rsslib.path`: `candidate_path=/home/mingi/RSS_2026/candidates`, `code_path=/home/mingi/RSS_2026`, `obj_path=/home/mingi/shared_data/RSS2026_Mingi/object/paradex`.

### Setcover Versions (no duplicates except attached_container → use revalidate)

- `revalidate`: 33 objects
- `v2`: 21 objects (20 unique after dedup)
- `v3`: 45 objects
- Total: 98 unique objects

### Output Layout (episode-wise for HuggingFace/GitHub Pages)

```
data/{obj_name}/{rank:03d}/turntable.mp4
```

### Commands

```bash
# Single grasp
python src/visualization/turntable_grasp.py --version revalidate --obj soap_dispenser --scene shelf/1/11

# Top N from setcover
python src/visualization/turntable_grasp.py --version revalidate --obj soap_dispenser --top 100

# All 98 objects × top 100
python src/visualization/turntable_grasp.py --batch-all --top 100
```

### Camera Auto-framing

Uses bounding sphere of combined object+robot mesh. Camera distance = `sphere_radius * padding / sin(effective_half_fov)`. Guarantees no clipping at any turntable angle.

## Planning Validation

### Reachability Grid Search (`src/validation/planning/reachability_set.py`)

Runs IK-only checks over a grid of (x_offset, z_rotation, tabletop_pose) per object.
- Grid: x_offset [0.2–0.5, step 0.05] × z_rotation [0°–330°, step 30°] × all tabletop poses × 10 trials/point
- Output: `outputs/reachability/{obj_name}/reachability_selected_100.json` (grid results) + `*_viz.json` (IK solutions for visualization)
- 14 objects processed. IK is ~97% deterministic (all-or-nothing), ~3% partial at boundary configs.

### Reachability Viewer (`src/validation/planning/reachability_viewer.py`)

Interactive viser viewer for reachability results. Two modes:
- **Single**: Robot at IK qpos + object mesh + 5 grasp candidate hands (green=forward, red=backward). Sliders for pose, x_offset, z_rotation, IK solution #.
- **Heatmap**: 1D row of color-coded spheres along x_offset (green=reachable, red=unreachable, yellow=partial). Z_rotation controlled by slider. Robot shown at INIT_STATE default pose. Object mesh shown offset in y for reference.
- **Filter/Navigate**: Jump between Reachable/Unreachable/Partial points.

```bash
python src/validation/planning/reachability_viewer.py --port 8080
```

### Planning Success Rate (`src/validation/planning/success_rate.py`)

Compares IK reachability vs full planning success. Loads IK-reachable points from `outputs/reachability/`, runs `planner.plan()` only on those. Breaks early on success (only retries on failure). Saves per-stage timing breakdown (all/success/fail) to JSON.

```bash
python src/validation/planning/success_rate.py --obj attached_container --version selected_100 --n_trials 1
```

Output: `outputs/planning_success_rate/{obj_name}/plan_vs_ik_{version}.json`

### Key Constants

- `TABLE_POSE_XYZ = [1.1, 0, -0.1]`, `TABLE_DIMS = [2, 3, 0.2]`
- `INIT_STATE`: from `autodex.utils.robot_config` — xarm6 + allegro default joint config
- Grasp candidates: `selected_100` version, 100 per object via `load_candidate()`
- Backward filter: `wrist_se3[:, 0, 2] < 0.3`

## External Dependency: `rsslib` (legacy, being replaced)

Old shared library at `~/RSS_2026/rsslib/`. Still imported by `src/`, `Visualization/`, and BODex scripts. **All core functionality already ported to `autodex/`:**

- `rsslib.path` → `autodex.utils.path` (identical functions, `project_dir` changed to `~/shared_data/AutoDex`)
- `rsslib.conversion` → `autodex.utils.conversion` (identical: `cart2se3`, `se32cart`, `se32action`)
- `rsslib.robot_config` → `autodex.utils.robot_config` (identical: `INIT_STATE`, `LINK6_TO_WRIST`)
- `rsslib.scene` → `autodex.utils.scene` (partial: `overlay_scene`, `get_scene_image_dict_template`)
- `rsslib.curobo_util` → `autodex.planner.planner.GraspPlanner` (fully superseded — `CuroboPlanner`/`CuroboIkSolver`/`filter_collision`/`get_traj` are all methods on `GraspPlanner`)
- `rsslib.planner` → `autodex.planner.planner` (superseded)
- `rsslib.visualizer`, `rsslib.gui_player` — not yet ported (Viser GUI helpers)

Migration: mechanically replace `from rsslib.xxx import` → `from autodex.utils.xxx import` in `src/` scripts. BODex internal imports need separate handling (it's a forked cuRobo with its own `rsslib` refs).

## Grasp Generation: BODex (`src/grasp_generation/BODex/`)

GPU-accelerated dexterous grasp generation built on a **forked NVIDIA cuRobo**. Optimizes Allegro hand joint angles + wrist pose for force-closure grasps with collision avoidance. See `src/grasp_generation/BODex/CLAUDE.md` for architecture details.

## Grasp Pipeline: BODex → Sim Filter → Candidate → Selection

BODex raw outputs go through sim validation and selection before planning. See `src/grasp_generation/sim_filter/CLAUDE.md` for details.

```
bodex_outputs/ → MuJoCo sim eval → candidates/ → set cover selection → selected_100/
```

Key data (currently at `~/RSS_2026/`): `candidates/{version}/`, `order/{version}/`, `candidates/selected_100/`.

## Perception Evaluation Pipeline (`src/validation/execution/eval_perception/`)

Evaluates per-view 6D pose quality to select the best camera views.
See `src/validation/execution/eval_perception/CLAUDE.md` for details.

### CRITICAL: Reference Implementation

**`/home/mingi/shared_data/_object_6d_tracking/`** is the reference (ground truth) pipeline. When writing perception code, ALWAYS read and follow the reference implementation first. Do NOT improvise or write from scratch.

Key reference files:
- `run/models/depth_server.py` — DA3 depth: `DepthAnything3.from_pretrained()`, intrinsics + extrinsics, fallback on exception
- `run/models/foundationpose_server.py` — FPose: `trimesh.load(process=False)`, downscale=0.5, `mask.astype(bool)`
- `run/models/silhouette_server.py` — Differentiable silhouette optimization (200 iters, MSE + IoU loss, rotation 6d parameterization)
- `run/run_object_6d_pipeline_distributed.py` — Full pipeline orchestration, NMS, visualization with `nvdiffrast_render`

### HARD RULES (learned from mistakes)

1. **NEVER use `DepthAnything3(model_name=...)` — ALWAYS use `DepthAnything3.from_pretrained("depth-anything/DA3-LARGE")`**. The constructor creates random-init weights. `from_pretrained` loads actual trained weights.
2. **NEVER omit extrinsics from DA3** — multi-view alignment with extrinsics gives correct metric depth. Without extrinsics, depth scale is wrong.
3. **NEVER use `trimesh.load(force="mesh")`** for FoundationPose — use `process=False`. `force="mesh"` merges/deduplicates vertices (7944 vs 22743), changing the mesh geometry.
4. **NEVER rewrite rendering code** — use `Utils.py`'s `nvdiffrast_render` and `make_mesh_tensors` directly. Import requires pytorch3d in the env.
5. **When something doesn't work, read the reference code first** — don't blame external factors (DA3, extrinsics, calibration, xformers).

## Conda Environments (Updated)

- `foundation_stereo`: FoundationStereo TRT (`tensorrt` + `pycuda`)
- `foundationpose`: FoundationPose, YOLOE, pytorch3d, nvdiffrast
- `sam3`: SAM3 segmentation
- `dav3`: Depth-Anything-3 (separate env with all DA3 dependencies)

## Daemon Setup (Perception Pipeline)

### Architecture

- **Main PC** (mingi, RTX 3090): DA3/stereo depth + silhouette matching + planning
- **capture1, 2, 3**: SAM3 daemons (ZMQ, port 5001)
- **capture4, 5, 6**: FPose daemons (ZMQ, port 5003)

### First-time setup on each capture PC

```bash
# 1. Clone repo
git clone https://github.com/willi19/AutoDex.git ~/AutoDex
cd ~/AutoDex

# 2. Download weights from NAS
bash scripts/setup_weights.sh

# 3. For FPose PCs: copy mycpp build (python 3.9 required)
mkdir -p ~/AutoDex/autodex/perception/thirdparty/FoundationPose/mycpp/build
cp ~/shared_data/AutoDex/weights/foundationpose/mycpp_build/mycpp*.so \
   ~/AutoDex/autodex/perception/thirdparty/FoundationPose/mycpp/build/
```

### SAM3 daemon (capture1, 2, 3)

```bash
# Conda env setup (once)
conda create -n sam3 python=3.12 -y
conda activate sam3
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install pyzmq ftfy regex psutil pycocotools einops iopath hydra-core timm tqdm pillow scipy huggingface_hub opencv-python numpy
python -c "from huggingface_hub import login; login(token='<HF_TOKEN>')"

# Run daemon
conda activate sam3
cd ~/AutoDex
python src/execution/daemon/perception_daemon.py --model sam3 --port 5001
```

### FPose daemon (capture4, 5, 6)

```bash
# Conda env setup (once)
conda create -n foundationpose python=3.9 -y
conda activate foundationpose
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install pyzmq opencv-python numpy trimesh nvdiffrast omegaconf
pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation

# Run daemon
conda activate foundationpose
cd ~/AutoDex
python src/execution/daemon/perception_daemon.py --model fpose --port 5003 \
    --mesh ~/shared_data/object_6d/data/mesh/attached_container/attached_container.obj
```

### Update on capture PCs

```bash
cd ~/AutoDex && git fetch origin && git reset --hard origin/main
```

### NAS Weight Structure

```
~/shared_data/AutoDex/weights/
├── foundationpose/
│   ├── 2023-10-28-18-33-37/   # RefinePredictor
│   ├── 2024-01-11-20-02-45/   # ScorePredictor
│   └── mycpp_build/           # Pre-built C++ extension
├── sam3/
│   └── sam3.pt                # SAM3 checkpoint (3.3GB)
├── da3/
│   └── model.safetensors      # DA3-LARGE (1.6GB)
└── yoloe/
    ├── yoloe-26x-seg.pt       # YOLOE (164MB)
    └── mobileclip2_b.ts       # MobileCLIP (243MB)
```

## Ongoing Refactoring

`src/process/` scripts have heavy code duplication with `autodex/perception/`.
Plan: consolidate core logic into `autodex/`, make `src/process/` thin CLI wrappers.
See `autodex/perception/CLAUDE.md` for detailed plan.
