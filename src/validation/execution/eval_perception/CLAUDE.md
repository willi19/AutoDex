# Perception Evaluation Pipeline

Evaluates per-view 6D pose quality across all cameras to determine the best views for the local perception pipeline.

## Pipeline Steps

```
Step 0: Undistort raw images (step0_undistort.py, any env)
Step 1: SAM3 image model masks, 24 views (step1_mask.py, sam3 env)
Step 2: DA3 depth, 24 views batch (step2_depth_da3.py, dav3 env)
Step 3: FPose register all 24 views (step3_pose.py, foundationpose env)
Step 4: NMS + silhouette optimization → GT pose (step4_silhouette.py, foundationpose env)
Step 5: Rank views by ADD error vs GT (step5_evaluate.py, foundationpose env)
Step 6: Verify best 5 views (step6_verify.py, foundationpose env)
```

## Running

```bash
# Single episode
bash run_eval.sh /home/mingi/shared_data/mingi_object_test attached_container 20260317_172644

# All episodes for one object
bash run_eval.sh /home/mingi/shared_data/mingi_object_test attached_container

# All objects
bash run_eval.sh /home/mingi/shared_data/mingi_object_test
```

## Data Layout

Results are saved in-place within each capture directory:
```
{capture_dir}/
├── raw/images/          # Original (distorted) images
├── images/              # Undistorted images
├── cam_param/           # intrinsics.json, extrinsics.json, C2R.npy
├── masks/{serial}.png   # SAM3 binary masks
├── depth_da3/{serial}.png    # DA3 depth (uint16 mm)
├── depth_da3_vis/grid.png    # Depth colormap visualization
├── pose/
│   ├── {serial}.npy          # Per-view world pose (4x4)
│   ├── {serial}_cam.npy      # Per-view cam pose (4x4)
│   ├── gt.npy                # GT pose from silhouette optimization
│   ├── fpose_vis/            # Cross-view overlay grids
│   └── silhouette_vis/       # GT silhouette overlays
└── view_ranking.json         # Per-view error ranking
```

## Reference Implementation

**ALL perception code must follow `/home/mingi/shared_data/_object_6d_tracking/`.**

This is the distributed pipeline that has been validated to work correctly. Key files:
- `run/models/depth_server.py` — DA3 depth server
- `run/models/foundationpose_server.py` — FPose server
- `run/models/silhouette_server.py` — Silhouette optimization server
- `run/run_object_6d_pipeline_distributed.py` — Pipeline orchestration

## Rendering

Uses `Utils.py` from FoundationPose (`nvdiffrast_render`, `make_mesh_tensors`).
Requires pytorch3d installed in the env (foundationpose env has it).
`render_utils.py` is a thin wrapper that imports from `Utils.py`.

## Lessons Learned / Mistakes NOT to Repeat

### 1. `from_pretrained` vs `__init__` (CRITICAL)
- **Mistake**: Used `DepthAnything3(model_name="da3-large")` which creates a random-initialized model. Depth output was flat (0.94-1.00m range, 6cm variation) instead of correct (0.48-2.8m range).
- **Fix**: Must use `DepthAnything3.from_pretrained("depth-anything/DA3-LARGE")` to load pretrained weights.
- **Wasted time**: ~2 hours debugging "why FPose is wrong" when depth was garbage from random weights.
- **Root cause**: Did not read the reference `depth_server.py` which clearly uses `from_pretrained`.

### 2. Extrinsics must be passed to DA3
- **Mistake**: Claimed "intrinsics only gives accurate depth (1.16m)" and insisted on removing extrinsics. Actual correct depth was 1.37m for that view — 20% error.
- **Fix**: DA3 needs both intrinsics AND extrinsics for multi-view metric alignment. The reference `depth_server.py` passes both, with exception fallback.
- **Wasted time**: ~1 hour arguing that extrinsics weren't needed, when the reference code clearly passes them.

### 3. Mesh loading: `process=False` not `force="mesh"`
- **Mistake**: Used `trimesh.load(path, force="mesh")` which merges/deduplicates vertices (7944 vs 22743 vertices). FoundationPose uses the original unprocessed mesh.
- **Fix**: Use `trimesh.load(path, process=False)` like the reference `foundationpose_server.py`.

### 4. Do NOT rewrite rendering code
- **Mistake**: Created `render_utils.py` as a standalone nvdiffrast renderer instead of using the existing `Utils.py`. The custom renderer had bugs (missing rasterization mask, wrong y-axis handling) causing overlays to be completely wrong.
- **Fix**: Import directly from `Utils.py` via `render_utils.py` wrapper. Requires pytorch3d in the env.
- **Wasted time**: ~2 hours debugging "wrong overlay" that was caused by broken custom renderer.

### 5. Do NOT blame external factors
- **Mistake**: When results were wrong, blamed DA3 model, extrinsics convention, xformers, calibration, mesh scale — everything except my own code. Every time, the actual cause was my code not matching the reference.
- **Fix**: When something doesn't work, FIRST diff against the reference implementation. The answer is almost always in the reference code.

### 6. Read the reference code FIRST
- **Mistake**: Wrote eval pipeline from scratch without reading the reference `_object_6d_tracking` code. Made every possible mistake that the reference code already solved.
- **Fix**: Before writing ANY perception code, read the corresponding reference implementation end-to-end. Copy the exact function calls, parameters, and preprocessing steps.

### General Pattern
Every bug in this pipeline was caused by:
1. Not reading the reference code
2. Making assumptions instead of checking
3. Blaming external factors instead of checking my own code
4. "It's close enough" instead of making it exact

The reference code exists and works. Follow it exactly.
