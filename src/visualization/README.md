# Visualization

## What we visualize

### 1. Object 6D Pose Estimation Pipeline (`paradex/object6d/`)
Visualizes each step of the perception pipeline as a 1x6 grid:
1. **Object Mesh** - 3D mesh rendered from front view with tabletop pose
2. **Input RGB** - Raw camera frame from the multi-camera system
3. **Foundation Stereo** - Stereo depth estimation (TURBO colormap blended on RGB)
4. **SAM Mask** - Segmentation mask overlaid on RGB (green tint + contour)
5. **Foundation Pose** - 6D pose estimate rendered as mesh overlay
6. **Silhouette Opt.** - Silhouette-based pose refinement (WIP)

Output: `output/{hand}/{obj}/{index}/` with individual panels + combined grid + per-camera images + stereo pairs + sampled frames.

Usage:
```bash
python src/visualization/paradex/object6d/plot_pipeline.py --hand inspire_f1 --obj attached_container --index 1
```

### 2. Tabletop Pose Generation (`mesh_process/table_top.py`)
Interactive 3D viewer (viser) for inspecting object stable placement results:
- Objects arranged on a grid, each showing a candidate tabletop pose
- Color coded: green = valid stable pose, red = failed
- OBB (oriented bounding box) wireframes drawn on each object
- Local coordinate axes (XYZ) and face center/normal vectors shown
- Supports trajectory animation and video recording

### 3. Grasp Planning Results (`autodex/visualizer/`)
Interactive 3D viewer (viser) for inspecting motion planning output:
- **Overview mode**: All grasp candidates shown with color coding
  - Green = successful plan, Yellow = planning failed, Red = collision
  - Filterable via checkboxes
- **Trajectory mode**: Select a successful grasp and scrub through the arm trajectory with a timeline slider
- Scene objects (table, target mesh) rendered alongside robot URDFs

### 4. Result Visualization (`result/`)
(Empty - placeholder for experiment result plots)

### 5. Grasp Generation (`grasp_generation/`)
(Empty - placeholder for grasp candidate visualization)

## Shared Infrastructure

All 3D viewers use **paradex's `ViserViewer`** (`paradex.visualization.visualizer.viser`), a web-based viewer built on [viser](https://github.com/nerfstudio-project/viser).

Scene config quaternion convention: `[w, x, y, z]` in `scene_cfg`, converted to scipy `[x, y, z, w]` via `R.from_quat([qx, qy, qz, w])`.
