# Visualization

All 3D viewers use paradex's `ViserViewer` (`paradex.visualization.visualizer.viser`) unless noted otherwise.

## Scripts

### bodex_output.py
BODex grasp results with contact points. Shows object mesh, Allegro hand at pre-grasp/grasp poses, contact points, and optional trajectory animation. Filterable by success/fail.

### candidate.py
Grasp candidates with simulation evaluation. Shows collision-checked grasps color-coded by sim success/fail, with animated simulation trajectory and contact points from `sim_eval/viz_data.json`.

### scene.py
Browse scene configurations (table, packed, wall). Shows target mesh and cuboid obstacles with pose transforms. Scene type dropdown + index slider.

### scene_coverage.py
Scene coverage analysis across multiple objects. Grid layout of scenes with Xarm+Allegro robot and grasp trajectories from `coverage_scene/` directory.

### sim_eval.py
Grasp evaluation with contact analysis. Shows contact points between hand and object (green=object, red=robot), with metrics (grasp_error, dist_error) from `bodex_info.npy`.

### squeeze_result.py
Squeeze execution results. Multiple Allegro hands color-coded (green=success, red=fail), with squeeze motion animation (10 phases) and collision detection against table/obstacles.

### table_top.py
Object tabletop pose generation. Grid of candidate poses with OBB wireframes, local coordinate axes, face center/normal vectors. Color-coded success/failure.

### test_scene.py
Test scenes with Xarm+Allegro at initial pose. Browse `test_scene_test/` scenes with cuboid obstacles.

## Subdirectories

### paper/
Publication figure generation. Recording scripts for figures 1, 2, 4 — each captures video sequences of grasp execution in wall/shelf/box scenes. `make_final.py` and `make_stats.py` produce final figures.

### supple/
Supplementary material visualizations:
- `grid_view/` — large-scale grid of scenes with auto-capture for video (fade in/out grasp sequences)
- `dataset/` — dataset visualization and recording
- `selection/` — grasp selection and ordering
- `system/` — full system demo (grasp selection, planning, introduction)
- `motivation/` — motivation visualizations (wall zoom-out, sim recording)

### quant/
Quantitative plots (matplotlib):
- `validity.py` — coverage vs. number of grasps (log-scale), comparing real methods (teleop, ours) vs. sim methods (tabletop, collision, sampling, floating). Outputs `coverage_vs_grasps.pdf`.
