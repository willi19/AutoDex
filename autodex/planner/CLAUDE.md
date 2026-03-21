# Planner

## Architecture

`GraspPlanner` in `planner.py` wraps cuRobo for collision-free trajectory generation.

### plan() Flow (Current)

1. Load grasp candidates (100 per object, `selected_100` version)
2. World setup (motion_gen + ik_solver)
3. Filter: backward (`wrist_se3[:, 0, 2] < 0.3`) + hand-table collision
4. **IK solve** on valid candidates → arm joints + pregrasp finger joints = `ik_qpos` (22 DOF)
5. **plan_single_js** on each IK-reachable candidate (INIT_STATE → ik_qpos) until success

### Previous Flow (Deprecated)

1. Filter → plan_batch (50 candidates at a time) → _refine_fingers (plan_single_js)
2. Problems:
   - plan_batch found arm trajectory, then _refine_fingers re-planned entire 22 DOF from scratch — arm plan result was thrown away
   - No IK pre-filter: IK-infeasible candidates wasted plan_batch time (timeout=60s each)
   - ~59s per plan() call vs current ~1-2s

### Key Insight: IK Reachability ≈ Planning Feasibility

- IK success → plan_single_js success rate: **95-100%**
- Failures only at boundary configurations (small x_offset 0.2-0.3, specific z_rotations)
- Boundary failures are stochastic (seed-dependent), not systematic
- Increasing cuRobo attempts/seeds doesn't meaningfully help boundary cases

## Timing (single GPU, no contention)

| Stage | Time |
|-------|------|
| First-time warmup (MotionGen + CUDA graph) | **~7s** (once per process) |
| GraspPlanner() constructor | 0.02s (lazy init) |
| load_candidates (100) | 0.01-0.05s |
| world_setup | 0.01-0.02s |
| filter (backward + collision) | 0.04-0.09s |
| IK solve (100 candidates batch) | **~0.1s** |
| plan_single_js (1 call) | **0.6-1.4s** |
| **Total per plan() call** | **~1-2s** |

Practical deployment: pre-initialize planner (warmup 7s), then each plan() is ~1-2s.

## cuRobo Configuration

### MotionGenConfig (init)
- `num_trajopt_seeds=1024`, `num_graph_seeds=1`, `num_ik_seeds=32`
- `trajopt_tsteps=64`, `interpolation_dt=0.01`
- `ik_opt_iters=200`, `grad_trajopt_iters=200`

### MotionGenPlanConfig (per plan call)
- `timeout=60.0`, `max_attempts=10`
- `num_trajopt_seeds=32`, `num_ik_seeds=32`
- `enable_graph=True`, `enable_finetune_trajopt=True`

### IKSolverConfig
- `num_seeds=32`, `collision_activation_distance=0.01`
- Uses table-only world (no target mesh — hand should be near object)

## cuRobo API Notes

- **plan_batch**: N independent start→goal pairs, N trajectories. Expensive. (deprecated in our flow)
- **plan_goalset**: 1 start, N goals, picks best goal each optimization iteration. 1 trajectory.
- **plan_single_js**: Joint-space planning, no IK needed. Fast. Used for current flow.
- All three share the same `_plan_cfg` (timeout, max_attempts).
- cuRobo is gradient-based: seed determines initial trajectory, different seeds can find different local optima.

## Robot

- 22 DOF: xarm6 (6) + allegro hand (16)
- Config: `{project_dir}/content/configs/robot/xarm_allegro.yml`
- INIT_STATE from `autodex.utils.robot_config`
- BATCH_SIZE = 50 (for IK solve_batch chunking)

## Validation

Results in `outputs/planning_success_rate/`. Visualization:
- Heatmap plots: `python src/validation/planning/plot_success_heatmap.py`
- Interactive viser viewer: `python src/validation/planning/planning_viewer.py --port 8080`

Grid: x_offset [0.2-0.5, step 0.05] × z_rotation [0°-330°, step 30°] × all tabletop poses × 5 trials/point.
