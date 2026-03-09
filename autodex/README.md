# AutoDex

Perception-planning-simulation pipeline for robotic dexterous grasping.

## Modules

### `perception/`
Computer vision for object detection, depth estimation, and 6D pose tracking.

- **`mask.py`** — Object segmentation via YOLO-E (`get_mask_yoloe`) and SAM3 (`get_mask_sam3`)
- **`depth.py`** — Stereo depth (FoundationStereo / TensorRT) and monocular depth (Depth-Anything-3)
- **`pose.py`** — 6D pose initialization and tracking using FoundationPose (`PoseTracker`)

### `planner/`
Collision-aware grasp planning and trajectory optimization.

- **`planner.py`** — `GraspPlanner` loads grasp candidates, filters via collision checking (cuRobo), and optimizes arm trajectories. `PlanResult` stores the output (trajectory, wrist pose, hand joints).

### `executor/`
Trajectory visualization (no physics).

- **`sim.py`** — `SimExecutor.visualize()` replays a planned trajectory in a Viser 3D viewer with the robot and a static object mesh.

### `simulator/`
Physics simulation for grasp validation.

- **`mujoco_sim.py`** — `Simulator` runs MuJoCo-based physics with position-controlled robots and free-joint objects. Supports multiple robots/objects per environment. Used to verify grasps under gravity, friction, and contact dynamics.

### `utils/`
Shared file I/O utilities.

## Pipeline

```
Perception (camera input)
    → mask, depth, 6D pose
        → Planner (grasp candidates + trajectory optimization)
            → Executor (visualize plan)
            → Simulator (validate with physics)
```

## Demo (`src/demo/`)

### `sim.py` — Sanity check
Runs the full pipeline end-to-end to verify the code is working properly:
1. Capture scene → estimate object 6D pose
2. Load grasp candidates → filter collisions → plan trajectories
3. Visualize successful/failed plans in Viser

This is a **code validation** tool, not the real demo loop.

### Real demo workflow (planned)
The actual demo runs a continuous grasp-and-lift loop:
1. Place an object randomly on the table
2. Perceive the object (mask, depth, 6D pose)
3. Plan a grasp trajectory
4. Execute: robot grasps and lifts the object
5. Reset the robot to home position
6. Swap in a new object and repeat
