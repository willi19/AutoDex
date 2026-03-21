# Sim Filter — Scene Collision Check + MuJoCo Grasp Validation

Filters BODex raw grasps through (1) scene collision check and (2) MuJoCo force-closure simulation, then copies passing grasps as candidates.

## Running

**Requires `mingi` conda env** (for cuRobo collision check + MuJoCo).

```bash
# Single object
python src/grasp_generation/sim_filter/run_sim_filter.py --hand allegro --version v3 --obj attached_container

# Inspire hand
python src/grasp_generation/sim_filter/run_sim_filter.py --hand inspire --version v3 --obj attached_container

# All objects (reads src/grasp_generation/obj_list.txt)
python src/grasp_generation/sim_filter/run_sim_filter.py --hand allegro --version v3

# With MuJoCo viewer for debugging
python src/grasp_generation/sim_filter/run_sim_filter.py --hand allegro --version v3 --obj attached_container --viewer
```

### Re-running (clear previous results)

```bash
find ~/AutoDex/bodex_outputs/allegro/v3/attached_container -name "sim_eval.json" -delete -o -name "sim_traj.json" -delete
```

Skip logic: seeds with existing `sim_eval.json` are skipped.

## Pipeline (2 stages in one script)

### Stage 1: Scene Collision Check (cuRobo, GPU, fast)

For each scene, batch-checks all grasp seeds against scene obstacles (table, box walls, shelf, etc.) using `autodex.planner.GraspPlanner._check_collision()`.

- Transforms `wrist_se3` (object frame) to world frame using scene's object pose
- Checks hand-scene collision via cuRobo SDF
- Seeds that collide → immediately saved as `{"success": false, "reason": "scene_collision"}`
- Typically filters out 30-70% of seeds before sim eval

### Stage 2: MuJoCo Simulation (CPU, slow — collision-free seeds only)

Tests each surviving seed in MuJoCo physics:

1. **Reset** to pregrasp pose (wrist via mocap weld, fingers via position control)
2. **Pregrasp → Grasp** interpolation (10 outer steps × 10 inner steps)
3. **Grasp → Squeeze** interpolation (squeeze = `grasp * 2 - pregrasp`)
4. **6-direction force test**: apply `1 × obj_mass` force at object CoM in ±x, ±y, ±z
5. Each force held for 2 seconds (50 × 10 steps × 0.004s)
6. **Success** = object stays within **5cm translation** and **15° rotation** for ALL 6 directions
7. Early stop on first failure

### Physics Settings (matching RSS_2026)

| Parameter | Value |
|-----------|-------|
| Gravity | **Disabled** (`mjDSBL_GRAVITY`) |
| Object mass | 0.1 kg (fixed for all objects) |
| External force | `1.0 × obj_mass` = 0.1N per direction |
| Timestep | 0.004s |
| Friction | `[0.6, 0.02]` (tangent, torsion) |
| Contact cone | Elliptic |
| noslip_iterations | 2 |
| impratio | 10 |
| Actuator kp | 5.0 (position control) |

### Hand Models

| Hand | Model file | Weld body | Notes |
|------|-----------|-----------|-------|
| Allegro | `assets/hand/allegro/right_hand.xml` | `world` | MuJoCo XML with actuators (kp=5) |
| Inspire | BODex URDF (`inspire_hand_right.urdf`) | `wrist` | Actuators auto-generated, mimic joints via equality constraints |

**IMPORTANT**: Allegro uses a MuJoCo XML (from `mujoco_menagerie`), NOT the URDF. The XML has proper actuator definitions (`<position kp="5"/>`), joint naming (`ffj0`, `mfj0`, etc.), and collision geometry tuned for simulation. Using the URDF directly with manually created actuators (`gainprm=500`) causes instability.

### Wrist Coordinate Transform

BODex's `wrist_se3` uses a different coordinate convention. Before simulation:
```python
wrist_se3[:3, :3] = wrist_se3[:3, :3] @ inv(R_DELTA)
```
where `R_DELTA = quat2mat([0, 1, 0, 1] / norm)`. This matches the `quat="0 1 0 1"` on the palm body in the Allegro XML.

## Output

Per seed (saved in the bodex_outputs seed directory):
- `sim_eval.json` — `{"success": bool, "hand": str, "version": str}` (or `"reason": "scene_collision"` if coll-filtered)
- `sim_traj.json` — `{"robot_qpos": [...], "object_pose": [...], "phase": [...]}` for replay in viewer

Passing seeds are copied to `bodex_outputs/{hand}/{version}_candidates/{obj}/{scene_type}/{scene_id}/{seed}/`.

## Data Flow

```
bodex_outputs/{hand}/{version}/{obj}/
    ↓ scene collision check (cuRobo GPU, per-scene batch)
    ↓ MuJoCo sim eval (collision-free seeds only)
    sim_eval.json + sim_traj.json per seed
    ↓ passing seeds copied
bodex_outputs/{hand}/{version}_candidates/{obj}/
    ↓ set cover selection (src/grasp_generation/order/compute_order.py)
bodex_outputs/{hand}/{version}_order/{obj}/setcover_order.json
```

## Key Files

- `run_sim_filter.py` — Main script (collision check + sim eval + candidate extraction)
- `autodex/simulator/hand_object.py` — `MjHO` class (ported from RSS_2026 `hand_util.py`)
- `autodex/simulator/rot_util.py` — Interpolation and pose delta utilities
- `assets/hand/allegro/right_hand.xml` — Allegro hand MuJoCo XML
- `assets/hand/allegro/meshes/` — Allegro hand mesh STLs