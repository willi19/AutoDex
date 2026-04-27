# robothome viewer — handoff notes

The goal of this directory is a viser-based GUI for FR3 + Inspire-left
**grasp planning validation**: given an object pose from perception, can the
robot reach a stored BODex grasp candidate without colliding, and what does
the trajectory look like? Everything else here exists to support that goal.

## What I was actually trying to do (intent)

The user wanted a tool to:

1. Load real perception output (robothome `tracking_*.json`) and place objects
   at those poses.
2. For a chosen object, run *all* its BODex grasp candidates through cuRobo IK
   (which is collision-aware), see which succeed, and visualize each candidate
   with a floating Inspire hand colored by success/failure.
3. Pick a candidate, plan from `ARM_HOME` to that grasp, and step through the
   trajectory frame by frame.
4. Save/load arbitrary arm+hand configurations as **waypoints** (drop point
   over the trash bin, pre-grasp staging poses, etc.).
5. Build a **sequential** trajectory between two waypoints — joint 0 first,
   then joint 1, ..., then joint 6 — and check collisions per frame. The user
   asked for this because cuRobo plans were "always cutting it close" and they
   wanted hand-designed safe paths instead.

## Why so many transformation hacks

Perception in robothome uses the **raw zip mesh** (mm-scale, asymmetric
origin), but BODex grasp candidates were generated against the **AutoDex
mesh** (m-scale, symmetry-axis aligned, origin centered on the symmetry axis).
The two mesh frames differ by a rigid transform per object:

```
v_autodex = R_M @ v_orig_in_metres + t_M
```

`build_orig_transforms.py` recovers `R_M, t_M` per object via SVD and saves
`orig_to_autodex.npy` next to each object dir. Without that, grasp candidates
land in the wrong place relative to the perceived object.

**Caveat that bit me:** Procrustes/SVD on a *cylindrically symmetric* mesh
gives a rotation that's correct on the symmetry axis but arbitrary in the
spin around it — there are infinitely many valid Rs. paperCup and Jp_Water
are both axially symmetric. The "right" R only matters when the asset is
non-symmetric. For the symmetric cases the JSON pose's rotation around the
symmetry axis is also meaningless, so the ambiguity cancels out *if* you pick
any consistent convention. I didn't get that all the way to working; the
viewer currently overrides the rotation with a lying tabletop pose for
visual sanity (see "Known issues" below).

The handeye `Z` from `hand_eye_result.pkl` is `base→world` (per the user's
robothome convention), so to bring world poses into the robot base frame you
apply `Z^-1`. This trips people up — the *name* of the variable in robothome
is `Z_world2base` but the stored matrix is base→world. There's a snippet in
robothome that shows it being inverted before use.

## Why the threading and locking machinery exists

I learned this the hard way by getting native crashes:

- cuRobo allocates per-thread CUDA state on first use. **Calling cuRobo from a
  viser slider callback segfaults** because viser callbacks run on a worker
  thread. Errors I saw:
  - `corrupted size vs. prev_size` (heap corruption)
  - `cudaErrorStreamCaptureInvalidated` (CUDA graph reuse across threads)
- The fix was to make every viser callback *only* enqueue a job + set a dirty
  flag, and let the main thread (which constructed cuRobo) drain the queue at
  ~30 Hz inside `_tick()`. All cuRobo work happens in the main thread.
- Inside `CuroboChecker` itself I use an `RLock` (not a plain `Lock`) because
  some helper methods recursively call each other (e.g. earlier versions of
  `query` and `world_collide` would each grab the lock).
- `IKSolverConfig` and `MotionGenConfig` are built with `use_cuda_graph=False`.
  Captured CUDA graphs can't be safely shared, and the small extra cost is
  fine for an interactive tool.

Bottom line: **don't add a new cuRobo call inside a viser callback**. Enqueue
it. There's already a pattern (`_pending_ik_target`, `_world_jobs`,
`_pending_seq_traj`) — copy that.

## Why the collision viz is what it is

Initially I used trimesh `ProximityQuery` against `scene_mesh.obj` for
per-sphere collision. With ~85 robot collision spheres against a 76k-face
scene mesh, this took **213 ms per query**. That's why the viewer felt
sluggish and the sequential trajectory build (210 frames × 213 ms ≈ 45 s)
appeared to hang.

The right answer was already in cuRobo:

```python
wc = self.rw.collision_cost.world_coll_checker
d = wc.get_sphere_distance(
    x_sph_bhn4, buf, weight, activation_dist,
    sum_collisions=False,   # ← key: per-sphere, not summed cost
)
# d > 0 means colliding
```

This runs in **0.16 ms** for the same workload. The
`CollisionQueryBuffer` needs to be cached per `x_sph` shape and invalidated
on world rebuild — see `_sphere_buf` in `CuroboChecker`.

The **high-level** `RobotWorld.get_collision_distance` and
`get_self_collision_distance` return summed cost (a scalar per (batch, horizon)).
They are not what you want for sphere-level highlighting. They also expect a
4D shape `(B, H, N, 4)`, not 3D.

## Why the scene mesh is what it is

The user gave me an SDF voxel grid (`scene_voxel.npz`). I extracted the zero
isosurface via marching cubes and saved it as `scene_mesh.obj`. Two issues
came up:

1. The robot's own arm was baked into the SDF (the scan included the robot).
   That made initial-pose collisions appear out of nowhere. Fix: zero out the
   SDF region the robot lives in before marching cubes (`x < 0.2 ∧ z > 0`,
   per the user's call). Then drop any face from the result whose vertex is
   inside the masked region (this avoids creating a fake wall on the mask
   boundary).
2. Marching cubes on a truncated SDF leaves a lot of small floating components.
   I kept only the four largest connected components.

The mesh ends up not-watertight. `trimesh.signed_distance` lies on
not-watertight meshes, which is part of why the trimesh-based collision
attempt was unreliable on top of being slow.

## Why the JSON pose pipeline is gnarly

Even after applying `Z^-1` and `M`, the rendered objects looked rotated wrong.
Reasons:

- The cylindrical-symmetric ambiguity in `M` (above).
- I bounced through several conventions (T orig @ M vs. T orig @ M^-1, scale
  vs. no-scale) and the user (correctly) called me out for guessing.
- For the immediate goal — sanity-checking the *rest* of the pipeline — we
  hardcoded a known lying tabletop pose from
  `{obj}/processed_data/info/tabletop/*.npy` and just kept the perception
  translation. That puts the object in a believable lying pose at the right
  spot so the user can test the grasp + planning loop.

The "real" fix is to either (a) use a non-cylindrical reference object for the
fit, or (b) ditch `M` for symmetric objects and snap them onto a tabletop pose
matched by tilt. We didn't get there.

## Performance gotchas, summarized

- Don't use trimesh proximity queries against the scene mesh. Use cuRobo's
  per-sphere distance.
- Don't call cuRobo from viser callbacks. Enqueue.
- Don't keep using `IKSolver` with `use_cuda_graph=True` in this app.
- `CollisionQueryBuffer` must be invalidated on world rebuild (object add/remove).
- `MotionGen.warmup` is ~7 s. Done lazily on the first `plan_to_grasp` so
  startup is fast; first plan is slow.
- `build_visual_mesh.py` is only for robothome — paradex assets are already
  small enough.

## Known unfinished things

- **Cylindrical M ambiguity**: `orig_to_autodex.npy` rotations are not unique
  for symmetric objects. Viewer compensates with a debug tabletop override.
  Real fix: snap to the tabletop pose whose tilt best matches the perception
  pose, or fit M with non-symmetric vertex features.
- ~~**Slider NaN error**~~ — fixed 2026-04-27. Root cause: int sliders
  (`grasp_slider`, `traj_slider`, `seq_traj_slider`) were initialized with
  `min=0, max=0, step=1`. With `min == max`, the client UI computes a
  position fraction `(v-min)/(max-min) = 0/0 = NaN` and ships NaN back; viser
  hits `int(NaN)` in `_handle_gui_updates` and raises. Fix: init with
  `max=1` and use `max(1, len(...)-1)` on every update so the slider is
  never degenerate. The on-update callbacks already clamp idx to the valid
  data range, so allowing max=1 with 0 or 1 items is safe.
- **Sequential trajectory** builds are saved to
  `waypoints/{start}__to_{goal}.npz`. The user said they don't actually want
  the saved trajectory, just the live animation; the save can be removed
  later if they ask.
- **Hand-eye Z is a single snapshot.** If recalibrated, replace
  `hand_eye_result.pkl` and the viewer picks it up next launch.
- **Tracking JSON**: only the most recent file is used. Older files clutter
  the dropdown — they were deleted manually. Consider auto-pruning.
- **paradex objects** mentioned in tracking JSONs (e.g. `pepsi can`) have
  meshes under `shared_data/AutoDex/object/paradex/`, not `robothome/`. The
  viewer searches both. The simple-name resolver (`find_mesh_dir`) does
  case-insensitive exact-name matching only, no substring fallback — this
  was deliberate after a substring fallback once matched `Jp_WaterCrush2`
  for `jp_water`.

## How to launch

```bash
python src/validation/robothome/viewer.py --port 8090
# open http://localhost:8090 in a browser
```

Expect this on first boot:
```
[curobo] initializing RobotWorld...
[curobo] ready: 85 spheres (85 valid), dof=13
```

First plan invokes `MotionGen.warmup` (~7 s). Subsequent plans 1–2 s. IK is
~50 ms.

## What to do next (concrete punch list)

In rough priority order. Don't start a new feature without finishing the one
above it unless the user asks for something else explicitly.

### 1. Fix the rotation pipeline for symmetric objects (blocking real use)
The viewer currently *fakes* object rotation by snapping to a lying tabletop
pose. The user wants real perception poses. Two options:

- **Option A (recommended):** snap the perception rotation to the tabletop
  pose whose tilt best matches it. Concretely:
  1. Compute `axis_world = T_obj_base[:3,:3] @ +y_autodex`.
  2. Among `{obj}/processed_data/info/tabletop/*.npy`, pick the one whose
     own `+y_world` is most parallel (or anti-parallel) to `axis_world`.
  3. Replace `T_obj_base[:3,:3]` with that tabletop's rotation. Translation
     stays from perception, plus the tabletop's `t.z` so the object rests on
     the table.
- **Option B:** for non-symmetric objects, the SVD `M` is correct as-is — use
  `T_orig @ M`. Decide per-object whether it's symmetric (you can detect this
  from mesh principal moments).

The current `_load_tracking_json` is the place to change. Right now everything
after the `1cm lift` line should be replaced.

### 2. ~~Fix the slider NaN crash~~ — done 2026-04-27
The crash:
```
prop_value = type(handle_state.value)(prop_value)
ValueError: cannot convert float NaN to integer
```
Root cause: `grasp_slider`, `traj_slider`, `seq_traj_slider` were created
with `min=0, max=0, step=1`. When `min == max`, the client UI's position
fraction `(v-min)/(max-min)` is `0/0 = NaN`, and any drag/click sends NaN
back. viser then runs `int(NaN)` in `_handle_gui_updates` and raises. It
was *not* a `len(...)-1` going negative — `max(0, ...)` already guarded
that; the bug was the degenerate range itself.

Fix: init each slider with `max=1`, and use `max(1, len(...)-1)` on every
update so the slider is never `min == max`. The on-update callbacks
(`_on_grasp_slider`, `_on_traj_slider`, `_show_candidate`) already clamp
idx into the valid data range, so allowing max=1 with 0 or 1 items is
safe — dragging past the end just clamps back.

### 3. Wire grasp planning into the trash-drop flow (the real pipeline)
Current viewer builds these pieces independently:
- `Check grasps` finds an IK-feasible grasp.
- `Plan to this grasp` plans `ARM_HOME → grasp_qpos`.
- `Sequential start→goal` plans waypoint→waypoint.
- `Drop release (right/left)` jumps to a fixed drop pose.

The user wants a **chained** trajectory: `ARM_HOME → pre-grasp → grasp → lift
→ drop_release`. Approach:
1. After a successful grasp plan, append a 10cm lift in world z.
2. From lift pose, plan to the appropriate `drop_*` waypoint.
3. Concatenate the three trajectories so the trajectory slider scrubs the
   whole pick-and-place.

Don't try to do this with one giant cuRobo call — do it as separate
`plan_single_js` calls and concatenate the resulting interpolated qpos arrays.

### 4. Sequential trajectory: keep it useful
The user only wanted live animation + collision viz, not saved files. The
build currently saves `waypoints/{start}__to_{goal}.npz` regardless. Either
make saving opt-in (checkbox), or remove it. Don't just leave it as is.

### 5. Visualize collisions during the sequential animation
Sequential build already collision-checks every frame and stores
`self._traj_coll`. The trajectory slider doesn't use it yet. When scrubbing,
the collision spheres should reflect the *current frame's* collision state,
not the live FK state. Update `_on_traj_slider` to look up `self._traj_coll`
and tint accordingly.

### 6. Per-object visual mesh for paradex if needed
`build_visual_mesh.py` only runs over robothome. If paradex objects start
showing up large, generalize the script. Right now it's fine.

### 7. Auto-prune old tracking JSONs
Currently a manual `rm`. The viewer reads the latest one, but old files in
the directory clutter source control. Either gitignore them, or have the
viewer move old files to a `_archive/` subfolder on startup.

## Things I won't pretend I did right

- I started with trimesh-based collision instead of looking at cuRobo's
  per-sphere API first. That cost the user a lot of time and made the tool
  feel broken when it was just my wrapper.
- I shipped guesses about transform conventions instead of pinning down what
  the user actually meant before writing code, which caused multiple rounds
  of "is it M or M⁻¹? on the left or the right?".
- I told the user a slider was already there when it was buried in a folder
  they hadn't expanded, instead of just adding a visible one in the obvious
  place.

If you're picking this up, please don't repeat any of those.
