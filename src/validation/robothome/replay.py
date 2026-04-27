"""Replay saved pick-and-trash trajectories.

Loads segment files from `traj/`, concatenates them per pick, animates the
robot via viser, and rigidly attaches each grasped object to the wrist
from its grasp frame onward.

Pick segments expected under `traj/`:
  - `home__to_pick_start.npz`             (cached, plan)
  - `place_start__to_drop_left.npz`       (cached, sequential)
  - `place_start__to_drop_right.npz`      (cached, sequential)
  - `{ts}_pick_start__to_grasp_{obj}_{scene}_{scene_id}_{grasp_id}.npz`
        — one per pick (timestamp prefix gives the pick order)

`grasp → place_start` and `drop → home` are linearly interpolated at
load time (no cuRobo needed, sequential is fine per the user's spec).

Object initial poses come from `traj/object_poses.npz` written by the
"Save object poses" button in viewer.py.

Trash classification per object base name → `TRASH_DROP`. Unknown
objects default to drop_left with a warning.

Usage:
    python src/validation/robothome/replay.py [--port 8091]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch
import trimesh
import viser
from viser.extras import ViserUrdf
import yourdfpy
from scipy.spatial.transform import Rotation as R

from curobo.geom.sdf.world import CollisionQueryBuffer
from curobo.geom.types import Mesh as CuroboMesh
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

ROBOT_CFG_NAME = "fr3_inspire_left.yml"


HERE = Path(__file__).resolve().parent
TRAJ_DIR = HERE / "traj"
URDF_PATH = HERE / "fr3_inspire_left.urdf"
SCENE_MESH_PATH = HERE / "scene_mesh.obj"
OBJECT_DIRS = [
    Path("/home/mingi/shared_data/AutoDex/object/robothome"),
    Path("/home/mingi/shared_data/AutoDex/object/paradex"),
]

ARM_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
HAND_ACTUATED = [
    "left_thumb_1_joint", "left_thumb_2_joint",
    "left_index_1_joint", "left_middle_1_joint",
    "left_ring_1_joint", "left_little_1_joint",
]
ARM_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

TRASH_DROP = {
    "Jp_Water": "drop_left",
    "paperCup": "drop_right",
    "big_Oi_OchaCrush": "drop_left",
}
DEFAULT_DROP = "drop_left"

WRIST_LINK = "base_link"  # cuRobo ee_link in fr3_inspire_left.yml

PICK_FILE_PREFIX_RE = re.compile(
    r"^(?P<ts>\d{8}_\d{6})_pick_start__to_grasp_"
)


def parse_pick_filename(name: str, known_bases):
    """Pick filename: `{ts}_pick_start__to_grasp_{base}_{scene}_{scene_id}_{grasp_id}.npz`.

    base may itself contain underscores (Jp_Water, big_Oi_OchaCrush) so we
    match by longest known-base prefix instead of a brittle regex.
    grasp_id may contain `_y{angle}` from the Y-rotation sweep.
    """
    if not name.endswith(".npz"):
        return None
    m = PICK_FILE_PREFIX_RE.match(name)
    if not m:
        return None
    payload = name[m.end():-4]  # strip prefix + .npz
    bases_sorted = sorted(known_bases, key=len, reverse=True)
    for base in bases_sorted:
        if payload == base or payload.startswith(base + "_"):
            rest = payload[len(base):].lstrip("_")
            parts = rest.split("_")
            if len(parts) < 2:
                continue
            scene = parts[0]
            scene_id = parts[1]
            grasp_id = "_".join(parts[2:]) if len(parts) > 2 else ""
            return {
                "ts": m.group("ts"),
                "obj": base,
                "scene": scene,
                "scene_id": scene_id,
                "grasp_id": grasp_id,
            }
    return None


def mat_to_pos_wxyz(T):
    T = np.asarray(T, dtype=np.float64)
    pos = tuple(T[:3, 3].tolist())
    qxyzw = R.from_matrix(T[:3, :3]).as_quat()
    return pos, (float(qxyzw[3]), float(qxyzw[0]), float(qxyzw[1]), float(qxyzw[2]))


def load_segment(name: str):
    """Load a {start}__to_{goal}.npz segment, return (traj, joint_names) or None."""
    path = TRAJ_DIR / name
    if not path.exists():
        print(f"[replay] missing: {path.name}")
        return None
    d = np.load(path, allow_pickle=True)
    return np.asarray(d["traj_curobo"]), list(d["curobo_joint_names"])


def load_pick_segments(known_bases):
    """Return list of dicts (sorted by timestamp): {path, ts, instance, obj, traj, joint_names, ...}.

    Reads metadata (instance, base_name, scene, ...) from inside the npz
    when present (new viewer.py saves include them); falls back to a
    known-base prefix parse on the filename for legacy segments.
    """
    out = []
    for p in sorted(TRAJ_DIR.glob("*_pick_start__to_grasp_*.npz")):
        parsed = parse_pick_filename(p.name, known_bases)
        if parsed is None:
            print(f"[replay] skipping unparseable filename: {p.name}")
            continue
        d = np.load(p, allow_pickle=True)
        keys = set(d.files)
        # npz-stored metadata wins over filename parse — that's the only
        # place we know paperCup_2 vs paperCup_1.
        instance = str(d["instance"]) if "instance" in keys else parsed["obj"]
        base = str(d["base_name"]) if "base_name" in keys else parsed["obj"]
        scene = str(d["scene"]) if "scene" in keys else parsed["scene"]
        scene_id = str(d["scene_id"]) if "scene_id" in keys else parsed["scene_id"]
        grasp_id = str(d["grasp_id"]) if "grasp_id" in keys else parsed["grasp_id"]
        out.append({
            "path": p,
            "ts": parsed["ts"],
            "instance": instance,
            "obj": base,                # base class for TRASH_DROP lookup
            "scene": scene,
            "scene_id": scene_id,
            "grasp_id": grasp_id,
            "traj": np.asarray(d["traj_curobo"]),
            "joint_names": list(d["curobo_joint_names"]),
        })
    out.sort(key=lambda r: r["ts"])
    return out


def linear_interp_qpos(q0: np.ndarray, q1: np.ndarray, n: int) -> np.ndarray:
    """N-step linear interpolation between two configs (inclusive of both ends)."""
    n = max(n, 2)
    alphas = np.linspace(0.0, 1.0, n)
    return (1 - alphas)[:, None] * q0[None, :] + alphas[:, None] * q1[None, :]


def sequential_arm_interp(q0: np.ndarray, q1: np.ndarray, joint_names,
                          steps_per_joint: int = 30) -> np.ndarray:
    """Move FR3 arm joints 0→6 to goal one at a time (hand DOF stay at q0).

    Mirrors viewer.py's `_build_sequential_trajectory`: avoids the wrist
    swinging up/around that linear all-joint interp produces when the two
    endpoints sit near opposite joint limits.
    """
    arm_idx = [joint_names.index(jn) for jn in ARM_JOINTS if jn in joint_names]
    out = [q0.copy()]
    cur = q0.copy()
    for ci in arm_idx:
        v0, v1 = float(cur[ci]), float(q1[ci])
        if abs(v1 - v0) < 1e-6:
            continue
        for s in range(1, steps_per_joint + 1):
            a = s / steps_per_joint
            cur[ci] = v0 + a * (v1 - v0)
            out.append(cur.copy())
    return np.asarray(out)


def fk_link(urdf: yourdfpy.URDF, qpos: np.ndarray, link_name: str) -> np.ndarray:
    """Compute world-frame transform of `link_name` for the given full joint cfg."""
    actuated = list(urdf.actuated_joint_names)
    cfg = np.zeros(len(actuated))
    for jn, v in zip(actuated, qpos):
        cfg[actuated.index(jn)] = float(v)
    urdf.update_cfg(cfg)
    return urdf.get_transform(link_name, urdf.base_link)


def find_mesh(obj_name: str):
    for root in OBJECT_DIRS:
        d = root / obj_name
        if not d.exists():
            continue
        for sub in ("visual_mesh", "raw_mesh"):
            p = d / sub / f"{obj_name}.obj"
            if p.exists():
                return p
    return None


def build_full_plan(picks):
    """Concatenate segments for every pick into one (T, n_dof) trajectory.

    Returns:
        traj_full (T, n_dof) np.ndarray
        joint_names list[str]
        events list of {"obj": str, "grasp_frame": int, "release_frame": int,
                        "T_obj_in_wrist": (4,4) | None, "drop": str}
    """
    if not picks:
        return None, None, []
    joint_names = picks[0]["joint_names"]
    n_dof = len(joint_names)

    # Build a curobo-order qpos vector from a waypoint npz (fallback to zeros).
    def waypoint_qpos(name):
        path = HERE / "waypoints" / f"{name}.npz"
        if not path.exists():
            print(f"[replay] missing waypoint: {name}")
            return np.zeros(n_dof, dtype=np.float64)
        wp = np.load(path, allow_pickle=True)
        q = np.zeros(n_dof, dtype=np.float64)
        for jn, v in zip(wp["actuated_joint_names"], wp["q_actuated"]):
            jn = str(jn)
            if jn in joint_names:
                q[joint_names.index(jn)] = float(v)
        return q

    q_home = waypoint_qpos("home")
    q_pick_start = waypoint_qpos("pick_start")
    q_place_start = waypoint_qpos("place_start")
    q_drop = {
        "drop_left": waypoint_qpos("drop_left"),
        "drop_right": waypoint_qpos("drop_right"),
    }

    home_to_ps = load_segment("home__to_pick_start.npz")
    place_to_drop = {
        "drop_left": load_segment("place_start__to_drop_left.npz"),
        "drop_right": load_segment("place_start__to_drop_right.npz"),
    }
    # pick_start → place_start: same shape both ways, so its reverse is
    # the inter-pick `place_start → pick_start` transition.
    pick_to_place = load_segment("pick_start__to_place_start.npz")

    # If a cached cuRobo plan home→pick_start is missing, fall back to linear.
    if home_to_ps is None:
        home_to_ps_traj = linear_interp_qpos(q_home, q_pick_start, 60)
    else:
        home_to_ps_traj = home_to_ps[0]

    chunks = []
    events = []
    cursor = 0  # global frame index after concatenation

    last_q = q_home.copy()

    for i, pick in enumerate(picks):
        traj_pg = pick["traj"]
        # 1) home → pick_start (first pick) OR drop → place_start →
        # pick_start (subsequent picks). For the latter we reverse the
        # cached segments so we don't need a separate forward plan.
        if i == 0:
            seg = home_to_ps_traj
            chunks.append(seg)
            cursor += len(seg)
            last_q = seg[-1].copy()
        else:
            # drop → place_start = reverse of place_start → drop
            prev_drop = TRASH_DROP.get(picks[i - 1]["obj"], DEFAULT_DROP)
            d2p = place_to_drop.get(prev_drop)
            if d2p is None:
                seg_a = linear_interp_qpos(last_q, q_place_start, 60)
            else:
                seg_a = d2p[0][::-1].copy()
                # Hand stays open (release happened before this transition).
                for jn in HAND_ACTUATED:
                    seg_a[:, joint_names.index(jn)] = 0.0
            chunks.append(seg_a)
            cursor += len(seg_a)
            last_q = seg_a[-1].copy()
            # place_start → pick_start = reverse of pick_start → place_start
            if pick_to_place is None:
                seg_b = linear_interp_qpos(last_q, q_pick_start, 60)
            else:
                seg_b = pick_to_place[0][::-1].copy()
                for jn in HAND_ACTUATED:
                    seg_b[:, joint_names.index(jn)] = 0.0
            chunks.append(seg_b)
            cursor += len(seg_b)
            last_q = seg_b[-1].copy()

        # 2) pick_start → grasp_qpos (loaded)
        # Stitch first frame to previous tail to avoid jumps in case of
        # waypoint qpos mismatch.
        seg = traj_pg
        chunks.append(seg)
        grasp_frame = cursor + len(seg) - 1  # frame at which hand reaches grasp
        cursor += len(seg)
        last_q = seg[-1].copy()

        # 3) grasp → place_start (joint-by-joint sequential, hand carries)
        # Linear all-joint interp swings the wrist up when endpoints sit
        # near opposite joint limits; sequential moves one arm joint at
        # a time so the hand path stays roughly straight.
        q_place_with_carry = q_place_start.copy()
        for jn in HAND_ACTUATED:
            ci = joint_names.index(jn)
            q_place_with_carry[ci] = last_q[ci]
        seg = sequential_arm_interp(last_q, q_place_with_carry, joint_names)
        chunks.append(seg)
        cursor += len(seg)
        last_q = seg[-1].copy()

        # 4) place_start → drop (cached sequential)
        drop_name = TRASH_DROP.get(pick["obj"], DEFAULT_DROP)
        if pick["obj"] not in TRASH_DROP:
            print(f"[replay] WARN: no TRASH_DROP entry for {pick['obj']!r} → defaulting to {DEFAULT_DROP}")
        d = place_to_drop.get(drop_name)
        if d is None:
            seg = linear_interp_qpos(last_q, q_drop[drop_name], 60)
        else:
            seg = d[0]
        # Preserve the carried-hand pose throughout the drop segment until
        # the release event at the very end.
        seg = seg.copy()
        for jn in HAND_ACTUATED:
            ci = joint_names.index(jn)
            seg[:, ci] = last_q[ci]
        chunks.append(seg)
        release_frame = cursor + len(seg) - 1
        cursor += len(seg)
        last_q = seg[-1].copy()

        # 5) Open hand at the very last drop frame (release).
        last_q_open = last_q.copy()
        for jn in HAND_ACTUATED:
            last_q_open[joint_names.index(jn)] = 0.0
        chunks.append(last_q_open[None, :])
        cursor += 1
        last_q = last_q_open

        events.append({
            "obj": pick["obj"],
            "instance": pick.get("instance"),
            "scene": pick["scene"],
            "scene_id": pick["scene_id"],
            "grasp_id": pick["grasp_id"],
            "grasp_frame": grasp_frame,
            "release_frame": release_frame,
            "drop": drop_name,
            "T_obj_in_wrist": None,  # filled at animate time
        })

    traj_full = np.concatenate(chunks, axis=0)
    return traj_full, joint_names, events


def check_full_trajectory_collisions(traj, joint_names):
    """Per-frame robot collision over the concatenated trajectory.

    Matches the viewer's check: scene-mesh world + self-collision via
    cuRobo's RobotWorld. Returns dict with (T,) bool arrays:
        robot_world  — robot spheres vs scene
        robot_self   — robot self-collision
    """
    print("[collision] initializing cuRobo (RobotWorld, scene-only)...")
    tensor_args = TensorDeviceType()
    cfg_dict = load_yaml(join_path(get_robot_configs_path(), ROBOT_CFG_NAME))
    qxyzw = R.from_matrix(np.eye(3)).as_quat()
    scene_mesh = CuroboMesh(
        name="scene_collision", file_path=str(SCENE_MESH_PATH),
        pose=[0.0, 0.0, 0.0,
              float(qxyzw[3]), float(qxyzw[0]), float(qxyzw[1]), float(qxyzw[2])],
    )
    wc = WorldConfig(mesh=[scene_mesh])
    rw_cfg = RobotWorldConfig.load_from_config(
        cfg_dict, wc, tensor_args=tensor_args,
        collision_activation_distance=0.0,
    )
    rw = RobotWorld(rw_cfg)
    curobo_joint_names = list(rw.kinematics.joint_names)

    if curobo_joint_names != joint_names:
        idx = [joint_names.index(jn) for jn in curobo_joint_names]
        traj_curobo = traj[:, idx]
    else:
        traj_curobo = traj

    n_frames = len(traj_curobo)
    print(f"[collision] checking robot collisions across {n_frames} frames...")
    q_all = torch.tensor(traj_curobo.astype(np.float32),
                         device=tensor_args.device, dtype=tensor_args.dtype)
    # Combined helper returns (d_world, d_self) for a (B, dof) batch.
    d_world, d_self = rw.get_world_self_collision_distance_from_joints(q_all)
    robot_world = (d_world.view(n_frames, -1) > 0).any(dim=-1).cpu().numpy()
    robot_self = (d_self.view(n_frames, -1) > 0).any(dim=-1).cpu().numpy()

    return {"robot_world": robot_world, "robot_self": robot_self}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    # Known bases come from object_poses.npz (set in viewer "Save object
    # poses") — that's what defines which base names exist in the scene.
    obj_poses_path = TRAJ_DIR / "object_poses.npz"
    if not obj_poses_path.exists():
        raise SystemExit(f"missing {obj_poses_path}; click 'Save object poses' in viewer first")
    op = np.load(obj_poses_path, allow_pickle=True)
    known_bases = set(str(b) for b in op["base_names"])

    picks = load_pick_segments(known_bases)
    print(f"[replay] {len(picks)} pick segment(s) under {TRAJ_DIR}")
    for p in picks:
        print(f"  {p['ts']}  {p['obj']}/{p['scene']}/{p['scene_id']}/{p['grasp_id']}  ({len(p['traj'])} frames)")

    traj, joint_names, events = build_full_plan(picks)
    if traj is None:
        print("[replay] no picks to replay; exiting.")
        return

    # Object initial poses already loaded above (op variable).
    obj_T = {str(inst): np.asarray(t, dtype=np.float64)
             for inst, t in zip(op["instances"], op["T"])}
    obj_base = {str(inst): str(b) for inst, b in zip(op["instances"], op["base_names"])}
    obj_mesh = {str(inst): str(p) for inst, p in zip(op["instances"], op["mesh_paths"])}

    # Bind each pick event to a saved object instance. Prefer the exact
    # `instance` already on the event (from the npz metadata); if missing
    # (legacy file), fall back to first-unused-instance-of-base-class.
    used = set()
    for ev in events:
        wanted = ev.get("instance")
        if wanted and wanted in obj_T and wanted not in used:
            used.add(wanted)
            continue
        candidates = [i for i, b in obj_base.items() if b == ev["obj"] and i not in used]
        if not candidates:
            print(f"[replay] WARN: no object instance for pick {ev['obj']}")
            ev["instance"] = None
            continue
        ev["instance"] = candidates[0]
        used.add(candidates[0])

    # Per-frame collision check (cuRobo, scene-only world).
    n_frames = len(traj)
    coll = check_full_trajectory_collisions(traj, joint_names)
    n_coll_world = int(coll["robot_world"].sum())
    n_coll_self = int(coll["robot_self"].sum())
    print(f"[collision] world: {n_coll_world}/{n_frames} frames  "
          f"self: {n_coll_self}/{n_frames} frames")

    # Spin up viser.
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    print(f"[replay] http://localhost:{args.port}")

    # Scene mesh (table/environment).
    if SCENE_MESH_PATH.exists():
        m = trimesh.load(str(SCENE_MESH_PATH), force="mesh", process=False)
        server.scene.add_mesh_simple(
            "/scene", vertices=m.vertices, faces=m.faces,
            color=(70, 130, 220), opacity=0.5,
        )

    # Robot URDF via viser's built-in ViserUrdf — handles nested frames and
    # mesh placement correctly. Driving with update_cfg(actuated_qpos)
    # propagates joint transforms through the kinematic chain.
    urdf = yourdfpy.URDF.load(
        str(URDF_PATH), mesh_dir=str(URDF_PATH.parent),
        build_collision_scene_graph=False, load_collision_meshes=False,
    )
    viser_urdf = ViserUrdf(server, urdf, root_node_name="/robot")

    # Object meshes.
    obj_handles = {}
    for inst, T in obj_T.items():
        mesh_path = obj_mesh[inst]
        if not Path(mesh_path).exists():
            print(f"[replay] missing mesh: {mesh_path}")
            continue
        m = trimesh.load(mesh_path, force="mesh", process=False)
        pos, wxyz = mat_to_pos_wxyz(T)
        h = server.scene.add_mesh_simple(
            f"/objects/{inst}",
            vertices=m.vertices, faces=m.faces,
            color=(220, 190, 100),
            position=pos, wxyz=wxyz,
        )
        obj_handles[inst] = h

    # Compute T_obj_in_wrist at the grasp frame for each event.
    for ev in events:
        if ev["instance"] is None:
            continue
        q_at_grasp = traj[ev["grasp_frame"]]
        T_wrist = fk_link(urdf, q_at_grasp, WRIST_LINK)
        T_obj_world = obj_T[ev["instance"]]
        ev["T_obj_in_wrist"] = np.linalg.inv(T_wrist) @ T_obj_world
    print(f"[replay] full trajectory: {n_frames} frames, {len(events)} pick events")
    for ev in events:
        print(f"  {ev['obj']:24} grasp_frame={ev['grasp_frame']:5d}  "
              f"release_frame={ev['release_frame']:5d}  drop={ev['drop']}")

    # GUI: timestep slider + play/pause.
    play = server.gui.add_checkbox("Playing", initial_value=True)
    fps = server.gui.add_slider("FPS", min=1, max=120, step=1, initial_value=args.fps)
    t = server.gui.add_slider("Frame", min=0, max=n_frames - 1, step=1, initial_value=0)
    info = server.gui.add_text("Phase", "—", disabled=True)
    rec_btn = server.gui.add_button("Record MP4 (full traj)")
    rec_status = server.gui.add_text("Record", "idle", disabled=True)
    rec_height = server.gui.add_slider("Rec height", min=240, max=1440, step=60, initial_value=720)
    rec_width = server.gui.add_slider("Rec width", min=320, max=2560, step=80, initial_value=1280)

    def render(frame_idx: int):
        q = traj[frame_idx]
        # Build actuated qpos in URDF order from saved curobo-order traj.
        actuated = list(urdf.actuated_joint_names)
        cfg = np.zeros(len(actuated))
        for jn in actuated:
            if jn in joint_names:
                cfg[actuated.index(jn)] = float(q[joint_names.index(jn)])
        viser_urdf.update_cfg(cfg)
        urdf.update_cfg(cfg)  # also drives FK for object attachment below

        # Object animation: stationary before grasp_frame, attached to wrist
        # from grasp_frame to release_frame, then back to a free pose at
        # whatever world location the wrist happened to be at release.
        T_wrist_now = urdf.get_transform(WRIST_LINK, urdf.base_link)
        for ev in events:
            inst = ev["instance"]
            if inst is None or inst not in obj_handles:
                continue
            if frame_idx < ev["grasp_frame"]:
                T_world = obj_T[inst]
            elif frame_idx <= ev["release_frame"] and ev["T_obj_in_wrist"] is not None:
                T_world = T_wrist_now @ ev["T_obj_in_wrist"]
            else:
                T_world = obj_T.get(inst + "__released", obj_T[inst])
                if frame_idx == ev["release_frame"] + 1:
                    obj_T[inst + "__released"] = T_wrist_now @ ev["T_obj_in_wrist"]
                    T_world = obj_T[inst + "__released"]
            pos, wxyz = mat_to_pos_wxyz(T_world)
            obj_handles[inst].position = pos
            obj_handles[inst].wxyz = wxyz

        # Phase label + collision flag
        phase = "transit"
        for ev in events:
            if ev["grasp_frame"] <= frame_idx <= ev["release_frame"]:
                phase = f"carrying {ev['obj']} → {ev['drop']}"
                break
        flags = []
        if coll["robot_world"][frame_idx]:
            flags.append("WORLD_COLL")
        if coll["robot_self"][frame_idx]:
            flags.append("SELF_COLL")
        flag_str = (" [" + ",".join(flags) + "]") if flags else ""
        info.value = f"frame {frame_idx + 1}/{n_frames}  {phase}{flag_str}"

    @t.on_update
    def _(_):
        if not play.value:
            render(int(t.value))

    @rec_btn.on_click
    def _record(_):
        clients = list(server.get_clients().values())
        if not clients:
            rec_status.value = "no client connected"
            return
        client = clients[0]
        was_playing = bool(play.value)
        play.value = False
        h, w = int(rec_height.value), int(rec_width.value)
        out_dir = TRAJ_DIR / "recordings"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"replay_{int(time.time())}.mp4"
        try:
            import imageio
        except ImportError:
            rec_status.value = "pip install imageio[ffmpeg] first"
            play.value = was_playing
            return
        rec_status.value = f"recording 0/{n_frames}..."
        writer = imageio.get_writer(str(out_path), fps=int(fps.value), codec="libx264", quality=8)
        try:
            for i in range(n_frames):
                t.value = float(i)
                render(i)
                img = client.get_render(height=h, width=w)
                if img.shape[-1] == 4:
                    img = img[..., :3]
                writer.append_data(img)
                if i % 30 == 0:
                    rec_status.value = f"recording {i}/{n_frames}..."
        finally:
            writer.close()
        rec_status.value = f"saved → {out_path.name}"
        print(f"[record] saved {out_path}")
        play.value = was_playing

    # Initial frame.
    render(0)

    import time
    while True:
        if play.value:
            nxt = (int(t.value) + 1) % n_frames
            t.value = nxt
            render(nxt)
        time.sleep(1.0 / max(1, int(fps.value)))


if __name__ == "__main__":
    main()
