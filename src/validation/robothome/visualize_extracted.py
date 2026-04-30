"""Verify extracted per-capture summaries from ``extract_traj.py``.

Loads ``src/validation/robothome/extracted/<obj>/<idx>.npz`` and shows in
viser, all in robot-base frame:

- Inspire-left palm mesh attached to the wrist SE3 (so the user can
  visually verify the extracted wrist trajectory is correct).
- Wrist SE3 trajectory as a polyline + sliding playback frame axes.
- Occluded centroid as a small sphere (orange) and the object mesh
  recentered at it.
- ChArUco plane as a translucent rectangle (double-sided) at the fitted
  plane centroid; the +normal arrow is rooted at the occluded centroid.

Usage:
    /home/mingi/miniconda3/envs/foundationpose/bin/python \
        src/validation/robothome/visualize_extracted.py \
        --obj choco --idx 0 [--port 8200]

    --obj/--idx omitted → dropdown lets you pick interactively.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import viser
from scipy.spatial.transform import Rotation as R

PARADEX_ROOT = Path("/home/mingi/paradex")
if str(PARADEX_ROOT) not in sys.path:
    sys.path.insert(0, str(PARADEX_ROOT))

HERE = Path(__file__).resolve().parent
EXTRACT_DIR = HERE / "extracted"
SUB_DIR = HERE / "subsampled"

# Floating inspire-left URDF: 6-DOF prismatic+revolute root → wrist → hand.
FLOATING_URDF = Path(
    "/home/mingi/AutoDex/autodex/planner/src/curobo/content/assets/robot/"
    "inspire_description/inspire_left_floating.urdf"
)
FLOATING_ROOT_JOINTS = [
    "x_joint", "y_joint", "z_joint",
    "x_rotation_joint", "y_rotation_joint", "z_rotation_joint",
]
# Inspire-left finger driver joints in URDF/RobotModule order.
FLOATING_HAND_JOINTS = [
    "left_thumb_1_joint", "left_thumb_2_joint", "left_index_1_joint",
    "left_middle_1_joint", "left_ring_1_joint", "left_little_1_joint",
]
RAW_HAND_TO_URDF = [5, 4, 3, 2, 1, 0]


def _hand_qpos_order(d: np.lib.npyio.NpzFile) -> str:
    return str(d["hand_qpos_order"].item()) if "hand_qpos_order" in d.files else "raw"


def _hand_qpos_urdf_order(hand: np.ndarray, order: str) -> np.ndarray:
    hand = np.asarray(hand, dtype=np.float64)
    if order == "urdf":
        return hand
    if order == "raw":
        return hand[:, RAW_HAND_TO_URDF]
    raise ValueError(f"unknown hand_qpos_order={order!r}")


def se3_to_floating_cfg(T: np.ndarray):
    """Decompose 4x4 SE3 into (x, y, z, rx, ry, rz) for the floating URDF.

    The floating URDF's rotation chain is x_rot → y_rot → z_rot in
    intrinsic order, so R = R_x(rx) · R_y(ry) · R_z(rz). scipy's
    ``'XYZ'`` (uppercase = intrinsic) returns angles in this convention.
    """
    xyz = T[:3, 3]
    rxyz = R.from_matrix(T[:3, :3]).as_euler("XYZ", degrees=False)
    return float(xyz[0]), float(xyz[1]), float(xyz[2]), \
           float(rxyz[0]), float(rxyz[1]), float(rxyz[2])


def _import_vc():
    spec = importlib.util.spec_from_file_location("vc", HERE / "visualize_capture.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def list_records():
    out = []
    if EXTRACT_DIR.exists():
        for obj_dir in sorted(p for p in EXTRACT_DIR.iterdir() if p.is_dir()):
            for f in sorted(obj_dir.glob("*.npz")):
                out.append((obj_dir.name, f.stem))
    return out


def _rect_mesh(center, normal, half_extents=(0.30, 0.30)):
    """Rectangle mesh centered at ``center`` lying in the plane with given
    ``normal``. Returns (verts(4,3), faces(2*2,3)) — double-sided."""
    nrm = normal / max(np.linalg.norm(normal), 1e-12)
    a = np.array([1.0, 0.0, 0.0]) if abs(nrm[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = a - nrm * (a @ nrm); u /= np.linalg.norm(u)
    v = np.cross(nrm, u)
    hu, hv = half_extents
    corners = [
        center - hu * u - hv * v,
        center + hu * u - hv * v,
        center + hu * u + hv * v,
        center - hu * u + hv * v,
    ]
    verts = np.stack(corners).astype(np.float32)
    faces = np.array(
        [[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]],
        dtype=np.int32,
    )
    return verts, faces


def _arrow_positions(start, direction, length=0.15):
    end = start + direction / max(np.linalg.norm(direction), 1e-12) * length
    return np.stack([start, end], axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", type=str, default=None)
    ap.add_argument("--idx", type=str, default=None)
    ap.add_argument("--port", type=int, default=8200)
    args = ap.parse_args()

    records = list_records()
    if not records:
        raise SystemExit(f"No extracted records under {EXTRACT_DIR}. "
                         f"Run extract_traj.py first.")

    vc = _import_vc()

    import yourdfpy
    from viser.extras import ViserUrdf
    import trimesh as _tm

    server = viser.ViserServer(port=args.port)
    print(f"[viser] http://localhost:{args.port}")

    # Floating inspire-left URDF: 6-DOF root drives the wrist; finger
    # drivers come from the captured hand_qpos.
    floating_urdf = yourdfpy.URDF.load(
        str(FLOATING_URDF),
        mesh_dir=str(FLOATING_URDF.parent),
        build_collision_scene_graph=False,
        load_collision_meshes=False,
    )
    actuated = list(floating_urdf.actuated_joint_names)
    root_idx = [actuated.index(n) for n in FLOATING_ROOT_JOINTS]
    finger_idx = [actuated.index(n) for n in FLOATING_HAND_JOINTS]
    viser_urdf = ViserUrdf(server, floating_urdf, root_node_name="/inspire")

    state = {"key": None, "data": None, "T": 0, "idx": 0, "dirty": True,
             "wf": None, "obj_handle": None,
             "obj_v": None, "obj_bbox_center": None, "obj_R0": np.eye(3)}

    keys = [f"{o}/{i}" for o, i in records]
    init_key = (f"{args.obj}/{args.idx}" if args.obj and args.idx
                else (f"{records[0][0]}/{records[0][1]}"))
    if init_key not in keys:
        init_key = keys[0]

    record_dd = server.gui.add_dropdown("Record", keys, initial_value=init_key)

    play_cb = server.gui.add_checkbox("Playing", initial_value=False)
    fps_sl = server.gui.add_slider("FPS", min=1, max=120, step=1, initial_value=30)
    frame_sl = server.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
    prev_btn = server.gui.add_button("Prev frame")
    next_btn = server.gui.add_button("Next frame")
    sub_start_btn = server.gui.add_button("Subsample start")
    sub_end_btn = server.gui.add_button("Subsample end")
    info_txt = server.gui.add_text("Info", "—", disabled=True)
    hand_order_dd = server.gui.add_dropdown(
        "Hand order", ["auto", "raw→urdf", "as stored"], initial_value="auto",
    )
    hand_txt = server.gui.add_text("Hand q", "—", disabled=True)

    def set_frame(idx: int, update_slider: bool = True):
        state["idx"] = int(np.clip(idx, 0, max(1, state["T"]) - 1))
        if update_slider:
            frame_sl.value = state["idx"]
        state["dirty"] = True

    def load(key: str):
        obj, idx = key.split("/")
        path = EXTRACT_DIR / obj / f"{idx}.npz"
        d = np.load(path)
        wrist = d["wrist_se3_traj"]
        T = len(wrist)
        occ = d["occluded_centroid"].astype(np.float64)
        plane_n = d["plane_normal"].astype(np.float64)
        plane_c = d["plane_centroid"].astype(np.float64)
        if "hand_qpos" not in d.files:
            raise SystemExit(
                f"{path.name} has no 'hand_qpos' — re-run extract_traj.py --force"
            )
        hand_qpos_stored = np.asarray(d["hand_qpos"], dtype=np.float64)
        hand_order = _hand_qpos_order(d)
        state["data"] = {"wrist": wrist, "occ": occ, "plane_c": plane_c,
                         "plane_n": plane_n, "hand_qpos_stored": hand_qpos_stored,
                         "hand_qpos_order": hand_order}
        state["key"] = key
        state["T"] = T
        state["idx"] = 0

        # Rebuild scene fixtures.
        for handle_key in ("wf", "obj_handle"):
            h = state.get(handle_key)
            if h is not None:
                try: h.remove()
                except Exception: pass
                state[handle_key] = None
        try: server.scene.remove("/extracted")
        except Exception: pass

        # Occluded centroid (orange sphere).
        server.scene.add_icosphere(
            "/extracted/occluded", radius=0.020,
            position=tuple(occ.astype(np.float32).tolist()),
            color=(255, 140, 30),
        )

        # ChArUco plane: rectangle, double-sided. Anchored at plane_c with
        # the fitted normal; sized from the spread of detected corners.
        rect_v, rect_f = _rect_mesh(plane_c, plane_n, half_extents=(0.25, 0.25))
        server.scene.add_mesh_simple(
            "/extracted/plane", vertices=rect_v, faces=rect_f,
            color=(80, 160, 255), opacity=0.35,
        )

        # Plane normal arrow rooted at OCCLUDED centroid (so the user
        # immediately sees the up-direction at the object).
        arrow = _arrow_positions(occ, plane_n, length=0.15)
        server.scene.add_spline_catmull_rom(
            "/extracted/normal", positions=arrow, color=(80, 160, 255),
            line_width=5.0,
        )

        # Wrist trajectory polyline.
        wpath = wrist[:, :3, 3].astype(np.float32)
        server.scene.add_spline_catmull_rom(
            "/extracted/wrist_path", positions=wpath, color=(40, 200, 80),
            line_width=2.0,
        )

        # Wrist frame axes for orientation reference (the floating URDF
        # itself is the inspire-left hand visualization).
        state["wf"] = server.scene.add_frame(
            "/extracted/wrist_frame",
            wxyz=(1.0, 0.0, 0.0, 0.0), position=(0.0, 0.0, 0.0),
            axes_length=0.06, axes_radius=0.003,
        )

        # Object mesh at occluded centroid (recentered, like
        # visualize_capture.py). Uses the auto mapping CAPTURE_TO_OBJECT.
        obj_name = vc.CAPTURE_TO_OBJECT.get(obj)
        mp = vc.resolve_object_mesh(obj_name)
        if mp is not None:
            m = _tm.load(str(mp), force="mesh", process=False)
            v = np.asarray(m.vertices, dtype=np.float64)
            bbox_c = (v.min(axis=0) + v.max(axis=0)) / 2
            v_centered = (v - bbox_c).astype(np.float32)
            tts = vc.load_tabletop_poses(obj_name)
            R0 = (next(iter(tts.values()))[:3, :3] if tts else np.eye(3))
            state["obj_v"] = v_centered
            state["obj_bbox_center"] = bbox_c
            state["obj_R0"] = R0
            qxyzw = R.from_matrix(R0).as_quat()
            wxyz = (float(qxyzw[3]), float(qxyzw[0]),
                    float(qxyzw[1]), float(qxyzw[2]))
            state["obj_handle"] = server.scene.add_mesh_simple(
                "/extracted/object", vertices=v_centered,
                faces=np.asarray(m.faces, dtype=np.int32),
                color=(220, 190, 100), opacity=0.9,
                position=tuple(occ.astype(np.float32).tolist()),
                wxyz=wxyz,
            )
            print(f"[load] object mesh: {mp.name}, default tabletop pose applied")

        sub_path = SUB_DIR / obj / f"{idx}.npz"
        if sub_path.exists():
            sd = np.load(sub_path)
            sub_info = {
                "start": int(sd["start"]) if "start" in sd.files else 0,
                "end": int(sd["end"]) if "end" in sd.files else T,
                "stride": int(sd["stride"]) if "stride" in sd.files else 1,
            }
        else:
            sub_info = None
        state["data"]["sub_info"] = sub_info
        frame_sl.max = max(1, T - 1)
        # Start on the exact frame used as demo frame 0 when a subsampled
        # slice exists; this matches test_demo_plan's source_frame display.
        set_frame(sub_info["start"] if sub_info is not None else 0)
        print(f"[load] {key}  T={T}  occ={occ}  plane_n={plane_n}  sub={sub_info}")

    @record_dd.on_update
    def _(_): load(record_dd.value)

    @frame_sl.on_update
    def _(_):
        set_frame(frame_sl.value, update_slider=False)

    @hand_order_dd.on_update
    def _(_):
        state["dirty"] = True

    @prev_btn.on_click
    def _(_):
        play_cb.value = False
        set_frame(state["idx"] - 1)

    @next_btn.on_click
    def _(_):
        play_cb.value = False
        set_frame(state["idx"] + 1)

    @sub_start_btn.on_click
    def _(_):
        play_cb.value = False
        data = state.get("data") or {}
        sub_info = data.get("sub_info")
        if sub_info is not None:
            set_frame(sub_info["start"])

    @sub_end_btn.on_click
    def _(_):
        play_cb.value = False
        data = state.get("data") or {}
        sub_info = data.get("sub_info")
        if sub_info is not None:
            set_frame(max(sub_info["start"], sub_info["end"] - 1))

    def render(idx: int):
        idx = int(np.clip(idx, 0, max(1, state["T"]) - 1))
        data = state["data"]
        if data is None:
            return
        T = data["wrist"][idx]
        if hand_order_dd.value == "auto":
            hand_qpos = _hand_qpos_urdf_order(
                data["hand_qpos_stored"], data["hand_qpos_order"],
            )
        elif hand_order_dd.value == "raw→urdf":
            hand_qpos = _hand_qpos_urdf_order(data["hand_qpos_stored"], "raw")
        else:
            hand_qpos = data["hand_qpos_stored"]
        # Drive the floating URDF: 6 root joints (xyz + intrinsic XYZ rot)
        # + 6 finger drivers from captured hand_qpos.
        cfg = np.zeros(len(actuated))
        x, y, z, rx, ry, rz = se3_to_floating_cfg(T)
        cfg[root_idx[0]] = x
        cfg[root_idx[1]] = y
        cfg[root_idx[2]] = z
        cfg[root_idx[3]] = rx
        cfg[root_idx[4]] = ry
        cfg[root_idx[5]] = rz
        cfg[finger_idx] = hand_qpos[idx]
        viser_urdf.update_cfg(cfg)
        # Wrist frame axes track the same SE3.
        if state["wf"] is not None:
            qxyzw = R.from_matrix(T[:3, :3]).as_quat()
            state["wf"].position = tuple(T[:3, 3].astype(float).tolist())
            state["wf"].wxyz = (float(qxyzw[3]), float(qxyzw[0]),
                                float(qxyzw[1]), float(qxyzw[2]))
        sub_info = data.get("sub_info")
        sub_txt = ""
        if sub_info is not None:
            start, end, stride = sub_info["start"], sub_info["end"], sub_info["stride"]
            if start <= idx < end and (idx - start) % stride == 0:
                sub_txt = f"  subsampled_demo_frame={(idx - start) // stride}"
            else:
                sub_txt = f"  subsample_range=[{start},{end})/{stride}"
        info_txt.value = (
            f"{state['key']}  frame={idx}/{state['T'] - 1}  "
            f"stored_order={data['hand_qpos_order']}  view_order={hand_order_dd.value}  "
            f"wrist=({T[0, 3]:+.3f}, {T[1, 3]:+.3f}, {T[2, 3]:+.3f})"
            f"{sub_txt}"
        )
        hand_txt.value = "  ".join(
            f"{jn}:{hand_qpos[idx, j]:+.3f}"
            for j, jn in enumerate(FLOATING_HAND_JOINTS)
        )

    load(init_key)
    last = time.time()
    frame_accum = 0.0
    while True:
        now = time.time(); dt = now - last; last = now
        if play_cb.value and state["T"] > 1:
            frame_accum += dt * float(fps_sl.value)
            adv = int(frame_accum)
            if adv > 0:
                frame_accum -= adv
                state["idx"] = (state["idx"] + adv) % state["T"]
                frame_sl.value = state["idx"]
                state["dirty"] = True
        if state["dirty"]:
            render(state["idx"])
            state["dirty"] = False
        time.sleep(1.0 / 120.0)


if __name__ == "__main__":
    main()
