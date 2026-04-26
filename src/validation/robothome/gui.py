"""
BODex Grasp Planner GUI — lightweight Viser interface for grasp planning + execution.

No MPC, no heavy curobo init at startup. LeftGraspPlanner is loaded lazily on first
"Plan grasp" click. This avoids GPU OOM on startup.

Pipeline:
  1. Load object 6D pose (ZMQ capture or JSON file)
  2. Convert world -> robot base via hand-eye Z
  3. Click "Plan grasp (BODex)" → LeftGraspPlanner generates trajectory
  4. Preview trajectory in viser
  5. Click "Execute on real robot" → sends to FR3 + Inspire via ZMQ/Modbus

Usage:
    conda activate robothome
    cd ~/robothome/junyoung/RobotHome
    python robothome/robot/curobo/grasp_planner_gui.py [--port 8086]
"""
import argparse
import glob
import json
import os
import sys
import time
import threading

import joblib
import msgpack
import numpy as np
import trimesh
import viser
import yaml
import zmq
from scipy.spatial.transform import Rotation as R

GRASP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(GRASP_DIR, "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
from robothome.robot.pinocchio_viser_robot import PinocchioViserRobot
from robothome.robot.robot_config import FRANKA_HOME_QPOS
from robothome.utils.object_info import load_objects_config, find_mesh_file
from pathlib import Path

def _find_latest_handeye():
    """Find the most recent hand_eye_result.pkl for jangja (grasp is
    jangja-only). Uses handeye_loader so it follows the franka_<tag>/<ts>/
    bucketed layout instead of the old flat timestamp directory scan."""
    from robothome.robot.handeye.handeye_loader import get_latest_handeye_pkl, HANDEYE_CACHE_DIR
    latest = get_latest_handeye_pkl("franka:jangja")
    if latest is not None:
        return str(latest)
    return str(HANDEYE_CACHE_DIR / "hand_eye_result.pkl")

DEFAULT_HANDEYE = _find_latest_handeye()
DEFAULT_OBJECTS_YAML = os.path.expanduser(
    "~/robothome/junyoung/object-6d-tracking/objects.yaml"
)
DEFAULT_MESH_DIR = os.path.expanduser("~/robothome/mesh")
DEFAULT_TRACKING_JSON_DIR = os.path.join(REPO_ROOT, "tracking")
DEFAULT_TRACKING_HOST = "localhost"
COMBINED_TRACKING_PUB_PORT = 9765

QPOS_INIT = FRANKA_HOME_QPOS

HAND_JOINT_NAMES = ["thumb_1", "thumb_2", "index_1", "middle_1", "ring_1", "little_1"]


def mat_to_pos_wxyz(T):
    """Convert a 4x4 SE(3) matrix to (position_tuple, wxyz_quaternion_tuple).

    Viser uses (w, x, y, z) quaternion order while scipy Rotation returns
    (x, y, z, w); this helper isolates the xyzw → wxyz shuffle and float
    cast so callers don't duplicate the 3-line conversion.
    """
    T = np.asarray(T, dtype=np.float64)
    pos = tuple(T[:3, 3].tolist())
    qxyzw = R.from_matrix(T[:3, :3]).as_quat()
    wxyz = (float(qxyzw[3]), float(qxyzw[0]), float(qxyzw[1]), float(qxyzw[2]))
    return pos, wxyz


def extract_base_class(instance_name: str) -> str:
    name = instance_name.split(":", 1)[-1]
    if "_" in name:
        prefix, suffix = name.rsplit("_", 1)
        if suffix.isdigit():
            return prefix
    return name


def normalize_object_name(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_")



class GraspPlannerGUI:
    def __init__(self, port: int, handeye_path: str, tracking_host: str):
        self.port = port
        self.tracking_host = tracking_host

        # Planner init lock: prewarm (background) and first user click both
        # construct LeftGraspPlanner, which allocates large GPU buffers.
        # Without a lock, a quick click during prewarm can trigger two
        # parallel inits → GPU OOM. The lock serializes construction.
        self._planner_init_lock = threading.Lock()
        self.grasp_planner = None

        # State
        self.qpos_current = QPOS_INIT.copy()
        self.captured_objects = {}
        self.captured_mesh_poses = {}
        self.discarded_objects = set()
        self.sphere_handles = {}
        self.collision_spheres = {}
        self.last_planned_object_name = None
        self.batch_plan_results = {}
        self.replay_sequences = {}
        self.replay_object_name = None
        self.replay_frame_idx = 0
        self.replay_lock = threading.Lock()
        self.object_bbox_extents = {}
        self.selected_bbox_handle = None

        self._run_all_in_progress = False

        # Rejection inspector state
        self.rejection_data = None
        self.rej_candidate_log = []   # list of per-attempt viz dicts
        self.rej_sphere_handles = []  # list of viser mesh handles
        self._rej_tick = 0            # unique-name counter for scene handles

        # Objects.yaml
        self.objects_config = load_objects_config(Path(DEFAULT_OBJECTS_YAML))
        self.bodex_names = {}
        for name, meta in self.objects_config.items():
            self.bodex_names[name] = meta.get("mesh_folder", name)
        print(f"[objects] Loaded objects config with {len(self.objects_config)} items")

        # Hand-eye
        try:
            handeye = joblib.load(handeye_path)
            self.Z_base2world = handeye["Z"]
            self.Z_world2base = np.linalg.inv(self.Z_base2world)
            print(f"[handeye] Loaded Z (translation {self.Z_base2world[:3, 3]})")
        except Exception as e:
            print(f"[handeye] WARNING: {e}. Using identity")
            self.Z_base2world = np.eye(4)
            self.Z_world2base = np.eye(4)

        self._setup_viser()

    # ----------------------------------------------------------- Viser

    def _setup_viser(self):
        print("[1/2] Setting up Viser...")
        self.server = viser.ViserServer(host="0.0.0.0", port=self.port)
        self.port = self.server.get_port()

        # Scene mesh
        scene_mesh_path = os.path.join(GRASP_DIR, "assets", "scene_mesh.obj")
        if os.path.exists(scene_mesh_path):
            scene_mesh = trimesh.load(scene_mesh_path)
            if hasattr(scene_mesh.visual, "face_colors"):
                scene_mesh.visual.face_colors = [200, 200, 200, 100]
            self.server.scene.add_mesh_trimesh("/scene", mesh=scene_mesh)

        # Robot — hand selected via ROBOTHOME_HAND env (see planner.HAND).
        from robothome.robot.grasp.planner import HAND as _HAND
        urdf_path = os.path.join(
            REPO_ROOT, "robothome/assets/franka_description", f"fr3_{_HAND}.urdf"
        )
        self.robot_viz = PinocchioViserRobot(
            server=self.server,
            robot_tag="fr3",
            urdf_path=urdf_path,
            base2world=np.eye(4),
            qpos_init=self.qpos_current,
            tcp_frame_name="hand_tcp",
        )

        # Target indicator
        self.target_frame = self.server.scene.add_frame(
            "/target", axes_length=0.12, axes_radius=0.005,
            position=(0, 0, 0), wxyz=(1, 0, 0, 0), visible=False,
        )
        self.target_label = self.server.scene.add_label(
            "/target_label", "TARGET", position=(0, 0, 0)
        )
        self.target_label.visible = False
        target_marker_mesh = trimesh.creation.uv_sphere(radius=0.015, count=[12, 12])
        if hasattr(target_marker_mesh.visual, "face_colors"):
            target_marker_mesh.visual.face_colors = [255, 140, 0, 220]
        self.target_marker = self.server.scene.add_mesh_trimesh(
            "/target_marker", mesh=target_marker_mesh, position=(0, 0, 0), wxyz=(1, 0, 0, 0)
        )
        self.target_marker.visible = False
        self.planner_center_frame = self.server.scene.add_frame(
            "/planner_center", axes_length=0.08, axes_radius=0.004,
            position=(0, 0, 0), wxyz=(1, 0, 0, 0), visible=False,
        )
        self.planner_center_label = self.server.scene.add_label(
            "/planner_center_label", "PLANNER CENTER", position=(0, 0, 0)
        )
        self.planner_center_label.visible = False
        planner_marker_mesh = trimesh.creation.uv_sphere(radius=0.012, count=[12, 12])
        if hasattr(planner_marker_mesh.visual, "face_colors"):
            planner_marker_mesh.visual.face_colors = [0, 220, 220, 220]
        self.planner_center_marker = self.server.scene.add_mesh_trimesh(
            "/planner_center_marker", mesh=planner_marker_mesh, position=(0, 0, 0), wxyz=(1, 0, 0, 0)
        )
        self.planner_center_marker.visible = False
        self.wrist_target_frame = self.server.scene.add_frame(
            "/wrist_target", axes_length=0.10, axes_radius=0.004,
            position=(0, 0, 0), wxyz=(1, 0, 0, 0), visible=False,
        )
        self.wrist_target_label = self.server.scene.add_label(
            "/wrist_target_label", "WRIST TARGET", position=(0, 0, 0)
        )
        self.wrist_target_label.visible = False
        wrist_marker_mesh = trimesh.creation.uv_sphere(radius=0.012, count=[12, 12])
        if hasattr(wrist_marker_mesh.visual, "face_colors"):
            wrist_marker_mesh.visual.face_colors = [255, 0, 180, 220]
        self.wrist_target_marker = self.server.scene.add_mesh_trimesh(
            "/wrist_target_marker", mesh=wrist_marker_mesh, position=(0, 0, 0), wxyz=(1, 0, 0, 0)
        )
        self.wrist_target_marker.visible = False

        # GUI: Object Capture
        with self.server.gui.add_folder("Object Capture"):
            self.txt_host = self.server.gui.add_text(
                "Tracking host", initial_value=self.tracking_host
            )
            self.btn_capture = self.server.gui.add_button("Capture from combined_tracking (ZMQ)")
            _jsons = sorted(glob.glob(os.path.join(DEFAULT_TRACKING_JSON_DIR, "tracking_*.json")))
            self.txt_json_path = self.server.gui.add_text(
                "Or load JSON", initial_value=_jsons[-1] if _jsons else ""
            )
            self.btn_load_json = self.server.gui.add_button("Load from JSON")
            self.btn_reset_discarded = self.server.gui.add_button("Reset discarded objects")
            self.dropdown = self.server.gui.add_dropdown(
                "Target object", options=("(none)",), initial_value="(none)"
            )
            self.capture_status = self.server.gui.add_text(
                "Capture status", initial_value="No capture yet", disabled=True
            )

        # GUI: Grasp Planner
        with self.server.gui.add_folder("Grasp Planner"):
            self.dropdown_scene_mode = self.server.gui.add_dropdown(
                "Planning mode",
                options=("bodex", "simple", "trash"),
                initial_value="bodex",
            )

            self.btn_plan_grasp = self.server.gui.add_button("Plan grasp")
            self.btn_plan_multi = self.server.gui.add_button("Plan multi (dropdown)")
            self.btn_plan_all_multi = self.server.gui.add_button("Plan all w/ options")
            self.btn_plan_all_preview = self.server.gui.add_button("Plan all sequential preview")
            self.btn_show_ik_candidates = self.server.gui.add_button("Show IK candidates")
            self.btn_hide_ik_candidates = self.server.gui.add_button("Hide IK candidates")
            self.gui_grasp_options = self.server.gui.add_dropdown(
                "Grasp option",
                options=("(none)",),
                initial_value="(none)",
            )
            self.grasp_status = self.server.gui.add_text(
                "Grasp status", initial_value="Idle", disabled=True
            )
            self.checkbox_show_spheres = self.server.gui.add_checkbox(
                "Show collision spheres", initial_value=False
            )
        with self.server.gui.add_folder("Playback"):
            self.gui_replay_object = self.server.gui.add_dropdown(
                "Replay object", options=("(none)",), initial_value="(none)"
            )
            self.gui_replay_timestep = self.server.gui.add_slider(
                "Timestep", min=0.0, max=0.0, step=1.0, initial_value=0.0, disabled=True
            )
            self.gui_replay_playing = self.server.gui.add_checkbox("Playing", initial_value=False)
            self.gui_replay_fps = self.server.gui.add_slider(
                "FPS", min=1, max=60, step=1, initial_value=20
            )
            self.replay_status = self.server.gui.add_text(
                "Replay status", initial_value="No replay loaded", disabled=True
            )

        # GUI: Real Robot Execution
        with self.server.gui.add_folder("Real Robot Execution"):
            self.btn_check_robot = self.server.gui.add_button("Check robot connection")
            self.btn_execute_robot = self.server.gui.add_button("Execute on real robot (CAUTION)")
            self.btn_run_all_robot = self.server.gui.add_button("Run all sequential discard")
            self.btn_release_robot = self.server.gui.add_button("Release + return to init")
            # pylibfranka commander treats `speed` as a TIME-SCALE factor
            # (exec_duration = plan_duration / speed). The commander now uses
            # a Quintic Hermite interpolator (C² continuous) — acceleration
            # matches at each cuRobo knot, so there are no jerk spikes at
            # knot boundaries. This eliminates the libfranka accel/velocity
            # discontinuity reflex that C¹ cubic hit at speed≈1.0. We can
            # now safely run at cuRobo's planned rate.
            # Default 0.5 keeps a 2× safety margin; raise to 1.0 for full
            # cuRobo plan speed (~15s cycle).
            self.slider_robot_speed = self.server.gui.add_slider(
                "Speed factor", min=0.05, max=1.0, step=0.05, initial_value=0.3
            )
            # Waypoint skip = stride in cuRobo's 20ms grid before handing to
            # commander's Hermite interpolator. 3 → dt_eff=60ms (matches
            # APPROACH_V2_MAX_SKIP in executor). Larger skip (e.g. 15 → dt_eff
            # 300ms) makes the concat trajectory coarser and visually sluggish.
            self.slider_waypoint_skip = self.server.gui.add_slider(
                "Waypoint skip", min=1, max=15, step=1, initial_value=1
            )
            self.robot_exec_status = self.server.gui.add_text(
                "Exec status", initial_value="Not connected", disabled=True
            )

        with self.server.gui.add_folder("Debug Rejection"):
            self.btn_pull_rejection = self.server.gui.add_button("Pull candidate log")
            self.dropdown_rej_candidate = self.server.gui.add_dropdown(
                "Candidate", options=("(none)",), initial_value="(none)",
            )
            # max=1 so the slider has a valid integer range at startup.
            self.slider_rej_wp = self.server.gui.add_slider(
                "Waypoint", min=0, max=1, step=1, initial_value=0
            )
            self.slider_rej_clear_filter_mm = self.server.gui.add_slider(
                "Show spheres < mm", min=10, max=200, step=5, initial_value=50,
            )
            self.rej_info = self.server.gui.add_text(
                "Rejection info", initial_value="No rejection captured", disabled=True
            )

        # Event handlers
        self.btn_capture.on_click(self._on_capture)
        self.btn_load_json.on_click(self._on_load_json)
        self.btn_reset_discarded.on_click(self._on_reset_discarded)
        self.btn_plan_grasp.on_click(self._on_plan_grasp)
        self.btn_plan_multi.on_click(self._on_plan_multi)
        self.btn_plan_all_multi.on_click(self._on_plan_all_multi)
        self.btn_plan_all_preview.on_click(self._on_plan_all_preview)
        self.btn_show_ik_candidates.on_click(self._on_show_ik_candidates)
        self.btn_hide_ik_candidates.on_click(self._on_hide_ik_candidates)
        self.gui_grasp_options.on_update(self._on_grasp_option_change)
        self.checkbox_show_spheres.on_update(self._on_toggle_spheres)
        self.btn_check_robot.on_click(self._on_check_robot)
        self.btn_execute_robot.on_click(self._on_execute_robot)
        self.btn_run_all_robot.on_click(self._on_run_all_robot)
        self.btn_release_robot.on_click(self._on_release_robot)
        self.btn_pull_rejection.on_click(self._on_pull_rejection)
        self.slider_rej_wp.on_update(self._on_rej_slider)
        self.dropdown_rej_candidate.on_update(self._on_rej_candidate_pick)
        self.slider_rej_clear_filter_mm.on_update(self._on_rej_slider)
        self.dropdown.on_update(self._on_select_object)
        self.gui_replay_object.on_update(self._on_replay_object_change)
        self.gui_replay_timestep.on_update(self._on_replay_timestep)
        self.gui_replay_playing.on_update(self._on_replay_playing_update)

        self.object_frames = {}
        self.object_meshes = {}
        self.last_plan_result = None
        # Multi-grasp dropdown state. "Plan all with options" populates
        # `multi_grasp_per_object[name] = {label: PlanResult, ...}` for every
        # captured object. When the user selects an object in the main
        # dropdown, the grasp-option dropdown is repopulated with that
        # object's labels. Changing the grasp option writes the chosen
        # PlanResult to `batch_plan_results[name]`, which the existing
        # "Run all sequential discard" button already uses — so one-shot
        # deploy works without any new run-all logic.
        self.multi_grasp_per_object = {}   # {name: {label: PlanResult}}
        # IK-candidate viz: list of viser scene handles to clear on re-draw.
        self._ik_viz_handles = []
        self.collision_spheres = self._load_collision_spheres()
        threading.Thread(target=self._playback_loop, daemon=True).start()
        threading.Thread(target=self._prewarm_planner, daemon=True).start()

        print(f"[1/2] Viser ready at http://localhost:{self.port}")
        print("[2/2] Pre-warming planner in background (first run compiles CUDA kernels)...")

    # ----------------------------------------------------------- Object capture

    def _update_captured_objects(self, poses_world, source, mesh_poses_world=None):
        self.captured_objects = {}
        self.captured_mesh_poses = {}
        for name, mat in poses_world.items():
            if name in self.discarded_objects:
                continue
            T_world = np.array(mat, dtype=np.float64)
            if T_world.shape != (4, 4):
                continue
            self.captured_objects[name] = self.Z_world2base @ T_world

        mesh_src = mesh_poses_world if mesh_poses_world else poses_world
        for name, mat in mesh_src.items():
            if name in self.discarded_objects:
                continue
            T_world = np.array(mat, dtype=np.float64)
            if T_world.shape != (4, 4):
                continue
            self.captured_mesh_poses[name] = self.Z_world2base @ T_world

        for name, T in self.captured_objects.items():
            pos, wxyz = mat_to_pos_wxyz(T)
            if name in self.object_frames:
                self.object_frames[name].visible = True
                self.object_frames[name].position = pos
                self.object_frames[name].wxyz = wxyz
            else:
                self.object_frames[name] = self.server.scene.add_frame(
                    f"/objects/{name}/frame", axes_length=0.08, axes_radius=0.004,
                    position=pos, wxyz=wxyz,
                )

        for name, T in self.captured_mesh_poses.items():
            mesh_pos, mesh_wxyz = mat_to_pos_wxyz(T)
            base_class = extract_base_class(name)
            mesh_path = self._resolve_mesh_path(base_class)
            if mesh_path and name not in self.object_meshes:
                try:
                    m = trimesh.load(mesh_path, force="mesh")
                    self.object_bbox_extents[name] = np.asarray(m.bounding_box.extents, dtype=np.float64)
                    print(f"[objects] add mesh {name} from {mesh_path}")
                    self.object_meshes[name] = self.server.scene.add_mesh_trimesh(
                        f"/objects/{name}/mesh",
                        mesh=m,
                        position=mesh_pos, wxyz=mesh_wxyz,
                    )
                except Exception as e:
                    print(f"[objects] Failed to load mesh for {name}: {e}")
            elif name in self.object_meshes:
                print(f"[objects] update mesh {name}")
                self.object_meshes[name].visible = True
                self.object_meshes[name].position = mesh_pos
                self.object_meshes[name].wxyz = mesh_wxyz
            else:
                print(f"[objects] No mesh path for {name} (base_class={base_class!r})")

        active_names = set(self.captured_objects.keys())
        for name, handle in self.object_frames.items():
            if name not in active_names:
                handle.visible = False
        for name, handle in self.object_meshes.items():
            if name not in active_names:
                handle.visible = False

        names = sorted(self.captured_objects.keys())
        self.dropdown.options = tuple(names) if names else ("(none)",)
        if names:
            self.dropdown.value = names[0]
            self._update_selected_object_highlight(names[0])
        else:
            self._update_selected_object_highlight("(none)")
        self.capture_status.value = f"[{source}] Loaded {len(names)}: {', '.join(names)}"

    def _on_reset_discarded(self, _):
        self.discarded_objects.clear()
        self.batch_plan_results = {}
        self.replay_sequences = {}
        self.last_plan_result = None
        self.last_planned_object_name = None
        self._refresh_replay_object_options()
        self._set_active_replay(None)
        self.capture_status.value = "Discarded-object filter reset. Capture again to reload all objects."

    def _remove_active_object(self, name: str):
        self.captured_objects.pop(name, None)
        self.captured_mesh_poses.pop(name, None)
        if name in self.object_frames:
            self.object_frames[name].visible = False
        if name in self.object_meshes:
            self.object_meshes[name].visible = False
        self.object_bbox_extents.pop(name, None)
        self.batch_plan_results.pop(name, None)
        self.replay_sequences.pop(name, None)
        names = sorted(self.captured_objects.keys())
        self.dropdown.options = tuple(names) if names else ("(none)",)
        self.dropdown.value = names[0] if names else "(none)"
        self._update_selected_object_highlight(self.dropdown.value)
        self._refresh_replay_object_options()
        if self.replay_object_name == name:
            self._set_active_replay(self.dropdown.value if self.dropdown.value != "(none)" else None)

    def _on_select_object(self, _):
        name = self.dropdown.value
        self._update_selected_object_highlight(name)
        self._load_cached_plan_for_selected_object()
        # Keep the grasp-option dropdown in sync with whatever object is
        # active, so picking a grasp per object is literally main-dropdown-
        # then-grasp-dropdown without any extra button press.
        if getattr(self, "multi_grasp_per_object", None):
            self._populate_grasp_options_for_object(name, auto_select_first=False)

    def _update_selected_object_highlight(self, name: str):
        if name == "(none)" or name not in self.captured_objects:
            self.target_frame.visible = False
            self.target_label.visible = False
            self.target_marker.visible = False
            if self.selected_bbox_handle is not None:
                self.selected_bbox_handle.visible = False
            return

        T = self.captured_mesh_poses.get(name, self.captured_objects[name])
        pos, wxyz = mat_to_pos_wxyz(T)
        self.target_frame.position = pos
        self.target_frame.wxyz = wxyz
        self.target_frame.visible = True
        self.target_label.position = (pos[0], pos[1], pos[2] + 0.10)
        self.target_label.visible = True
        self.target_marker.position = pos
        self.target_marker.visible = True

        extents = self.object_bbox_extents.get(name)
        if extents is None:
            base_class = extract_base_class(name)
            mesh_path = self._resolve_mesh_path(base_class)
            if mesh_path and os.path.exists(mesh_path):
                try:
                    m = trimesh.load(mesh_path, force="mesh")
                    extents = np.asarray(m.bounding_box.extents, dtype=np.float64)
                    self.object_bbox_extents[name] = extents
                except Exception:
                    extents = None
        if extents is None:
            extents = np.array([0.08, 0.08, 0.08], dtype=np.float64)

        ext_arr = np.asarray(extents, dtype=np.float64).flatten()
        if ext_arr.size != 3 or not np.all(np.isfinite(ext_arr)) or np.any(ext_arr <= 0):
            ext_arr = np.array([0.08, 0.08, 0.08], dtype=np.float64)
        dx, dy, dz = float(ext_arr[0]), float(ext_arr[1]), float(ext_arr[2])
        corners = np.array([
            [-dx/2, -dy/2, -dz/2], [+dx/2, -dy/2, -dz/2],
            [+dx/2, +dy/2, -dz/2], [-dx/2, +dy/2, -dz/2],
            [-dx/2, -dy/2, +dz/2], [+dx/2, -dy/2, +dz/2],
            [+dx/2, +dy/2, +dz/2], [-dx/2, +dy/2, +dz/2],
        ], dtype=np.float32)
        edge_idx = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ], dtype=np.int32)
        segments = corners[edge_idx].astype(np.float32)  # (12, 2, 3)
        if self.selected_bbox_handle is not None:
            try:
                self.selected_bbox_handle.remove()
            except Exception:
                pass
        self.selected_bbox_handle = self.server.scene.add_line_segments(
            "/target_bbox",
            points=segments,
            colors=np.asarray([255, 160, 0], dtype=np.uint8),
            line_width=2.0,
            position=pos,
            wxyz=wxyz,
            visible=True,
        )

    def _load_cached_plan_for_selected_object(self):
        name = self.dropdown.value
        if name == "(none)":
            self.last_plan_result = None
            self.last_planned_object_name = None
            self._update_plan_debug_markers(None)
            return
        result = self.batch_plan_results.get(name)
        if result is None:
            self.last_plan_result = None
            self.last_planned_object_name = None
            self._update_plan_debug_markers(None)
            return
        self.last_plan_result = result
        self.last_planned_object_name = name
        self._update_plan_debug_markers(result)
        self._cache_replay_sequence(name, result)
        self.grasp_status.value = f"Cached plan ready for {name}: {result.scene_info}, traj {result.traj.shape}"

    def _update_plan_debug_markers(self, result):
        if result is None:
            self.planner_center_frame.visible = False
            self.wrist_target_frame.visible = False
            self.planner_center_label.visible = False
            self.wrist_target_label.visible = False
            self.planner_center_marker.visible = False
            self.wrist_target_marker.visible = False
            return
        debug = getattr(result, "debug_info", None) or {}
        planner_center = debug.get("planner_center")
        if planner_center is not None:
            planner_pos = tuple(np.asarray(planner_center, dtype=np.float64).tolist())
            self.planner_center_frame.position = planner_pos
            self.planner_center_frame.wxyz = (1.0, 0.0, 0.0, 0.0)
            self.planner_center_frame.visible = True
            self.planner_center_label.position = (planner_pos[0], planner_pos[1], planner_pos[2] + 0.08)
            self.planner_center_label.visible = True
            self.planner_center_marker.position = planner_pos
            self.planner_center_marker.visible = True
        else:
            self.planner_center_frame.visible = False
            self.planner_center_label.visible = False
            self.planner_center_marker.visible = False
        wrist_target = debug.get("wrist_target")
        if wrist_target is not None:
            wrist_pos, wrist_wxyz = mat_to_pos_wxyz(wrist_target)
            self.wrist_target_frame.position = wrist_pos
            self.wrist_target_frame.wxyz = wrist_wxyz
            self.wrist_target_frame.visible = True
            self.wrist_target_label.position = (wrist_pos[0], wrist_pos[1], wrist_pos[2] + 0.08)
            self.wrist_target_label.visible = True
            self.wrist_target_marker.position = wrist_pos
            self.wrist_target_marker.visible = True
        else:
            self.wrist_target_frame.visible = False
            self.wrist_target_label.visible = False
            self.wrist_target_marker.visible = False

    def _refresh_replay_object_options(self):
        names = sorted(self.replay_sequences.keys())
        self.gui_replay_object.options = tuple(names) if names else ("(none)",)
        if self.replay_object_name in names:
            self.gui_replay_object.value = self.replay_object_name
        else:
            self.gui_replay_object.value = names[0] if names else "(none)"

    def _cache_replay_sequence(self, name: str, result):
        self.replay_sequences[name] = self._build_replay_sequence(result)
        self._refresh_replay_object_options()
        self._set_active_replay(name)

    def _set_active_replay(self, name):
        if not name or name == "(none)" or name not in self.replay_sequences:
            self.replay_object_name = None
            self.replay_frame_idx = 0
            self.gui_replay_playing.value = False
            # Set max BEFORE value to avoid NaN (viser slider requires max >= value)
            self.gui_replay_timestep.max = 1.0
            self.gui_replay_timestep.min = 0.0
            self.gui_replay_timestep.value = 0.0
            self.gui_replay_timestep.disabled = True
            self.replay_status.value = "No replay loaded"
            return
        seq = self.replay_sequences[name]
        n = len(seq)
        if n == 0:
            self._set_active_replay(None)
            return
        self.replay_object_name = name
        self.replay_frame_idx = min(self.replay_frame_idx, n - 1)
        # Order matters: set max BEFORE value to prevent NaN
        self.gui_replay_timestep.max = float(n - 1)
        self.gui_replay_timestep.min = 0.0
        self.gui_replay_timestep.step = 1.0
        self.gui_replay_timestep.value = float(self.replay_frame_idx)
        self.gui_replay_timestep.disabled = False
        self._show_replay_frame(name, self.replay_frame_idx)

    def _on_replay_object_change(self, _):
        self._set_active_replay(self.gui_replay_object.value)

    def _on_replay_timestep(self, _):
        if self.gui_replay_playing.value:
            return  # playback loop owns rendering; skip to avoid double _show_replay_frame
        name = self.gui_replay_object.value
        if name == "(none)" or name not in self.replay_sequences:
            return
        try:
            value = float(self.gui_replay_timestep.value)
        except (ValueError, TypeError):
            return
        if not np.isfinite(value):
            return
        seq = self.replay_sequences.get(name, [])
        if not seq:
            return
        idx = max(0, min(int(round(value)), len(seq) - 1))
        self.replay_frame_idx = idx
        if float(self.gui_replay_timestep.value) != float(idx):
            self.gui_replay_timestep.value = float(idx)
        self._show_replay_frame(name, idx)

    def _on_replay_playing_update(self, _):
        self.gui_replay_timestep.disabled = self.gui_replay_playing.value or self.replay_object_name is None

    def _append_replay_joint(self, frames, q7, hand6, label):
        q7 = np.asarray(q7, dtype=np.float64)
        hand6 = np.asarray(hand6, dtype=np.float64)
        frames.append({
            "q_arm": q7.copy(),
            "q_hand": hand6.copy(),
            "label": label,
        })

    def _build_replay_sequence(self, result):
        frames = []
        traj = np.asarray(result.traj)
        open_hand = np.asarray(result.pregrasp if result.pregrasp is not None else np.zeros(6), dtype=np.float64)
        grasp_hand = np.asarray(result.grasp if result.grasp is not None else open_hand, dtype=np.float64)
        for q in traj:
            q = np.asarray(q, dtype=np.float64)
            arm = q[:7]
            hand = q[7:13] if q.shape[0] >= 13 else open_hand
            self._append_replay_joint(frames, arm, hand, "approach")

        if result.pregrasp is not None and result.grasp is not None:
            for i in range(31):
                alpha = i / 30
                hand = open_hand * (1 - alpha) + grasp_hand * alpha
                self._append_replay_joint(frames, traj[-1][:7], hand, "close")
            squeeze_target = grasp_hand * 2 - open_hand
            for i in range(16):
                alpha = i / 15
                hand = grasp_hand * (1 - alpha) + squeeze_target * alpha
                self._append_replay_joint(frames, traj[-1][:7], hand, "squeeze")

        post_trajs = getattr(result, "post_grasp_trajs", None) or {}
        carried_hand = grasp_hand
        for phase_key in ("lift", "retreat", "rotate", "transit", "drop"):
            phase_traj = post_trajs.get(phase_key)
            if phase_traj is None:
                continue
            for q in np.asarray(phase_traj):
                q = np.asarray(q, dtype=np.float64)
                arm = q[:7]
                hand = q[7:13] if q.shape[0] >= 13 else carried_hand
                self._append_replay_joint(frames, arm, hand, phase_key)

        if frames:
            self._append_replay_joint(frames, frames[-1]["q_arm"], np.zeros(6, dtype=np.float64), "release")

        home_traj = post_trajs.get("home")
        if home_traj is not None:
            for q in np.asarray(home_traj):
                q = np.asarray(q, dtype=np.float64)
                arm = q[:7]
                hand = q[7:13] if q.shape[0] >= 13 else np.zeros(6, dtype=np.float64)
                self._append_replay_joint(frames, arm, hand, "home")
        return frames

    def _show_replay_frame(self, name: str, idx: int):
        seq = self.replay_sequences.get(name)
        if not seq:
            self.replay_status.value = "No replay loaded"
            return
        idx = max(0, min(idx, len(seq) - 1))
        frame = seq[idx]
        q_arm = np.asarray(frame["q_arm"], dtype=np.float64)
        q_hand = np.asarray(frame["q_hand"], dtype=np.float64)
        if not np.all(np.isfinite(q_arm)) or not np.all(np.isfinite(q_hand)):
            self.replay_status.value = f"{name}: frame {idx + 1}/{len(seq)} [{frame['label']}] SKIPPED (NaN)"
            return
        self.qpos_current = q_arm.copy()
        self.robot_viz.set_qpos(self.qpos_current)
        self._set_hand_joints(q_hand)
        self.robot_viz.update_viser_link_frames()
        self._update_collision_spheres()
        self.replay_status.value = f"{name}: frame {idx + 1}/{len(seq)} [{frame['label']}]"

    def _playback_loop(self):
        while True:
            try:
                if self.gui_replay_playing.value and self.replay_object_name in self.replay_sequences:
                    seq = self.replay_sequences.get(self.replay_object_name, [])
                    if seq:
                        next_idx = (self.replay_frame_idx + 1) % len(seq)
                        self.replay_frame_idx = next_idx
                        # Only update slider if value is within valid range
                        max_val = float(self.gui_replay_timestep.max)
                        if np.isfinite(max_val) and max_val > 0:
                            self.gui_replay_timestep.value = float(min(next_idx, int(max_val)))
                        self._show_replay_frame(self.replay_object_name, next_idx)
                fps = max(1.0, float(self.gui_replay_fps.value))
                time.sleep(1.0 / fps)
            except Exception:
                time.sleep(0.1)

    def _on_load_json(self, _):
        threading.Thread(target=self._load_json, daemon=True).start()

    def _load_collision_spheres(self):
        from robothome.robot.grasp.planner import HAND_SPHERES_YAML
        spheres_path = HAND_SPHERES_YAML
        if not os.path.exists(spheres_path):
            return {}
        with open(spheres_path) as f:
            data = yaml.safe_load(f)
        out = {}
        for link, sphere_list in data.get("collision_spheres", {}).items():
            out[link] = [
                (np.array(s["center"], dtype=np.float64), float(s["radius"]))
                for s in sphere_list
            ]
        return out

    def _on_toggle_spheres(self, _):
        if self.checkbox_show_spheres.value:
            self._show_collision_spheres()
        else:
            self._hide_collision_spheres()

    def _show_collision_spheres(self):
        self._hide_collision_spheres()
        urdf = self.robot_viz.urdf
        base_link = urdf.base_link
        for link_name, spheres in self.collision_spheres.items():
            try:
                T_base_link = urdf.get_transform(link_name, base_link)
            except Exception:
                continue
            handles = []
            for i, (center_local, radius) in enumerate(spheres):
                center_h = np.array([*center_local, 1.0])
                center_base = (T_base_link @ center_h)[:3]
                sphere_mesh = trimesh.creation.uv_sphere(radius=radius, count=[10, 10])
                if hasattr(sphere_mesh.visual, "face_colors"):
                    sphere_mesh.visual.face_colors = [0, 200, 255, 100]
                h = self.server.scene.add_mesh_trimesh(
                    f"/spheres/{link_name}/{i}",
                    mesh=sphere_mesh,
                    position=tuple(center_base.tolist()),
                    wxyz=(1.0, 0.0, 0.0, 0.0),
                )
                handles.append(h)
            if handles:
                self.sphere_handles[link_name] = handles

    def _hide_collision_spheres(self):
        for handles in self.sphere_handles.values():
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass
        self.sphere_handles = {}

    def _update_collision_spheres(self):
        if not self.checkbox_show_spheres.value or not self.sphere_handles:
            return
        urdf = self.robot_viz.urdf
        base_link = urdf.base_link
        for link_name, handles in self.sphere_handles.items():
            try:
                T_base_link = urdf.get_transform(link_name, base_link)
            except Exception:
                continue
            spheres = self.collision_spheres.get(link_name, [])
            for handle, (center_local, _radius) in zip(handles, spheres):
                center_h = np.array([*center_local, 1.0])
                center_base = (T_base_link @ center_h)[:3]
                handle.position = tuple(center_base.tolist())

    def _resolve_mesh_path(self, base_class: str):
        mesh_path = find_mesh_file(Path(DEFAULT_MESH_DIR), base_class, self.objects_config)
        return str(mesh_path) if mesh_path else None

    def _build_multi_object_world(self, exclude_name: str = None):
        """Build cuRobo world config with scene mesh + all captured objects as obstacles.

        Args:
            exclude_name: object to exclude from obstacles (the one being grasped).
                         Its mesh is handled separately by the planner as 'target_object'.
        """
        from robothome.robot.grasp.planner import SCENE_MESH_PATH, _se3_to_curobo_pose_list
        world_cfg = {"mesh": {
            "scene": {"pose": [0, 0, 0, 1, 0, 0, 0], "file_path": SCENE_MESH_PATH},
        }}
        for name, T_base in self.captured_mesh_poses.items():
            if name == exclude_name:
                continue
            base_class = extract_base_class(name)
            mesh_path = self._resolve_mesh_path(base_class)
            if mesh_path and os.path.exists(mesh_path):
                world_cfg["mesh"][f"obj_{name}"] = {
                    "pose": _se3_to_curobo_pose_list(T_base),
                    "file_path": mesh_path,
                }
        n_obstacles = len(world_cfg["mesh"]) - 1  # exclude scene
        print(f"[world] Built multi-object world: {n_obstacles} object obstacles (excl={exclude_name})")
        return world_cfg

    def _load_json(self):
        path = self.txt_json_path.value.strip()
        if not path or not os.path.exists(path):
            self.capture_status.value = f"File not found: {path}"
            return
        self.capture_status.value = f"Loading {os.path.basename(path)}..."
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            self.capture_status.value = f"JSON parse error: {e}"
            return
        obj_section = data.get("object", {})
        mesh_poses = None
        if isinstance(obj_section, dict) and "poses" in obj_section:
            obj_world = obj_section["poses"]
            mesh_poses = obj_section.get("mesh_poses")
        else:
            obj_world = obj_section
        if not obj_world:
            self.capture_status.value = "No object pose in file"
            return
        self._update_captured_objects(obj_world, source=os.path.basename(path),
                                      mesh_poses_world=mesh_poses)

    def _on_capture(self, _):
        threading.Thread(target=self._capture_objects, daemon=True).start()

    def _capture_objects(self):
        host = self.txt_host.value or self.tracking_host
        url = f"tcp://{host}:{COMBINED_TRACKING_PUB_PORT}"
        self.capture_status.value = f"Connecting to {url}..."
        ctx = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.setsockopt(zmq.RCVHWM, 1)
        sock.connect(url)
        try:
            poller = zmq.Poller()
            poller.register(sock, zmq.POLLIN)
            deadline = time.time() + 5.0
            obj_world = {}
            while time.time() < deadline:
                evts = dict(poller.poll(timeout=500))
                if sock in evts:
                    msg = sock.recv()
                    data = msgpack.unpackb(msg, raw=False)
                    objects = data.get("object", {})
                    if objects:
                        obj_world = objects
                        break
            if not obj_world:
                self.capture_status.value = "No object data within 5s"
                return
            self._update_captured_objects(obj_world, source="ZMQ")
        finally:
            sock.close()
            ctx.term()

    # ----------------------------------------------------------- BODex grasp planner

    def _prewarm_planner(self):
        try:
            self.grasp_status.value = "Pre-warming planner (compiling CUDA kernels)..."
            self._get_grasp_planner()
            self.grasp_status.value = "Planner ready."
            print("[planner] Pre-warm complete — Plan grasp is ready.", flush=True)
        except Exception as e:
            print(f"[planner] Pre-warm failed: {e}", flush=True)
            self.grasp_status.value = f"Pre-warm failed: {e}"

    def _get_grasp_planner(self):
        # Double-checked locking: cheap read on hot path, lock only if we
        # really need to construct. Guarantees exactly one LeftGraspPlanner
        # across prewarm thread + click threads.
        if self.grasp_planner is not None:
            return self.grasp_planner
        with self._planner_init_lock:
            if self.grasp_planner is None:
                from robothome.robot.grasp.planner import LeftGraspPlanner
                self.grasp_planner = LeftGraspPlanner()
        return self.grasp_planner

    def _on_plan_grasp(self, _):
        threading.Thread(target=self._plan_grasp_worker, daemon=True).start()

    def _on_plan_multi(self, _):
        threading.Thread(target=self._plan_multi_worker, daemon=True).start()

    def _on_plan_all_preview(self, _):
        threading.Thread(target=self._plan_all_preview_worker, daemon=True).start()

    def _plan_multi_worker(self):
        """Run plan_all for the currently selected object."""
        name = self.dropdown.value
        if not name or name == "(none)":
            self.grasp_status.value = "No object selected"
            return
        ok = self._plan_multi_for_object(name)
        if ok:
            self._populate_grasp_options_for_object(name, auto_select_first=True)

    def _on_plan_all_multi(self, _):
        threading.Thread(target=self._plan_all_multi_worker, daemon=True).start()

    def _plan_all_multi_worker(self):
        """Plan M grasps per captured object, nearest-first, with optimistic
        obstacle removal between iterations.

        Mirrors `_plan_all_preview_worker`'s behavior: after a successful
        plan the object is hidden from captured_mesh_poses so the next
        object's world matches what the real robot will see at execution
        time (the already-grasped object is gone). Without this, the farther
        objects are planned with every closer object still present and often
        fail because of artificial obstruction.
        """
        if not self.captured_objects:
            self.grasp_status.value = "No active objects"
            return

        def _pos_of(n):
            T = self.captured_mesh_poses.get(n, self.captured_objects[n])
            return np.asarray(T[:3, 3], dtype=np.float64)

        # First item: nearest to robot base. After that, prefer the cluster
        # partner of whatever was just planned (i.e., the remaining object
        # closest to the just-removed one). This way physically clustered
        # pairs resolve together — grabbing cup_2 immediately helps its
        # neighbor pepsi_2 rather than a far-away pepsi_1 that still has
        # its own cluster partner as an obstacle.
        remaining = list(self.captured_objects.keys())
        remaining.sort(key=lambda n: float(np.linalg.norm(_pos_of(n))))
        queue = [remaining.pop(0)] if remaining else []
        last_pos = _pos_of(queue[0]) if queue else None
        while remaining:
            remaining.sort(key=lambda n: float(np.linalg.norm(_pos_of(n) - last_pos)))
            nxt = remaining.pop(0)
            queue.append(nxt)
            last_pos = _pos_of(nxt)
        print(f"[plan_all_multi] cluster-aware queue: {queue}", flush=True)
        total = len(queue)
        self.multi_grasp_per_object = {}
        self.batch_plan_results = {}
        hidden_mesh = {}
        t0 = time.time()
        per_object_timing = []
        try:
            for i, name in enumerate(queue, start=1):
                if name not in self.captured_objects:
                    continue
                # Aggressive per-object cleanup: cuRobo accumulates Warp-
                # allocated VRAM across plan() calls that Python-level
                # empty_cache can't free. Before starting a new object,
                # force a torch sync + gc pass so the driver has a chance
                # to reclaim what it can.
                try:
                    import torch, gc
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    gc.collect()
                    vram_line = ""
                    if torch.cuda.is_available():
                        free_gb, total_gb = torch.cuda.mem_get_info()
                        vram_line = (f"VRAM {(total_gb-free_gb)/1e9:.1f}GB used / "
                                     f"{total_gb/1e9:.1f}GB")
                    try:
                        import psutil
                        vm = psutil.virtual_memory()
                        ram_line = (f"RAM {(vm.total-vm.available)/1e9:.1f}GB used / "
                                    f"{vm.total/1e9:.1f}GB ({vm.percent:.0f}%)")
                    except Exception:
                        ram_line = ""
                    print(f"[plan_all_multi] before {name}: {vram_line} | {ram_line}",
                          flush=True)
                except Exception:
                    pass
                self.dropdown.value = name
                n_obstacles = len(self.captured_mesh_poses) - 1  # self excluded
                self.grasp_status.value = (
                    f"[{i}/{total}] plan_multi {name} "
                    f"(optimistic: {n_obstacles} obstacles remain)..."
                )
                obj_t0 = time.time()
                MAX_PLAN_RETRIES = 3
                ok = False
                for _attempt in range(1, MAX_PLAN_RETRIES + 1):
                    ok = self._plan_multi_for_object(name)
                    if ok:
                        if _attempt > 1:
                            print(f"[plan_all_multi] {name}: succeeded on attempt {_attempt}/{MAX_PLAN_RETRIES}",
                                  flush=True)
                        break
                    if _attempt < MAX_PLAN_RETRIES:
                        print(f"[plan_all_multi] {name}: attempt {_attempt}/{MAX_PLAN_RETRIES} FAIL — "
                              f"retrying with fresh cuRobo seeds...", flush=True)
                        # GPU cleanup between attempts so cuRobo has room to reseed.
                        try:
                            import torch, gc
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                            gc.collect()
                        except Exception:
                            pass
                obj_dt = time.time() - obj_t0
                per_object_timing.append((name, obj_dt, ok))
                if not ok:
                    print(f"[plan_all_multi] {i}/{total} {name}: FAIL after {MAX_PLAN_RETRIES} attempts "
                          f"in {obj_dt:.2f}s", flush=True)
                    continue
                # Optimistic removal: hide the just-planned object so the next
                # (farther) object plans against a world with one less
                # obstacle — the same world the robot will see post-grasp.
                if name in self.captured_mesh_poses:
                    hidden_mesh[name] = self.captured_mesh_poses.pop(name)
                print(f"[plan_all_multi] {i}/{total} {name}: OK in {obj_dt:.2f}s", flush=True)
        finally:
            # Restore hidden objects so the preview has no lasting side effect
            # on scene state (run-all will do its own removal when executing).
            for n, T in hidden_mesh.items():
                self.captured_mesh_poses[n] = T
            total_dt = time.time() - t0
            print(f"[plan_all_multi] total {total_dt:.2f}s for {len(queue)} objects:", flush=True)
            for n, dt, ok in per_object_timing:
                print(f"  - {n}: {dt:.2f}s {'OK' if ok else 'FAIL'}", flush=True)

        n_ok = sum(1 for v in self.multi_grasp_per_object.values() if v)
        self.grasp_status.value = (
            f"plan_all_multi done — {n_ok}/{total} objects have grasp options "
            f"(total {total_dt:.1f}s). Pick per object, then Run all."
        )
        # Prefer to land on an object that actually has grasps so the user
        # sees a usable dropdown immediately. If the loop ended on a failed
        # object (e.g., final object hit "no candidates"), jump to the first
        # object that succeeded instead.
        succeeded = [n for n, v in self.multi_grasp_per_object.items() if v]
        if succeeded:
            target = self.dropdown.value
            if target not in self.multi_grasp_per_object or not self.multi_grasp_per_object.get(target):
                target = succeeded[0]
                self.dropdown.value = target
            self._populate_grasp_options_for_object(target, auto_select_first=True)

    def _plan_multi_for_object(self, name):
        """Run plan_all for a single object; store results in the per-object map.

        Returns True on success (at least one feasible grasp stored).
        """
        if name not in self.captured_objects:
            return False
        # Match _plan_current_object's exact name-mapping logic. Previously
        # this method did its own ad-hoc slicing which skipped the raw-key
        # lookup in bodex_names, so "pepsi can" was normalized to "pepsi_can"
        # and then directly passed as the BODex folder name — which doesn't
        # exist (the actual folder is "pepsi"), causing silent per-object
        # failures with "No BODex output for pepsi_can".
        T_base = self.captured_mesh_poses.get(name, self.captured_objects[name])
        base_class = extract_base_class(name)
        bodex_name = self.bodex_names.get(base_class, base_class)
        if bodex_name == base_class:
            bodex_name = self.bodex_names.get(normalize_object_name(base_class), bodex_name)
        if " " in bodex_name:
            bodex_name = normalize_object_name(bodex_name)
        scene_mode = str(self.dropdown_scene_mode.value)
        try:
            planner = self._get_grasp_planner()
        except Exception as e:
            import traceback; traceback.print_exc()
            self.grasp_status.value = f"Init failed: {type(e).__name__}: {str(e)[:60]}"
            return False
        obj_mesh_path = self._resolve_mesh_path(base_class)
        world_cfg = self._build_multi_object_world(exclude_name=name)
        try:
            results = planner.plan_all(
                obj_name=bodex_name, obj_pose_base=T_base,
                obj_mesh_path=obj_mesh_path,
                world_cfg=world_cfg,
                scene_mode=scene_mode,
                max_successes=8,
            )
        except FileNotFoundError as e:
            print(f"[plan_multi] {name}: no candidates — {e}")
            self.multi_grasp_per_object[name] = {}
            return False
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[plan_multi] {name}: {type(e).__name__}: {e}")
            self.multi_grasp_per_object[name] = {}
            return False

        if not results:
            self.multi_grasp_per_object[name] = {}
            return False

        labeled = {}
        for i, res in enumerate(results):
            info = res.debug_info or {}
            score = info.get("score", 0.0)
            bq = info.get("bodex_quality", 0.0)
            label = f"#{i + 1}  score={score:.2f}  bodex={bq:.2f}"
            labeled[label] = res
        self.multi_grasp_per_object[name] = labeled
        # Default selection = top-scored grasp so run-all works without any
        # manual picking. The user can override per object via the dropdown.
        first_label = next(iter(labeled))
        self._apply_selected_grasp(name, labeled[first_label])
        # If plan_all_multi set the main dropdown to this object BEFORE we
        # populated multi_grasp_per_object[name] (the normal case in the
        # sequential loop), the grasp-option dropdown will have fired with
        # an empty map and is now stuck on "(none)". Refresh it here so that
        # finishing an object makes its options immediately selectable.
        if self.dropdown.value == name:
            self._populate_grasp_options_for_object(name, auto_select_first=False)
        return True

    def _populate_grasp_options_for_object(self, name, auto_select_first=False):
        """Refresh the grasp-option dropdown to match the given object."""
        options_map = self.multi_grasp_per_object.get(name) or {}
        if not options_map:
            self.gui_grasp_options.options = ("(none)",)
            self.gui_grasp_options.value = "(none)"
            return
        labels = tuple(options_map.keys())
        self.gui_grasp_options.options = labels
        # Prefer the label matching whatever is currently recorded as the
        # user's selection for this object (batch_plan_results), otherwise
        # fall back to the first (top-scored) label.
        current = self.batch_plan_results.get(name)
        chosen_label = labels[0]
        if current is not None:
            for lbl, res in options_map.items():
                if res is current:
                    chosen_label = lbl
                    break
        self.gui_grasp_options.value = chosen_label
        if auto_select_first:
            self._apply_selected_grasp(name, options_map[chosen_label])

    def _apply_selected_grasp(self, name, result):
        """Set this result as the active plan for preview + execution."""
        self.last_plan_result = result
        self.last_planned_object_name = name
        self.batch_plan_results[name] = result
        self._update_plan_debug_markers(result)
        self._cache_replay_sequence(name, result)

    def _on_grasp_option_change(self, _):
        label = self.gui_grasp_options.value
        if label == "(none)":
            return
        name = self.dropdown.value
        options_map = self.multi_grasp_per_object.get(name) or {}
        if label not in options_map:
            return
        result = options_map[label]
        self._apply_selected_grasp(name, result)
        info = result.debug_info or {}
        self.grasp_status.value = (
            f"{name}: picked {label} | cand_idx={info.get('candidate_idx', '?')}"
        )

    # --------------------- IK candidate visualization ---------------------

    def _on_show_ik_candidates(self, _):
        self._render_ik_candidates()

    def _on_hide_ik_candidates(self, _):
        self._clear_ik_candidates()

    def _clear_ik_candidates(self):
        for h in self._ik_viz_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._ik_viz_handles = []

    def _render_ik_candidates(self):
        """Draw top-N NMS-surviving IK candidates as colored TCP markers +
        wrist frames so the user can see how each candidate would grasp
        without running motion planning first.

        Reads `planner._last_nms_candidates` populated in plan()/plan_all().
        """
        planner = getattr(self, "grasp_planner", None)
        cands = getattr(planner, "_last_nms_candidates", None) if planner else None
        if not cands or len(cands.get("ik_valid", [])) == 0:
            self.grasp_status.value = "No IK candidates to show — run Plan first"
            return

        self._clear_ik_candidates()

        # Limit to top-K so the scene doesn't get overwhelming. The NMS list
        # is already sorted by quality_score descending (top-3 log confirms).
        K = min(24, len(cands["ik_valid"]))
        wrist = cands["wrist_se3"][:K]
        scores = cands["quality_score"][:K]
        bodex = cands["bodex_quality"][:K]
        ik_qpos = cands["ik_qpos"][:K]
        ik_valid = cands["ik_valid"][:K]

        # Normalize score → [0, 1] for color gradient (green=best, red=worst).
        if len(scores) > 1 and scores.max() > scores.min():
            t = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            t = np.ones_like(scores)

        # Build markers per candidate: wrist frame + TCP sphere + approach arrow
        # end sphere to show the direction the hand would come from.
        # wrist → hand_tcp offset; sign depends on hand variant (LEFT: -Z, F1: +Z).
        from robothome.robot.grasp.planner import (
            HAND as _HAND_GUI,
            HAND_TCP_LOCAL_VEC,
            _HAND_TCP_LOCAL_Z_SIGN as _HAND_TCP_LOCAL_Z_SIGN_GUI,
        )
        tcp_local = HAND_TCP_LOCAL_VEC
        _approach_z_sign = _HAND_TCP_LOCAL_Z_SIGN_GUI.get(_HAND_GUI, -1.0)
        for i in range(K):
            R = wrist[i, :3, :3]
            t_world = wrist[i, :3, 3]
            tcp_world = t_world + R @ tcp_local
            # Approach axis = wrist local -Z for LEFT, +Z for F1 (hand moves along this into obj)
            approach = R @ np.array([0.0, 0.0, _approach_z_sign])
            # Point 5cm back along approach = where hand "comes from"
            back = tcp_world - 0.05 * approach

            # Color: green (good) to red (poor). viser colors are RGB 0-1.
            col = (float(1.0 - t[i]), float(t[i]), 0.0)

            # Tiny sphere at TCP, colored by score
            sphere_mesh = trimesh.creation.uv_sphere(radius=0.008, count=[8, 8])
            if hasattr(sphere_mesh.visual, "face_colors"):
                sphere_mesh.visual.face_colors = [
                    int(col[0] * 255), int(col[1] * 255), int(col[2] * 255), 230
                ]
            h_tcp = self.server.scene.add_mesh_trimesh(
                f"/ik_cand/{i}/tcp", mesh=sphere_mesh, position=tuple(tcp_world),
            )
            self._ik_viz_handles.append(h_tcp)

            # Wrist frame axes at wrist pose
            pos, wxyz = mat_to_pos_wxyz(wrist[i])
            h_frame = self.server.scene.add_frame(
                f"/ik_cand/{i}/wrist",
                axes_length=0.04, axes_radius=0.002,
                position=tuple(pos), wxyz=tuple(wxyz),
            )
            self._ik_viz_handles.append(h_frame)

            # Approach arrow: line segment from "back" to "tcp"
            seg = np.asarray([[back, tcp_world]], dtype=np.float32)
            color_rgb = np.asarray(
                [[int(col[0] * 255), int(col[1] * 255), int(col[2] * 255)]],
                dtype=np.uint8,
            )
            h_line = self.server.scene.add_line_segments(
                f"/ik_cand/{i}/approach", points=seg, colors=color_rgb, line_width=3.0,
            )
            self._ik_viz_handles.append(h_line)

            # Text label: rank + bodex quality
            h_lbl = self.server.scene.add_label(
                f"/ik_cand/{i}/label",
                f"#{i+1} s={scores[i]:.2f} q={bodex[i]:.2f}",
                position=tuple(tcp_world + np.array([0, 0, 0.02])),
            )
            self._ik_viz_handles.append(h_lbl)

        self.grasp_status.value = (
            f"Showing top-{K} IK candidates (green=best score). "
            f"Hide via 'Hide IK candidates'."
        )

    def _plan_grasp_worker(self):
        name = self.dropdown.value
        self._plan_current_object(name, replay=False)

    def _plan_all_preview_worker(self):
        if not self.captured_objects:
            self.grasp_status.value = "No active objects to preview"
            return
        # Sort by distance to robot base (closer first → easier to reach, clears path for farther objects)
        queue = sorted(
            self.captured_objects.keys(),
            key=lambda n: float(np.linalg.norm(
                self.captured_mesh_poses.get(n, self.captured_objects[n])[:3, 3]
            )),
        )
        self.batch_plan_results = {}
        total = len(queue)
        # Optimistic sequential preview: each successful plan hides that object
        # from the world for subsequent plans, matching what run-all actually
        # sees at execution time. Without this, the last-grasped object in a
        # clutter is planned with every other object still present and often
        # fails even though execution would find the scene empty by then.
        # Objects are restored in a finally block so preview never permanently
        # alters world state on failure/exception.
        hidden_mesh = {}
        hidden_obj = {}
        preview_t0 = time.time()
        per_object_timing = []
        try:
            for idx, name in enumerate(queue, start=1):
                if name not in self.captured_objects:
                    continue
                self.dropdown.value = name
                n_present = len(self.captured_mesh_poses)
                self.grasp_status.value = (
                    f"[{idx}/{total}] Planning {name} "
                    f"(optimistic: {n_present} obstacles remain)..."
                )
                obj_t0 = time.time()
                result = self._plan_current_object(name, replay=False)
                obj_dt = time.time() - obj_t0
                per_object_timing.append((name, obj_dt, result.success if result else False))
                if result is None or not result.success:
                    self.grasp_status.value = (
                        f"SKIP: planning failed for {name} ({obj_dt:.1f}s), continuing..."
                    )
                    print(f"[preview] {idx}/{total} {name}: FAIL in {obj_dt:.2f}s", flush=True)
                    continue
                self.batch_plan_results[name] = result
                self.last_plan_result = result
                self.last_planned_object_name = name
                self._cache_replay_sequence(name, result)
                # Hide this object for remaining plans (optimistic removal).
                if name in self.captured_mesh_poses:
                    hidden_mesh[name] = self.captured_mesh_poses.pop(name)
                if name in self.captured_objects:
                    hidden_obj[name] = self.captured_objects[name]
                    # keep in captured_objects so later iteration's `if name not in
                    # captured_objects` gating still sees it — but mesh_poses
                    # removal is what _build_multi_object_world uses for obstacles.
                print(f"[preview] {idx}/{total} {name}: OK in {obj_dt:.2f}s", flush=True)
                self.grasp_status.value = f"[{idx}/{total}] Cached preview for {name} ({obj_dt:.1f}s)"
        finally:
            # Restore all hidden objects so preview leaves no side effects.
            for n, T in hidden_mesh.items():
                self.captured_mesh_poses[n] = T
            total_dt = time.time() - preview_t0
            print(f"[preview] total {total_dt:.2f}s for {len(queue)} objects:", flush=True)
            for n, dt, ok in per_object_timing:
                print(f"  - {n}: {dt:.2f}s {'OK' if ok else 'FAIL'}", flush=True)

        # Build combined replay: all objects' trajectories concatenated in order
        if self.batch_plan_results:
            combined = []
            for name in queue:
                obj_seq = self.replay_sequences.get(name)
                if obj_seq:
                    short_name = name.split(":")[-1] if ":" in name else name
                    for frame in obj_seq:
                        combined.append({
                            "q_arm": frame["q_arm"],
                            "q_hand": frame["q_hand"],
                            "label": f"{short_name} | {frame['label']}",
                        })
            if combined:
                self.replay_sequences["[all]"] = combined
                self._refresh_replay_object_options()
                self._set_active_replay("[all]")
                self.gui_replay_playing.value = True
        n_ok = len(self.batch_plan_results)
        n_fail = total - n_ok
        self.grasp_status.value = f"Preview ready: {n_ok} planned, {n_fail} failed. Execute individually or run all."

    def _plan_current_object(self, name: str, replay: bool = False):
        if name == "(none)" or name not in self.captured_objects:
            self.grasp_status.value = "No object selected"
            return None
        T_base = self.captured_mesh_poses.get(name, self.captured_objects[name])
        base_class = extract_base_class(name)
        # Map tracking name (e.g. "pepsi can") to BODex folder name (e.g. "pepsi")
        bodex_name = self.bodex_names.get(base_class, base_class)
        if bodex_name == base_class:
            bodex_name = self.bodex_names.get(normalize_object_name(base_class), bodex_name)
        if " " in bodex_name:
            bodex_name = normalize_object_name(bodex_name)
        scene_mode = str(self.dropdown_scene_mode.value)
        # Planner handles "auto" directly (tries simple_topdown → bodex fallback)
        self.grasp_status.value = f"Planning {bodex_name} [{scene_mode}]... (first run ~7s for GPU warmup)"
        try:
            planner = self._get_grasp_planner()
        except Exception as e:
            import traceback; traceback.print_exc()
            self.grasp_status.value = f"Init failed: {type(e).__name__}: {str(e)[:60]}"
            return None
        obj_mesh_path = self._resolve_mesh_path(base_class)
        world_cfg = self._build_multi_object_world(exclude_name=name)

        # Stream partial successes to the replay dropdown as they arrive so
        # the user can start previewing the first good grasp while cuRobo
        # keeps searching for alternatives. Uses a counter-based unique
        # label per stream so multiple partials don't overwrite each other.
        _partial_counter = [0]

        def _on_partial(partial_result):
            _partial_counter[0] += 1
            label = f"{name}#{_partial_counter[0]}"
            try:
                self._cache_replay_sequence(label, partial_result)
            except Exception as _e:
                print(f"[gui] partial cache failed: {_e}")

        try:
            result = planner.plan(obj_name=bodex_name, obj_pose_base=T_base,
                                  obj_mesh_path=obj_mesh_path,
                                  world_cfg=world_cfg,
                                  scene_mode=scene_mode,
                                  on_success=_on_partial)
        except FileNotFoundError as e:
            self.grasp_status.value = f"No candidates: {str(e)[:80]}"
            return None
        except Exception as e:
            import traceback; traceback.print_exc()
            self.grasp_status.value = f"Plan failed: {type(e).__name__}: {str(e)[:60]}"
            return None
        if not result.success:
            self.grasp_status.value = f"No feasible grasp (timing={result.timing})"
            return None
        self.last_plan_result = result
        self.last_planned_object_name = name
        self.batch_plan_results[name] = result
        self._update_plan_debug_markers(result)
        self.grasp_status.value = (
            f"Success: {result.scene_info} | traj {result.traj.shape} | "
            f"timing={result.timing}"
        )
        self._cache_replay_sequence(name, result)
        return result

    def _set_hand_joints(self, joints_6):
        """Forward 6-DOF Inspire finger targets to the viz robot.

        Mimic multipliers live in PinocchioViserRobot.set_hand_qpos; this
        keeps them in one place instead of duplicating them here.
        """
        self.robot_viz.set_hand_qpos(joints_6)

    # ----------------------------------------------------------- Real robot execution

    def _get_executor(self):
        from robothome.robot.grasp.executor import GraspExecutor
        if getattr(self, "executor", None) is None:
            self.executor = GraspExecutor(
                speed=float(self.slider_robot_speed.value),
                waypoint_skip=int(self.slider_waypoint_skip.value),
            )
        else:
            self.executor.speed = float(self.slider_robot_speed.value)
            self.executor.accel = max(0.05, float(self.slider_robot_speed.value) * 0.7)
            self.executor.waypoint_skip = int(self.slider_waypoint_skip.value)
        return self.executor

    def _on_check_robot(self, _):
        threading.Thread(target=self._check_robot_worker, daemon=True).start()

    def _check_robot_worker(self):
        self.robot_exec_status.value = "Checking..."
        try:
            executor = self._get_executor()
            ok, msg = executor.check_connection()
            self.robot_exec_status.value = msg
        except Exception as e:
            self.robot_exec_status.value = f"Error: {e}"

    def _on_execute_robot(self, _):
        name = self.dropdown.value
        result = self.batch_plan_results.get(name, self.last_plan_result)
        if result is None or not result.success:
            self.robot_exec_status.value = "No plan yet - run 'Plan grasp' first"
            return
        threading.Thread(target=self._execute_robot_worker, args=(result,),
                         daemon=True).start()

    def _execute_robot_worker(self, result, name=None):
        """Execute a plan. If name is given, discard THAT object on success
        (avoids a race where dropdown-callback-driven last_planned_object_name
        is stale and we remove the wrong object in run-all).
        """
        executor = self._get_executor()
        ok, msg = executor.execute(
            result,
            callback=lambda m: setattr(self.robot_exec_status, "value", m),
        )
        self.robot_exec_status.value = f"{'SUCCESS' if ok else 'FAIL'}: {msg}"
        discarded = name if name is not None else self.last_planned_object_name
        if ok and discarded:
            self.discarded_objects.add(discarded)
            self._remove_active_object(discarded)
            self.last_plan_result = None
            self.last_planned_object_name = None
            self.grasp_status.value = f"Discarded {discarded}. Select the next object."
        return ok, msg

    def _on_run_all_robot(self, _):
        # Set flag synchronously here to close the race window between
        # check and thread start — a rapid second click would otherwise
        # pass the check before the worker can set the flag.
        if self._run_all_in_progress:
            self.robot_exec_status.value = "Run all already in progress — wait for it to finish"
            return
        self._run_all_in_progress = True
        threading.Thread(target=self._run_all_robot_worker, daemon=True).start()

    def _run_all_robot_worker(self):
        try:
            self._run_all_robot_worker_impl()
        finally:
            self._run_all_in_progress = False

    def _run_all_robot_worker_impl(self):
        # Plan all is the single source of truth: only objects it successfully
        # planned are eligible here.  Anything missing from batch_plan_results
        # was a planning failure and is intentionally skipped — we never
        # re-plan synchronously here, or the user would see the robot move
        # for objects Plan all had marked as failed.
        planned_names = [
            n for n in self.batch_plan_results.keys()
            if n in self.captured_objects
            and self.batch_plan_results[n] is not None
            and self.batch_plan_results[n].success
        ]
        queue = sorted(
            planned_names,
            key=lambda n: float(np.linalg.norm(
                self.captured_mesh_poses.get(n, self.captured_objects[n])[:3, 3]
            )),
        )
        if not queue:
            self.robot_exec_status.value = "No planned objects — run 'Plan all' first"
            return
        total_active = len(self.captured_objects)
        self.robot_exec_status.value = (
            f"Running sequential discard for {len(queue)}/{total_active} "
            f"planned objects (nearest-first)..."
        )
        print(f"[run-all] queue ({len(queue)} items): {queue}", flush=True)
        print(f"[run-all] batch_plan_results keys: {list(self.batch_plan_results.keys())}", flush=True)
        completed = 0
        run_all_t0 = time.time()
        per_object_exec = []  # (name, seconds, ok)

        try:
            for i, name in enumerate(list(queue)):
                print(f"[run-all] iter {i}/{len(queue)-1}: name={name}", flush=True)
                if name not in self.captured_objects:
                    print(f"[run-all] iter {i}: SKIP — {name} not in captured_objects "
                          f"(current: {sorted(self.captured_objects.keys())})", flush=True)
                    continue
                self.dropdown.value = name

                # Use the pre-computed batch plan result. No re-planning.
                result = self.batch_plan_results.get(name)
                if result is None or not result.success:
                    print(f"[run-all] iter {i}: SKIP — no valid plan for {name} "
                          f"(result={'None' if result is None else f'success={result.success}'})",
                          flush=True)
                    self.robot_exec_status.value = f"SKIP: no valid plan for {name}, continuing..."
                    continue

                # Execute current object on robot.
                obj_t0 = time.time()
                try:
                    ok, msg = self._execute_robot_worker(result, name=name)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    ok, msg = False, f"exception: {type(e).__name__}: {e}"
                obj_dt = time.time() - obj_t0
                per_object_exec.append((name, obj_dt, ok))
                if not ok:
                    # Try to recover the robot from any error state (torque trip,
                    # etc.) so the next object in the queue can still execute.
                    try:
                        ex = self._get_executor()
                        if hasattr(ex, "recover_from_error"):
                            ex.recover_from_error()
                    except Exception:
                        pass
                    self.robot_exec_status.value = f"SKIP: execution failed for {name}: {msg}"
                    continue
                completed += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[run-all] WORKER CRASHED: {type(e).__name__}: {e}", flush=True)
            self.robot_exec_status.value = f"CRASHED: {type(e).__name__}: {e}"

        total_dt = time.time() - run_all_t0
        self.robot_exec_status.value = (
            f"SUCCESS: {completed}/{len(queue)} objects in {total_dt:.1f}s"
        )
        print(f"[run-all] total wall time: {total_dt:.2f}s")
        print(f"[run-all] per-object execution:")
        for n, dt, okf in per_object_exec:
            print(f"  - {n}: {dt:.2f}s {'OK' if okf else 'FAIL'}")

    # ── Rejection inspector ───────────────────────────────────────────────

    def _on_pull_rejection(self, _):
        planner = getattr(self, "grasp_planner", None)
        log = list(getattr(planner, "_candidate_viz_log", []) or []) if planner is not None else []
        if not log:
            # Fall back to the (legacy) single-rejection pointer
            rej = getattr(planner, "_last_rejection", None) if planner is not None else None
            if rej is not None:
                log = [rej]
        if not log:
            self.rej_info.value = "No candidates captured yet — run Plan all first"
            self.rejection_data = None
            self.dropdown_rej_candidate.options = ("(none)",)
            self.dropdown_rej_candidate.value = "(none)"
            return
        self.rej_candidate_log = log
        labels = []
        for i, rej in enumerate(log):
            reason = rej.get("reason", "?")
            cidx = rej.get("candidate_idx", -1)
            attempt = rej.get("attempt_n", i + 1)
            labels.append(f"#{attempt} idx={cidx} [{reason}]")
        self.dropdown_rej_candidate.options = tuple(labels)
        self.dropdown_rej_candidate.value = labels[0]
        self._load_rejection_candidate(0)

    def _on_rej_candidate_pick(self, _):
        val = self.dropdown_rej_candidate.value
        if not val or val == "(none)":
            return
        for i, rej in enumerate(self.rej_candidate_log):
            attempt = rej.get("attempt_n", i + 1)
            cidx = rej.get("candidate_idx", -1)
            reason = rej.get("reason", "?")
            if val == f"#{attempt} idx={cidx} [{reason}]":
                self._load_rejection_candidate(i)
                return

    def _load_rejection_candidate(self, i: int):
        if i < 0 or i >= len(self.rej_candidate_log):
            return
        rej = self.rej_candidate_log[i]
        self.rejection_data = rej
        n_wp = len(rej.get("waypoints", []))
        if n_wp <= 0:
            self.rej_info.value = "Candidate has no waypoints"
            return
        self.slider_rej_wp.max = max(1, n_wp - 1)
        target_wp = self._find_viz_idx_for_violation(rej)
        self.slider_rej_wp.value = target_wp
        self._render_rejection_wp(target_wp)

    def _on_rej_slider(self, _):
        if self.rejection_data is None:
            return
        v = self.slider_rej_wp.value
        try:
            wp_idx = int(v)
        except (TypeError, ValueError):
            return  # defensive guard (viser can send NaN on empty-range slider)
        self._render_rejection_wp(wp_idx)

    def _find_viz_idx_for_violation(self, rej):
        viol_t = rej["violating_wp"]
        for i, wp in enumerate(rej["waypoints"]):
            if wp["t_idx"] >= viol_t:
                return i
        return 0

    def _clearance_to_colors(self, clearances, margin):
        colors = np.zeros((len(clearances), 3), dtype=np.uint8)
        for i, c in enumerate(clearances):
            if c < 0.0:
                colors[i] = [255, 0, 0]
            elif c < margin:
                colors[i] = [255, 200, 0]
            else:
                g = int(min(220, 120 + c * 1000))
                colors[i] = [0, g, 60]
        return colors

    def _render_rejection_wp(self, wp_idx: int):
        rej = self.rejection_data
        if rej is None:
            return
        wps = rej["waypoints"]
        if not wps:
            return
        wp_idx = int(np.clip(wp_idx, 0, len(wps) - 1))
        wp = wps[wp_idx]
        q = wp["q"]
        q_arm = np.asarray(q[:7], dtype=np.float64)
        q_hand = np.asarray(q[7:], dtype=np.float64)
        self.qpos_current = q_arm.copy()
        self.robot_viz.set_qpos(q_arm)
        self._set_hand_joints(q_hand)
        self.robot_viz.update_viser_link_frames()
        self._update_collision_spheres()
        margin = rej["margin_m"]
        centers = np.asarray(wp["centers"], dtype=np.float32)
        radii = np.asarray(wp["radii"], dtype=np.float32)
        clearances = np.asarray(wp["clearances"], dtype=np.float64)
        colors = self._clearance_to_colors(clearances, margin)
        # Only render spheres that are reasonably close to the scene — safe
        # ones swamp the scene and hide the actual problem.
        threshold_m = float(self.slider_rej_clear_filter_mm.value) / 1000.0
        visible_mask = clearances < threshold_m
        # Unique path per scrub tick so re-adding with the same name doesn't
        # race against the previous batch's removal.
        self._rej_tick += 1
        tick = self._rej_tick
        for h in self.rej_sphere_handles:
            try:
                h.remove()
            except Exception:
                pass
        self.rej_sphere_handles = []
        proto = trimesh.creation.icosphere(radius=1.0, subdivisions=1)
        proto_verts = np.asarray(proto.vertices, dtype=np.float32)
        proto_faces = np.asarray(proto.faces, dtype=np.uint32)
        for i in range(len(centers)):
            if not bool(visible_mask[i]):
                continue
            r = float(radii[i])
            self.rej_sphere_handles.append(
                self.server.scene.add_mesh_simple(
                    f"/rejection/t{tick}/s{i}",
                    vertices=proto_verts * r,
                    faces=proto_faces,
                    color=tuple(int(x) for x in colors[i]),
                    opacity=0.55,
                    position=tuple(float(x) for x in centers[i]),
                )
            )
        worst_i = int(np.argmin(clearances))
        worst_mm = float(clearances[worst_i]) * 1000.0
        worst_link = wp["links"][worst_i]
        is_violating = wp["t_idx"] == rej["violating_wp"]
        tag = " !" if clearances[worst_i] < margin else ""
        self.rej_info.value = (
            f"wp={wp['t_idx']}/{len(rej['traj'])-1} "
            f"worst={worst_mm:.1f}mm ({worst_link}){tag}"
            + (f" — first violation at wp={rej['violating_wp']}" if is_violating else "")
        )

    def _on_release_robot(self, _):
        threading.Thread(target=self._release_robot_worker, daemon=True).start()

    def _release_robot_worker(self):
        executor = self._get_executor()
        executor.release(
            callback=lambda m: setattr(self.robot_exec_status, "value", m),
        )

    # ----------------------------------------------------------- Main loop

    def run(self):
        print(f"[2/2] Server running at http://localhost:{self.port}")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nShutting down...")


def main():
    parser = argparse.ArgumentParser(description="BODex Grasp Planner GUI")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--handeye", type=str, default=DEFAULT_HANDEYE)
    parser.add_argument("--tracking_host", type=str, default=DEFAULT_TRACKING_HOST)
    args = parser.parse_args()

    gui = GraspPlannerGUI(
        port=args.port,
        handeye_path=args.handeye,
        tracking_host=args.tracking_host,
    )
    gui.run()


if __name__ == "__main__":
    main()
