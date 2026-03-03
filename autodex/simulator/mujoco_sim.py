"""
MuJoCo Grasping Simulator

MuJoCo port of ~/paradex/paradex/simulator/isaac.py.
Interface is intentionally kept similar so callers can swap backends.

Usage:
    sim = Simulator(headless=True, table_z=0.037)
    sim.load_robot_asset("xarm", "allegro")
    sim.load_object_asset("apple")
    sim.add_env("env0", {
        "robot":  {"xarm_allegro": ("xarm", "allegro"),
                   "arm2":         ("xarm", "allegro")},
        "object": {"apple": "apple",
                   "cup":   "cup"},
    }, obj_poses={"apple": T_apple, "cup": T_cup})
    sim.reset("env0", {
        "robot":  {"xarm_allegro": init_qpos},   # (22,)
        "object": {"apple": T_apple},            # (4,4)
    })
    for qpos in trajectory:
        sim.step("env0", {"robot": {"xarm_allegro": qpos}})
        sim.tick()
        state = sim.get_state("env0")
    sim.terminate()
"""

import os
import json
import numpy as np
import mujoco
import mujoco.viewer
import transforms3d.quaternions as tq

# ==================== Asset paths ====================

ROBOT_URDF_DIR = (
    "/home/mingi/shared_data/RSS2026_Mingi/content/assets/robot"
    "/allegro_description"
)
OBJ_BASE = "/home/mingi/shared_data/RSS2026_Mingi/object/paradex"

ARM_JOINTS  = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
HAND_JOINTS = [f"joint_{i}.0" for i in range(16)]

# Position control gains (matches isaac.py stiffness values)
KP_ARM  = 1000.0
KP_HAND = 500.0


# ==================== Helpers ====================

def _se3_to_pos_quat(T):
    """4x4 SE3 -> (pos [3], quat [qw,qx,qy,qz])."""
    return T[:3, 3].copy(), tq.mat2quat(T[:3, :3])

def _pos_quat_to_se3(pos, quat):
    """(pos, [qw,qx,qy,qz]) -> 4x4 SE3."""
    T = np.eye(4)
    T[:3, :3] = tq.quat2mat(quat)
    T[:3, 3]  = pos
    return T

def _robot_urdf_path(arm_name, hand_name):
    if arm_name is None:
        name = hand_name
    elif hand_name is None:
        name = arm_name
    else:
        name = f"{arm_name}_{hand_name}"
    return os.path.join(ROBOT_URDF_DIR, f"{name}.urdf")

def _build_mesh_lookup(urdf_dir):
    """Walk urdf_dir and return {basename: abspath} for mesh files.
    MjSpec strips directory prefixes from mesh filenames, so we resolve them here.
    """
    lookup = {}
    for root, _, files in os.walk(urdf_dir):
        for f in files:
            if f.endswith((".obj", ".stl")):
                lookup.setdefault(f, os.path.join(root, f))
    return lookup


# ==================== Model builder ====================

def _compile_model(robots, objects, table_z=0.037, friction=(0.8, 0.02)):
    """
    Build a MuJoCo model with arbitrary numbers of robots and objects.

    robots:  [(actor_name, arm_name, hand_name), ...]
    objects: [(actor_name, obj_name, pose_se3, obj_mass), ...]

    Each robot actor gets a unique joint prefix  "<actor_name>-".
    Each object actor gets a freejoint named     "<actor_name>_freejoint".
    """
    spec = mujoco.MjSpec()

    # Robots
    for actor_name, arm_name, hand_name in robots:
        prefix      = f"{actor_name}-"
        urdf_path   = _robot_urdf_path(arm_name, hand_name)
        urdf_dir    = os.path.dirname(urdf_path)
        mesh_lookup = _build_mesh_lookup(urdf_dir)

        rspec = mujoco.MjSpec.from_file(urdf_path)
        rspec.compiler.balanceinertia = True
        for m in rspec.meshes:
            m.file = mesh_lookup.get(m.file, os.path.join(urdf_dir, m.file))
        for g in rspec.geoms:
            g.solimp[:3]   = [0.5, 0.99, 0.0001]
            g.solref[:2]   = [0.005, 1]
            g.friction[:2] = friction
            g.condim       = 4
        spec.worldbody.add_frame().attach_body(rspec.worldbody, prefix, "")

    # Table (infinite plane)
    spec.worldbody.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[0, 0, table_z],
        size=[0, 0, 1.0],
    )

    # Objects
    for actor_name, obj_name, pose_se3, obj_mass in objects:
        obj_dir  = os.path.join(OBJ_BASE, obj_name, "processed_data")
        mesh_dir = os.path.join(obj_dir, "urdf", "meshes")
        with open(os.path.join(obj_dir, "info", "simplified.json")) as f:
            info = json.load(f)
        obj_density = obj_mass / (info["mass"] / info["density"])

        obj_pos, obj_quat = _se3_to_pos_quat(pose_se3)
        obj_body = spec.worldbody.add_body(
            name=f"obj_{actor_name}",
            pos=obj_pos.tolist(), quat=obj_quat.tolist(),
        )
        obj_body.add_freejoint(name=f"{actor_name}_freejoint")
        for fname in sorted(os.listdir(mesh_dir)):
            if not fname.endswith(".obj"):
                continue
            piece = fname.replace(".obj", "")
            mname = f"obj_{actor_name}_{piece}"
            spec.add_mesh(name=mname, file=os.path.join(mesh_dir, fname), scale=[1, 1, 1])
            obj_body.add_geom(name=f"obj_vis_{actor_name}_{piece}",
                              type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mname,
                              density=0, contype=0, conaffinity=0,
                              rgba=[0.9, 0.45, 0.1, 1.0])
            cg = obj_body.add_geom(name=f"obj_col_{actor_name}_{piece}",
                                   type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mname,
                                   density=obj_density)
            cg.solimp[:3]   = [0.5, 0.99, 0.0001]
            cg.solref[:2]   = [0.005, 1]
            cg.friction[:2] = friction
            cg.condim       = 4

    # Physics
    spec.option.gravity    = [0, 0, -9.81]
    spec.option.timestep   = 0.004
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone       = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.noslip_iterations = 2
    spec.option.impratio   = 10

    # Actuators
    for actor_name, arm_name, hand_name in robots:
        prefix = f"{actor_name}-"
        for jname, kp in [(j, KP_ARM) for j in ARM_JOINTS] + [(j, KP_HAND) for j in HAND_JOINTS]:
            act = spec.add_actuator()
            act.name       = f"act_{prefix}{jname}"
            act.target     = prefix + jname
            act.trntype    = mujoco.mjtTrn.mjTRN_JOINT
            act.gaintype   = mujoco.mjtGain.mjGAIN_FIXED
            act.biastype   = mujoco.mjtBias.mjBIAS_AFFINE
            act.gainprm[0] = kp
            act.biasprm[1] = -kp

    spec.add_key()
    model = spec.compile()
    data  = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


def _build_actor_index(model, robots, objects):
    """
    Query the compiled model to build per-actor qpos/ctrl index maps.

    Returns dict:
        {actor_name: {"type": "robot",
                      "qpos_addrs": [int, ...],   # len 22
                      "dof_addrs":  [int, ...],   # len 22 (for qvel)
                      "ctrl_ids":   [int, ...]}}  # len 22
        {actor_name: {"type": "object",
                      "qpos_start": int}}         # freejoint qpos[s:s+7]
    """
    index = {}

    for actor_name, *_ in robots:
        prefix     = f"{actor_name}-"
        qpos_addrs = []
        dof_addrs  = []
        ctrl_ids   = []
        for jname in ARM_JOINTS + HAND_JOINTS:
            full_jname = prefix + jname
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, full_jname)
            qpos_addrs.append(int(model.jnt_qposadr[jnt_id]))
            dof_addrs.append(int(model.jnt_dofadr[jnt_id]))
            act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                        f"act_{full_jname}")
            ctrl_ids.append(int(act_id))
        index[actor_name] = {
            "type":       "robot",
            "qpos_addrs": qpos_addrs,
            "dof_addrs":  dof_addrs,
            "ctrl_ids":   ctrl_ids,
        }

    for actor_name, *_ in objects:
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                    f"{actor_name}_freejoint")
        index[actor_name] = {
            "type":       "object",
            "qpos_start": int(model.jnt_qposadr[jnt_id]),
        }

    return index


# ==================== Simulator ====================

class Simulator:
    """
    MuJoCo grasping simulator.

    Supports arbitrary numbers of robots and objects per environment.
    Each call to add_env() creates an independent MjModel + MjData pair.
    tick() advances all registered environments in serial.
    """

    def __init__(self, headless=True, table_z=0.037,
                 friction=(0.8, 0.02), obj_mass=0.1):
        self.headless  = headless
        self.table_z   = table_z
        self.friction  = friction
        self.obj_mass  = obj_mass

        self._model:  dict = {}
        self._data:   dict = {}
        self._viewer: dict = {}
        self._index:  dict = {}  # {env_name: {actor_name: index_info}}

        self.num_envs = 0

    # ---------- Asset registration ----------

    def load_robot_asset(self, arm_name, hand_name):
        urdf = _robot_urdf_path(arm_name, hand_name)
        assert os.path.exists(urdf), f"URDF not found: {urdf}"

    def load_object_asset(self, obj_name):
        path = os.path.join(OBJ_BASE, obj_name, "processed_data", "urdf", "meshes")
        assert os.path.isdir(path), f"Object meshes not found: {path}"

    # ---------- Environment ----------

    def add_env(self, name, env_info, obj_poses=None):
        """
        env_info = {
            "robot":  {actor_name: (arm_name, hand_name), ...},
            "object": {actor_name: obj_name, ...},
        }
        obj_poses = {actor_name: T_4x4, ...}   # omit for identity pose
        """
        if obj_poses is None:
            obj_poses = {}

        robots  = [(actor, arm, hand)
                   for actor, (arm, hand) in env_info.get("robot", {}).items()]
        objects = [(actor, oname, obj_poses.get(actor, np.eye(4)), self.obj_mass)
                   for actor, oname in env_info.get("object", {}).items()]

        model, data = _compile_model(
            robots, objects,
            table_z=self.table_z, friction=self.friction,
        )
        index = _build_actor_index(model, robots, objects)

        self._model[name]  = model
        self._data[name]   = data
        self._index[name]  = index

        viewer = None
        if not self.headless:
            viewer = mujoco.viewer.launch_passive(model, data)
            viewer.sync()
        self._viewer[name] = viewer
        self.num_envs += 1

    def destroy_env(self, name):
        if self._viewer.get(name):
            self._viewer[name].close()
        for d in [self._model, self._data, self._viewer, self._index]:
            d.pop(name, None)
        self.num_envs -= 1

    # ---------- Control ----------

    def reset(self, name, action_dict):
        """
        action_dict = {
            "robot":  {actor_name: qpos (22,)},
            "object": {actor_name: T (4,4)},
        }
        Sets qpos + ctrl for robots, qpos for objects, zeroes qvel.
        """
        data  = self._data[name]
        model = self._model[name]
        index = self._index[name]

        for actor, qpos in action_dict.get("robot", {}).items():
            if actor not in index:
                continue
            info  = index[actor]
            qpos  = np.asarray(qpos, dtype=float)
            for i, addr in enumerate(info["qpos_addrs"]):
                data.qpos[addr] = qpos[i]
            for i, cid in enumerate(info["ctrl_ids"]):
                data.ctrl[cid] = qpos[i]

        for actor, T in action_dict.get("object", {}).items():
            if actor not in index:
                continue
            info = index[actor]
            pos, quat = _se3_to_pos_quat(np.asarray(T))
            s = info["qpos_start"]
            data.qpos[s:s+3] = pos
            data.qpos[s+3:s+7] = quat

        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        if self._viewer.get(name):
            self._viewer[name].sync()

    def step(self, name, action_dict):
        """Set position control targets (call tick() to advance physics)."""
        data  = self._data[name]
        index = self._index[name]
        for actor, qpos in action_dict.get("robot", {}).items():
            if actor not in index:
                continue
            info = index[actor]
            qpos = np.asarray(qpos, dtype=float)
            for i, cid in enumerate(info["ctrl_ids"]):
                data.ctrl[cid] = qpos[i]

    def tick(self, n=1):
        """Advance all environments by n physics steps."""
        for name, model in self._model.items():
            data   = self._data[name]
            viewer = self._viewer.get(name)
            for _ in range(n):
                mujoco.mj_step(model, data)
            if viewer:
                viewer.sync()

    # ---------- State ----------

    def get_state(self, name):
        """
        Returns:
            {
              "robot":  {actor_name: {"qpos": (22,), "qvel": (22,)}},
              "object": {actor_name: T (4,4)},
            }
        """
        data  = self._data[name]
        index = self._index[name]
        result = {"robot": {}, "object": {}}

        for actor, info in index.items():
            if info["type"] == "robot":
                qpos = np.array([data.qpos[a] for a in info["qpos_addrs"]])
                qvel = np.array([data.qvel[a] for a in info["dof_addrs"]])
                result["robot"][actor] = {"qpos": qpos, "qvel": qvel}
            else:
                s = info["qpos_start"]
                result["object"][actor] = _pos_quat_to_se3(
                    data.qpos[s:s+3].copy(), data.qpos[s+3:s+7].copy()
                )

        return result

    # ---------- Cleanup ----------

    def terminate(self):
        for name in list(self._model.keys()):
            self.destroy_env(name)
        print("Simulation terminated")