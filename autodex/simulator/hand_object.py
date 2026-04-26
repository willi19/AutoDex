"""MuJoCo Hand-Object simulation (ported from RSS_2026/sim_eval/util/hand_util.py).

Uses MuJoCo XML hand models (not URDF) with mocap weld for wrist control.
"""

import os
import numpy as np
import mujoco
import mujoco.viewer

from autodex.simulator.rot_util import interplote_pose, interplote_qpos
from autodex.utils.path import obj_path

SIM_FILTER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "src", "grasp_generation", "sim_filter",
)


class MjHO:
    """MuJoCo Hand-Object environment."""

    hand_prefix = "child-"

    def __init__(
        self,
        obj_name,
        hand_path,
        weld_body_name="world",
        obj_mass=0.1,
        friction_coef=(0.6, 0.02),
        has_floor=False,
        debug_viewer=False,
        obj_root_dir=None,
    ):
        self._obj_root_dir = obj_root_dir or obj_path
        self.hand_mocap = True
        self._is_urdf = False
        self.spec = mujoco.MjSpec()
        self.spec.meshdir = SIM_FILTER_DIR
        self.spec.option.timestep = 0.004
        self.spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.spec.option.disableflags = mujoco.mjtDisableBit.mjDSBL_GRAVITY

        if debug_viewer:
            self.spec.add_texture(
                type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                rgb1=[0.3, 0.5, 0.7], rgb2=[0.3, 0.5, 0.7],
                width=512, height=512,
            )
            self.spec.worldbody.add_light(name="spotlight", pos=[0, -1, 2], castshadow=False)
            self.spec.worldbody.add_camera(name="closeup", pos=[0.0, -0.6, 0.0], xyaxes=[1, 0, 0, 0, 0, 1])

        self._add_hand(hand_path, weld_body_name=weld_body_name)
        self._add_object(obj_name, obj_mass, has_floor)
        self._set_friction(friction_coef)
        self.spec.add_key()

        # For URDF: compile once to discover joints, add actuators, recompile
        if self._is_urdf:
            tmp_model = self.spec.compile()
            # Find joints that are mimic (have equality constraints)
            mimic_joint_ids = set()
            for i in range(tmp_model.neq):
                if tmp_model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT:
                    # eq_obj1id is the mimic joint
                    mimic_joint_ids.add(tmp_model.eq_obj1id[i])
            # Only add actuators to non-free, non-mimic joints
            for i in range(tmp_model.njnt):
                jnt = tmp_model.joint(i)
                if jnt.type == mujoco.mjtJoint.mjJNT_FREE:
                    continue
                if i in mimic_joint_ids:
                    continue
                act = self.spec.add_actuator()
                act.name = f"act_{jnt.name}"
                act.target = jnt.name
                act.trntype = mujoco.mjtTrn.mjTRN_JOINT
                act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
                act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
                act.gainprm[0] = 5.0
                act.biasprm[1] = -5.0
            del tmp_model

        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)

        if self._is_urdf:
            # Disable collision on links that BODex treats as non-colliding.
            # base_link: raw STL convex hull too large.
            # inspire_f1 specific: plam_1 / plam_force_sensor / *_tip not in BODex collision set
            #   (BODex uses plam_2, plam_3, *_force_sensor instead). If left enabled, palm/tip
            #   collision prevents the hand from closing as BODex intended.
            disable_substrings = ('base_link', 'plam_1', 'plam_force_sensor',
                                  'thumb_tip', 'index_tip', 'middle_tip',
                                  'ring_tip', 'little_tip')
            for i in range(self.model.ngeom):
                body_id = self.model.geom_bodyid[i]
                body_name = self.model.body(body_id).name
                if any(s in body_name for s in disable_substrings):
                    self.model.geom_contype[i] = 0
                    self.model.geom_conaffinity[i] = 0
            # Add joint damping (URDF has 0 damping → unstable with position actuators)
            for i in range(self.model.njnt):
                if self.model.joint(i).type != mujoco.mjtJoint.mjJNT_FREE:
                    self.model.dof_damping[self.model.jnt_dofadr[i]] = 0.1

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        # qpos → ctrl mapping
        qpos2ctrl_matrix = np.zeros((self.model.nu, self.model.nv))
        mujoco.mju_sparse2dense(
            qpos2ctrl_matrix,
            self.data.actuator_moment,
            self.data.moment_rownnz,
            self.data.moment_rowadr,
            self.data.moment_colind,
        )
        self._qpos2ctrl_matrix = qpos2ctrl_matrix[..., :-6]  # exclude object freejoint

        self.viewer = None
        if debug_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def _add_hand(self, hand_path, weld_body_name="world"):
        """Load hand from XML or URDF. XML must have actuators defined.
        URDF: actuators are auto-generated with kp=5."""
        is_urdf = hand_path.lower().endswith(".urdf")

        child_spec = mujoco.MjSpec.from_file(hand_path)

        if is_urdf:
            # URDF: resolve mesh paths
            urdf_dir = os.path.dirname(hand_path)
            mesh_lookup = {}
            for root, _, files in os.walk(urdf_dir):
                for f in files:
                    if f.lower().endswith((".obj", ".stl")):
                        mesh_lookup.setdefault(f, os.path.join(root, f))
            for m in child_spec.meshes:
                m.file = mesh_lookup.get(m.file, os.path.join(urdf_dir, m.file))
        else:
            # XML: resolve relative meshdir
            for m in child_spec.meshes:
                m.file = os.path.join(os.path.dirname(hand_path), child_spec.meshdir, m.file)
        child_spec.meshdir = self.spec.meshdir

        for g in child_spec.geoms:
            g.solimp[:3] = [0.5, 0.99, 0.0001]
            g.solref[:2] = [0.005, 1]

        attach_frame = self.spec.worldbody.add_frame()
        child_world = attach_frame.attach_body(child_spec.worldbody, self.hand_prefix, "")

        child_world.add_freejoint(name="hand_freejoint")
        self.spec.worldbody.add_body(name="mocap_body", mocap=True)
        self.spec.add_equality(
            type=mujoco.mjtEq.mjEQ_WELD,
            name1="mocap_body",
            name2=f"{self.hand_prefix}{weld_body_name}",
            objtype=mujoco.mjtObj.mjOBJ_BODY,
            solimp=[0.9, 0.95, 0.001, 0.5, 2],
            data=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        )

        self._is_urdf = is_urdf

    def _add_object(self, obj_name, obj_mass, has_floor):
        if has_floor:
            self.spec.worldbody.add_geom(
                name="object_collision_floor",
                type=mujoco.mjtGeom.mjGEOM_PLANE,
                pos=[0, 0, 0], size=[0, 0, 1.0],
            )

        obj_dir = os.path.join(self._obj_root_dir, obj_name, "processed_data")
        mesh_dir = os.path.join(obj_dir, "urdf", "meshes")

        import json
        with open(os.path.join(obj_dir, "info", "simplified.json")) as f:
            info = json.load(f)
        obj_density = obj_mass / (info["mass"] / info["density"])

        obj_body = self.spec.worldbody.add_body(name="object")
        obj_body.add_freejoint(name="obj_freejoint")
        for fname in sorted(os.listdir(mesh_dir)):
            if not fname.endswith(".obj"):
                continue
            mesh_name = fname.replace(".obj", "")
            mesh_id = mesh_name.replace("convex_piece_", "")
            self.spec.add_mesh(name=mesh_name, file=os.path.join(mesh_dir, fname))
            obj_body.add_geom(
                name=f"object_visual_{mesh_id}",
                type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh_name,
                density=0, contype=0, conaffinity=0,
            )
            obj_body.add_geom(
                name=f"object_collision_{mesh_id}",
                type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh_name,
                density=obj_density,
            )

    def _set_friction(self, friction_coef):
        self.spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        self.spec.option.noslip_iterations = 2
        self.spec.option.impratio = 10
        for g in self.spec.geoms:
            g.friction[:2] = friction_coef
            g.condim = 4

    def _qpos2ctrl(self, hand_qpos):
        return self._qpos2ctrl_matrix[:, 6:] @ hand_qpos[7:]

    def get_obj_pose(self):
        return self.data.qpos[-7:]

    def set_ext_force_on_obj(self, ext_force):
        self.data.xfrc_applied[-1] = ext_force

    def reset_pose_qpos(self, hand_qpos, obj_pose):
        self.model.key_qpos[0] = np.concatenate([hand_qpos, obj_pose])
        self.model.key_ctrl[0] = self._qpos2ctrl(hand_qpos)
        self.model.key_qvel[0] = 0
        self.model.key_act[0] = 0
        self.model.key_mpos[0] = hand_qpos[:3]
        self.model.key_mquat[0] = hand_qpos[3:7]
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

    def control_hand_with_interp(self, hand_qpos1, hand_qpos2, step_outer=10, step_inner=10):
        pose_interp = interplote_pose(hand_qpos1[:7], hand_qpos2[:7], step_outer)
        qpos_interp = interplote_qpos(self._qpos2ctrl(hand_qpos1), self._qpos2ctrl(hand_qpos2), step_outer)

        for j in range(step_outer):
            self.data.mocap_pos[0] = pose_interp[j, :3]
            self.data.mocap_quat[0] = pose_interp[j, 3:7]
            self.data.ctrl[:] = qpos_interp[j]
            mujoco.mj_forward(self.model, self.data)
            self.control_hand_step(step_inner)

    def get_contact_info(self):
        """Get hand-object contacts. Returns list of {pos, normal, force, hand_body}."""
        object_id = self.model.nbody - 1
        contacts = []
        for c in self.data.contact:
            b1 = self.model.geom(c.geom1).bodyid
            b2 = self.model.geom(c.geom2).bodyid
            b1_name = self.model.body(b1).name
            b2_name = self.model.body(b2).name

            is_ho = (b2 == object_id and b1 != object_id) or (b1 == object_id and b2 != object_id)
            if not is_ho:
                continue

            if b2 == object_id:
                normal = c.frame[0:3]
                hand_body = b1_name.removeprefix(self.hand_prefix)
            else:
                normal = -c.frame[0:3]
                hand_body = b2_name.removeprefix(self.hand_prefix)

            contacts.append({
                "pos": c.pos.tolist(),
                "normal": normal.tolist(),
                "dist": float(c.dist),
                "hand_body": hand_body,
            })
        return contacts

    def control_hand_step(self, step_inner):
        for _ in range(step_inner):
            mujoco.mj_step(self.model, self.data)
        if self.viewer:
            self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None