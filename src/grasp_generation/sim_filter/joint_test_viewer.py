"""Mujoco-based hand viewer with one slider per active joint.
Move one slider, see which finger moves."""
import os, sys, time, numpy as np
import mujoco, viser, trimesh
from scipy.spatial.transform import Rotation as Rot

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
from autodex.simulator.hand_object import MjHO

HAND_PATH = f"{REPO_ROOT}/src/grasp_generation/BODex/src/curobo/content/assets/robot/inspire_f1_description/inspire_f1_hand_right.urdf"
INSPIRE_F1_MIMIC_MAP = [None, None, (1,1.2953,0), (1,1.1610,0),
                         None, (2,1.1545,0), None, (3,1.1545,0),
                         None, (4,1.1545,0), None, (5,1.1545,0)]
ACTIVE_NAMES = ["thumb_1","thumb_2","index_1","middle_1","ring_1","little_1"]

def expand(j6):
    out=[]; ai=0
    for e in INSPIRE_F1_MIMIC_MAP:
        if e is None: out.append(j6[ai]); ai+=1
        else: out.append(j6[e[0]]*e[1]+e[2])
    return np.array(out)

# Use any obj (just for mujoco setup; we won't draw it)
mj = MjHO("Jp_Water", HAND_PATH, weld_body_name="base_link", obj_mass=0.1, debug_viewer=False,
          obj_root_dir="/home/mingi/shared_data/AutoDex/object/robothome")
wrist_cart = np.array([0,0,0,1,0,0,0], dtype=float)
active = np.zeros(6)
mj.reset_pose_qpos(np.concatenate([wrist_cart, expand(active)]),
                    np.array([100,100,100,1,0,0,0], dtype=float))  # obj far away
mujoco.mj_forward(mj.model, mj.data)

server = viser.ViserServer(port=8082)
print("Viser at http://localhost:8082")

mesh_handles = {}
def redraw():
    mujoco.mj_forward(mj.model, mj.data)
    urdf_dir = os.path.dirname(HAND_PATH)
    mesh_lookup = {}
    for root,_,files in os.walk(urdf_dir):
        for f in files:
            if f.lower().endswith(".stl"): mesh_lookup.setdefault(f, os.path.join(root,f))
    for i in range(mj.model.nbody):
        b = mj.model.body(i)
        if mj.hand_prefix not in b.name: continue
        name = b.name.replace(mj.hand_prefix, "")
        cands = [f"{name}.STL", f"{name}.stl"]
        stl = next((mesh_lookup[c] for c in cands if c in mesh_lookup), None)
        if stl is None: continue
        m = trimesh.load(stl, force="mesh")
        T = np.eye(4); T[:3,:3] = mj.data.xmat[i].reshape(3,3); T[:3,3] = mj.data.xpos[i]
        m.apply_transform(T)
        if name in mesh_handles:
            try: mesh_handles[name].remove()
            except: pass
        mesh_handles[name] = server.scene.add_mesh_simple(
            f"/hand/{name}", vertices=np.asarray(m.vertices),
            faces=np.asarray(m.faces, dtype=np.uint32),
            color=(220,150,80), opacity=0.9, side="double")

redraw()

sliders = []
for i, nm in enumerate(ACTIVE_NAMES):
    s = server.gui.add_slider(f"[{i}] {nm}", min=-0.5, max=2.0, step=0.01, initial_value=0.0)
    sliders.append(s)
    @s.on_update
    def _(event, idx=i):
        active[idx] = sliders[idx].value
        full12 = expand(active)
        # Set qpos[7:19] = finger 12
        mj.data.qpos[7:19] = full12
        redraw()

while True:
    time.sleep(1)
