"""Dump a combined OBJ + PNG showing inspire hand meshes with their collision spheres.

Loads inspire_hand_{right,left}.urdf at zero pose, places both side-by-side
along x, paints each link mesh light gray and overlays its collision spheres
(red on right, cyan on left). Export to:

  bodex_outputs/viz/inspire_collision.obj
  bodex_outputs/viz/inspire_collision_{front,top,side}.png   (if pyrender works)

Open the OBJ in MeshLab/Blender, or just view the PNGs.

Run:
    /home/mingi/miniconda3/envs/bodex/bin/python src/grasp_generation/BODex/viz_inspire_collision.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import trimesh
import yaml
import yourdfpy

REPO = Path(__file__).resolve().parent
ASSET_ROOT = REPO / "src/curobo/content/assets/robot/inspire_description"
SPHERE_ROOT = REPO / "src/curobo/content/configs/robot/spheres"
OUT_DIR = REPO.parent.parent.parent / "bodex_outputs/viz"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HANDS = {
    "right": {
        "urdf": ASSET_ROOT / "inspire_hand_right.urdf",
        "spheres": SPHERE_ROOT / "inspire.yml",
        "x_offset": -0.12,
        "sphere_color": [220, 40, 40, 200],   # red
        "mesh_color":   [200, 200, 200, 255],
        "link_prefix": "right",
    },
    "left": {
        "urdf": ASSET_ROOT / "inspire_hand_left.urdf",
        "spheres": SPHERE_ROOT / "inspire_left.yml",
        "x_offset": +0.12,
        "sphere_color": [40, 160, 220, 200],  # cyan
        "mesh_color":   [200, 200, 200, 255],
        "link_prefix": "left",
    },
}


def collect_hand_geometry(cfg: dict) -> list[trimesh.Trimesh]:
    urdf = yourdfpy.URDF.load(
        str(cfg["urdf"]),
        build_scene_graph=True,
        load_meshes=True,
        build_collision_scene_graph=False,
        load_collision_meshes=False,
    )
    urdf.update_cfg({})  # zero joint angles

    spheres = yaml.safe_load(cfg["spheres"].read_text())["collision_spheres"]

    geoms: list[trimesh.Trimesh] = []
    side_offset = np.eye(4)
    side_offset[0, 3] = cfg["x_offset"]

    # 1) link visual meshes at zero pose
    for link_name in urdf.link_map:
        T_link_to_base = urdf.get_transform(link_name, "base_link")
        link = urdf.link_map[link_name]
        for v in link.visuals:
            if v.geometry.mesh is None:
                continue
            mesh_path = v.geometry.mesh.filename
            # urdf paths are relative to URDF dir; resolve.
            if mesh_path.startswith("./"):
                mesh_path = ASSET_ROOT / mesh_path[2:]
            else:
                mesh_path = Path(mesh_path)
            try:
                m = trimesh.load(mesh_path, force="mesh")
            except Exception:
                continue
            T_visual = np.eye(4)
            if v.origin is not None:
                T_visual = v.origin
            T = side_offset @ T_link_to_base @ T_visual
            m.apply_transform(T)
            m.visual.face_colors = cfg["mesh_color"]
            geoms.append(m)

    # 2) collision spheres in their parent link frames
    for link_name, sphere_list in spheres.items():
        T_link_to_base = urdf.get_transform(link_name, "base_link")
        for s in sphere_list:
            r = float(s["radius"])
            if r <= 0.0:
                continue
            c_link = np.array(s["center"], dtype=float)
            c_base = (T_link_to_base[:3, :3] @ c_link) + T_link_to_base[:3, 3]
            sph = trimesh.creation.icosphere(subdivisions=2, radius=r)
            T = np.eye(4)
            T[:3, 3] = c_base
            T = side_offset @ T
            sph.apply_transform(T)
            sph.visual.face_colors = cfg["sphere_color"]
            geoms.append(sph)

    return geoms


def main() -> None:
    all_geoms: list[trimesh.Trimesh] = []
    for side, cfg in HANDS.items():
        gs = collect_hand_geometry(cfg)
        print(f"{side}: {len(gs)} geometries")
        all_geoms.extend(gs)

    combined = trimesh.util.concatenate(all_geoms)
    obj_path = OUT_DIR / "inspire_collision.obj"
    combined.export(obj_path)
    print(f"\n=> {obj_path}")
    print("   Open in MeshLab/Blender. Right hand on left side (red spheres),")
    print("   left hand on right side (cyan spheres).")

    # Optional: render a few orthographic-ish views via pyrender (best-effort).
    try:
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        import pyrender  # noqa: F401
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.4, 0.4, 0.4])
        for g in all_geoms:
            scene.add(pyrender.Mesh.from_trimesh(g, smooth=False))

        bounds = combined.bounds
        center = bounds.mean(axis=0)
        size = float(np.linalg.norm(bounds[1] - bounds[0]))
        cam = pyrender.PerspectiveCamera(yfov=np.pi / 4)

        def look_at(eye, target, up):
            f = (target - eye); f /= np.linalg.norm(f)
            r = np.cross(f, up); r /= np.linalg.norm(r)
            u = np.cross(r, f)
            T = np.eye(4)
            T[:3, 0] = r; T[:3, 1] = u; T[:3, 2] = -f; T[:3, 3] = eye
            return T

        views = {
            "front": center + np.array([0, -size, 0]),
            "top":   center + np.array([0, 0, size]),
            "side":  center + np.array([size, 0, 0]),
        }
        ups = {"front": [0, 0, 1], "top": [0, 1, 0], "side": [0, 0, 1]}
        renderer = pyrender.OffscreenRenderer(viewport_width=1200, viewport_height=600)
        for name, eye in views.items():
            cam_node = scene.add(cam, pose=look_at(eye, center, np.array(ups[name], dtype=float)))
            light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0)
            light_node = scene.add(light, pose=look_at(eye, center, np.array(ups[name], dtype=float)))
            color, _ = renderer.render(scene)
            scene.remove_node(cam_node)
            scene.remove_node(light_node)
            from PIL import Image
            png = OUT_DIR / f"inspire_collision_{name}.png"
            Image.fromarray(color).save(png)
            print(f"=> {png}")
        renderer.delete()
    except Exception as e:
        print(f"\n[pyrender skipped] {type(e).__name__}: {e}")
        print("OBJ is the canonical output; the PNGs are optional.")


if __name__ == "__main__":
    main()
