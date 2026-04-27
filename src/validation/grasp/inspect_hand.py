"""Interactive viewer for a hand URDF — slider per actuated joint.

Optionally overlays collision spheres and highlights self-collisions
between sphere pairs that are NOT in self_collision_ignore (from a
cuRobo robot YAML). Used to calibrate self_collision_ignore.

Usage:
    python src/validation/grasp/inspect_hand.py \
        --urdf src/grasp_generation/BODex/src/curobo/content/assets/robot/inspire_f1_description/inspire_f1_hand_right.urdf \
        --robot_yml src/grasp_generation/BODex/src/curobo/content/configs/robot/inspire_f1.yml
"""
import argparse
import os
import time

import numpy as np
import trimesh
import viser
import yaml
import yourdfpy
from scipy.spatial.transform import Rotation as Rsc


def _R_to_wxyz(R):
    q = Rsc.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def _mesh_filename_to_path(urdf_path, mesh_fn):
    if mesh_fn.startswith("package://"):
        mesh_fn = mesh_fn.split("/", 3)[-1]
    if os.path.isabs(mesh_fn):
        return mesh_fn
    return os.path.normpath(os.path.join(os.path.dirname(urdf_path), mesh_fn))


def _load_link_meshes(urdf_path, robot):
    out = {}
    for link_name, link in robot.link_map.items():
        if not link.visuals:
            continue
        items = []
        for visual in link.visuals:
            if visual.geometry.mesh is None:
                continue
            mesh_path = _mesh_filename_to_path(urdf_path, visual.geometry.mesh.filename)
            if not os.path.exists(mesh_path):
                print(f"  WARN: mesh not found: {mesh_path}")
                continue
            mesh = trimesh.load(mesh_path, force="mesh", process=False)
            T = visual.origin if visual.origin is not None else np.eye(4)
            items.append((T, np.asarray(mesh.vertices), np.asarray(mesh.faces)))
        if items:
            out[link_name] = items
    return out


def _load_robot_cfg(robot_yml_path):
    """Return (collision_spheres_path, ignore_pairs_set, collision_links_set, sphere_buffer)."""
    with open(robot_yml_path) as f:
        cfg = yaml.safe_load(f)
    kin = cfg["robot_cfg"]["kinematics"]
    sphere_rel = kin["collision_spheres"]
    sphere_path = os.path.normpath(
        os.path.join(os.path.dirname(robot_yml_path), sphere_rel)
    )
    ignore = kin.get("self_collision_ignore", {}) or {}
    pairs = set()
    for a, lst in ignore.items():
        for b in lst or []:
            pairs.add(frozenset([a, b]))
    links = set(kin.get("collision_link_names", []) or [])
    buffer = kin.get("collision_sphere_buffer", 0.0) or 0.0
    return sphere_path, pairs, links, buffer


def _load_sphere_yaml(sphere_yml_path):
    """Return {link_name: [(center3, radius), ...]} from a cuRobo spheres yml."""
    with open(sphere_yml_path) as f:
        data = yaml.safe_load(f)
    spheres = data.get("collision_spheres", data)
    out = {}
    for link, items in (spheres or {}).items():
        if not items:
            continue
        out[link] = [(np.asarray(it["center"], dtype=np.float64), float(it["radius"])) for it in items]
    return out


def _icosphere(radius, subdivisions=1):
    return trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)


def _hsv_to_rgb(h, s, v):
    import colorsys
    return colorsys.hsv_to_rgb(h, s, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--robot_yml", default=None,
                    help="cuRobo robot yml — enables sphere overlay + self-collision check")
    ap.add_argument("--contact", default="",
                    help="Comma-separated 'link/idx' tokens to highlight as contact points "
                         "(e.g. thumb_force_sensor/0,index_force_sensor/0,...)")
    ap.add_argument("--rainbow_links", default="",
                    help="Comma-separated link names whose spheres should be colored by index "
                         "(rainbow), with each sphere labeled. Useful for picking contact idx.")
    ap.add_argument("--show_axes_for", default="",
                    help="Comma-separated link names to draw coordinate axes (XYZ frame).")
    ap.add_argument("--port", type=int, default=8081)
    args = ap.parse_args()

    robot = yourdfpy.URDF.load(args.urdf)
    actuated = list(robot.actuated_joint_names)
    print(f"actuated joints ({len(actuated)}):")
    for n in actuated:
        j = robot.joint_map[n]
        lo = j.limit.lower if j.limit else 0.0
        hi = j.limit.upper if j.limit else 0.0
        print(f"  {n}: limit=[{lo:.4f}, {hi:.4f}]")

    link_meshes = _load_link_meshes(args.urdf, robot)
    base_link = robot.base_link

    contact_set = set()
    for tok in (args.contact or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        link, idx = tok.split("/")
        contact_set.add((link, int(idx)))

    rainbow_links = set()
    for tok in (args.rainbow_links or "").split(","):
        tok = tok.strip()
        if tok:
            rainbow_links.add(tok)

    axis_links = set()
    for tok in (args.show_axes_for or "").split(","):
        tok = tok.strip()
        if tok:
            axis_links.add(tok)
    axis_handles = {}

    spheres = {}
    ignore_pairs = set()
    coll_links = set()
    sphere_buffer = 0.0
    retract_cfg = None
    if args.robot_yml:
        sphere_path, ignore_pairs, coll_links, sphere_buffer = _load_robot_cfg(args.robot_yml)
        spheres = _load_sphere_yaml(sphere_path)
        with open(args.robot_yml) as _f:
            _full = yaml.safe_load(_f)
        cspace = _full["robot_cfg"]["kinematics"].get("cspace") or {}
        joint_names = cspace.get("joint_names") or []
        retract_list = cspace.get("retract_config") or []
        if joint_names and retract_list and len(joint_names) == len(retract_list):
            retract_cfg = {n: float(v) for n, v in zip(joint_names, retract_list)}
        print(f"loaded {sum(len(v) for v in spheres.values())} spheres "
              f"on {len(spheres)} links from {sphere_path}")
        print(f"self_collision_ignore pairs: {len(ignore_pairs)}, "
              f"collision_link_names: {len(coll_links)}, buffer: {sphere_buffer}")
        if retract_cfg:
            print(f"using retract_config from yml: {retract_cfg}")

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/grid", width=0.5, height=0.5, cell_size=0.05)
    server.scene.add_frame("/world", show_axes=True, axes_length=0.03, axes_radius=0.001)

    sliders = {}
    with server.gui.add_folder("Joint sliders"):
        for n in actuated:
            j = robot.joint_map[n]
            lo = j.limit.lower if j.limit else -3.14
            hi = j.limit.upper if j.limit else 3.14
            init = float(retract_cfg[n]) if retract_cfg and n in retract_cfg else float(lo)
            init = max(float(lo), min(float(hi), init))
            sliders[n] = server.gui.add_slider(
                n, min=float(lo), max=float(hi), step=0.005, initial_value=init
            )

    show_mesh = server.gui.add_checkbox("Show mesh", initial_value=True)
    show_spheres = server.gui.add_checkbox("Show spheres", initial_value=bool(spheres))
    mesh_opacity = server.gui.add_slider("Mesh opacity", min=0.0, max=1.0, step=0.05, initial_value=0.4)
    print_btn = server.gui.add_button("Print joints + collisions")

    mesh_handles = {}
    sphere_handles = {}

    def _world_sphere_centers(cfg):
        robot.update_cfg(cfg)
        out = {}
        for link, items in spheres.items():
            T = robot.get_transform(frame_to=link, frame_from=base_link)
            R, t = T[:3, :3], T[:3, 3]
            cs = []
            for c, r in items:
                cs.append((R @ c + t, r + sphere_buffer, link))
            out[link] = cs
        return out

    def _find_collisions(world_spheres):
        flat = []
        for link in world_spheres:
            if coll_links and link not in coll_links:
                continue
            for i, (c, r, _) in enumerate(world_spheres[link]):
                flat.append((link, i, c, r))
        hits = set()
        for ai in range(len(flat)):
            la, ia, ca, ra = flat[ai]
            for bi in range(ai + 1, len(flat)):
                lb, ib, cb, rb = flat[bi]
                if la == lb:
                    continue
                if frozenset([la, lb]) in ignore_pairs:
                    continue
                d = np.linalg.norm(ca - cb)
                if d < ra + rb:
                    hits.add((la, ia))
                    hits.add((lb, ib))
        return hits

    def update():
        cfg = np.array([float(sliders[n].value) for n in actuated], dtype=np.float64)

        for axlink in axis_links:
            try:
                T = robot.get_transform(frame_to=axlink, frame_from=base_link)
            except Exception:
                continue
            key = f"/axes/{axlink}"
            if key in axis_handles:
                axis_handles[key].position = T[:3, 3]
                axis_handles[key].wxyz = _R_to_wxyz(T[:3, :3])
            else:
                axis_handles[key] = server.scene.add_frame(
                    key, position=T[:3, 3], wxyz=_R_to_wxyz(T[:3, :3]),
                    show_axes=True, axes_length=0.025, axes_radius=0.0015,
                )

        for link_name, items in link_meshes.items():
            T_link = robot.get_transform(frame_to=link_name, frame_from=base_link)
            for i, (T_visual, verts, faces) in enumerate(items):
                T = T_link @ T_visual
                key = f"/robot/{link_name}/{i}"
                if key in mesh_handles:
                    h = mesh_handles[key]
                    h.position = T[:3, 3]
                    h.wxyz = _R_to_wxyz(T[:3, :3])
                    h.visible = show_mesh.value
                    h.opacity = float(mesh_opacity.value)
                else:
                    mesh_handles[key] = server.scene.add_mesh_simple(
                        key, verts, faces,
                        position=T[:3, 3], wxyz=_R_to_wxyz(T[:3, :3]),
                        color=(0.7, 0.7, 0.85), opacity=float(mesh_opacity.value),
                        visible=show_mesh.value,
                    )

        if not spheres:
            return
        world_spheres = _world_sphere_centers(cfg)
        hits = _find_collisions(world_spheres)
        for link, items in world_spheres.items():
            n_link = len(items)
            for i, (c, r, _) in enumerate(items):
                key = f"/sphere/{link}/{i}"
                colliding = (link, i) in hits
                is_contact = (link, i) in contact_set
                in_rainbow = link in rainbow_links
                if is_contact:
                    color = (0.1, 0.4, 1.0)
                    opacity = 0.95
                elif in_rainbow:
                    # hue ramp from red(0) to violet(N-1)
                    h = (i / max(1, n_link - 1)) * 0.85
                    color = _hsv_to_rgb(h, 0.85, 1.0)
                    opacity = 0.95
                elif colliding:
                    color = (1.0, 0.1, 0.1)
                    opacity = 0.85
                else:
                    color = (0.6, 0.6, 0.6)
                    opacity = 0.25
                # Label sphere index for rainbow links
                if in_rainbow:
                    label_key = f"/label/{link}/{i}"
                    if label_key in sphere_handles:
                        sphere_handles[label_key].position = c + np.array([0, 0, r * 1.5])
                    else:
                        sphere_handles[label_key] = server.scene.add_label(
                            label_key, text=str(i),
                            position=c + np.array([0, 0, r * 1.5]),
                        )

                ico = _icosphere(r)
                if key in sphere_handles:
                    h = sphere_handles[key]
                    h.position = c
                    h.color = color
                    h.opacity = opacity
                    h.visible = show_spheres.value
                else:
                    sphere_handles[key] = server.scene.add_mesh_simple(
                        key, np.asarray(ico.vertices) + (c - c),  # vertices already centered
                        np.asarray(ico.faces),
                        position=c, wxyz=np.array([1.0, 0, 0, 0]),
                        color=color, opacity=opacity,
                        visible=show_spheres.value,
                    )

    @print_btn.on_click
    def _(_evt):
        cfg = np.array([float(sliders[n].value) for n in actuated], dtype=np.float64)
        print("\ncurrent joints:")
        for n in actuated:
            print(f"  {n}: {float(sliders[n].value):.4f}")
        cli = " ".join(f"{n}={float(sliders[n].value):.4f}" for n in actuated)
        print(f"\nCLI form: --joints {cli}")

        if not spheres:
            return
        world_spheres = _world_sphere_centers(cfg)
        # report violating link pairs (not just sphere indices)
        link_pairs = {}
        flat = []
        for link in world_spheres:
            if coll_links and link not in coll_links:
                continue
            for i, (c, r, _) in enumerate(world_spheres[link]):
                flat.append((link, i, c, r))
        for ai in range(len(flat)):
            la, ia, ca, ra = flat[ai]
            for bi in range(ai + 1, len(flat)):
                lb, ib, cb, rb = flat[bi]
                if la == lb or frozenset([la, lb]) in ignore_pairs:
                    continue
                d = np.linalg.norm(ca - cb)
                if d < ra + rb:
                    key = tuple(sorted([la, lb]))
                    link_pairs.setdefault(key, []).append((ia, ib, ra + rb - d))
        if not link_pairs:
            print("\nNo self-collisions detected.")
        else:
            print(f"\nSelf-collision pairs ({len(link_pairs)}):")
            for (la, lb), hits in sorted(link_pairs.items()):
                worst = max(h[2] for h in hits)
                print(f"  {la} <-> {lb}  ({len(hits)} sphere pairs, worst overlap {worst*1000:.2f}mm)")

    for n in actuated:
        sliders[n].on_update(lambda _e: update())
    show_mesh.on_update(lambda _e: update())
    show_spheres.on_update(lambda _e: update())
    mesh_opacity.on_update(lambda _e: update())

    update()

    print(f"\nViser running at http://localhost:{args.port}")
    print("Ctrl+C to exit.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
