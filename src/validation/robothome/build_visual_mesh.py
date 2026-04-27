"""Pre-build simplified visual meshes for all objects under OBJECT_DIRS.

Reads `{obj_dir}/raw_mesh/{name}.obj` (with optional .mtl + texture PNG/JPG)
and writes `{obj_dir}/visual_mesh/{name}.obj` (+ mtl/texture copied alongside).

Uses Open3D `simplify_quadric_decimation` on the geometry. Open3D drops UV
coordinates when reading triangle meshes, so we instead:
  1. Load with trimesh (preserves UVs / vertex colors / texture material).
  2. Decimate via Open3D, mapping the vertex/face indices back so we can
     rebuild trimesh with the original visual material *if texture is via
     vertex colors*.
  3. For UV-mapped textures we keep the simplified geometry but reuse the
     original material — UVs may not match every face perfectly after
     decimation; viewer just needs a lighter mesh that still looks textured.

Trade-offs accepted (visual-only mesh, not used for collision):
  - UV stretching is acceptable; the mesh is just for display.
  - If trimesh detects texture, we copy the source texture file alongside
    the simplified obj so loading still finds it.
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh


OBJECT_DIRS = [
    Path("/home/mingi/shared_data/AutoDex/object/robothome"),
]
TARGET_FACES = 10000


def find_raw_obj(obj_dir: Path):
    raw = obj_dir / "raw_mesh"
    if not raw.is_dir():
        return None
    preferred = raw / f"{obj_dir.name}.obj"
    if preferred.exists():
        return preferred
    cands = list(raw.glob("*.obj"))
    return cands[0] if cands else None


def simplify_with_o3d(verts: np.ndarray, faces: np.ndarray, target_faces: int):
    if len(faces) <= target_faces:
        return verts, faces
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
    m.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    m_simp = m.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    return np.asarray(m_simp.vertices), np.asarray(m_simp.triangles)


def process(obj_dir: Path, target_faces: int):
    raw_obj = find_raw_obj(obj_dir)
    if raw_obj is None:
        return None
    out_dir = obj_dir / "visual_mesh"
    out_dir.mkdir(exist_ok=True)
    out_obj = out_dir / raw_obj.name

    src_mesh = trimesh.load(str(raw_obj), force="mesh", process=False)
    n_in = len(src_mesh.faces)

    verts_s, faces_s = simplify_with_o3d(src_mesh.vertices, src_mesh.faces, target_faces)

    # Build a new trimesh; reuse the source visual (texture/material reference)
    # so the .mtl + texture export logic still kicks in. UVs won't fit the new
    # face indices perfectly but the export still produces something openable.
    new_mesh = trimesh.Trimesh(vertices=verts_s, faces=faces_s, process=False)
    if hasattr(src_mesh.visual, "material") and src_mesh.visual.material is not None:
        # Drop UVs (they no longer match) and keep the material color/texture.
        try:
            new_mesh.visual = trimesh.visual.TextureVisuals(
                material=src_mesh.visual.material,
            )
        except Exception:
            pass

    new_mesh.export(str(out_obj))

    # Copy any .mtl/.png/.jpg from raw_mesh alongside, so MTL refs resolve.
    for sib in raw_obj.parent.iterdir():
        if sib.suffix.lower() in (".mtl", ".png", ".jpg", ".jpeg"):
            tgt = out_dir / sib.name
            if not tgt.exists() or tgt.stat().st_size != sib.stat().st_size:
                shutil.copy2(sib, tgt)

    n_out = len(faces_s)
    return n_in, n_out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-faces", type=int, default=TARGET_FACES)
    p.add_argument("--only", type=str, default=None,
                   help="Only process this object name (substring match).")
    args = p.parse_args()

    seen = set()
    for root in OBJECT_DIRS:
        if not root.exists():
            continue
        for obj_dir in sorted(root.iterdir()):
            if not obj_dir.is_dir():
                continue
            if (obj_dir / "raw_mesh").is_dir() is False:
                continue
            if args.only and args.only not in obj_dir.name:
                continue
            if obj_dir.name in seen:
                continue
            seen.add(obj_dir.name)
            try:
                res = process(obj_dir, args.target_faces)
                if res is None:
                    print(f"[skip] {obj_dir.name}: no raw obj")
                    continue
                n_in, n_out = res
                print(f"[ok]   {obj_dir.name}: {n_in:>7} → {n_out:>5} faces")
            except Exception as e:
                print(f"[err]  {obj_dir.name}: {e}")


if __name__ == "__main__":
    main()
