"""Compute T_orig_to_autodex for each object in robothome zips.

For each object dir under shared_data/AutoDex/object/robothome/{name}, find
the matching shared_data/robothome/{name}_obj.zip, unzip the raw mesh, and
fit a similarity transform (scale*R + t) so that

    autodex_verts ≈ T_orig_to_autodex @ orig_verts

Saves to {obj_dir}/orig_to_autodex.npy (4x4).

Run with --only paperCup,Jp_Water for the JSON-relevant subset.
"""
import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import trimesh


AUTODEX_ROOT = Path("/home/mingi/shared_data/AutoDex/object/robothome")
ROBOTHOME_ZIP_DIR = Path("/home/mingi/shared_data/robothome")


def fit_similarity(src: np.ndarray, dst: np.ndarray):
    """Fit a *rigid* transform after pre-scaling src to metres.

    Perception loads the raw zip mesh with a mm→m unit conversion, so the
    pose it produces is in the rescaled-to-m frame. The AutoDex mesh is also
    in metres but with an extra (R, t) normalization applied. The relevant
    correction is therefore a pure rigid transform between the two metre-frames.
    """
    src_m = src * 0.001  # zip is mm; perception sees metres
    src_c = src_m.mean(axis=0)
    dst_c = dst.mean(axis=0)
    src0 = src_m - src_c
    dst0 = dst - dst_c
    M = src0.T @ dst0
    U, S, Vt = np.linalg.svd(M)
    D = np.eye(3)
    if np.linalg.det(Vt.T @ U.T) < 0:
        D[2, 2] = -1
    R = Vt.T @ D @ U.T
    t = dst_c - R @ src_c
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    pred = (R @ src_m.T).T + t
    rms = np.sqrt(((pred - dst) ** 2).sum(axis=1).mean())
    return T, rms


def find_zip(obj_name: str) -> Path | None:
    # Try canonical name first, then case-insensitive search.
    p = ROBOTHOME_ZIP_DIR / f"{obj_name}_obj.zip"
    if p.exists():
        return p
    target = f"{obj_name}_obj.zip".lower()
    for c in ROBOTHOME_ZIP_DIR.glob("*_obj.zip"):
        if c.name.lower() == target:
            return c
    return None


def first_obj_in(folder: Path) -> Path | None:
    cands = sorted(folder.rglob("*.obj"))
    return cands[0] if cands else None


def process(obj_name: str):
    obj_dir = AUTODEX_ROOT / obj_name
    if not obj_dir.is_dir():
        print(f"[skip] {obj_name}: no AutoDex dir")
        return
    autodex_obj = obj_dir / "raw_mesh" / f"{obj_name}.obj"
    if not autodex_obj.exists():
        # fallback: any *.obj
        cands = sorted((obj_dir / "raw_mesh").glob("*.obj"))
        if not cands:
            print(f"[skip] {obj_name}: no AutoDex raw_mesh obj")
            return
        autodex_obj = cands[0]

    zip_path = find_zip(obj_name)
    if zip_path is None:
        print(f"[skip] {obj_name}: no zip in {ROBOTHOME_ZIP_DIR}")
        return

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tdp)
        orig_obj = first_obj_in(tdp)
        if orig_obj is None:
            print(f"[skip] {obj_name}: no obj in zip")
            return

        a = trimesh.load(orig_obj, force="mesh", process=False)
        b = trimesh.load(autodex_obj, force="mesh", process=False)
        if len(a.vertices) != len(b.vertices):
            print(f"[skip] {obj_name}: vertex count mismatch "
                  f"(orig={len(a.vertices)}, autodex={len(b.vertices)})")
            return
        T, rms = fit_similarity(np.asarray(a.vertices), np.asarray(b.vertices))

    out = obj_dir / "orig_to_autodex.npy"
    np.save(out, T)
    print(f"[ok] {obj_name}: rms={rms:.2e} t={T[:3,3]} → {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated object names (default: all under AutoDex root).")
    args = p.parse_args()

    if args.only:
        names = [n.strip() for n in args.only.split(",") if n.strip()]
    else:
        names = sorted([d.name for d in AUTODEX_ROOT.iterdir() if d.is_dir()])

    for n in names:
        try:
            process(n)
        except Exception as e:
            print(f"[err] {n}: {e}")


if __name__ == "__main__":
    main()
