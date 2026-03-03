#!/usr/bin/env python3
"""Generate processing status CSV for Google Sheets.

Scans the NAS (network FS) and checks which steps are done.
Columns: hand, name, index, segmentation, depth, pose_estimation

Cell values:
  X  = file missing (not processed yet)
  (empty) = file exists (processed)
  O  = manually verified by user (preserved across re-runs)

Safe to re-run: preserves manual 'O' marks, updates X->empty when
processing completes, and adds new rows for new indices.

Usage:
    python src/perception/status_sheet.py \
        --base /home/mingi/paradex1/capture/eccv2026/inspire_f1 \
        --serials 22684755 23263780

    # Multiple hands into one CSV:
    python src/perception/status_sheet.py \
        --base /home/mingi/paradex1/capture/eccv2026/inspire_f1 \
        --serials 22684755 23263780 \
        --base /home/mingi/paradex1/capture/eccv2026/allegro \
        --serials 22641005 22641023 \
        --output all_status.csv
"""

import os
import csv
import argparse
from pathlib import Path

HEADER = ["hand", "name", "index", "segmentation", "depth", "pose_estimation"]
STATUS_COLS = ["segmentation", "depth", "pose_estimation"]


def load_existing(csv_path):
    """Load existing CSV into a dict keyed by (hand, name, index)."""
    existing = {}
    if not os.path.exists(csv_path):
        return existing
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["hand"], row["name"], row["index"])
            existing[key] = row
    return existing


def scan_hand(base_dir, serials):
    """Scan one hand's NAS directory and return rows."""
    base = Path(base_dir)
    hand_name = base.name

    if not base.is_dir():
        print(f"Warning: not found: {base}")
        return []

    rows = []
    for obj_dir in sorted(base.iterdir()):
        if not obj_dir.is_dir():
            continue
        obj_name = obj_dir.name
        for idx_dir in sorted(obj_dir.iterdir(), key=lambda p: (len(p.name), p.name)):
            if not idx_dir.is_dir():
                continue
            idx_name = idx_dir.name

            has_video = any((idx_dir / "videos" / f"{s}.avi").exists() for s in serials)
            if not has_video:
                rows.append({"hand": hand_name, "name": obj_name, "index": idx_name,
                             "segmentation": "X", "depth": "X", "pose_estimation": "X"})
                continue

            has_seg = any((idx_dir / "obj_mask" / f"{s}.avi").exists() for s in serials)
            has_depth = any((idx_dir / "depth" / f"{s}.avi").exists() for s in serials)
            has_pose = any((idx_dir / "pose" / f"{s}.npy").exists() for s in serials)

            rows.append({
                "hand": hand_name,
                "name": obj_name,
                "index": idx_name,
                "segmentation": "" if has_seg else "X",
                "depth": "" if has_depth else "X",
                "pose_estimation": "" if has_pose else "X",
            })

    return rows


def merge(existing, scanned_rows):
    """Merge scanned rows with existing CSV data.

    Rules per status column:
      - Existing 'O' -> keep 'O' (manual verification preserved)
      - Existing 'X', scanned '' -> update to '' (processing completed)
      - Otherwise -> use scanned value
    New rows are appended.
    """
    merged = {}

    # Start with existing rows (preserves order for known entries)
    for key, row in existing.items():
        merged[key] = dict(row)

    n_new = 0
    n_updated = 0
    for row in scanned_rows:
        key = (row["hand"], row["name"], row["index"])
        if key in merged:
            old = merged[key]
            changed = False
            for col in STATUS_COLS:
                if old[col] == "O":
                    # Preserve manual verification
                    pass
                elif old[col] != row[col]:
                    old[col] = row[col]
                    changed = True
            if changed:
                n_updated += 1
        else:
            merged[key] = dict(row)
            n_new += 1

    return merged, n_new, n_updated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, action="append",
                        help="Base dir (can repeat for multiple hands)")
    parser.add_argument("--serials", required=True, nargs="+", action="append",
                        help="Serial numbers for the preceding --base (can repeat)")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: {hand_name}_status.csv)")
    args = parser.parse_args()

    if len(args.base) != len(args.serials):
        parser.error("Each --base must have a matching --serials")

    # Determine output path
    if args.output:
        out_path = args.output
    elif len(args.base) == 1:
        out_path = f"{Path(args.base[0]).name}_status.csv"
    else:
        out_path = "status.csv"

    # Load existing CSV if present
    existing = load_existing(out_path)

    # Scan NAS
    all_scanned = []
    for base, serials in zip(args.base, args.serials):
        rows = scan_hand(base, serials)
        all_scanned.extend(rows)
        print(f"  {Path(base).name}: {len(rows)} entries on NAS")

    # Merge
    merged, n_new, n_updated = merge(existing, all_scanned)

    # Write — sort by hand, name, index
    sorted_rows = sorted(merged.values(), key=lambda r: (r["hand"], r["name"], len(r["index"]), r["index"]))

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(sorted_rows)

    print(f"Saved {len(sorted_rows)} rows to {out_path} ({n_new} new, {n_updated} updated)")


if __name__ == "__main__":
    main()
