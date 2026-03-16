#!/usr/bin/env python3
"""Visualize stereo pair selection on a viser 3D viewer.

Shows all cameras as frustums, colored by focal group (wide/mid/tele),
with selected adjacent pairs drawn as lines between camera centers.

Usage:
    python src/validation/perception/viz_stereo_pairs.py \
        --capture_dir /home/mingi/shared_data/RSS2026_Mingi/experiment/selected_100/attached_container/20260121_163413
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

AUTODEX_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(AUTODEX_ROOT))

from autodex.perception.depth import _to_4x4, load_cam_param

# Colors per focal group (RGB 0-1)
GROUP_COLORS = {
    "wide": (0.2, 0.8, 0.2),   # green
    "mid":  (0.2, 0.5, 1.0),   # blue
    "tele": (1.0, 0.3, 0.3),   # red
}
PAIR_COLOR = (1.0, 1.0, 0.0)  # yellow for pair lines


def focal_group(K):
    f = float(K[0, 0])
    if f < 2500:
        return "wide"
    elif f < 4000:
        return "mid"
    return "tele"


def build_adjacent_pairs(serials, intrinsics, extrinsics, C2R_inv, max_angle_gap=40.0):
    """Build adjacent pairs: same focal group, sorted by angle, skip large gaps.

    Only pairs cameras that are horizontally adjacent (by rig angle).
    No z-level grouping — just angle-sorted within each focal group,
    skipping pairs where the angle gap exceeds max_angle_gap.
    """
    rig_data = {}
    for s in serials:
        T = _to_4x4(extrinsics[s])
        pos_world = -T[:3, :3].T @ T[:3, 3]
        pos_rig = C2R_inv[:3, :3] @ pos_world + C2R_inv[:3, 3]
        angle = float(np.degrees(np.arctan2(pos_rig[1], pos_rig[0])))
        z = float(pos_rig[2])
        z_level = "low" if z < 0.7 else ("mid" if z < 1.1 else "high")
        rig_data[s] = {"angle": angle, "z_level": z_level}

    # Group by (focal group, z-level), sort by angle
    buckets = defaultdict(list)
    for s in serials:
        key = (focal_group(intrinsics[s]), rig_data[s]["z_level"])
        buckets[key].append(s)
    for key in buckets:
        buckets[key].sort(key=lambda s: rig_data[s]["angle"])

    pairs = []
    for (grp, zlev), ss in sorted(buckets.items()):
        # Adjacent only (no wrap-around), skip large angle gaps
        for i in range(len(ss) - 1):
            s1, s2 = ss[i], ss[i + 1]
            gap = abs(rig_data[s1]["angle"] - rig_data[s2]["angle"])
            if gap <= max_angle_gap:
                pairs.append((s1, s2))
    return pairs, rig_data


def main():
    parser = argparse.ArgumentParser(description="Visualize stereo pairs in viser")
    parser.add_argument("--capture_dir", type=str, required=True)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    capture_dir = Path(args.capture_dir)
    intrinsics, extrinsics = load_cam_param(capture_dir)
    serials = sorted(intrinsics.keys())

    c2r_path = capture_dir / "C2R.npy"
    C2R_inv = np.linalg.inv(np.load(str(c2r_path))) if c2r_path.exists() else np.eye(4)

    pairs, rig_data = build_adjacent_pairs(serials, intrinsics, extrinsics, C2R_inv)

    # Transform extrinsics to robot frame, then compute cam-to-robot poses
    # extrinsics[s] = T_w2c (world→cam)
    # cam→robot = C2R_inv @ cam→world = C2R_inv @ inv(T_w2c)
    cam_positions = {}  # in robot frame
    cam_c2robot = {}    # cam-to-robot 4x4
    for s in serials:
        T_w2c = _to_4x4(extrinsics[s])
        c2w = np.linalg.inv(T_w2c)
        c2robot = C2R_inv @ c2w
        cam_c2robot[s] = c2robot
        cam_positions[s] = c2robot[:3, 3]

    # Set of paired serials for highlighting
    paired_serials = set()
    for s1, s2 in pairs:
        paired_serials.add(s1)
        paired_serials.add(s2)

    # Launch viser
    from paradex.visualization.visualizer.viser import ViserViewer
    viewer = ViserViewer(port_number=args.port)

    # Origin marker
    viewer.server.scene.add_frame(
        "/origin",
        position=(0, 0, 0),
        show_axes=True,
        axes_length=0.1,
        axes_radius=0.005,
    )
    viewer.server.scene.add_icosphere(
        "/origin/sphere",
        radius=0.015,
        color=(1.0, 1.0, 1.0),
        position=(0, 0, 0),
    )

    # Add cameras
    for s in serials:
        grp = focal_group(intrinsics[s])
        color = GROUP_COLORS[grp]
        # Dim unpaired cameras
        if s not in paired_serials:
            color = tuple(c * 0.3 for c in color)

        # extrinsic: paradex expects cam-to-world (here cam-to-robot)
        viewer.add_camera(
            name=s,
            extrinsic=cam_c2robot[s],
            intrinsic=intrinsics[s],
            color=color,
            size=0.05,
        )

        # Add serial label
        viewer.server.scene.add_label(
            f"/cameras/{s}_frame/label",
            text=f"{s} [{grp}]",
            position=(0, 0, -0.01),
            font_screen_scale=0.4,
        )

    # Draw pair connections
    for i, (s1, s2) in enumerate(pairs):
        p1 = cam_positions[s1]
        p2 = cam_positions[s2]
        bl = float(np.linalg.norm(p1 - p2))

        viewer.server.scene.add_spline_catmull_rom(
            f"/pairs/pair_{i}_{s1}_{s2}",
            positions=np.array([p1, p2]),
            color=PAIR_COLOR,
            line_width=3.0,
        )
        # Midpoint label with baseline
        mid = (p1 + p2) / 2
        viewer.server.scene.add_label(
            f"/pairs/pair_{i}_{s1}_{s2}_label",
            text=f"{bl:.2f}m",
            position=mid,
            font_screen_scale=0.3,
        )

    print(f"\n{len(serials)} cameras, {len(pairs)} pairs")
    print(f"Viser running at http://localhost:{args.port}")
    print("Press Ctrl+C to stop")

    # Keep alive
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nDone")


if __name__ == "__main__":
    main()