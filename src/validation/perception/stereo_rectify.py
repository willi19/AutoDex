#!/usr/bin/env python3
"""Dump stereo rectification for ALL camera pairs in a capture.

Saves one image per ordered pair: original on top, rectified with epipolar
lines on bottom.  Pairs without geometric overlap get only the original row.
Also writes a CSV with all pair stats for threshold tuning.

Output layout:
    {output_dir}/
    ├── {s1}_{s2}.jpg          # 24×23 = 552 images
    └── stats.csv              # one row per pair

Usage:
    python src/validation/perception/stereo_rectify.py \
        --capture_dir /home/mingi/shared_data/RSS2026_Mingi/experiment/selected_100/attached_container/20260121_163413
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

AUTODEX_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(AUTODEX_ROOT))

from autodex.perception.depth import (
    _to_4x4,
    load_cam_param,
)


def compute_pair_stats(K1, K2, T1, T2, image_size):
    """Compute pair quality metrics and check rectified overlap."""
    T1_4, T2_4 = _to_4x4(T1), _to_4x4(T2)
    W, H = image_size

    # Camera positions & forward directions
    pos1 = -T1_4[:3, :3].T @ T1_4[:3, 3]
    pos2 = -T2_4[:3, :3].T @ T2_4[:3, 3]
    fwd1 = T1_4[2, :3] / (np.linalg.norm(T1_4[2, :3]) + 1e-9)
    fwd2 = T2_4[2, :3] / (np.linalg.norm(T2_4[2, :3]) + 1e-9)

    baseline = float(np.linalg.norm(pos1 - pos2))
    cos_sim = float(np.dot(fwd1, fwd2))

    baseline_dir = (pos2 - pos1) / (np.linalg.norm(pos2 - pos1) + 1e-9)
    mean_fwd = (fwd1 + fwd2)
    mean_fwd /= np.linalg.norm(mean_fwd) + 1e-9
    perp = 1.0 - abs(float(np.dot(baseline_dir, mean_fwd)))

    f1, f2 = float(K1[0, 0]), float(K2[0, 0])
    f_ratio = max(f1, f2) / (min(f1, f2) + 1e-9)

    # stereoRectify (alpha=0 for reliable metrics)
    T_rel = T2_4 @ np.linalg.inv(T1_4)
    R1, R2, P1_cv, _, _, _, _ = cv2.stereoRectify(
        K1, None, K2, None, (W, H),
        T_rel[:3, :3], T_rel[:3, 3],
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )
    f_rect_cv = float(P1_cv[0, 0])
    orig_f = max(f1, f2)
    rect_f_ratio = f_rect_cv / orig_f if f_rect_cv > 0 else -1.0

    def rot_angle(R):
        return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))

    rot1, rot2 = rot_angle(R1), rot_angle(R2)

    # Check overlap: project corners through R1/R2 with original focal length
    f_probe = max(f1, f2)
    P_probe = np.array([[f_probe, 0, 0], [0, f_probe, 0], [0, 0, 1]], dtype=np.float64)
    corners = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float64).reshape(-1, 1, 2)
    cl = cv2.undistortPoints(corners, K1, None, R=R1, P=P_probe)
    cr = cv2.undistortPoints(corners, K2, None, R=R2, P=P_probe)
    x_min = max(float(cl[:, 0, 0].min()), float(cr[:, 0, 0].min()))
    x_max = min(float(cl[:, 0, 0].max()), float(cr[:, 0, 0].max()))
    y_min = max(float(cl[:, 0, 1].min()), float(cr[:, 0, 1].min()))
    y_max = min(float(cl[:, 0, 1].max()), float(cr[:, 0, 1].max()))
    has_overlap = x_max > x_min and y_max > y_min

    return {
        "baseline": baseline,
        "cos_sim": cos_sim,
        "perp": perp,
        "f_ratio": f_ratio,
        "rect_f_ratio": rect_f_ratio,
        "rot1_deg": rot1,
        "rot2_deg": rot2,
        "has_overlap": has_overlap,
    }


def read_frame0(capture_dir, serial):
    """Read frame 0 from video or image."""
    video_path = capture_dir / "videos" / f"{serial}.avi"
    if video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        cap.release()
        if ret:
            return frame
    img_path = capture_dir / "images" / f"{serial}.png"
    if img_path.exists():
        return cv2.imread(str(img_path))
    return None


def main():
    parser = argparse.ArgumentParser(description="Dump all stereo pair rectifications")
    parser.add_argument("--capture_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir (default: {capture_dir}/stereo_rectify_debug)")
    args = parser.parse_args()

    capture_dir = Path(args.capture_dir)
    default_out = AUTODEX_ROOT / "output" / "stereo_rectify_debug"
    output_dir = Path(args.output_dir) if args.output_dir else default_out
    output_dir.mkdir(parents=True, exist_ok=True)

    intrinsics, extrinsics = load_cam_param(capture_dir)
    serials = sorted(intrinsics.keys())

    n_cams = len(serials)
    n_pairs = n_cams * (n_cams - 1)
    print(f"{n_cams} cameras, {n_pairs} pairs, output -> {output_dir}")

    # Load all frame-0 images
    images = {}
    for s in serials:
        img = read_frame0(capture_dir, s)
        if img is not None:
            images[s] = img
    print(f"Loaded {len(images)}/{n_cams} images")

    # Use actual image dimensions (NOT intrinsics cx*2 which can be wrong)
    sample_img = next(iter(images.values()))
    H_img, W_img = sample_img.shape[:2]
    image_size = (W_img, H_img)
    print(f"Image size: {W_img}x{H_img}")

    # Group cameras by focal length
    def focal_group(serial):
        f = float(intrinsics[serial][0, 0])
        if f < 2500:
            return "wide"
        elif f < 4000:
            return "mid"
        return "tele"

    # Compute rig-frame positions for adjacency ordering
    c2r_path = capture_dir / "C2R.npy"
    if c2r_path.exists():
        C2R_inv = np.linalg.inv(np.load(str(c2r_path)))
    else:
        C2R_inv = np.eye(4)

    rig_data = {}
    for s in serials:
        T = _to_4x4(extrinsics[s])
        pos_world = -T[:3, :3].T @ T[:3, 3]
        pos_rig = C2R_inv[:3, :3] @ pos_world + C2R_inv[:3, 3]
        angle = float(np.degrees(np.arctan2(pos_rig[1], pos_rig[0])))
        z = float(pos_rig[2])
        z_level = "low" if z < 0.7 else ("mid" if z < 1.1 else "high")
        rig_data[s] = {"angle": angle, "z_level": z_level}

    # Group by (focal group, z-level), sort by angle, pair adjacent only
    from collections import defaultdict
    MAX_ANGLE_GAP = 40.0
    buckets = defaultdict(list)
    for s in serials:
        buckets[(focal_group(s), rig_data[s]["z_level"])].append(s)
    for key in buckets:
        buckets[key].sort(key=lambda s: rig_data[s]["angle"])

    pairs = []
    for (grp, zlev), ss in sorted(buckets.items()):
        print(f"  {grp}/{zlev}: {len(ss)} cameras — {ss}")
        for i in range(len(ss) - 1):
            s1, s2 = ss[i], ss[i + 1]
            gap = abs(rig_data[s1]["angle"] - rig_data[s2]["angle"])
            if gap <= MAX_ANGLE_GAP:
                pairs.append((s1, s2))

    n_pairs = len(pairs)
    print(f"{n_pairs} adjacent pairs to process")

    # Target width for saved images (both original and rectified rows)
    TARGET_W = 1600

    csv_rows = []
    done = 0
    for s1, s2 in pairs:
        if s1 not in images or s2 not in images:
            done += 1
            continue

        stats = compute_pair_stats(
            intrinsics[s1], intrinsics[s2],
            extrinsics[s1], extrinsics[s2], image_size,
        )
        angle_deg = np.degrees(np.arccos(np.clip(stats["cos_sim"], -1, 1)))
        csv_rows.append({
            "cam1": s1, "cam2": s2,
            "group": focal_group(s1),
            "baseline": f"{stats['baseline']:.4f}",
            "angle_deg": f"{angle_deg:.1f}",
            "cos_sim": f"{stats['cos_sim']:.3f}",
            "perp": f"{stats['perp']:.3f}",
            "f_ratio": f"{stats['f_ratio']:.2f}",
            "rect_f_ratio": f"{stats['rect_f_ratio']:.2f}",
            "rot1": f"{stats['rot1_deg']:.1f}",
            "rot2": f"{stats['rot2_deg']:.1f}",
            "has_overlap": stats["has_overlap"],
        })

        img1, img2 = images[s1], images[s2]
        K1, K2 = intrinsics[s1], intrinsics[s2]
        T1, T2 = extrinsics[s1], extrinsics[s2]

        # --- Original pair (top row) ---
        orig_pair = np.hstack([img1, img2])
        scale_orig = min(1.0, TARGET_W / orig_pair.shape[1])
        if scale_orig < 1.0:
            orig_pair = cv2.resize(orig_pair, None, fx=scale_orig, fy=scale_orig)
        info = (f"ORIGINAL  {s1} - {s2} [{focal_group(s1)}]  "
                f"bl={stats['baseline']:.3f}m  angle={angle_deg:.1f}deg  "
                f"perp={stats['perp']:.2f}")
        cv2.putText(orig_pair, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if not stats["has_overlap"]:
            cv2.putText(orig_pair, "NO OVERLAP — cannot rectify",
                        (10, orig_pair.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(str(output_dir / f"{s1}_{s2}.jpg"), orig_pair)
            done += 1
            if done % 20 == 0:
                print(f"  [{done}/{n_pairs}]")
            continue

        # --- Rectified pair (bottom row) ---
        # Determine left/right from rig y-coordinate:
        # y > 0 = left side, y < 0 = right side (from origin perspective)
        # So camera with LARGER y (more left in rig frame) is LEFT in stereo
        y1 = rig_data[s1]["angle"]  # angle ∝ y position
        y2 = rig_data[s2]["angle"]
        if y1 < y2:
            # s1 has smaller angle → s1=left in stereo (right side of rig when looking inward)
            l_img, r_img = img1, img2
            l_K, r_K = K1, K2
            l_T, r_T = T1, T2
            swapped = False
        else:
            l_img, r_img = img2, img1
            l_K, r_K = K2, K1
            l_T, r_T = T2, T1
            swapped = True

        W_in, H_in = image_size
        T_lr = _to_4x4(r_T) @ np.linalg.inv(_to_4x4(l_T))

        # Use alpha=0 to get valid R1, R2 (rotation matrices are alpha-independent,
        # but alpha=1 can produce negative focal lengths in P matrices).
        # We'll use our own focal length and union bounding box for no-crop output.
        R1, R2, P1_cv, P2_cv, _, _, _ = cv2.stereoRectify(
            l_K, None, r_K, None, image_size,
            T_lr[:3, :3], T_lr[:3, 3],
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
        )
        f_rect_cv = float(P1_cv[0, 0])
        if f_rect_cv <= 0:
            # Wrong order — swap and retry
            l_img, r_img = r_img, l_img
            l_K, r_K = r_K, l_K
            l_T, r_T = r_T, l_T
            swapped = not swapped
            T_lr = _to_4x4(r_T) @ np.linalg.inv(_to_4x4(l_T))
            R1, R2, P1_cv, P2_cv, _, _, _ = cv2.stereoRectify(
                l_K, None, r_K, None, image_size,
                T_lr[:3, :3], T_lr[:3, 3],
                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
            )
            f_rect_cv = float(P1_cv[0, 0])

        if f_rect_cv <= 0:
            cv2.putText(orig_pair,
                        f"DEGENERATE (f={f_rect_cv:.0f}) — not a valid stereo pair",
                        (10, orig_pair.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(str(output_dir / f"{s1}_{s2}.jpg"), orig_pair)
            done += 1
            continue

        # Use original focal length. Find valid output region via remap (not
        # undistortPoints which fails for large rotations where z→0).
        f_orig = max(float(l_K[0, 0]), float(r_K[0, 0]))
        Tx_phys = float(P2_cv[0, 3]) / f_rect_cv  # physical Tx

        # First pass: remap at original size with cx=W/2, cy=H/2 to find content bbox
        P1_tmp = np.array([
            [f_orig, 0, W_in / 2.0, 0],
            [0, f_orig, H_in / 2.0, 0],
            [0, 0, 1, 0],
        ], dtype=np.float64)
        P2_tmp = P1_tmp.copy()
        P2_tmp[0, 3] = f_orig * Tx_phys

        # Use oversized canvas to capture all content
        W_big, H_big = int(W_in * 3), int(H_in * 3)
        P1_big = P1_tmp.copy()
        P1_big[0, 2] = W_big / 2.0
        P1_big[1, 2] = H_big / 2.0
        P2_big = P1_big.copy()
        P2_big[0, 3] = f_orig * Tx_phys

        map_l_big = cv2.initUndistortRectifyMap(l_K, None, R1, P1_big, (W_big, H_big), cv2.CV_32FC1)
        map_r_big = cv2.initUndistortRectifyMap(r_K, None, R2, P2_big, (W_big, H_big), cv2.CV_32FC1)

        # Find valid regions: where map points back into [0, W_in) x [0, H_in)
        valid_l = ((map_l_big[0] >= 0) & (map_l_big[0] < W_in) &
                   (map_l_big[1] >= 0) & (map_l_big[1] < H_in)).astype(np.uint8)
        valid_r = ((map_r_big[0] >= 0) & (map_r_big[0] < W_in) &
                   (map_r_big[1] >= 0) & (map_r_big[1] < H_in)).astype(np.uint8)
        valid_union = valid_l | valid_r

        ys, xs = np.where(valid_union)
        if len(xs) == 0:
            cv2.putText(orig_pair, "NO VALID REGION",
                        (10, orig_pair.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(str(output_dir / f"{s1}_{s2}.jpg"), orig_pair)
            done += 1
            continue

        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        W_rect = x1 - x0
        H_rect = y1 - y0

        # Shift principal point so content starts at (0, 0)
        cx_out = W_big / 2.0 - x0
        cy_out = H_big / 2.0 - y0

        scale_ratio = max(W_rect / W_in, H_rect / H_in)
        print(f"  {s1}-{s2}: f_orig={f_orig:.0f} rect={W_rect}x{H_rect}"
              f" ({W_rect/W_in:.2f}x{H_rect/H_in:.2f})"
              f"{' swapped' if swapped else ''}")

        P1_out = np.array([
            [f_orig, 0, cx_out, 0],
            [0, f_orig, cy_out, 0],
            [0, 0, 1, 0],
        ], dtype=np.float64)
        P2_out = P1_out.copy()
        P2_out[0, 3] = f_orig * Tx_phys

        map_l = cv2.initUndistortRectifyMap(l_K, None, R1, P1_out, (W_rect, H_rect), cv2.CV_32FC1)
        map_r = cv2.initUndistortRectifyMap(r_K, None, R2, P2_out, (W_rect, H_rect), cv2.CV_32FC1)

        left_rect = cv2.remap(l_img, map_l[0], map_l[1], cv2.INTER_LINEAR)
        right_rect = cv2.remap(r_img, map_r[0], map_r[1], cv2.INTER_LINEAR)

        rect_pair = np.hstack([left_rect, right_rect])
        for y in range(0, H_rect, max(H_rect // 12, 1)):
            color = tuple(int(c) for c in np.random.randint(80, 255, 3))
            cv2.line(rect_pair, (0, y), (2 * W_rect, y), color, 1)

        scale_rect = min(1.0, TARGET_W / rect_pair.shape[1])
        if scale_rect < 1.0:
            rect_pair = cv2.resize(rect_pair, None, fx=scale_rect, fy=scale_rect)

        swap_tag = "SWAPPED" if swapped else "as-is"
        info = (f"RECTIFIED ({swap_tag})  f={f_orig:.0f}  {W_rect}x{H_rect}  "
                f"rot={stats['rot1_deg']:.1f}/{stats['rot2_deg']:.1f}deg  "
                f"bl={stats['baseline']:.3f}m  angle={angle_deg:.1f}deg")
        cv2.putText(rect_pair, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Match widths, then stack vertically
        w_orig, w_rect = orig_pair.shape[1], rect_pair.shape[1]
        if w_orig != w_rect:
            target = max(w_orig, w_rect)
            if w_orig < target:
                orig_pair = cv2.resize(orig_pair, (target, int(orig_pair.shape[0] * target / w_orig)))
            else:
                rect_pair = cv2.resize(rect_pair, (target, int(rect_pair.shape[0] * target / w_rect)))

        combined = np.vstack([orig_pair, rect_pair])
        cv2.imwrite(str(output_dir / f"{s1}_{s2}.jpg"), combined)
        done += 1
        if done % 20 == 0:
            print(f"  [{done}/{n_pairs}]")

    # Write CSV
    csv_path = output_dir / "stats.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "cam1", "cam2", "group", "baseline", "angle_deg", "cos_sim", "perp",
            "f_ratio", "rect_f_ratio", "rot1", "rot2", "has_overlap",
        ])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n{done} images saved, stats -> {csv_path}")


if __name__ == "__main__":
    main()
