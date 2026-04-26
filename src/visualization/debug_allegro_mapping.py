"""Debug allegro position mapping: render frames with different mappings across cameras.

Output: debug_mapping/{frame_idx}.png — each row is a mapping, each column is a camera.
Finger colors: index=cyan, middle=green, ring=pink, thumb=orange, arm=gray.

Usage:
    python src/visualization/debug_allegro_mapping.py --obj banana
    python src/visualization/debug_allegro_mapping.py --obj banana --frame 100 200 300
    python src/visualization/debug_allegro_mapping.py --obj banana --serial 25305460 25322645
"""
import argparse
import sys
import numpy as np
import cv2
from pathlib import Path
from scipy.interpolate import interp1d

PARADEX_ROOT = Path.home() / "paradex"
sys.path.insert(0, str(PARADEX_ROOT))

from paradex.visualization.robot import RobotModule
from paradex.calibration.utils import load_camparam

EXP_BASE = Path.home() / "shared_data" / "AutoDex" / "experiment" / "selected_100"

FINGER_COLORS_BGR = {
    "index":  (255, 200,   0),
    "middle": (100, 255,   0),
    "ring":   (200,   0, 255),
    "thumb":  (  0, 140, 255),
}
ARM_COLOR_BGR = (40, 200, 40)

# URDF link -> finger mapping
# trimesh appends _1, _2 suffixes when the same mesh is reused across fingers:
#   no suffix = index (links 0-3), _1 = middle (links 4-7), _2 = ring (links 8-11)
#   thumb (links 12-15) has unique mesh names
ALLEGRO_LINK_TO_FINGER = {}
for i in range(4):
    ALLEGRO_LINK_TO_FINGER[f"link_{i}.0.obj"] = "index"
    ALLEGRO_LINK_TO_FINGER[f"link_{i}.0.obj_1"] = "middle"
    ALLEGRO_LINK_TO_FINGER[f"link_{i}.0.obj_2"] = "ring"
ALLEGRO_LINK_TO_FINGER["link_3.0_tip.obj"] = "index"
ALLEGRO_LINK_TO_FINGER["link_3.0_tip.obj_1"] = "middle"
ALLEGRO_LINK_TO_FINGER["link_3.0_tip.obj_2"] = "ring"
for name in ["link_12.0_right.obj", "link_12.0_left.obj", "link_13.0.obj", "link_14.0.obj", "link_15.0.obj", "link_15.0_tip.obj"]:
    ALLEGRO_LINK_TO_FINGER[name] = "thumb"


def resample(src_time, src_values, target_time):
    t = np.clip(target_time, float(src_time[0]), float(src_time[-1]))
    return interp1d(src_time, src_values, axis=0)(t)


def get_link_color(link_name):
    finger = ALLEGRO_LINK_TO_FINGER.get(link_name)
    if finger:
        return FINGER_COLORS_BGR[finger]
    if "link" in link_name or "base.obj" in link_name or "palm" in link_name:
        return ARM_COLOR_BGR
    return ARM_COLOR_BGR


def render_overlay(frame, robot, qpos, cam_from_robot, K, alpha=0.5):
    robot.update_cfg(qpos[:robot.get_num_joints()])
    scene = robot.scene
    overlay = frame.copy()

    for ln in scene.geometry:
        mesh = scene.geometry[ln]
        T = scene.graph.get(ln)[0]
        verts_world = (T[:3, :3] @ mesh.vertices.T + T[:3, 3:4]).T
        verts_cam = (cam_from_robot[:3, :3] @ verts_world.T + cam_from_robot[:3, 3:4]).T

        valid = verts_cam[:, 2] > 0.01
        if not valid.any():
            continue
        pts = (K @ verts_cam.T).T
        pts_2d = pts[:, :2] / pts[:, 2:3]

        color = get_link_color(ln)
        for face in mesh.faces:
            if not (valid[face[0]] and valid[face[1]] and valid[face[2]]):
                continue
            tri = pts_2d[face].astype(np.int32)
            cv2.fillPoly(overlay, [tri], color)

    return cv2.addWeighted(frame, 1.0 - alpha, overlay, alpha, 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hand", default="allegro")
    p.add_argument("--obj", required=True)
    p.add_argument("--ep", default=None)
    p.add_argument("--serial", nargs="+", default=None)
    p.add_argument("--frame", nargs="+", type=int, default=None)
    p.add_argument("--n-serials", type=int, default=4)
    p.add_argument("--arm-time-offset", type=float, default=0.03)
    p.add_argument("--hand-time-offset", type=float, default=0.03)
    p.add_argument("--out-dir", default="debug_mapping")
    args = p.parse_args()

    hand_dir = EXP_BASE / args.hand / args.obj
    if args.ep:
        ep = hand_dir / args.ep
    else:
        ep = sorted(d for d in hand_dir.iterdir() if d.is_dir())[0]
    print(f"Episode: {ep}")

    # Load raw data
    arm_time = np.load(ep / "raw/arm/time.npy") + args.arm_time_offset
    arm_pos = np.load(ep / "raw/arm/position.npy")
    hand_time = np.load(ep / "raw/hand/time.npy") + args.hand_time_offset
    hand_pos = np.load(ep / "raw/hand/position.npy")
    hand_act = np.load(ep / "raw/hand/action.npy")
    video_times = np.load(ep / "raw/timestamps/timestamp.npy")

    # Frame indices
    if args.frame:
        frames = args.frame
    else:
        n = len(video_times)
        frames = [n // 4, n // 2, n * 3 // 4]

    # Camera setup
    intrinsic, extrinsic = load_camparam(str(ep))
    c2r = np.load(ep / "C2R.npy")
    all_serials = sorted(intrinsic.keys())

    if args.serial:
        serials = [s for s in args.serial if s in all_serials]
    else:
        serials = all_serials[:args.n_serials]
    print(f"Cameras: {serials}")

    cam_data = {}
    for s in serials:
        K = np.array(intrinsic[s]["intrinsics_undistort"])
        ext4 = np.eye(4)
        ext4[:3, :] = extrinsic[s]
        cam_from_robot = ext4 @ c2r
        cam_data[s] = (K, cam_from_robot)

    # Mappings
    POS_TO_URDF = [5, 2, 0, 1, 7, 15, 14, 12, 11, 13, 4, 9, 8, 6, 10, 3]
    ACT_TO_URDF = list(range(4, 16)) + list(range(0, 4))
    OLD_MAPPING = [4, 2, 0, 1, 7, 15, 14, 12, 11, 13, 4, 9, 8, 6, 10, 3]

    ACT_TO_URDF = list(range(4, 16)) + list(range(0, 4))
    mappings = [
        ("POSITION", POS_TO_URDF, "pos"),
        ("ACTION", ACT_TO_URDF, "act"),
    ]

    # Robot
    urdf_base = Path.home() / "AutoDex" / "autodex" / "planner" / "src" / "curobo" / "content" / "assets" / "robot"
    urdf_path = str(urdf_base / f"{args.hand}_description" / f"xarm_{args.hand}.urdf")
    robot = RobotModule(urdf_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fidx in frames:
        if fidx >= len(video_times):
            print(f"Frame {fidx} out of range ({len(video_times)}), skipping")
            continue

        vt = video_times[fidx]
        arm_q = resample(arm_time, arm_pos, np.array([vt]))[0]
        hand_pos_q = resample(hand_time, hand_pos, np.array([vt]))[0]
        hand_act_q = resample(hand_time, hand_act, np.array([vt]))[0]

        # Read frames from all cameras
        cam_frames = {}
        for s in serials:
            cap = cv2.VideoCapture(str(ep / "videos" / f"{s}.avi"))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, img = cap.read()
            cap.release()
            if ret:
                cam_frames[s] = img

        if not cam_frames:
            continue

        print(f"\n--- Frame {fidx} ---")
        print(f"hand_pos raw: {np.round(hand_pos_q, 3)}")
        print(f"hand_act raw: {np.round(hand_act_q, 3)}")
        print(f"pos[POS_TO_URDF]: {np.round(hand_pos_q[POS_TO_URDF], 3)}")
        print(f"act[ACT_TO_URDF]: {np.round(hand_act_q[ACT_TO_URDF], 3)}")
        print(f"diff: {np.round(np.abs(hand_pos_q[POS_TO_URDF] - hand_act_q[ACT_TO_URDF]), 3)}")

        # Rows = mappings, Cols = cameras
        rows = []
        for map_name, mapping, source in mappings:
            if source == "act":
                hand_urdf = hand_act_q[mapping]
            else:
                hand_urdf = hand_pos_q[mapping]

            qpos = np.concatenate([arm_q, hand_urdf])
            cols = []
            for s in serials:
                if s not in cam_frames:
                    continue
                K, cam_from_robot = cam_data[s]
                img = render_overlay(cam_frames[s], robot, qpos, cam_from_robot, K)
                # Label
                label = f"{map_name} | {s}"
                cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cols.append(img)

            if cols:
                rows.append(np.hstack(cols))

        if rows:
            grid = np.vstack(rows)
            max_w = 3840
            if grid.shape[1] > max_w:
                scale = max_w / grid.shape[1]
                grid = cv2.resize(grid, None, fx=scale, fy=scale)

            out_path = out_dir / f"{fidx:06d}.png"
            cv2.imwrite(str(out_path), grid)
            print(f"Saved: {out_path} ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()