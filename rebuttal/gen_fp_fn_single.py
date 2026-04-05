"""Generate FP/FN figure for a single experiment: all camera views × 4 phases.

Outputs per-serial 1×4 strip images + sim trajectory for later rendering.

Usage:
    python rebuttal/gen_fp_fn_single.py \
        --obj pepsi_light --dir_idx 20260326_214007 \
        --out rebuttal/figure/2/false_positive/pepsi_light_20260326_214007
"""

import argparse
import json
import os
import sys
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

EXPERIMENT_ROOT = os.path.expanduser(
    "~/shared_data/AutoDex/experiment/selected_100/allegro"
)
CANDIDATE_ROOT = os.path.expanduser(
    "~/shared_data/AutoDex/candidates/allegro/selected_100"
)

N_FRAMES = 4


def load_frame_mapping(exp_dir):
    """Load timestamps and build time->frame mapping."""
    ts = np.load(os.path.join(exp_dir, "raw", "timestamps", "timestamp.npy"))
    return ts


def state_to_frame(ts, state_time_str):
    """Map ISO timestamp to nearest frame index."""
    t = datetime.fromisoformat(state_time_str).timestamp()
    return int(np.argmin(np.abs(ts - t)))


def extract_frames(video_path, frame_indices):
    """Extract specific frames from video. Returns list of BGR images."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frames.append(np.zeros((h, w, 3), dtype=np.uint8))
    cap.release()
    return frames


def make_strip(frames, target_h=512):
    """Resize frames to same height, concatenate horizontally."""
    resized = []
    for f in frames:
        h, w = f.shape[:2]
        s = target_h / h
        resized.append(cv2.resize(f, (int(w * s), target_h)))
    return np.concatenate(resized, axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--dir_idx", type=str, required=True)
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--target_h", type=int, default=512)
    args = parser.parse_args()

    exp_dir = os.path.join(EXPERIMENT_ROOT, args.obj, args.dir_idx)
    if not os.path.isdir(exp_dir):
        print(f"Experiment dir not found: {exp_dir}")
        return

    # Load result + timing
    with open(os.path.join(exp_dir, "result.json")) as f:
        result = json.load(f)

    states = {s["state"]: s["time"] for s in result["timing"]["execution_states"]}
    ts = load_frame_mapping(exp_dir)

    # Extract from all cameras
    video_dir = os.path.join(exp_dir, "videos")

    # 4 frames: end-1.5s, end-1.0s, end-0.5s, end
    # Use actual video frame count (timestamps can exceed video length)
    sample_vid = os.path.join(video_dir, os.listdir(video_dir)[0])
    cap = cv2.VideoCapture(sample_vid)
    n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    fps = len(ts) / (ts[-1] - ts[0])
    end_frame = n_video_frames - 1
    indices_list = [max(0, int(end_frame - sec * fps)) for sec in [1.5, 1.0, 0.5, 0]]
    print(f"Frames: {indices_list} (end={end_frame}, video_frames={n_video_frames}, fps={fps:.1f})")
    serials = sorted([f.replace(".avi", "") for f in os.listdir(video_dir) if f.endswith(".avi")])

    os.makedirs(args.out, exist_ok=True)

    for serial in serials:
        video_path = os.path.join(video_dir, f"{serial}.avi")
        frames = extract_frames(video_path, indices_list)
        strip = make_strip(frames, args.target_h)

        out_path = os.path.join(args.out, f"{serial}.png")
        cv2.imwrite(out_path, strip)

    print(f"Saved {len(serials)} strips to {args.out}")

    # Also save sim trajectory for this case
    scene_info = result.get("scene_info")
    if scene_info and len(scene_info) == 3:
        cand_dir = os.path.join(CANDIDATE_ROOT, args.obj, *scene_info)
        if os.path.isdir(cand_dir):
            print(f"Candidate: {cand_dir}")
            # Save trajectory with median params
            from src.validation.simulation.sim_dr_confusion import (
                eval_grasp_with_params, DR_GRID, FRICTION_TORSION, HAND_PATH, FORCE_SCALE,
            )
            from autodex.simulator.hand_object import MjHO

            median_ft = float(np.median(DR_GRID["friction_tangent"]))
            median_m = float(np.median(DR_GRID["obj_mass"]))

            wrist_se3 = np.load(os.path.join(cand_dir, "wrist_se3.npy"))
            pregrasp = np.load(os.path.join(cand_dir, "pregrasp_pose.npy"))
            grasp = np.load(os.path.join(cand_dir, "grasp_pose.npy"))

            mj = MjHO(
                args.obj, HAND_PATH, weld_body_name="world",
                obj_mass=median_m, friction_coef=(median_ft, FRICTION_TORSION),
            )
            succ, traj = eval_grasp_with_params(
                mj, wrist_se3, pregrasp, grasp, median_m, record_traj=True,
            )
            mj.close()

            traj_path = os.path.join(args.out, "sim_traj.json")
            with open(traj_path, "w") as f:
                json.dump(traj, f)

            info = {
                "obj": args.obj, "dir_idx": args.dir_idx,
                "scene_info": scene_info,
                "real_success": result["success"],
                "sim_success": succ,
                "params": {"friction_tangent": median_ft, "obj_mass": median_m,
                           "force_scale": FORCE_SCALE},
                "phases": {p: frame_indices.get(p) for p in PHASES},
            }
            with open(os.path.join(args.out, "info.json"), "w") as f:
                json.dump(info, f, indent=2)

            print(f"Sim result: {'pass' if succ else 'fail'} (ft={median_ft}, m={median_m})")


if __name__ == "__main__":
    main()
