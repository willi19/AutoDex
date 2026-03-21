#!/usr/bin/env python3
"""ZMQ daemon for perception models.

Runs a single model (SAM3 or FPose) as a ZMQ REP server.
Model stays loaded in GPU memory, accepts requests via ZMQ.
Communication uses NAS file paths — no image serialization over network.

Usage:
    # SAM3 daemon (on capture2/3/4)
    conda activate sam3
    python -m autodex.perception.daemon --model sam3 --port 5001

    # FPose daemon (on capture4/5/6)
    conda activate foundationpose
    python -m autodex.perception.daemon --model fpose --port 5003 --mesh /path/to/mesh.obj
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import zmq

logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')
logger = logging.getLogger("daemon")

import re
_HOME = str(Path.home())

def _localize_path(path: str) -> str:
    """Replace /home/*/shared_data/ with local home's shared_data/."""
    return re.sub(r'/home/[^/]+/shared_data/', f'{_HOME}/shared_data/', path)


def run_sam3_daemon(port: int, gpu: int = 0):
    """SAM3 image model daemon. Receives image path, returns mask path."""
    logger.info(f"Loading SAM3 image model on GPU {gpu}...")
    from autodex.perception import Sam3ImageSegmentor
    seg = Sam3ImageSegmentor(gpu=gpu)
    logger.info("SAM3 loaded")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://0.0.0.0:{port}")
    logger.info(f"SAM3 daemon listening on port {port}")

    while True:
        try:
            req = json.loads(sock.recv_string())
            t0 = time.perf_counter()

            if req.get("command") == "ping":
                sock.send_string(json.dumps({"status": "ok"}))
                continue

            image_path = _localize_path(req["image_path"])
            prompt = req.get("prompt", "object on the checkerboard")
            output_path = _localize_path(req["output_path"]) if req.get("output_path") else None

            rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            mask = seg.segment(rgb, prompt)

            dt = time.perf_counter() - t0
            found = mask is not None

            if found and output_path:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(output_path, mask)

            sock.send_string(json.dumps({
                "found": found,
                "mask_path": output_path if found else None,
                "time": dt,
            }))
            logger.info(f"  {Path(image_path).stem}: {'found' if found else 'none'} [{dt:.3f}s]")

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            try:
                sock.send_string(json.dumps({"error": str(e)}))
            except:
                pass

    sock.close()
    ctx.term()


def run_fpose_daemon(port: int, mesh_path: str, gpu: int = 0):
    """FoundationPose daemon. Receives image/depth/mask paths, returns pose."""
    logger.info(f"Loading FoundationPose on GPU {gpu}...")
    from autodex.perception import PoseTracker
    tracker = PoseTracker(mesh_path, device_id=gpu)
    logger.info("FPose loaded")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://0.0.0.0:{port}")
    logger.info(f"FPose daemon listening on port {port}")

    while True:
        try:
            req = json.loads(sock.recv_string())
            t0 = time.perf_counter()

            if req.get("command") == "ping":
                sock.send_string(json.dumps({"status": "ok"}))
                continue

            if req.get("command") == "reset_mesh":
                new_mesh = _localize_path(req["mesh_path"])
                tracker = PoseTracker(new_mesh, device_id=gpu)
                logger.info(f"Mesh reset to {new_mesh}")
                sock.send_string(json.dumps({"status": "ok"}))
                continue

            image_path = _localize_path(req["image_path"])
            depth_path = _localize_path(req["depth_path"])
            mask_path = _localize_path(req["mask_path"])
            K = np.array(req["K"], dtype=np.float32)
            mode = req.get("mode", "register")
            iteration = req.get("iteration", 5)
            downscale = req.get("downscale", 0.5)

            rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
            mask = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)

            # Downscale
            H, W = rgb.shape[:2]
            nH, nW = int(H * downscale), int(W * downscale)
            rgb = cv2.resize(rgb, (nW, nH))
            depth = cv2.resize(depth, (nW, nH), interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, (nW, nH), interpolation=cv2.INTER_NEAREST)
            K_ds = K.copy()
            K_ds[0, :] *= downscale
            K_ds[1, :] *= downscale

            depth[(depth < 0.001) | (depth >= 100)] = 0

            if mode == "register":
                tracker.reset()
                pose = tracker.init(rgb, depth, mask, K_ds, iteration=iteration)
            else:
                pose = tracker.track(rgb, depth, K_ds, iteration=iteration)

            dt = time.perf_counter() - t0

            # Save pose if output_path given
            output_path = _localize_path(req["output_path"]) if req.get("output_path") else None
            if output_path:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                np.save(output_path, pose)

            sock.send_string(json.dumps({
                "pose": pose.tolist(),
                "pose_path": output_path,
                "time": dt,
            }))
            logger.info(f"  {Path(image_path).stem}: {mode} [{dt:.3f}s]")

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            try:
                sock.send_string(json.dumps({"error": str(e)}))
            except:
                pass

    sock.close()
    ctx.term()


def main():
    parser = argparse.ArgumentParser(description="Perception model daemon")
    parser.add_argument("--model", type=str, required=True, choices=["sam3", "fpose"])
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--mesh", type=str, default=None, help="Mesh path (fpose only)")
    args = parser.parse_args()

    if args.model == "sam3":
        run_sam3_daemon(args.port, args.gpu)
    elif args.model == "fpose":
        if not args.mesh:
            parser.error("--mesh required for fpose")
        run_fpose_daemon(args.port, args.mesh, args.gpu)


if __name__ == "__main__":
    main()