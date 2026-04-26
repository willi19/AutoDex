"""Robot-PC GoTrack tracker (stages 5-6).

Collects per-frame, per-camera anchor observations from the 6 capture-PC
``gotrack_daemon`` processes, synchronises by frame_id, runs multi-view
triangulation + robust Kabsch fit, and publishes the resulting world pose
back to the daemons as the next prior.

Use after a successful FoundPose-based init pose has been produced. Init
pose is sent once via CommandSender (or just published as the initial
prior on the prior-pose PUB channel).

Channels (mirror gotrack_daemon defaults):
    SUB obs:          DataCollector  port 1235  (subscribes to 6 capture PCs)
    PUB prior_pose:   custom PUB      port 1236  (binds, daemons subscribe)
    REQ commands:     CommandSender   port 6892  (init/start/stop)

Run inside the `gotrack` conda env on the robot PC.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import zmq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_GOTRACK_ROOT = Path(__file__).resolve().parent / "thirdparty/MV-GoTrack"
if str(_GOTRACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_GOTRACK_ROOT))

logger = logging.getLogger(__name__)

# Default sync / timing parameters.
_DEFAULT_FRAME_TIMEOUT_S = 0.5     # drop a frame if not all cams arrive in time
_DEFAULT_MAX_INFLIGHT_FRAMES = 8   # buffer cap


def _bytes_to_np(buf: bytes, shape: List[int], dtype: str) -> Optional[np.ndarray]:
    if not shape or not dtype or buf == b"":
        return None
    arr = np.frombuffer(buf, dtype=np.dtype(dtype))
    return arr.reshape(shape) if arr.size else None


def _unpack_payload(meta_item: Dict[str, Any], parts: List[bytes]) -> Dict[str, Any]:
    """Reconstruct numpy arrays for one cam from a multipart message."""
    out: Dict[str, Any] = {
        "frame_id": int(meta_item.get("frame_id", -1)),
        "prior_frame_id": int(meta_item.get("prior_frame_id", -1)),
        "status": str(meta_item.get("status", "")),
        "engine_sec": float(meta_item.get("engine_sec", 0.0)),
        "serial": str(meta_item.get("name", "")),
    }
    arrays = meta_item.get("arrays", {}) or {}
    for k, info in arrays.items():
        idx = int(info.get("data_index", -1))
        # data_index in meta is the per-message binary slot (0-based across all
        # arrays sent in this message). DataPublisher prepends [topic, json] so
        # binary parts start at parts[2 + idx].
        if 0 <= idx < len(parts):
            out[k] = _bytes_to_np(parts[idx], info["shape"], info["dtype"])
        else:
            out[k] = None
    return out


class PriorPosePublisher:
    """Bind PUB socket so all capture PCs can subscribe."""

    def __init__(self, port: int):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.bind(f"tcp://*:{port}")
        time.sleep(0.1)  # let subscribers connect
        logger.info(f"[prior_pub] bound on tcp://*:{port}")

    def publish(self, pose_world: np.ndarray, frame_id: int) -> None:
        msg = {
            "frame_id": int(frame_id),
            "pose_world": pose_world.tolist(),
            "ts": time.time(),
        }
        self.sock.send_json(msg)

    def close(self) -> None:
        self.sock.close()


class FrameSyncBuffer:
    """Per-frame buffer: collects per-cam payloads keyed by frame_id.

    Pops oldest frame once it has >= min_cams payloads, or after timeout.
    """

    def __init__(self, min_cams: int, timeout_s: float, max_inflight: int):
        self.min_cams = int(min_cams)
        self.timeout_s = float(timeout_s)
        self.max_inflight = int(max_inflight)
        self._buf: Dict[int, Dict[str, Dict[str, Any]]] = {}
        self._first_seen: Dict[int, float] = {}
        self._lock = threading.Lock()

    def add(self, frame_id: int, serial: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            slot = self._buf.setdefault(frame_id, {})
            slot[serial] = payload
            self._first_seen.setdefault(frame_id, time.time())

    def pop_ready(self) -> Optional[Tuple[int, Dict[str, Dict[str, Any]]]]:
        """Return (frame_id, payloads) for oldest frame that satisfies threshold
        or has timed out. Caller decides whether to skip a timed-out frame.
        """
        with self._lock:
            if not self._buf:
                return None
            oldest = min(self._buf.keys())
            slot = self._buf[oldest]
            seen = self._first_seen[oldest]
            ready = len(slot) >= self.min_cams
            timed_out = time.time() - seen >= self.timeout_s
            if ready or timed_out:
                payloads = self._buf.pop(oldest)
                self._first_seen.pop(oldest, None)
                return oldest, payloads

            # Drop very old frames if buffer is overloaded.
            if len(self._buf) > self.max_inflight:
                # Drop everything older than half the buffer.
                cutoff = sorted(self._buf.keys())[len(self._buf) // 2]
                for fid in list(self._buf.keys()):
                    if fid < cutoff:
                        self._buf.pop(fid, None)
                        self._first_seen.pop(fid, None)
            return None


class GoTrackTracker:
    """Robot-PC tracker: receives anchor obs, fuses to world pose, publishes prior."""

    def __init__(
        self,
        capture_pc_ips: List[str],
        port_obs: int = 1235,
        port_prior: int = 1236,
        min_cams_per_frame: int = 6,
        max_triangulation_residual_mm: float = 25.0,
        kabsch_inlier_thresh_mm: float = 35.0,
        confidence_weight_mode: str = "linear",
        confidence_weight_alpha: float = 1.0,
        external_unit_scale_to_meter: float = 1.0,
        frame_timeout_s: float = _DEFAULT_FRAME_TIMEOUT_S,
        max_inflight_frames: int = _DEFAULT_MAX_INFLIGHT_FRAMES,
    ):
        self.capture_pc_ips = list(capture_pc_ips)
        self.port_obs = int(port_obs)
        self.port_prior = int(port_prior)
        self.max_triangulation_residual_mm = float(max_triangulation_residual_mm)
        self.kabsch_inlier_thresh_mm = float(kabsch_inlier_thresh_mm)
        self.confidence_weight_mode = str(confidence_weight_mode)
        self.confidence_weight_alpha = float(confidence_weight_alpha)
        self.external_unit_scale_to_meter = float(external_unit_scale_to_meter)
        self.min_cams_per_frame = int(min_cams_per_frame)

        self.sync_buffer = FrameSyncBuffer(
            min_cams=min_cams_per_frame,
            timeout_s=frame_timeout_s,
            max_inflight=max_inflight_frames,
        )

        # SUB sockets — one per capture PC.
        self.ctx = zmq.Context.instance()
        self.sub_sockets: Dict[str, zmq.Socket] = {}
        self.poller = zmq.Poller()
        for ip in self.capture_pc_ips:
            sock = self.ctx.socket(zmq.SUB)
            sock.setsockopt_string(zmq.SUBSCRIBE, "")
            sock.connect(f"tcp://{ip}:{port_obs}")
            self.sub_sockets[ip] = sock
            self.poller.register(sock, zmq.POLLIN)
        logger.info(f"[tracker] subscribed to {len(self.sub_sockets)} capture PCs")

        self.prior_pub = PriorPosePublisher(port_prior)

        self._stop = threading.Event()
        self._sub_thread = threading.Thread(target=self._sub_loop, daemon=True)
        self._sub_thread.start()

    def _sub_loop(self) -> None:
        while not self._stop.is_set():
            try:
                socks = dict(self.poller.poll(timeout=100))
            except zmq.ZMQError:
                continue
            for ip, sock in self.sub_sockets.items():
                if sock not in socks:
                    continue
                try:
                    parts = sock.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    continue
                if len(parts) < 2:
                    continue
                try:
                    msg = json.loads(parts[1].decode("utf-8"))
                except Exception:
                    continue
                # Binary parts begin at parts[2:].
                bin_parts = parts[2:]
                for item in msg.get("items", []):
                    if item.get("type") != "gotrack_obs":
                        continue
                    payload = _unpack_payload(item, bin_parts)
                    self.sync_buffer.add(payload["frame_id"], payload["serial"], payload)

    def publish_prior(self, pose_world: np.ndarray, frame_id: int) -> None:
        self.prior_pub.publish(pose_world, frame_id)

    def fuse_one_frame(
        self, payloads: Dict[str, Dict[str, Any]]
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Triangulate + Kabsch fit on one frame's payloads.

        Returns (pose_world or None, info_dict).
        """
        from utils.multiview_geometry import (
            robust_fit_pose_from_anchors,
            triangulate_anchor_observations,
            build_fit_weights_from_triangulation_records,
        )

        observations_by_anchor: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        intrinsics_map: Dict[str, np.ndarray] = {}
        extrinsics_map: Dict[str, np.ndarray] = {}

        for serial, p in payloads.items():
            uv = p.get("uv_curr")
            conf = p.get("confidence")
            sel = p.get("selected_mask")
            anchor_ids = p.get("anchor_ids")
            positions_o = p.get("positions_o")
            ci = p.get("crop_intrinsic")
            Tw = p.get("T_world_from_crop_cam")
            if uv is None or conf is None or sel is None or anchor_ids is None \
               or positions_o is None or ci is None or Tw is None:
                continue
            intrinsics_map[serial] = np.asarray(ci, dtype=np.float64)
            # extrinsics expected as world->cam by triangulate (matches paradex).
            extrinsics_map[serial] = np.linalg.inv(np.asarray(Tw, dtype=np.float64))

            sel_idx = np.where(sel)[0]
            for i in sel_idx:
                aid = int(anchor_ids[i])
                observations_by_anchor[aid].append({
                    "camera_id": serial,
                    "uv_curr": np.asarray(uv[i], dtype=np.float32),
                    "confidence": float(conf[i]),
                    "position_o": np.asarray(positions_o[i], dtype=np.float32),
                    "valid_flag": True,
                })

        if not observations_by_anchor:
            return None, {"reason": "no_observations"}

        tri = triangulate_anchor_observations(
            observations_by_anchor=observations_by_anchor,
            intrinsics_map=intrinsics_map,
            extrinsics_map=extrinsics_map,
            min_views=2,
            external_unit_scale_to_meter=self.external_unit_scale_to_meter,
            weight_mode=self.confidence_weight_mode,
            weight_alpha=self.confidence_weight_alpha,
        )
        records = tri.get("records", [])
        if not records:
            return None, {"reason": "triangulation_empty", "tri": tri}

        # Optional residual filter.
        if self.max_triangulation_residual_mm > 0.0:
            keep = []
            for r in records:
                resid = r.get("max_residual_mm", 0.0)
                if resid is None or resid <= self.max_triangulation_residual_mm:
                    keep.append(r)
            records = keep
        if not records:
            return None, {"reason": "all_filtered_by_residual"}

        weights = build_fit_weights_from_triangulation_records(
            records,
            mode="geometry",
            params={},
        )
        fit = robust_fit_pose_from_anchors(
            triangulation_records=records,
            inlier_threshold_mm=self.kabsch_inlier_thresh_mm,
            weights=weights,
        )
        pose_world = fit.get("pose_world")
        if pose_world is None:
            return None, {"reason": "fit_failed", "fit": fit}
        return np.asarray(pose_world, dtype=np.float64), {
            "n_triangulated": len(records),
            "n_inliers": int(fit.get("num_inliers", 0)),
            "mean_residual_mm": float(fit.get("mean_residual_mm", -1)),
            "fit": fit,
        }

    def track(
        self, init_pose_world: np.ndarray
    ) -> Iterator[Tuple[int, np.ndarray, Dict[str, Any]]]:
        """Generator: yield (frame_id, pose_world, info) every frame."""
        # Send initial prior so daemons can start processing.
        self.publish_prior(init_pose_world, frame_id=-1)
        prev_pose = init_pose_world.astype(np.float64).copy()

        while not self._stop.is_set():
            ready = self.sync_buffer.pop_ready()
            if ready is None:
                time.sleep(0.005)
                continue
            frame_id, payloads = ready
            if len(payloads) < self.min_cams_per_frame:
                # Timed-out frame with too few cams — still try; otherwise skip.
                logger.debug(f"[track] frame {frame_id}: only {len(payloads)} cams")
            pose_world, info = self.fuse_one_frame(payloads)
            if pose_world is None:
                logger.warning(f"[track] frame {frame_id} fit failed: {info.get('reason')}")
                # Republish previous prior to keep daemons moving.
                self.publish_prior(prev_pose, frame_id=frame_id)
                continue
            self.publish_prior(pose_world, frame_id=frame_id)
            prev_pose = pose_world
            yield frame_id, pose_world, info

    def close(self) -> None:
        self._stop.set()
        self._sub_thread.join(timeout=1)
        for s in self.sub_sockets.values():
            s.close()
        self.prior_pub.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-ips", type=str, nargs="+", required=True,
                        help="IPs of capture1..6 PCs (one per PC).")
    parser.add_argument("--port-obs", type=int, default=1235)
    parser.add_argument("--port-prior", type=int, default=1236)
    parser.add_argument("--min-cams-per-frame", type=int, default=6)
    parser.add_argument("--init-pose-npy", type=str, required=True,
                        help="Path to .npy with 4x4 init pose_world.")
    parser.add_argument("--max-frames", type=int, default=-1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(message)s")

    init_pose = np.load(args.init_pose_npy)
    tracker = GoTrackTracker(
        capture_pc_ips=args.capture_ips,
        port_obs=args.port_obs,
        port_prior=args.port_prior,
        min_cams_per_frame=args.min_cams_per_frame,
    )
    try:
        n = 0
        for frame_id, pose, info in tracker.track(init_pose):
            print(f"frame {frame_id}: t={pose[:3, 3].tolist()}  "
                  f"n_inl={info.get('n_inliers')}  "
                  f"resid_mm={info.get('mean_residual_mm', -1):.2f}")
            n += 1
            if args.max_frames > 0 and n >= args.max_frames:
                break
    finally:
        tracker.close()


if __name__ == "__main__":
    main()
