"""Tiny Flask dashboard for GoTrackTracker live status.

Shares the same process as `GoTrackTracker` and reads its `status` dict
under a lock. One HTML page polls /api/status at 1Hz.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from flask import Flask, jsonify

if TYPE_CHECKING:
    from autodex.perception.gotrack_tracker import GoTrackTracker


_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>GoTrack Tracking Monitor</title>
<style>
body { font-family: -apple-system, sans-serif; background: #1a1a1a; color: #eee; padding: 20px; }
h1 { color: #00d4ff; margin-bottom: 16px; }
.card { background: #2a2a2a; border-radius: 8px; padding: 16px; margin-bottom: 12px; }
.row { display: flex; gap: 12px; }
.col { flex: 1; }
.k { color: #888; font-size: 0.85em; }
.v { font-size: 1.4em; font-weight: bold; margin-top: 2px; }
.ok { color: #4ade80; }
.bad { color: #f87171; }
.warn { color: #facc15; }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 6px 10px; text-align: left; border-bottom: 1px solid #3a3a3a; }
th { color: #888; font-weight: normal; font-size: 0.85em; }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }
.pose { font-family: ui-monospace, monospace; font-size: 0.85em; white-space: pre; }
.foot { color: #555; font-size: 0.8em; margin-top: 12px; }
</style>
</head>
<body>
<h1>GoTrack Tracking Monitor</h1>
<div id="root">loading...</div>
<div class="foot">polling /api/status @ 1Hz</div>
<script>
const pad = (x, n=8) => x.toFixed(4).padStart(n);
const dotColor = age => age < 0.3 ? '#4ade80' : age < 1.5 ? '#facc15' : '#f87171';

async function tick() {
  try {
    const r = await fetch('/api/status');
    const s = await r.json();
    const init = s.init_done ? '<span class="ok">YES</span>' : '<span class="bad">NO</span>';
    const fitBadge = s.last_fit_ok === null ? '<span class="warn">—</span>'
        : s.last_fit_ok ? '<span class="ok">OK</span>' : '<span class="bad">FAIL</span>';
    let pcRows = '';
    for (const [ip, info] of Object.entries(s.per_pc_last_frame || {})) {
      const age = info.age_s.toFixed(2);
      pcRows += `<tr><td><span class="dot" style="background:${dotColor(info.age_s)}"></span>${ip}</td>
                <td>${info.frame_id}</td><td>${age}s</td></tr>`;
    }
    let poseStr = '—';
    if (s.current_pose) {
      poseStr = s.current_pose.map(row => row.map(x => pad(x)).join('  ')).join('\\n');
    }
    document.getElementById('root').innerHTML = `
      <div class="row">
        <div class="card col">
          <div class="k">Object</div><div class="v">${s.obj_name || '—'}</div>
          <div class="k" style="margin-top:10px">Init</div><div class="v">${init}</div>
        </div>
        <div class="card col">
          <div class="k">Frame ID</div><div class="v">${s.frame_id}</div>
          <div class="k" style="margin-top:10px">FPS (rolling)</div><div class="v">${s.fps.toFixed(2)}</div>
        </div>
        <div class="card col">
          <div class="k">Last Fit</div><div class="v">${fitBadge}</div>
          <div class="k" style="margin-top:10px">Inliers / Resid (mm)</div>
          <div class="v">${s.n_inliers} / ${s.mean_residual_mm.toFixed(2)}</div>
          ${s.fail_reason ? `<div class="bad" style="margin-top:6px">reason: ${s.fail_reason}</div>` : ''}
        </div>
      </div>
      <div class="card">
        <div class="k">Capture PCs</div>
        <table><thead><tr><th>IP</th><th>Last frame</th><th>Age</th></tr></thead>
          <tbody>${pcRows || '<tr><td colspan=3 class="k">no data yet</td></tr>'}</tbody>
        </table>
      </div>
      <div class="card">
        <div class="k">Current Pose (4×4, world)</div>
        <div class="pose">${poseStr}</div>
      </div>`;
  } catch (e) {
    document.getElementById('root').innerHTML = `<div class="bad">fetch error: ${e}</div>`;
  }
}
tick(); setInterval(tick, 1000);
</script>
</body>
</html>"""


def create_app(tracker: "GoTrackTracker") -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return _HTML

    @app.route("/api/status")
    def status():
        with tracker._status_lock:
            s = dict(tracker.status)
            now = time.time()
            pc = {}
            for ip, info in s.get("per_pc_last_frame", {}).items():
                pc[ip] = {
                    "frame_id": info["frame_id"],
                    "age_s": now - info["ts"],
                }
            s["per_pc_last_frame"] = pc
        return jsonify(s)

    return app


def run_dashboard(tracker: "GoTrackTracker", port: int = 8090) -> None:
    app = create_app(tracker)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False, threaded=True)
