"""Classify fail modes for sim_filter:
  - no_contact: hand never touches obj after squeeze
  - lost_in_force: contact OK after squeeze, but obj escapes during force test
  - scene_collision: filtered by stage 1
"""
import os, sys, json, argparse
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="inspire_f1")
    ap.add_argument("--obj", required=True)
    ap.add_argument("--version", default="v3")
    args = ap.parse_args()

    base = f"{REPO_ROOT}/bodex_outputs/{args.hand}/{args.version}/{args.obj}"
    if not os.path.isdir(base):
        print(f"No outputs at {base}"); return

    counts = {"success": 0, "scene_coll": 0, "no_contact": 0, "lost_in_force": 0,
              "no_traj": 0, "other_fail": 0}

    for root, dirs, files in os.walk(base):
        if "sim_eval.json" not in files:
            continue
        e = json.load(open(os.path.join(root, "sim_eval.json")))
        if e.get("success"):
            counts["success"] += 1; continue
        if e.get("reason") == "scene_collision":
            counts["scene_coll"] += 1; continue
        traj_p = os.path.join(root, "sim_traj.json")
        if not os.path.exists(traj_p):
            counts["no_traj"] += 1; continue
        try:
            traj = json.load(open(traj_p))
        except Exception:
            counts["other_fail"] += 1; continue

        contacts = traj.get("contacts", [])
        phases = traj.get("phase", [])
        # Find indices in 'squeeze' phase (last entries before 'force_*')
        last_sqz_idx = -1
        for i, ph in enumerate(phases):
            if ph == "squeeze":
                last_sqz_idx = i
        n_contact_at_sqz = len(contacts[last_sqz_idx]) if last_sqz_idx >= 0 and last_sqz_idx < len(contacts) else 0

        if n_contact_at_sqz == 0:
            counts["no_contact"] += 1
        else:
            counts["lost_in_force"] += 1

    total = sum(counts.values())
    print(f"=== {args.hand} / {args.obj} fail-mode breakdown ({total} seeds) ===")
    for k, v in counts.items():
        pct = v/total*100 if total else 0
        print(f"  {k:20s} {v:6d}  ({pct:.2f}%)")


if __name__ == "__main__":
    main()
