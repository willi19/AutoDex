#!/usr/bin/env bash
set -euo pipefail

# Re-run distributed FoundPose init on one existing episode (disk mode).
#
# Usage:
#   bash scripts/rerun_foundpose_episode.sh \
#     /home/robot/shared_data/AutoDex/experiment/selected_100/allegro/attached_container/20260330_164351
#
# Optional:
#   PROMPT="object on the checkerboard" bash scripts/rerun_foundpose_episode.sh <episode_dir>
#   OUT_DIR=/tmp/fp_test bash scripts/rerun_foundpose_episode.sh <episode_dir>

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <episode_dir>"
  exit 1
fi

EP_DIR="$(python - <<'PY' "$1"
from pathlib import Path
import sys
p = Path(sys.argv[1]).expanduser().resolve()
print(p)
PY
)"

if [[ ! -d "$EP_DIR" ]]; then
  echo "episode dir not found: $EP_DIR"
  exit 1
fi

OBJ="$(basename "$(dirname "$EP_DIR")")"
EP_NAME="$(basename "$EP_DIR")"
EXP_ROOT="$(dirname "$(dirname "$EP_DIR")")"
PROMPT="${PROMPT:-object on the checkerboard}"
OUT_DIR="${OUT_DIR:-$HOME/shared_data/AutoDex/experiment/object6d_test_foundpose}"

echo "[rerun] obj=$OBJ"
echo "[rerun] ep =$EP_NAME"
echo "[rerun] root=$EXP_ROOT"
echo "[rerun] out =$OUT_DIR"
echo "[rerun] prompt=$PROMPT"

python src/validation/perception/init_interactive.py \
  --obj "$OBJ" \
  --mode disk \
  --exp-root "$EXP_ROOT" \
  --ep "$EP_NAME" \
  --prompt "$PROMPT" \
  --out "$OUT_DIR"

