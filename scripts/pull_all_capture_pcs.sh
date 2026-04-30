#!/usr/bin/env bash
set -euo pipefail

PCS=(capture1 capture2 capture3 capture4 capture5 capture6)
REPO_DIR="\$HOME/AutoDex"
BRANCH="${1:-main}"

for pc in "${PCS[@]}"; do
  echo "=== $pc ==="
  ssh -o ConnectTimeout=5 "$pc" "bash -lc '
    set -euo pipefail
    cd \"$REPO_DIR\"
    echo \"[\$(hostname)] before: \$(git rev-parse --short HEAD)\"
    git fetch origin
    git checkout \"$BRANCH\"
    git pull --ff-only origin \"$BRANCH\"
    echo \"[\$(hostname)] after : \$(git rev-parse --short HEAD)\"
  '" || echo "[$pc] FAILED"
done
