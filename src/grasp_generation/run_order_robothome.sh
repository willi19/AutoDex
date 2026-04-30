#!/bin/bash
# Run set cover ordering + extract top-N for robothome objects.
#
# Usage:
#   bash src/grasp_generation/run_order_robothome.sh [HAND] [N]
#
# Defaults: HAND=inspire_left, N=100.
# Skips objects whose setcover_order.json already exists.

set -e

HAND="${1:-inspire}"
N="${2:-100}"
VERSION="robothome"

PYTHON=/home/mingi/miniconda3/envs/mingi/bin/python
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OBJ_ROOT_DIR="/home/mingi/shared_data/AutoDex/object/robothome"
OBJ_LIST="$REPO_ROOT/src/grasp_generation/obj_list_robothome.txt"

cd "$REPO_ROOT"

echo "=== Step 1: Set cover ordering ==="
"$PYTHON" src/grasp_generation/order/compute_order.py \
    --hand "$HAND" --version "$VERSION" \
    --obj_root_dir "$OBJ_ROOT_DIR" \
    --obj_list_file "$OBJ_LIST"

echo
echo "=== Step 2: Extract top $N ==="
"$PYTHON" src/grasp_generation/order/extract_selected.py \
    --hand "$HAND" --version "$VERSION" --n "$N"

echo
echo "Done. Selected candidates at: $REPO_ROOT/candidates/$HAND/selected_$N/"
