#!/bin/bash
# Run sim_filter on robothome objects with any BODex output.
#
# Usage:
#   bash src/grasp_generation/run_sim_filter_robothome.sh [HAND]
#
# Default HAND is inspire_left.
# Safe to run while BODex is still going. sim_filter scans whatever scenes
# already exist under bodex_outputs/{HAND}/v3/{obj}/, so partial completion
# is fine. Re-running picks up new scenes via per-seed skip logic
# (sim_eval.json existence).

set -e

HAND="${1:-inspire_left}"
VERSION="v3"
WORKERS=8

PYTHON=/home/mingi/miniconda3/envs/mingi/bin/python
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OBJ_ROOT_DIR="/home/mingi/shared_data/AutoDex/object/robothome"
ALL_LIST="$REPO_ROOT/src/grasp_generation/obj_list_robothome.txt"
READY_LIST=$(mktemp)
trap 'rm -f "$READY_LIST"' EXIT

# Object is "ready" if its bodex output dir exists with at least one scene.
echo "=== Scanning BODex outputs ($HAND/$VERSION) ==="
ready_count=0
empty_count=0
for obj in $(cat "$ALL_LIST"); do
    bodex_obj="$REPO_ROOT/bodex_outputs/$HAND/$VERSION/$obj"
    n_scenes=0
    if [ -d "$bodex_obj" ]; then
        for st in box shelf wall; do
            d="$bodex_obj/$st"
            [ -d "$d" ] && n_scenes=$((n_scenes + $(ls "$d" 2>/dev/null | wc -l)))
        done
    fi
    if [ "$n_scenes" -gt 0 ]; then
        echo "$obj" >> "$READY_LIST"
        ready_count=$((ready_count + 1))
    else
        empty_count=$((empty_count + 1))
    fi
done

echo "Has bodex output: $ready_count   Empty: $empty_count"
[ "$ready_count" -eq 0 ] && { echo "Nothing to do."; exit 0; }

echo
echo "=== Running sim_filter (HAND=$HAND, WORKERS=$WORKERS) ==="

cd "$REPO_ROOT"
"$PYTHON" src/grasp_generation/sim_filter/run_sim_filter.py \
    --hand "$HAND" --version "$VERSION" \
    --workers "$WORKERS" \
    --obj_root_dir "$OBJ_ROOT_DIR" \
    --obj_list_file "$READY_LIST"
