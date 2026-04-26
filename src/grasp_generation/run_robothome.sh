#!/bin/bash
# Run BODex grasp generation on all robothome objects.
#
# Usage:
#   bash src/grasp_generation/run_robothome.sh [HAND] [SCENE_TYPES...]
#
# Examples:
#   bash src/grasp_generation/run_robothome.sh                          # default: inspire_left, all 3 scenes
#   bash src/grasp_generation/run_robothome.sh allegro                  # allegro, all 3 scenes
#   bash src/grasp_generation/run_robothome.sh inspire_left box wall    # only box + wall
#   bash src/grasp_generation/run_robothome.sh inspire_f1               # inspire_f1, all 3 scenes
#
# Skip logic: BODex skips scenes where grasp_pose.npy already exists.
# Safe to interrupt and re-run.

set -e

HAND="${1:-inspire_left}"
shift || true
SCENE_TYPES=("$@")
[ ${#SCENE_TYPES[@]} -eq 0 ] && SCENE_TYPES=(box shelf wall)

PYTHON=/home/mingi/miniconda3/envs/bodex/bin/python
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OBJ_LIST="$REPO_ROOT/src/grasp_generation/obj_list_robothome.txt"
OBJ_ROOT_DIR="/home/mingi/shared_data/AutoDex/object/robothome"

case "$HAND" in
    allegro)      CFG_DIR="sim_allegro" ;;
    inspire)      CFG_DIR="sim_inspire" ;;
    inspire_left) CFG_DIR="sim_inspire_left" ;;
    inspire_f1)   CFG_DIR="sim_inspire_f1" ;;
    *) echo "Unknown hand: $HAND. Use: allegro, inspire, inspire_left, inspire_f1." >&2; exit 1 ;;
esac

cd "$REPO_ROOT/src/grasp_generation/BODex"

echo "================================================"
echo "Hand:        $HAND"
echo "Scenes:      ${SCENE_TYPES[*]}"
echo "Obj list:    $OBJ_LIST"
echo "Obj root:    $OBJ_ROOT_DIR"
echo "================================================"

for scene in "${SCENE_TYPES[@]}"; do
    cfg="${CFG_DIR}/paradex_${scene}.yml"
    # Shelf is heavier per-scene; box/wall safe at -w 20.
    case "$scene" in
        shelf) parallel=10 ;;
        *)     parallel=20 ;;
    esac

    echo
    echo "[$(date '+%H:%M:%S')] === $scene  (config=$cfg, parallel=$parallel) ==="
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" generate.py \
        -c "$cfg" -w "$parallel" \
        --obj_list_file "$OBJ_LIST" \
        --obj_root_dir "$OBJ_ROOT_DIR"
done

echo
echo "[$(date '+%H:%M:%S')] All done. Output at: $REPO_ROOT/bodex_outputs/$HAND/v3/"
