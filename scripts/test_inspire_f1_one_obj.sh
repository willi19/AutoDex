#!/bin/bash
# BODex + sim_filter for one object, then report success count.
# Usage: bash scripts/test_inspire_f1_one_obj.sh [OBJ_NAME]
set -e

OBJ="${1:-Jp_Water}"
HAND="inspire_f1"
VERSION="v3"
OBJ_ROOT="/home/mingi/shared_data/AutoDex/object/robothome"
REPO=/home/mingi/AutoDex

echo "=== Cleaning previous outputs for $OBJ ==="
rm -rf "$REPO/bodex_outputs/$HAND/$VERSION/$OBJ"

# One-object obj_list
LIST=$(mktemp)
trap "rm -f $LIST" EXIT
echo "$OBJ" > "$LIST"

echo "=== BODex generate (box, shelf, wall) ==="
cd "$REPO/src/grasp_generation/BODex"
for scene in box shelf wall; do
    echo "  -- $scene --"
    case "$scene" in
        shelf) parallel=5 ;;
        *)     parallel=10 ;;
    esac
    CUDA_VISIBLE_DEVICES=0 /home/mingi/miniconda3/envs/bodex/bin/python generate.py \
        -c "sim_inspire_f1/paradex_${scene}.yml" -w "$parallel" \
        --obj_list_file "$LIST" \
        --obj_root_dir "$OBJ_ROOT"
done

echo "=== sim_filter ==="
cd "$REPO"
/home/mingi/miniconda3/envs/mingi/bin/python src/grasp_generation/sim_filter/run_sim_filter.py \
    --hand "$HAND" --version "$VERSION" --obj "$OBJ" \
    --obj_root_dir "$OBJ_ROOT"

echo
echo "=== Results for $OBJ ==="
total=$(find "$REPO/bodex_outputs/$HAND/$VERSION/$OBJ" -name sim_eval.json 2>/dev/null | wc -l)
succ=$(grep -lr '"success": true' "$REPO/bodex_outputs/$HAND/$VERSION/$OBJ" --include sim_eval.json 2>/dev/null | wc -l)
coll=$(grep -lr 'scene_collision' "$REPO/bodex_outputs/$HAND/$VERSION/$OBJ" --include sim_eval.json 2>/dev/null | wc -l)
echo "Total seeds:    $total"
echo "Success:        $succ"
echo "Scene coll:     $coll"
echo "Physics fail:   $((total - succ - coll))"
if [ "$total" -gt 0 ]; then
    echo "Rate:           $(awk -v s=$succ -v t=$total 'BEGIN{printf "%.2f%%", s*100/t}')"
fi
