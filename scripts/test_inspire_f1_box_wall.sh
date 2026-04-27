#!/bin/bash
set -e
OBJ="${1:-Jp_Water}"
HAND="inspire_f1"
VERSION="v3"
OBJ_ROOT="/home/mingi/shared_data/AutoDex/object/robothome"
REPO=/home/mingi/AutoDex

rm -rf "$REPO/bodex_outputs/$HAND/$VERSION/$OBJ"
LIST=$(mktemp); trap "rm -f $LIST" EXIT
echo "$OBJ" > "$LIST"

cd "$REPO/src/grasp_generation/BODex"
for scene in box wall; do
    echo "  -- $scene --"
    CUDA_VISIBLE_DEVICES=0 /home/mingi/miniconda3/envs/bodex/bin/python generate.py \
        -c "sim_inspire_f1/paradex_${scene}.yml" -w 10 \
        --obj_list_file "$LIST" --obj_root_dir "$OBJ_ROOT"
done

cd "$REPO"
/home/mingi/miniconda3/envs/mingi/bin/python src/grasp_generation/sim_filter/run_sim_filter.py \
    --hand "$HAND" --version "$VERSION" --obj "$OBJ" --obj_root_dir "$OBJ_ROOT"

echo
echo "=== Results for $OBJ (box+wall) ==="
total=$(find "$REPO/bodex_outputs/$HAND/$VERSION/$OBJ" -name sim_eval.json | wc -l)
succ=$(grep -lr '"success": true' "$REPO/bodex_outputs/$HAND/$VERSION/$OBJ" --include sim_eval.json 2>/dev/null | wc -l)
coll=$(grep -lr scene_collision "$REPO/bodex_outputs/$HAND/$VERSION/$OBJ" --include sim_eval.json 2>/dev/null | wc -l)
echo "Total: $total  Success: $succ  Coll: $coll  PhysFail: $((total-succ-coll))"
[ "$total" -gt 0 ] && echo "Rate: $(awk -v s=$succ -v t=$total 'BEGIN{printf "%.2f%%", s*100/t}')"
