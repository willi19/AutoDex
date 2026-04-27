#!/bin/bash
set -e
OBJ="Jp_Water"
HAND="inspire_f1"
VERSION="v3"
OBJ_ROOT="/home/mingi/shared_data/AutoDex/object/robothome"
REPO=/home/mingi/AutoDex
MANIP_DIR="$REPO/src/grasp_generation/BODex/src/curobo/content/configs/manip/sim_inspire_f1"
RESULT="/tmp/sweep_ring.txt"
echo "Variant,Total,Success,Coll,PhysFail,Rate" > "$RESULT"

LIST=$(mktemp); trap "rm -f $LIST" EXIT
echo "$OBJ" > "$LIST"

run() {
    name="$1"
    rm -rf "$REPO/bodex_outputs/$HAND/$VERSION/$OBJ"
    cd "$REPO/src/grasp_generation/BODex"
    for scene in box wall; do
        CUDA_VISIBLE_DEVICES=0 /home/mingi/miniconda3/envs/bodex/bin/python generate.py \
            -c "sim_inspire_f1/paradex_${scene}.yml" -w 10 \
            --obj_list_file "$LIST" --obj_root_dir "$OBJ_ROOT"
    done
    cd "$REPO"
    /home/mingi/miniconda3/envs/mingi/bin/python src/grasp_generation/sim_filter/run_sim_filter.py \
        --hand "$HAND" --version "$VERSION" --obj "$OBJ" --obj_root_dir "$OBJ_ROOT"
    total=$(find "$REPO/bodex_outputs/$HAND/$VERSION/$OBJ" -name sim_eval.json | wc -l)
    succ=$(grep -lr '"success": true' "$REPO/bodex_outputs/$HAND/$VERSION/$OBJ" --include sim_eval.json 2>/dev/null | wc -l)
    coll=$(grep -lr scene_collision "$REPO/bodex_outputs/$HAND/$VERSION/$OBJ" --include sim_eval.json 2>/dev/null | wc -l)
    rate=$(awk -v s=$succ -v t=$total 'BEGIN{if(t>0) printf "%.2f%%", s*100/t}')
    echo "$name,$total,$succ,$coll,$((total-succ-coll)),$rate" >> "$RESULT"
    echo "[$name] $succ/$total ($rate)"
}

# Ensure original setup (thumb 11, finger 14)
sed -i "s|thumb_force_sensor/[0-9]*|thumb_force_sensor/11|g; s|index_force_sensor/[0-9]*|index_force_sensor/14|g; s|middle_force_sensor/[0-9]*|middle_force_sensor/14|g; s|ring_force_sensor/[0-9]*|ring_force_sensor/14|g; s|little_force_sensor/[0-9]*|little_force_sensor/14|g" "$MANIP_DIR"/*.yml

# Variant 1: ring constraint present (original)
for f in "$MANIP_DIR"/*.yml; do
    python3 -c "
s=open('$f').read()
if '[[3], 0.5]' not in s:
    s=s.replace('[[0, 1], 0.4],\n      ]', '[[0, 1], 0.4],\n        [[3], 0.5],\n      ]')
open('$f','w').write(s)
"
done
run "with_ring"

# Variant 2: ring removed
for f in "$MANIP_DIR"/*.yml; do
    python3 -c "
import re
s=open('$f').read()
s=re.sub(r'        \[\[3\], 0\.5\],\n', '', s)
open('$f','w').write(s)
"
done
run "no_ring"

# Restore ring (clean state)
for f in "$MANIP_DIR"/*.yml; do
    python3 -c "
s=open('$f').read()
if '[[3], 0.5]' not in s:
    s=s.replace('[[0, 1], 0.4],\n      ]', '[[0, 1], 0.4],\n        [[3], 0.5],\n      ]')
open('$f','w').write(s)
"
done

echo
echo "=== Summary ==="
cat "$RESULT"
