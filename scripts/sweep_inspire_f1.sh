#!/bin/bash
# Sweep multiple inspire_f1 config variants on Jp_Water box only (faster).
# Each variant: modify spheres yml + manip yml, run BODex (box only) + sim_filter, log result.
set -e

OBJ="Jp_Water"
HAND="inspire_f1"
VERSION="v3"
OBJ_ROOT="/home/mingi/shared_data/AutoDex/object/robothome"
REPO=/home/mingi/AutoDex
BODEX_DIR="$REPO/src/grasp_generation/BODex"
SPHERES_BODEX="$BODEX_DIR/src/curobo/content/configs/robot/spheres/inspire_f1.yml"
SPHERES_PLANNER="$REPO/autodex/planner/src/curobo/content/configs/robot/spheres/inspire_f1.yml"
MANIP_DIR="$BODEX_DIR/src/curobo/content/configs/manip/sim_inspire_f1"
RESULT_LOG="/tmp/sweep_inspire_f1_results.txt"
echo "Variant,Total,Success,SceneColl,PhysFail,Rate" > "$RESULT_LOG"

# Backup original sphere yml + manip yml so we can restore between variants
BACKUP=/tmp/inspire_f1_backup
mkdir -p "$BACKUP"
cp "$SPHERES_BODEX" "$BACKUP/spheres.yml"
cp "$MANIP_DIR/paradex_box.yml" "$BACKUP/paradex_box.yml"
cp "$MANIP_DIR/paradex_shelf.yml" "$BACKUP/paradex_shelf.yml"
cp "$MANIP_DIR/paradex_wall.yml" "$BACKUP/paradex_wall.yml"

restore() {
    cp "$BACKUP/spheres.yml" "$SPHERES_BODEX"
    cp "$BACKUP/spheres.yml" "$SPHERES_PLANNER"
    cp "$BACKUP/paradex_box.yml" "$MANIP_DIR/paradex_box.yml"
    cp "$BACKUP/paradex_shelf.yml" "$MANIP_DIR/paradex_shelf.yml"
    cp "$BACKUP/paradex_wall.yml" "$MANIP_DIR/paradex_wall.yml"
}

LIST=$(mktemp)
trap "rm -f $LIST; restore" EXIT
echo "$OBJ" > "$LIST"

run_variant() {
    local name="$1"
    echo "================================================"
    echo "=== Variant: $name ==="
    echo "================================================"
    rm -rf "$REPO/bodex_outputs/$HAND/$VERSION/$OBJ"

    cd "$BODEX_DIR"
    CUDA_VISIBLE_DEVICES=0 /home/mingi/miniconda3/envs/bodex/bin/python generate.py \
        -c sim_inspire_f1/paradex_box.yml -w 10 \
        --obj_list_file "$LIST" --obj_root_dir "$OBJ_ROOT"

    cd "$REPO"
    /home/mingi/miniconda3/envs/mingi/bin/python src/grasp_generation/sim_filter/run_sim_filter.py \
        --hand "$HAND" --version "$VERSION" --obj "$OBJ" --obj_root_dir "$OBJ_ROOT"

    total=$(find "$REPO/bodex_outputs/$HAND/$VERSION/$OBJ" -name sim_eval.json | wc -l)
    succ=$(grep -lr '"success": true' "$REPO/bodex_outputs/$HAND/$VERSION/$OBJ" --include sim_eval.json 2>/dev/null | wc -l)
    coll=$(grep -lr scene_collision "$REPO/bodex_outputs/$HAND/$VERSION/$OBJ" --include sim_eval.json 2>/dev/null | wc -l)
    phys=$((total - succ - coll))
    rate=$(awk -v s=$succ -v t=$total 'BEGIN{if(t>0) printf "%.2f%%", s*100/t; else printf "0"}')
    echo "$name,$total,$succ,$coll,$phys,$rate" >> "$RESULT_LOG"
    echo "[$name] total=$total succ=$succ rate=$rate"
}

# === Variant A: baseline (current state — sphere 13 +z 2mm, thumb 10 +z 1mm, no ring constraint) ===
run_variant "A_current"

# === Variant B: revert spheres only (sphere 11/14 + thumb_12 stays?) — actually use 'pristine' starting state ===
# Pristine = backup (which has thumb=12, fingers=13, with +z shifts already applied + ring constraint removed).
# Let's define cleaner variants:

# Variant C: revert sphere positions (use original spheres before our +z shifts), keep ring constraint removed.
# Need original sphere positions: thumb_force_sensor/11 (idx 11 from initial yml) and fingers idx 14.
# We don't have that backup, so reconstruct via python:
python3 << 'EOF'
import yaml, copy
path = "/home/mingi/AutoDex/src/grasp_generation/BODex/src/curobo/content/configs/robot/spheres/inspire_f1.yml"
sd = yaml.safe_load(open(path))
sp = sd['collision_spheres']
# Revert thumb idx 10 z (+0.001 was applied) — set back to -0.004
sp['thumb_force_sensor'][10]['center'][2] = -0.004
# Revert finger idx 13 z (+0.002 was applied) — set back
for f in ["index_force_sensor","middle_force_sensor","ring_force_sensor","little_force_sensor"]:
    sp[f][13]['center'][2] = -0.005 if f in ("index_force_sensor","little_force_sensor") else -0.0045
with open(path, 'w') as fp:
    yaml.safe_dump(sd, fp, default_flow_style=False, sort_keys=False)
import shutil
shutil.copy(path, "/home/mingi/AutoDex/autodex/planner/src/curobo/content/configs/robot/spheres/inspire_f1.yml")
EOF
# Manip config: use idx 11/14 (truly original)
sed -i "s|thumb_force_sensor/10|thumb_force_sensor/11|g; s|index_force_sensor/13|index_force_sensor/14|g; s|middle_force_sensor/13|middle_force_sensor/14|g; s|ring_force_sensor/13|ring_force_sensor/14|g; s|little_force_sensor/13|little_force_sensor/14|g" "$MANIP_DIR/paradex_box.yml"
run_variant "B_orig_idx_no_ring"

# Restore baseline (current with shifts + thumb 10 / finger 13 + no ring)
restore
run_variant "C_shifted_idx_no_ring_repeat"

# Variant D: sphere shifts kept, but ring constraint restored
restore
python3 -c "
import re
p = '$MANIP_DIR/paradex_box.yml'
s = open(p).read()
s = s.replace('[[0, 1], 0.4],\n      ]', '[[0, 1], 0.4],\n        [[3], 0.5],\n      ]')
open(p, 'w').write(s)
"
run_variant "D_shifted_idx_with_ring"

# Variant E: revert spheres back to true original (idx 11/14, no shift) WITH ring constraint
python3 << 'EOF'
import yaml
path = "/home/mingi/AutoDex/src/grasp_generation/BODex/src/curobo/content/configs/robot/spheres/inspire_f1.yml"
sd = yaml.safe_load(open(path))
sp = sd['collision_spheres']
sp['thumb_force_sensor'][10]['center'][2] = -0.004
for f in ["index_force_sensor","middle_force_sensor","ring_force_sensor","little_force_sensor"]:
    sp[f][13]['center'][2] = -0.005 if f in ("index_force_sensor","little_force_sensor") else -0.0045
with open(path, 'w') as fp:
    yaml.safe_dump(sd, fp, default_flow_style=False, sort_keys=False)
import shutil
shutil.copy(path, "/home/mingi/AutoDex/autodex/planner/src/curobo/content/configs/robot/spheres/inspire_f1.yml")
EOF
sed -i "s|thumb_force_sensor/10|thumb_force_sensor/11|g; s|index_force_sensor/13|index_force_sensor/14|g; s|middle_force_sensor/13|middle_force_sensor/14|g; s|ring_force_sensor/13|ring_force_sensor/14|g; s|little_force_sensor/13|little_force_sensor/14|g" "$MANIP_DIR/paradex_box.yml"
run_variant "E_orig_idx_with_ring"

# Variant F: thumb 12 (4,5 mid) + finger 14 (current default before z shift) — like best of earlier (succ=54)
restore
python3 << 'EOF'
import yaml
path = "/home/mingi/AutoDex/src/grasp_generation/BODex/src/curobo/content/configs/robot/spheres/inspire_f1.yml"
sd = yaml.safe_load(open(path))
sp = sd['collision_spheres']
sp['thumb_force_sensor'][10]['center'][2] = -0.004
for f in ["index_force_sensor","middle_force_sensor","ring_force_sensor","little_force_sensor"]:
    sp[f][13]['center'][2] = -0.005 if f in ("index_force_sensor","little_force_sensor") else -0.0045
with open(path, 'w') as fp:
    yaml.safe_dump(sd, fp, default_flow_style=False, sort_keys=False)
import shutil
shutil.copy(path, "/home/mingi/AutoDex/autodex/planner/src/curobo/content/configs/robot/spheres/inspire_f1.yml")
EOF
sed -i "s|thumb_force_sensor/10|thumb_force_sensor/12|g; s|index_force_sensor/13|index_force_sensor/14|g; s|middle_force_sensor/13|middle_force_sensor/14|g; s|ring_force_sensor/13|ring_force_sensor/14|g; s|little_force_sensor/13|little_force_sensor/14|g" "$MANIP_DIR/paradex_box.yml"
run_variant "F_thumb12_finger14_no_ring"

echo
echo "================================================"
echo "=== Sweep summary ==="
echo "================================================"
cat "$RESULT_LOG"
