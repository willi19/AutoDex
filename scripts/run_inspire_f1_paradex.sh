#!/bin/bash
# Run BODex inspire_f1 on all paradex objects (box + shelf + wall).
set -e
cd /home/mingi/AutoDex/src/grasp_generation/BODex
PYTHON=/home/mingi/miniconda3/envs/bodex/bin/python
OBJ_LIST=/home/mingi/AutoDex/src/grasp_generation/obj_list.txt

OBJ_ROOT=/home/mingi/shared_data/AutoDex/object/paradex
for scene in box shelf wall; do
    case "$scene" in
        shelf) parallel=10 ;;
        *)     parallel=20 ;;
    esac
    echo "[$(date '+%H:%M:%S')] === $scene (-w $parallel) ==="
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" generate.py \
        -c "sim_inspire_f1/paradex_${scene}.yml" -w "$parallel" \
        --obj_list_file "$OBJ_LIST" \
        --obj_root_dir "$OBJ_ROOT"
done

echo "[$(date '+%H:%M:%S')] BODex done. Output: bodex_outputs/inspire_f1/v3/"
