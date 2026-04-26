#!/bin/bash
set -e
cd "$(dirname "$0")"

PYTHON=/home/mingi/miniconda3/envs/bodex/bin/python
OBJ_LIST="$(dirname "$0")/../obj_list.txt"

# Allegro
CUDA_VISIBLE_DEVICES=0 $PYTHON generate.py -c sim_allegro/paradex_box.yml -w 20 --obj_list_file "$OBJ_LIST"
CUDA_VISIBLE_DEVICES=0 $PYTHON generate.py -c sim_allegro/paradex_shelf.yml -w 20 --obj_list_file "$OBJ_LIST"
CUDA_VISIBLE_DEVICES=0 $PYTHON generate.py -c sim_allegro/paradex_wall.yml -w 20 --obj_list_file "$OBJ_LIST"

# Inspire
CUDA_VISIBLE_DEVICES=0 $PYTHON generate.py -c sim_inspire/paradex_box.yml -w 20 --obj_list_file "$OBJ_LIST"
CUDA_VISIBLE_DEVICES=0 $PYTHON generate.py -c sim_inspire/paradex_shelf.yml -w 20 --obj_list_file "$OBJ_LIST"
CUDA_VISIBLE_DEVICES=0 $PYTHON generate.py -c sim_inspire/paradex_wall.yml -w 20 --obj_list_file "$OBJ_LIST"

# Inspire F1
CUDA_VISIBLE_DEVICES=0 $PYTHON generate.py -c sim_inspire_f1/paradex_box.yml -w 20 --obj_list_file "$OBJ_LIST"
CUDA_VISIBLE_DEVICES=0 $PYTHON generate.py -c sim_inspire_f1/paradex_shelf.yml -w 20 --obj_list_file "$OBJ_LIST"
CUDA_VISIBLE_DEVICES=0 $PYTHON generate.py -c sim_inspire_f1/paradex_wall.yml -w 20 --obj_list_file "$OBJ_LIST"