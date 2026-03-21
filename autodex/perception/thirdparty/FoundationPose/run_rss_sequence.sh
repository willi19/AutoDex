#!/usr/bin/env bash
set -euo pipefail

SRC_SEQ="${1:-}"
MESH_FILE="${2:-}"
OBJECT_NAME="${3:-}"
SCALE="${4:-0.5}"

if [[ -z "${SRC_SEQ}" || -z "${MESH_FILE}" ]]; then
  echo "Usage: $0 <sequence_path> <mesh_path> [object_name] [scale]"
  exit 1
fi

if [[ ! -d "${SRC_SEQ}" ]]; then
  echo "Sequence path not found: ${SRC_SEQ}"
  exit 1
fi

if [[ ! -f "${MESH_FILE}" ]]; then
  echo "Mesh file not found: ${MESH_FILE}"
  exit 1
fi

SEQ_NAME="$(basename "${SRC_SEQ}")"
if [[ -z "${OBJECT_NAME}" ]]; then
  OBJECT_NAME="$(basename "$(dirname "${SRC_SEQ}")")"
fi

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
SCALE_TAG="${SCALE//./p}"
DATA_DIR="${SCRIPT_DIR}/demo_data/rss2026/${OBJECT_NAME}/${SEQ_NAME}_s${SCALE_TAG}"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/${OBJECT_NAME}_${SEQ_NAME}_pose"

python "${SCRIPT_DIR}/prepare_rss_sequence.py" \
  --src "${SRC_SEQ}" \
  --dst "${DATA_DIR}" \
  --scale "${SCALE}"

MESH_FILE="${MESH_FILE}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
  "${SCRIPT_DIR}/run_demo_steps.sh" "${DATA_DIR}" "${OBJECT_NAME}"
