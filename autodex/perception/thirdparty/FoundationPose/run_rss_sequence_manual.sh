#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_rss_sequence_manual.sh --src-seq <sequence_path> --mesh-path <mesh_path> [options]

Options:
  --object-name <name>   Object name used for output folders (default: mesh filename).
  --scale <float>        Image scale factor (default: 0.5).
  --data-dir <path>      Override prepared data directory.
  --output-dir <path>    Override output directory.
  --overwrite            Overwrite prepared data directory if it exists.
EOF
}

SRC_SEQ=""
MESH_FILE=""
OBJECT_NAME=""
SCALE="0.5"
DATA_DIR=""
OUTPUT_DIR=""
OVERWRITE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src-seq)
      SRC_SEQ="${2:-}"
      shift 2
      ;;
    --mesh-path)
      MESH_FILE="${2:-}"
      shift 2
      ;;
    --object-name)
      OBJECT_NAME="${2:-}"
      shift 2
      ;;
    --scale)
      SCALE="${2:-}"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --overwrite)
      OVERWRITE="true"
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${SRC_SEQ}" || -z "${MESH_FILE}" ]]; then
  usage
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
  mesh_base="$(basename "${MESH_FILE}")"
  OBJECT_NAME="${mesh_base%.*}"
fi

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
SCALE_TAG="${SCALE//./p}"

if [[ -z "${DATA_DIR}" ]]; then
  DATA_DIR="${SCRIPT_DIR}/demo_data/rss2026/${OBJECT_NAME}/${SEQ_NAME}_s${SCALE_TAG}"
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${SCRIPT_DIR}/outputs/${OBJECT_NAME}_${SEQ_NAME}_pose"
fi

PREP_ARGS=(
  "${SCRIPT_DIR}/prepare_rss_sequence.py"
  --src "${SRC_SEQ}"
  --dst "${DATA_DIR}"
  --scale "${SCALE}"
)
if [[ "${OVERWRITE}" == "true" ]]; then
  PREP_ARGS+=(--overwrite)
fi

python "${PREP_ARGS[@]}"

MESH_FILE="${MESH_FILE}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
  "${SCRIPT_DIR}/run_demo_steps.sh" "${DATA_DIR}" "${OBJECT_NAME}"
