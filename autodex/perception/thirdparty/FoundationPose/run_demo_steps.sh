#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-/media/gunhee/DATA/robothome/FoundationPose/demo_data/clock_demo}"
OBJECT_NAME="${2:-}"
TEXT_PROMPT="${TEXT_PROMPT:-object on the checkerboard, excluding checkerboard}"
SAM3_REPO="${SAM3_REPO:-/media/gunhee/DATA/2025/3drecon/sam3}"
DA3_REPO="${DA3_REPO:-/media/gunhee/DATA/robothome/Depth-Anything-3}"
FOUNDATIONPOSE_REPO="${FOUNDATIONPOSE_REPO:-/media/gunhee/DATA/robothome/FoundationPose}"
ELOFTR_REPO="${ELOFTR_REPO:-/media/gunhee/DATA/robothome/EfficientLoFTR}"
ELOFTR_ENV="${ELOFTR_ENV:-eloftr}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

if [[ -z "${OBJECT_NAME}" ]]; then
  data_basename="$(basename "${DATA_DIR}")"
  if [[ "${data_basename}" == *_demo ]]; then
    OBJECT_NAME="${data_basename%_demo}"
  else
    OBJECT_NAME="clock"
  fi
fi

OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$0")/outputs/${OBJECT_NAME}_pose}"
MESH_PLY="${MESH_PLY:-${DATA_DIR}/mesh/${OBJECT_NAME}.ply}"
MESH_OBJ="${MESH_OBJ:-${DATA_DIR}/mesh/${OBJECT_NAME}.obj}"
MESH_FILE="${MESH_FILE:-}"

if [[ -z "${MESH_FILE}" ]]; then
  if [[ -f "${MESH_OBJ}" ]]; then
    MESH_FILE="${MESH_OBJ}"
  elif [[ -f "${MESH_PLY}" ]]; then
    MESH_FILE="${MESH_PLY}"
  else
    MESH_FILE="${MESH_OBJ}"
  fi
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH. Please initialize conda first."
  exit 1
fi

if [[ -n "${CONDA_BASE:-}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
else
  CONDA_BASE_FALLBACK="${HOME}/anaconda3"
  CONDA_BASE_DETECTED="$(CONDA_NO_PLUGINS=true conda info --base 2>/dev/null || true)"
  if [[ -n "${CONDA_BASE_DETECTED}" && -f "${CONDA_BASE_DETECTED}/etc/profile.d/conda.sh" ]]; then
    CONDA_SH="${CONDA_BASE_DETECTED}/etc/profile.d/conda.sh"
  elif [[ -f "${CONDA_BASE_FALLBACK}/etc/profile.d/conda.sh" ]]; then
    CONDA_SH="${CONDA_BASE_FALLBACK}/etc/profile.d/conda.sh"
  else
    echo "Unable to locate conda.sh. Set CONDA_BASE or initialize conda first."
    exit 1
  fi
fi

source "${CONDA_SH}"

echo "Step 1: SAM3 masks (env: sam3)"
conda activate sam3
export PYTHONPATH="${SAM3_REPO}:${PYTHONPATH:-}"
python "$(dirname "$0")/step1_sam3_masks.py" \
  --data-dir "$DATA_DIR" \
  --text-prompt "$TEXT_PROMPT"
conda deactivate

echo "Step 2: Depth Anything 3 (env: dav3)"
conda activate dav3
export PYTHONPATH="${DA3_REPO}:${PYTHONPATH:-}"
python "$(dirname "$0")/step2_depth_anything3.py" \
  --data-dir "$DATA_DIR"
conda deactivate

echo "Step 3: FoundationPose (env: foundationpose)"
conda activate foundationpose
export PYTHONPATH="${FOUNDATIONPOSE_REPO}:${PYTHONPATH:-}"
python "$(dirname "$0")/step3_foundationpose.py" \
  --data-dir "$DATA_DIR" \
  --object-name "$OBJECT_NAME" \
  --mesh-file "$MESH_FILE" \
  --output-dir "$OUTPUT_DIR"
conda deactivate

echo "Step 4: Pose NMS (env: foundationpose)"
conda activate foundationpose
export PYTHONPATH="${FOUNDATIONPOSE_REPO}:${PYTHONPATH:-}"
python "$(dirname "$0")/step4_nms_pose.py" \
  --output-dir "$OUTPUT_DIR" \
  --data-dir "$DATA_DIR" \
  --object-name "$OBJECT_NAME" \
  --mesh-file "$MESH_FILE"
conda deactivate

echo "Step 5: Silhouette refine (env: foundationpose)"
conda activate foundationpose
export PYTHONPATH="${FOUNDATIONPOSE_REPO}:${PYTHONPATH:-}"
python "$(dirname "$0")/step5_silhouette_refine.py" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --object-name "$OBJECT_NAME" \
  --mesh-file "$MESH_FILE"
conda deactivate

echo "All requested steps completed."
