#!/bin/bash
# Perception validation pipeline
#
# Independently runs mask methods and depth methods, then combines for pose.
#
# Output structure:
#   validation_output/
#   ├── segmentation/sam3/      mask method 1
#   ├── segmentation/yoloe/     mask method 2
#   ├── depth/da3/              depth method 1
#   ├── depth/foundationstereo/ depth method 2 (optional)
#   └── pose/{seg}_{depth}/     pose results per combination
#
# Usage:
#   bash src/validation/perception/run_all.sh <data_dir> <mesh_dir>
#
# Args:
#   data_dir  : directory containing raw images + object_info.json
#   mesh_dir  : root dir containing {obj_name}/processed_data/mesh/simplified.obj

set -e

DATA_DIR="${1:?Usage: $0 <data_dir> <mesh_dir>}"
MESH_DIR="${2:?Usage: $0 <data_dir> <mesh_dir>}"
OUT="${DATA_DIR}/validation_output"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

SEG_SAM3="${OUT}/segmentation/sam3"
SEG_YOLOE="${OUT}/segmentation/yoloe"
DEPTH_DA3="${OUT}/depth/da3"

echo "============================================================"
echo "data_dir    : ${DATA_DIR}"
echo "mesh_dir    : ${MESH_DIR}"
echo "output      : ${OUT}"
echo "============================================================"

TOTAL_START=$SECONDS

# ============================================================
# Segmentation
# ============================================================
echo ""
echo "[Seg 1/2] SAM3"
conda activate sam3
python "${SCRIPT_DIR}/step1_mask.py" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${SEG_SAM3}" \
    --method sam3

echo ""
echo "[Seg 2/2] YOLO-E (reuses images from sam3)"
conda activate foundationpose
python "${SCRIPT_DIR}/step1_mask.py" \
    --reuse_images_from "${SEG_SAM3}" \
    --output_dir "${SEG_YOLOE}" \
    --method yoloe

# ============================================================
# Depth
# ============================================================
echo ""
echo "[Depth 1/1] DA3"
conda activate sam3
python "${SCRIPT_DIR}/step2_depth.py" \
    --output_dir "${DEPTH_DA3}" \
    --seg_dir "${SEG_SAM3}" \
    --method da3

DEPTH_FS="${OUT}/depth/foundationstereo"
echo ""
echo "[Depth 2/2] FoundationStereo (auto-pair by closest extrinsic)"
conda activate foundation_stereo
python "${SCRIPT_DIR}/step2_depth.py" \
    --output_dir "${DEPTH_FS}" \
    --seg_dir "${SEG_SAM3}" \
    --method stereo

# ============================================================
# Pose (all combinations)
# ============================================================
conda activate foundationpose

echo ""
echo "[Pose 1/4] sam3 + da3"
python "${SCRIPT_DIR}/step3_pose.py" \
    --output_dir "${OUT}/pose/sam3_da3" \
    --seg_dir "${SEG_SAM3}" \
    --depth_dir "${DEPTH_DA3}" \
    --mesh_dir "${MESH_DIR}"

echo ""
echo "[Pose 2/4] yoloe + da3"
python "${SCRIPT_DIR}/step3_pose.py" \
    --output_dir "${OUT}/pose/yoloe_da3" \
    --seg_dir "${SEG_YOLOE}" \
    --depth_dir "${DEPTH_DA3}" \
    --mesh_dir "${MESH_DIR}"

echo ""
echo "[Pose 3/4] sam3 + foundationstereo"
python "${SCRIPT_DIR}/step3_pose.py" \
    --output_dir "${OUT}/pose/sam3_fs" \
    --seg_dir "${SEG_SAM3}" \
    --depth_dir "${DEPTH_FS}" \
    --mesh_dir "${MESH_DIR}"

echo ""
echo "[Pose 4/4] yoloe + foundationstereo"
python "${SCRIPT_DIR}/step3_pose.py" \
    --output_dir "${OUT}/pose/yoloe_fs" \
    --seg_dir "${SEG_YOLOE}" \
    --depth_dir "${DEPTH_FS}" \
    --mesh_dir "${MESH_DIR}"

# ============================================================
# Compare
# ============================================================
echo ""
echo "[Compare]"
python "${SCRIPT_DIR}/step4_compare.py" \
    --output_base "${OUT}"

echo ""
echo "============================================================"
echo "Done! ($(( SECONDS - TOTAL_START ))s)"
echo "Output: ${OUT}"
echo "  compare/summary.csv"
echo "  compare/{serial}_compare.png"
echo "============================================================"