#!/bin/bash
# Perception experiment visualization for selected_100.
#
# Runs all steps with appropriate conda environments.
# Each step skips already-processed captures.
#
# Usage:
#   bash src/demo/run_perception_exp.sh                          # all objects
#   bash src/demo/run_perception_exp.sh attached_container       # one object
#   bash src/demo/run_perception_exp.sh attached_container 20260121_163413  # one capture

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${SCRIPT_DIR}/perception_exp.py"

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

OBJ="${1:-}"
IDX="${2:-}"
EXTRA=""
[[ -n "$OBJ" ]] && EXTRA="$EXTRA --obj $OBJ"
[[ -n "$IDX" ]] && EXTRA="$EXTRA --idx $IDX"

echo "============================================================"
echo "Perception Experiment Visualization"
echo "  Object: ${OBJ:-all}"
echo "  Index:  ${IDX:-all}"
echo "  Output: $(cd "${SCRIPT_DIR}/../.." && pwd)/exp/perception/"
echo "============================================================"

TOTAL_START=$SECONDS

# ── Step 1: SAM3 masks ────────────────────────────────────────────────
echo ""
echo "[1/5] SAM3 masks (env: sam3)"
conda activate sam3
python "$SCRIPT" --step mask_sam3 $EXTRA

# ── Step 2: YOLO-E masks ──────────────────────────────────────────────
echo ""
echo "[2/5] YOLO-E masks (env: foundationpose)"
conda activate foundationpose
python "$SCRIPT" --step mask_yoloe $EXTRA

# ── Step 3: DA3 depth ─────────────────────────────────────────────────
echo ""
echo "[3/5] DA3 depth (env: sam3)"
conda activate sam3
python "$SCRIPT" --step depth_da3 $EXTRA

# ── Step 4: FoundationStereo depth ────────────────────────────────────
echo ""
echo "[4/5] FoundationStereo depth (env: foundationpose)"
conda activate foundationpose
python "$SCRIPT" --step depth_stereo $EXTRA

# ── Step 5: Pose visualization ────────────────────────────────────────
echo ""
echo "[5/5] Pose overlays (env: foundationpose)"
conda activate foundationpose
python "$SCRIPT" --step pose_viz $EXTRA

echo ""
echo "============================================================"
echo "Done! ($(( SECONDS - TOTAL_START ))s)"
echo "Output: $(cd "${SCRIPT_DIR}/../.." && pwd)/exp/perception/"
echo "============================================================"