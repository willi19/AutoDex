#!/bin/bash
# Run mask generation for all objects under inspire_f1.
# YOLOE first (fast), then SAM3 for any videos YOLOE missed.
#
# Usage:
#   bash src/perception/run_mask_all.sh [GPU] [MODE] [SERIALS...]
#   MODE: "both" (default), "yoloe", or "sam3"
#
# Examples:
#   bash src/perception/run_mask_all.sh 0                              # all cameras
#   bash src/perception/run_mask_all.sh 0 both 22684755 23263780       # two cameras only

GPU="${1:-0}"
MODE="${2:-both}"
shift 2 2>/dev/null
SERIALS="$@"

BASE="/home/mingi/paradex1/capture/eccv2026/inspire_f1"

SERIAL_ARGS=""
if [ -n "$SERIALS" ]; then
    SERIAL_ARGS="--serials $SERIALS"
fi

eval "$(conda shell.bash hook)"

if [ "$MODE" != "sam3" ]; then
    echo "===== YOLOE pass ====="
    conda activate foundationpose
    python -u src/perception/batch_mask_yoloe.py --base "$BASE" --gpu "$GPU" $SERIAL_ARGS
fi

if [ "$MODE" != "yoloe" ]; then
    echo "===== SAM3 pass (fallback) ====="
    conda activate sam3
    python -u src/perception/batch_mask.py --base "$BASE" --gpu "$GPU" $SERIAL_ARGS
fi

echo "All done!"
