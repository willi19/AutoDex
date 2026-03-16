#!/bin/bash
# Benchmark all perception models, each in its correct conda env.
#
# Usage:
#   bash src/validation/execution/benchmark.sh \
#       /home/mingi/shared_data/RSS2026_Mingi/experiment/selected_100/attached_container/20260121_163623 \
#       /home/mingi/shared_data/object_6d/data/mesh/attached_container/attached_container.obj

set -e

CAPTURE_DIR="$1"
MESH="$2"
SERIAL="${3:-}"  # optional

if [ -z "$CAPTURE_DIR" ] || [ -z "$MESH" ]; then
    echo "Usage: $0 <capture_dir> <mesh_path> [serial]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH="$SCRIPT_DIR/benchmark.py"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

SERIAL_ARG=""
if [ -n "$SERIAL" ]; then
    SERIAL_ARG="--serial $SERIAL"
fi

echo "============================================================"
echo "Perception Benchmark"
echo "  Capture: $CAPTURE_DIR"
echo "  Mesh:    $MESH"
echo "============================================================"
echo ""

# YOLOE — foundationpose env
echo ">>> [1/7] YOLOE (foundationpose env)"
conda run --no-capture-output --cwd "$PROJECT_ROOT" -n foundationpose \
    python -u "$BENCH" --model yoloe --capture_dir "$CAPTURE_DIR" --mesh "$MESH" $SERIAL_ARG || \
    echo "  YOLOE FAILED"
echo ""

# SAM3 — sam3 env
echo ">>> [2/7] SAM3 (sam3 env)"
conda run --no-capture-output --cwd "$PROJECT_ROOT" -n sam3 \
    python -u "$BENCH" --model sam3 --capture_dir "$CAPTURE_DIR" --mesh "$MESH" $SERIAL_ARG || \
    echo "  SAM3 FAILED"
echo ""

# DA3 — sam3 env
echo ">>> [3/7] Depth-Anything-3 (sam3 env)"
conda run --no-capture-output --cwd "$PROJECT_ROOT" -n sam3 \
    python -u "$BENCH" --model da3 --capture_dir "$CAPTURE_DIR" --mesh "$MESH" $SERIAL_ARG || \
    echo "  DA3 FAILED"
echo ""

# Stereo PyTorch — sam3 env (has omegaconf)
echo ">>> [4/7] FoundationStereo PyTorch (sam3 env)"
conda run --no-capture-output --cwd "$PROJECT_ROOT" -n sam3 \
    python -u "$BENCH" --model stereo_pytorch --capture_dir "$CAPTURE_DIR" --mesh "$MESH" $SERIAL_ARG || \
    echo "  Stereo PyTorch FAILED"
echo ""

# Stereo TRT — foundation_stereo env
echo ">>> [5/7] FoundationStereo TRT (foundation_stereo env)"
conda run --no-capture-output --cwd "$PROJECT_ROOT" -n foundation_stereo \
    python -u "$BENCH" --model stereo_trt --capture_dir "$CAPTURE_DIR" --mesh "$MESH" $SERIAL_ARG || \
    echo "  Stereo TRT FAILED"
echo ""

# FoundationPose — foundationpose env
echo ">>> [6/7] FoundationPose (foundationpose env)"
conda run --no-capture-output --cwd "$PROJECT_ROOT" -n foundationpose \
    python -u "$BENCH" --model fpose --capture_dir "$CAPTURE_DIR" --mesh "$MESH" $SERIAL_ARG || \
    echo "  FoundationPose FAILED"
echo ""

# Silhouette — foundationpose env
echo ">>> [7/7] Silhouette rendering (foundationpose env)"
conda run --no-capture-output --cwd "$PROJECT_ROOT" -n foundationpose \
    python -u "$BENCH" --model silhouette --capture_dir "$CAPTURE_DIR" --mesh "$MESH" $SERIAL_ARG || \
    echo "  Silhouette FAILED"
echo ""

echo "============================================================"
echo "All benchmarks complete."