#!/bin/bash
# Perception evaluation pipeline.
#
# Steps:
#   1. SAM3 mask (sam3 env)
#   2a. DA3 depth (sam3 env)
#   2b. Stereo TRT depth (foundation_stereo env)
#   3. FPose register all 24 views with DA3 (foundationpose env)
#   4. NMS + silhouette matching → near-GT pose (foundationpose env)
#   5. Rank views, run FPose with stereo, select best 5 (foundationpose env)
#   6. Verify: rerun best 5 → NMS+sil → compare with GT (foundationpose env)
#
# Usage:
#   bash run_eval.sh <data_root> [obj_name] [episode]

set -e

DATA_ROOT="$1"
OBJ="${2:-}"
EPISODE="${3:-}"
if [ -z "$DATA_ROOT" ]; then
    echo "Usage: $0 <data_root> [obj_name] [episode]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Build list of (obj, episode) pairs
PAIRS=()
if [ -n "$OBJ" ] && [ -n "$EPISODE" ]; then
    PAIRS=("$OBJ/$EPISODE")
elif [ -n "$OBJ" ]; then
    for ep in "$DATA_ROOT/$OBJ"/*/; do
        ep_name=$(basename "$ep")
        PAIRS+=("$OBJ/$ep_name")
    done
else
    for obj_dir in "$DATA_ROOT"/*/; do
        obj_name=$(basename "$obj_dir")
        [ "$obj_name" = "cam_param" ] && continue
        for ep in "$obj_dir"*/; do
            ep_name=$(basename "$ep")
            PAIRS+=("$obj_name/$ep_name")
        done
    done
fi

echo "============================================================"
echo "Perception Evaluation Pipeline"
echo "  Data:     $DATA_ROOT"
echo "  Episodes: ${#PAIRS[@]}"
echo "============================================================"

# Step 0: Undistort raw images (any env with cv2)
echo ""
echo ">>> Step 0: Undistort raw images"
conda run --no-capture-output --cwd "$PROJECT_ROOT" -n sam3 \
    python -u "$SCRIPT_DIR/step0_undistort.py" \
    --data_root "$DATA_ROOT" || echo "  UNDISTORT FAILED"

# Step 1 + 2a: SAM3 masks + DA3 depth (sam3 env)
echo ""
echo ">>> Step 1+2a: SAM3 masks + DA3 depth (sam3 env)"
for pair in "${PAIRS[@]}"; do
    OBJ_NAME="${pair%%/*}"
    EP_NAME="${pair##*/}"
    echo "  --- $OBJ_NAME/$EP_NAME ---"
    conda run --no-capture-output --cwd "$PROJECT_ROOT" -n sam3 \
        python -u "$SCRIPT_DIR/step1_mask.py" \
        --data_root "$DATA_ROOT" --obj "$OBJ_NAME" --episode "$EP_NAME" \
        || echo "  MASK FAILED"

    conda run --no-capture-output --cwd "$PROJECT_ROOT" -n dav3 \
        python -u "$SCRIPT_DIR/step2_depth_da3.py" \
        --data_root "$DATA_ROOT" --obj "$OBJ_NAME" --episode "$EP_NAME" \
        || echo "  DA3 FAILED"
done

# Step 2b: Stereo TRT depth (foundation_stereo env)
echo ""
echo ">>> Step 2b: Stereo TRT depth (foundation_stereo env)"
for pair in "${PAIRS[@]}"; do
    OBJ_NAME="${pair%%/*}"
    EP_NAME="${pair##*/}"
    echo "  --- $OBJ_NAME/$EP_NAME ---"
    conda run --no-capture-output --cwd "$PROJECT_ROOT" -n foundation_stereo \
        python -u "$SCRIPT_DIR/step2_depth_stereo.py" \
        --data_root "$DATA_ROOT" --obj "$OBJ_NAME" --episode "$EP_NAME" \
        || echo "  STEREO FAILED"
done

# Step 3-6: FPose + NMS + sil + evaluate + verify (foundationpose env)
echo ""
echo ">>> Steps 3-6: FPose + silhouette + evaluate + verify (foundationpose env)"
for pair in "${PAIRS[@]}"; do
    OBJ_NAME="${pair%%/*}"
    EP_NAME="${pair##*/}"
    echo "  --- $OBJ_NAME/$EP_NAME ---"

    echo "  Step 3: FPose register all views (DA3 depth)"
    conda run --no-capture-output --cwd "$PROJECT_ROOT" -n foundationpose \
        python -u "$SCRIPT_DIR/step3_pose.py" \
        --data_root "$DATA_ROOT" --obj "$OBJ_NAME" --episode "$EP_NAME" \
        || echo "  POSE FAILED"

    echo "  Step 4: NMS + silhouette → GT pose"
    conda run --no-capture-output --cwd "$PROJECT_ROOT" -n foundationpose \
        python -u "$SCRIPT_DIR/step4_silhouette.py" \
        --data_root "$DATA_ROOT" --obj "$OBJ_NAME" --episode "$EP_NAME" \
        || echo "  SILHOUETTE FAILED"

    echo "  Step 5: Rank views + FPose with stereo"
    conda run --no-capture-output --cwd "$PROJECT_ROOT" -n foundationpose \
        python -u "$SCRIPT_DIR/step5_evaluate.py" \
        --data_root "$DATA_ROOT" --obj "$OBJ_NAME" --episode "$EP_NAME" \
        || echo "  EVALUATE FAILED"

    echo "  Step 6: Verify best 5 views"
    conda run --no-capture-output --cwd "$PROJECT_ROOT" -n foundationpose \
        python -u "$SCRIPT_DIR/step6_verify.py" \
        --data_root "$DATA_ROOT" --obj "$OBJ_NAME" --episode "$EP_NAME" \
        || echo "  VERIFY FAILED"
done

echo ""
echo "============================================================"
echo "All done. Results saved in capture directories under $DATA_ROOT"