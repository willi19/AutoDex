#!/usr/bin/env bash
# Manage distributed foundpose onboarding across robot + capture1-6.
#
# Usage:
#     bash scripts/foundpose_onboard.sh start    # launch all 7 PCs in parallel
#     bash scripts/foundpose_onboard.sh stop     # kill all batch_onboard procs
#     bash scripts/foundpose_onboard.sh status   # quick proc count per PC
#
# Monitor progress separately:
#     watch -n 5 python scripts/monitor_foundpose_onboard.py
set -euo pipefail

REF_JSON_REL="shared_data/AutoDex/experiment/selected_100/allegro/attached_container/20260330_164351/cam_param/intrinsics.json"
REF_CAM=24080331
OUT_REL="shared_data/AutoDex/foundpose_assets"
PY_REL="anaconda3/envs/gotrack_cu128/bin/python"
SCRIPT_REL="AutoDex/src/process/batch_onboard_foundpose.py"
WORKERS=3

declare -A CHUNKS=(
    [robot]="fruit_cutter_green fruit_cutter_light_green green_attached_container green_cactus_vase green_lamp green_soap_dispenser icecream_scoop jja_ramen knife_sharpner large_peg"
    [capture1]="lemon lemon_squeezer light_green_basket lilac_food_container magazine_file meat_thermometer metal_scoop_big metal_scoop_small mug_holder open_box"
    [capture2]="open_short_pringles orange organizer_beige paper_bowl paper_box paper_cup pastel_blue_cup pepper_tuna pepper_tuna_light pepsi"
    [capture3]="pepsi_light pingpong pink_clock plant_mister plant_pot potato_mesher pringles redcar rolling_pin screwdriver"
    [capture4]="servingbowl_small shoe_organizer smallbowl soap_dispenser soaptray spam_can spicemill spray_bottle standing_frame tea_case"
    [capture5]="tennis_ball thermo_clock tissue_box toilet_roll_holder_steel toothbrush_holder washing_brush washing_brush2 wateringcan white_candle_holder white_clock"
    [capture6]="white_hand_shower white_pen_cup white_plastic_box white_soap_dish white_table_lamp white_watering_can white_wood_handle_watering_can wood_organizer wood_tray_big wood_tray_small"
)

ACTION="${1:-status}"

run_cmd_for_pc() {
    local pc="$1"
    local objs="$2"
    local cmd="\$HOME/$PY_REL \$HOME/$SCRIPT_REL \
        --objects $objs \
        --reference-intrinsics-json \$HOME/$REF_JSON_REL \
        --reference-camera-id $REF_CAM \
        --output-root \$HOME/$OUT_REL \
        --workers $WORKERS"
    echo "$cmd"
}

case "$ACTION" in
    start)
        for pc in "${!CHUNKS[@]}"; do
            objs="${CHUNKS[$pc]}"
            CMD=$(run_cmd_for_pc "$pc" "$objs")
            if [ "$pc" = "robot" ]; then
                nohup bash -c "$CMD" >/tmp/onboard_${pc}.log 2>&1 &
                echo "[start] $pc pid=$! log=/tmp/onboard_${pc}.log"
            else
                ssh "$pc" "nohup bash -lc '$CMD' >/tmp/onboard_${pc}.log 2>&1 < /dev/null &" &
                echo "[start] $pc (ssh background)"
            fi
        done
        wait
        echo "[start] all 7 PCs launched. Monitor with:"
        echo "  watch -n 5 python scripts/monitor_foundpose_onboard.py"
        ;;
    stop)
        for pc in "${!CHUNKS[@]}"; do
            if [ "$pc" = "robot" ]; then
                pkill -f "onboard_custom_mesh|batch_onboard" || true
                echo "[stop] $pc killed"
            else
                ssh "$pc" "pkill -f 'onboard_custom_mesh|batch_onboard' || true" &
                echo "[stop] $pc kill sent"
            fi
        done
        wait
        ;;
    status)
        for pc in "${!CHUNKS[@]}"; do
            if [ "$pc" = "robot" ]; then
                n=$(pgrep -fc "onboard_custom_mesh|batch_onboard" || echo 0)
            else
                n=$(ssh -o ConnectTimeout=3 "$pc" "pgrep -fc 'onboard_custom_mesh|batch_onboard' || echo 0" 2>/dev/null || echo "?")
            fi
            echo "  $pc: $n procs"
        done
        ;;
    *)
        echo "Unknown action: $ACTION (use start|stop|status)"
        exit 1
        ;;
esac
