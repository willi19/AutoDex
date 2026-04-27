#!/usr/bin/env bash
# Watch outputs/foundpose_assets/ for newly-completed repre.pth files,
# rsync them to NAS, send desktop notification.
# Stops when batch_onboard_foundpose has finished AND no in-progress workers remain.
set -u

LOCAL=/home/mingi/AutoDex/outputs/foundpose_assets
NAS=/home/mingi/shared_data/AutoDex/foundpose_assets
LOG=/home/mingi/AutoDex/outputs/foundpose_onboard_logs/_nas_watcher.log
mkdir -p "$NAS" "$(dirname "$LOG")"

declare -A SYNCED

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "watcher started"

while true; do
    # Sync any new repre.pth files.
    for repre in "$LOCAL"/*/object_repre/v1/*/1/repre.pth; do
        [ -f "$repre" ] || continue
        obj=$(echo "$repre" | sed 's|.*/foundpose_assets/||;s|/object_repre/.*||')
        [ -n "${SYNCED[$obj]:-}" ] && continue

        nas_file="$NAS/$obj/object_repre/v1/$obj/1/repre.pth"
        if [ -f "$nas_file" ] && [ "$(stat -c %s "$repre")" -eq "$(stat -c %s "$nas_file")" ]; then
            SYNCED[$obj]=1
            continue
        fi

        mkdir -p "$NAS/$obj/"
        if rsync -rt --no-perms --no-owner --no-group \
                "$LOCAL/$obj/object_repre" "$NAS/$obj/"; then
            SYNCED[$obj]=1
            n=${#SYNCED[@]}
            log "synced $obj ($n total)"
            if [ "$obj" = "attached_container" ]; then
                notify-send -u critical "attached_container ✓" "uploaded to NAS"
            else
                notify-send "$obj synced ($n)" ""
            fi
        else
            log "FAILED rsync $obj"
        fi
    done

    # Stop condition: no batch runner AND no per-obj workers, for 5 consecutive checks (5 min).
    if ! pgrep -f batch_onboard_foundpose > /dev/null \
       && ! pgrep -f onboard_custom_mesh_for_foundpose > /dev/null; then
        IDLE_COUNT=$((${IDLE_COUNT:-0} + 1))
        if [ "$IDLE_COUNT" -ge 5 ]; then
            n=${#SYNCED[@]}
            log "all done — exiting (synced $n total, idle 5 min)"
            notify-send -u critical "FoundPose all done" "$n objects synced to NAS"
            break
        fi
    else
        IDLE_COUNT=0
    fi

    sleep 60
done
