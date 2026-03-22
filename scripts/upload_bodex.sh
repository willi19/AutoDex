#!/bin/bash
# Upload bodex outputs to NAS (zip each object)
# Usage: bash scripts/upload_bodex.sh

set -e

LOCAL_ROOT="/home/mingi/AutoDex/bodex_outputs"
NAS_ROOT="/home/mingi/shared_data/AutoDex/bodex_outputs"

for hand in allegro inspire; do
    SRC="$LOCAL_ROOT/$hand/v3"
    DST="$NAS_ROOT/$hand/v3"

    if [ ! -d "$SRC" ]; then
        echo "SKIP: $SRC not found"
        continue
    fi

    # Clear old zips on NAS
    echo "Clearing old $DST..."
    rm -rf "$DST"/*
    mkdir -p "$DST"

    objects=($(ls "$SRC"))
    total=${#objects[@]}
    echo "Uploading $hand v3: $total objects"

    for i in "${!objects[@]}"; do
        obj="${objects[$i]}"
        n=$((i + 1))

        if [ -f "$SRC/$obj" ]; then
            # Already a zip
            echo "[$n/$total] $obj (copy zip)"
            cp "$SRC/$obj" "$DST/$obj"
        elif [ -d "$SRC/$obj" ]; then
            # Directory — zip it
            echo "[$n/$total] $obj (zipping...)"
            cd "$SRC"
            zip -rq "$DST/${obj}.zip" "$obj"
            cd - > /dev/null
        fi
    done

    echo "$hand done: $total objects uploaded"
    echo ""
done

echo "All done."