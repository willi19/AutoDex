#!/bin/bash
# Start perception daemons on capture PCs.
# Usage:
#   bash scripts/start_daemons.sh          # start all 6
#   bash scripts/start_daemons.sh sam3     # SAM3 only (capture1,2,3)
#   bash scripts/start_daemons.sh fpose    # FPose only (capture4,5,6)

set -e

# Kill any existing daemons first
kill_existing() {
    for i in 1 2 3 4 5 6; do
        ssh capture${i} "pkill -f perception_daemon 2>/dev/null" 2>/dev/null || true
    done
    sleep 1
}

start_sam3() {
    for i in 1 2 3; do
        echo "Starting SAM3 daemon on capture${i}..."
        ssh capture${i} "source ~/anaconda3/etc/profile.d/conda.sh && conda activate sam3 && cd ~/AutoDex && python src/execution/daemon/perception_daemon.py --model sam3 --port 5001" &
    done
}

start_fpose() {
    for i in 4 5 6; do
        echo "Starting FPose daemon on capture${i}..."
        ssh capture${i} "source ~/anaconda3/etc/profile.d/conda.sh && conda activate foundationpose && cd ~/AutoDex && python src/execution/daemon/perception_daemon.py --model fpose --port 5003" &
    done
}

kill_existing

case "${1:-all}" in
    sam3)   start_sam3 ;;
    fpose)  start_fpose ;;
    all)    start_sam3; start_fpose ;;
    *)      echo "Usage: $0 [sam3|fpose|all]"; exit 1 ;;
esac

echo "All daemons launched. Press Ctrl+C to stop."
wait
