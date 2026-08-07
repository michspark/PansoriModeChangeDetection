#!/bin/bash
set -e

# Usage:
#   ./run_folds.sh                              # frame, mel_base, 10 folds
#   ./run_folds.sh --config_path segment --config_name crepe_best
#   ./run_folds.sh --folds 5
#   ./run_folds.sh --config_path frame --config_name pesto_best --folds 10 --sleep 10

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG_PATH="frame"
CONFIG_NAME="chroma_base"
FOLDS=10
SLEEP_SEC=5   # seconds to wait between folds for GPU memory to clear

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config_path) CONFIG_PATH="$2"; shift 2 ;;
        --config_name) CONFIG_NAME="$2"; shift 2 ;;
        --folds)       FOLDS="$2";       shift 2 ;;
        --sleep)       SLEEP_SEC="$2";   shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "Config path : configs/$CONFIG_PATH/$CONFIG_NAME.yaml"
echo "Folds       : 1 .. $FOLDS"
echo "Sleep       : ${SLEEP_SEC}s between folds"
echo ""

for i in $(seq 1 $FOLDS); do
    echo "========================================="
    echo " Starting Fold $i / $FOLDS"
    echo "========================================="
    python train.py \
        --config-path "configs/$CONFIG_PATH" \
        --config-name "$CONFIG_NAME" \
        "train.target_folds=[$i]"

    echo "Fold $i done."

    if [ $i -lt $FOLDS ]; then
        echo "Sleeping ${SLEEP_SEC}s to clear GPU memory..."
        sleep $SLEEP_SEC
    fi
done

echo ""
echo "All $FOLDS folds complete."
