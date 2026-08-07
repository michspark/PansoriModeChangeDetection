#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
set -o pipefail
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# CultureMERT layer-wise ablation on the Version stratified split.
# One run per layer: features are read from that transformer layer and only that
# layer (+ the classifier head) is fine-tuned.
#
# Usage:
#   ./run_layer_ablation.sh                      # layers 1..12
#   ./run_layer_ablation.sh --layers 1,6,12      # only specific layers
#   ./run_layer_ablation.sh --iters 10000        # longer training budget per layer
#   ./run_layer_ablation.sh --sleep 20           # longer GPU cooldown between runs

LAYERS="1 2 3 4 5 6 7 8 9 10 11 12"
ITERS=5000
EXTRA_ARGS=()     # additional Hydra overrides, e.g. --extra "train.num_workers=4"
SLEEP_SEC=10
OUT_ROOT="weights/cmert/layer_ablation_version"
LOG_DIR="outputs/layer_ablation_version"

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --layers)   LAYERS="${2//,/ }";  shift 2 ;;   # accept "1,6,12" or "1 6 12"
        --iters)    ITERS="$2";          shift 2 ;;
        --sleep)    SLEEP_SEC="$2";      shift 2 ;;
        --out_root) OUT_ROOT="$2";       shift 2 ;;
        --log_dir)  LOG_DIR="$2";        shift 2 ;;
        --extra)    EXTRA_ARGS+=("$2");  shift 2 ;;   # extra Hydra overrides, repeatable
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Count runs for progress display
TOTAL=$(echo $LAYERS | wc -w)

echo "Config     : configs/frame/cmert.yaml (train.selection=Version)"
echo "Layers     : $LAYERS  ($TOTAL runs)"
echo "Iterations : $ITERS per layer"
echo "Weights    : $OUT_ROOT/layer<NN>/"
echo "Logs       : $LOG_DIR/layer<NN>.log"
echo "Sleep      : ${SLEEP_SEC}s between runs"
echo "Extra args : ${EXTRA_ARGS[*]}"
echo ""

mkdir -p "$LOG_DIR"

IDX=0
for LAYER in $LAYERS; do
    IDX=$((IDX + 1))
    TAG=$(printf "layer%02d" "$LAYER")

    echo "========================================="
    echo " CultureMERT layer $LAYER  ($IDX / $TOTAL)"
    echo "========================================="
    # -u: unbuffered stdout so the summary prints show up live under `tee`/tmux
    python -u train.py \
        --config-path configs/frame \
        --config-name cmert \
        "train.selection=Version" \
        "train.num_iterations=$ITERS" \
        "model.params.layer_index=$LAYER" \
        "dir.save_dir=$OUT_ROOT/$TAG" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee "$LOG_DIR/$TAG.log"

    echo "Layer $LAYER done."

    if [[ $IDX -lt $TOTAL ]]; then
        echo "Sleeping ${SLEEP_SEC}s to clear GPU memory..."
        sleep $SLEEP_SEC
    fi
done

echo ""
echo "All runs complete: $LAYERS"
echo "Aggregate with: python scripts/eval/aggregate_layer_ablation.py"
