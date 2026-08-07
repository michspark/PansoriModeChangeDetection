#!/bin/bash
# Unattended driver for the song-stratified CultureMERT run.
#
# Keeps relaunching run_song_stratified.sh until every fold has produced a
# test_results.csv. Two layers of recovery:
#   - fold level : folds that already finished are skipped on the next attempt
#   - step level : a fold killed mid-training resumes from its checkpoint
#                  (needs train.auto_resume=True, set in configs/frame/cmert.yaml)
#
# Usage:
#   ./run_auto_song_stratified.sh                 # layer 8, 20 retries
#   ./run_auto_song_stratified.sh --layer 6
#   ./run_auto_song_stratified.sh --max_retry 50 --sleep 300

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

LAYER=8
ROOT=""          # defaults to weights/cmert/layer<NN>_song_stratified once LAYER is known
VT_OUT_DIR=""
MAX_RETRY=20
SLEEP_SEC=120

while [[ $# -gt 0 ]]; do
    case $1 in
        --layer)      LAYER="$2";      shift 2 ;;
        --root)       ROOT="$2";       shift 2 ;;
        --vt_out_dir) VT_OUT_DIR="$2"; shift 2 ;;
        --max_retry)  MAX_RETRY="$2";  shift 2 ;;
        --sleep)      SLEEP_SEC="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Derive paths from LAYER only after parsing, so --layer always picks the matching
# directory instead of silently writing into another layer's folder.
if [[ -z "$ROOT" ]]; then
    ROOT="$(printf 'weights/cmert/layer%02d_song_stratified' "$LAYER")"
fi

# Pin the version-test output dir, otherwise each retry would scatter its CSVs
# into a differently timestamped directory.
if [[ -z "$VT_OUT_DIR" ]]; then
    VT_OUT_DIR="$(pwd)/outputs/version_test_layer${LAYER}_song_stratified"
fi

# Fold index -> name, in the order init_selection() builds them
FOLD_NAMES=(ch_1v2t ch_2v1t hb_1v2t hb_2v1t jb_1v2t jb_2v1t sc_1v2t sc_2v1t sg_1v2t sg_2v1t)

# A fold counts as done once its test evaluation has written test_results.csv
remaining_folds() {
    local out=()
    for i in "${!FOLD_NAMES[@]}"; do
        local n=$((i + 1))
        if ! find "$ROOT" -path "*/fold${n}_posteriorgrams/test_results.csv" -print -quit 2>/dev/null | grep -q .; then
            out+=("${FOLD_NAMES[$i]}")
        fi
    done
    (IFS=','; echo "${out[*]}")
}

echo "Layer      : $LAYER"
echo "Weights    : $ROOT"
echo "VT out dir : $VT_OUT_DIR"
echo "Retries    : up to $MAX_RETRY, ${SLEEP_SEC}s apart"
echo ""

for attempt in $(seq 1 "$MAX_RETRY"); do
    LEFT="$(remaining_folds)"

    if [[ -z "$LEFT" ]]; then
        echo "=========================================="
        echo " All 10 folds complete."
        echo "=========================================="
        exit 0
    fi

    echo "=========================================="
    echo " Attempt $attempt/$MAX_RETRY — remaining: $LEFT"
    echo "=========================================="

    if ./run_song_stratified.sh \
            --config_name cmert \
            --folds "$LEFT" \
            --vt_out_dir "$VT_OUT_DIR" \
            --extra "model.params.layer_index=$LAYER" \
            --extra "dir.save_dir=$ROOT"; then
        continue          # loop back and re-check; empty list exits above
    fi

    echo "Attempt $attempt failed. Sleeping ${SLEEP_SEC}s (GPU/RAM may need to free up)..."
    sleep "$SLEEP_SEC"
done

echo "Gave up after $MAX_RETRY attempts. Still remaining: $(remaining_folds)"
exit 1
