#!/bin/bash
set -e
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PANSORI_DATA_ROOT="${PANSORI_DATA_ROOT:-$REPO_ROOT/../Pansori_Data}"
cd "$REPO_ROOT"


# Usage:
#   ./run_song_stratified.sh                                        # frame, mel_base, all 10 folds
#   ./run_song_stratified.sh --config_name pesto_best               # Pesto, all 10 folds
#   ./run_song_stratified.sh --config_name mel_base                 # Mel Original, all 10 folds
#   ./run_song_stratified.sh --config_path segment --config_name crepe_best
#   ./run_song_stratified.sh --folds ch_1v2t,sg_2v1t               # run only specific folds by name
#   ./run_song_stratified.sh --sleep 10

CONFIG_PATH="frame"
CONFIG_NAME="mel_base"
FOLD_DIR="$PANSORI_DATA_ROOT/song_stratified"
TARGET_FOLDS=""   # e.g. "ch_1v2t,sg_1v2t" — empty means all
SLEEP_SEC=5
VT_OUT_DIR=""     # set after arg parsing so CONFIG_NAME is resolved first
EXTRA_ARGS=()     # additional Hydra overrides, e.g. --extra "data.data_dir=/path/to/audio"

while [[ $# -gt 0 ]]; do
    case $1 in
        --config_path) CONFIG_PATH="$2"; shift 2 ;;
        --config_name) CONFIG_NAME="$2"; shift 2 ;;
        --fold_dir)    FOLD_DIR="$2";    shift 2 ;;
        --folds)       TARGET_FOLDS="$2"; shift 2 ;;
        --sleep)       SLEEP_SEC="$2";   shift 2 ;;
        --vt_out_dir)  VT_OUT_DIR="$2";  shift 2 ;;
        --extra)       EXTRA_ARGS+=("$2"); shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Default VT_OUT_DIR uses resolved CONFIG_NAME
if [[ -z "$VT_OUT_DIR" ]]; then
    RUN_TS=$(date +%m%d_%H%M)
    VT_OUT_DIR="$REPO_ROOT/outputs/version_test_${CONFIG_NAME}_${RUN_TS}"
fi

# Build ordered list of fold names: for each base genre, two entries (1v2t, 2v1t)
# Derived from the song_stratified directory: ch_1.txt,ch_2.txt -> ch_1v2t, ch_2v1t
declare -a ALL_FOLD_NAMES
declare -A GENRE_SEEN
for f in $(ls "$FOLD_DIR"/*_[12].txt 2>/dev/null | sort); do
    stem=$(basename "$f" .txt)          # e.g. ch_1
    base="${stem%_[12]}"                # e.g. ch
    if [[ -z "${GENRE_SEEN[$base]}" ]]; then
        GENRE_SEEN[$base]=1
        ALL_FOLD_NAMES+=("${base}_1v2t" "${base}_2v1t")
    fi
done
TOTAL=${#ALL_FOLD_NAMES[@]}

# Filter to target folds if specified
if [[ -n "$TARGET_FOLDS" ]]; then
    IFS=',' read -ra FOLD_NAMES <<< "$TARGET_FOLDS"
else
    FOLD_NAMES=("${ALL_FOLD_NAMES[@]}")
fi

echo "Config      : configs/$CONFIG_PATH/$CONFIG_NAME.yaml"
echo "Fold dir    : $FOLD_DIR"
echo "Folds       : ${FOLD_NAMES[*]}"
echo "Sleep       : ${SLEEP_SEC}s between folds"
echo "VT out dir  : $VT_OUT_DIR"
echo "Extra args  : ${EXTRA_ARGS[*]}"
echo ""

for FOLD_NAME in "${FOLD_NAMES[@]}"; do
    # Get 1-based fold index from full list
    FOLD_IDX=0
    for i in "${!ALL_FOLD_NAMES[@]}"; do
        if [[ "${ALL_FOLD_NAMES[$i]}" == "$FOLD_NAME" ]]; then
            FOLD_IDX=$((i + 1))
            break
        fi
    done

    if [[ $FOLD_IDX -eq 0 ]]; then
        echo "Unknown fold name: $FOLD_NAME"
        echo "Available: ${ALL_FOLD_NAMES[*]}"
        exit 1
    fi

    echo "========================================="
    echo " Starting Fold $FOLD_NAME ($FOLD_IDX / $TOTAL)"
    echo "========================================="
    python train.py \
        --config-path "configs/$CONFIG_PATH" \
        --config-name "$CONFIG_NAME" \
        "train.selection=SongStratified" \
        "train.song_stratified_dir=$FOLD_DIR" \
        "train.target_folds=[$FOLD_IDX]" \
        "+train.vt_out_dir=$VT_OUT_DIR" \
        "${EXTRA_ARGS[@]}"

    echo "Fold $FOLD_NAME done."

    if [[ "$FOLD_NAME" != "${FOLD_NAMES[-1]}" ]]; then
        echo "Sleeping ${SLEEP_SEC}s to clear GPU memory..."
        sleep $SLEEP_SEC
    fi
done

echo ""
echo "All folds complete: ${FOLD_NAMES[*]}"

echo "Aggregating version test results from $VT_OUT_DIR ..."
python3 - "$VT_OUT_DIR" <<'PYEOF'
import sys, pandas as pd, pathlib
d = pathlib.Path(sys.argv[1])
csvs = sorted(d.glob('fold*_vt.csv'))
if not csvs:
    print('No version test CSVs found — skipping aggregation.')
    sys.exit(0)
df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
out = d / 'version_test_results_all_folds.csv'
df.to_csv(out, index=False)
print(f'Aggregated {len(csvs)} fold CSVs ({len(df)} rows) -> {out}')
PYEOF
