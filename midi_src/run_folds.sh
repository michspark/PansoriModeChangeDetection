#!/bin/bash
set -e

#./run_folds.sh stratified        # stratified 3:1:1 split (single run)
#./run_folds.sh random            # 10 random folds (default)
#./run_folds.sh random 5          # 5 random folds
#./run_folds.sh stratified_half   # 10 half-stratified folds (2 per genre)

run_song_stratified() {
    local FOLDS=${1:-10}
    local VT_OUT_DIR="$2"
    for i in $(seq 1 $FOLDS); do
        echo "========================================="
        echo " Starting Song-Stratified Fold $i / $FOLDS"
        echo "========================================="
        if [[ -n "$VT_OUT_DIR" ]]; then
            python train.py data.split=song_stratified "train.fold=$i" "+train.vt_out_dir=$VT_OUT_DIR"
        else
            python train.py data.split=song_stratified "train.fold=$i"
        fi
        echo "Fold $i done."
        sleep 5
    done
    if [[ -n "$VT_OUT_DIR" ]]; then
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
    fi
    echo "All $FOLDS song-stratified folds complete."
}

run_random_folds() {
    local FOLDS=${1:-10}   # default 10, override with: ./run_folds.sh random 5
    for i in $(seq 1 $FOLDS); do
        echo "========================================="
        echo " Starting Fold $i / $FOLDS"
        echo "========================================="
        python train.py data.split=random train.fold=$i
        echo "Fold $i done."
    done
    echo "All $FOLDS random folds complete."
}

run_stratified() {
    local FOLDS=${1:-5}
    for i in $(seq 1 $FOLDS); do
        echo "========================================="
        echo " Starting Stratified Fold $i / $FOLDS (3:1:1)"
        echo "========================================="
        python train.py data.split=stratified train.fold=$i
        echo "Fold $i done."
    done
    echo "All $FOLDS stratified folds complete."
}

run_stratified_half() {
    local FOLDS=${1:-10}
    for i in $(seq 1 $FOLDS); do
        echo "========================================="
        echo " Starting Half-Stratified Fold $i / $FOLDS"
        echo "========================================="
        python train.py data.split=stratified_half train.fold=$i
        echo "Fold $i done."
    done
    echo "All $FOLDS half-stratified folds complete."
}

MODE=${1:-random}

case "$MODE" in
    song_stratified)
        RUN_TS=$(date +%m%d_%H%M)
        VT_OUT_DIR="/home/sangheon/Desktop/PansoriMIDIDetection/outputs/version_test_song_stratified_${RUN_TS}"
        run_song_stratified "${2:-10}" "$VT_OUT_DIR"
        ;;
    stratified)
        run_stratified "${2:-5}"
        ;;
    stratified_half)
        run_stratified_half "${2:-10}"
        ;;
    random)
        run_random_folds "${2:-10}"
        ;;
    *)
        echo "Usage: $0 [song_stratified [N_FOLDS] | random [N_FOLDS] | stratified | stratified_half [N_FOLDS]]"
        echo "  song_stratified [N] - run N song-stratified folds, saving version test CSV (default 10)"
        echo "  random [N]          - run N random folds (default 10)"
        echo "  stratified [N]      - run N stratified folds with 3:1:1 ratio (default 5)"
        echo "  stratified_half [N] - run N half-stratified folds, 2 per genre (default 10)"
        exit 1
        ;;
esac
