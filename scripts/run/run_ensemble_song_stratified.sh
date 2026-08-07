#!/bin/bash
set -e

# Usage:
#   ./run_ensemble_song_stratified.sh                          # all 10 folds, Mel+MIDI+Pesto
#   ./run_ensemble_song_stratified.sh --folds 1,2,3            # specific folds only
#   ./run_ensemble_song_stratified.sh --no_midi                # Mel+Pesto only
#   ./run_ensemble_song_stratified.sh --weights 0.35 0.35 0.30
#   ./run_ensemble_song_stratified.sh --no_posteriors
#   ./run_ensemble_song_stratified.sh --sleep 10
#   ./run_ensemble_song_stratified.sh --mel_dir /path/to/mel_weights
#   ./run_ensemble_song_stratified.sh --pesto_dir /path/to/pesto_weights

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PANSORI_DATA_ROOT="${PANSORI_DATA_ROOT:-$REPO_ROOT/../Pansori_Data}"
cd "$REPO_ROOT"

FOLD_DIR="$PANSORI_DATA_ROOT/song_stratified"
MIDI_DIR="$REPO_ROOT/weights/midi/MIDI_Song_Stratified"
MEL_DIR="$REPO_ROOT/weights/frame/Mel_Original_Song_Stratified"
PESTO_DIR="$REPO_ROOT/weights/frame/Pesto_Song_Stratified"
OUT_DIR="$REPO_ROOT/outputs/ensemble_song_stratified"
SLEEP_SEC=5
TARGET_FOLDS=""       # comma-separated, e.g. "1,2,3" — empty means all 10
EXTRA_ARGS=()         # passed through to the Python script

while [[ $# -gt 0 ]]; do
    case $1 in
        --folds)        TARGET_FOLDS="$2"; shift 2 ;;
        --midi_dir)     MIDI_DIR="$2";     shift 2 ;;
        --mel_dir)      MEL_DIR="$2";      shift 2 ;;
        --pesto_dir)    PESTO_DIR="$2";    shift 2 ;;
        --no_midi)      EXTRA_ARGS+=("--no_midi");      shift ;;
        --no_posteriors) EXTRA_ARGS+=("--no_posteriors"); shift ;;
        --weights)      EXTRA_ARGS+=("--weights" "$2" "$3" "$4"); shift 4 ;;
        --sleep)        SLEEP_SEC="$2";    shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Build ordered fold list (1..10) from song_stratified txt files
declare -a ALL_FOLD_NUMS
IDX=1
declare -A GENRE_SEEN
for f in $(ls "$FOLD_DIR"/*_[12].txt 2>/dev/null | sort); do
    stem=$(basename "$f" .txt)
    base="${stem%_[12]}"
    if [[ -z "${GENRE_SEEN[$base]}" ]]; then
        GENRE_SEEN[$base]=1
        ALL_FOLD_NUMS+=($IDX $((IDX + 1)))
        IDX=$((IDX + 2))
    fi
done
TOTAL=${#ALL_FOLD_NUMS[@]}

# Filter to requested folds
if [[ -n "$TARGET_FOLDS" ]]; then
    IFS=',' read -ra FOLD_NUMS <<< "$TARGET_FOLDS"
else
    FOLD_NUMS=("${ALL_FOLD_NUMS[@]}")
fi

echo "Script      : ensemble_inference_song_stratified.py"
echo "Mel dir     : $MEL_DIR"
echo "Pesto dir   : $PESTO_DIR"
echo "MIDI dir    : $MIDI_DIR"
echo "Out dir     : $OUT_DIR"
echo "Folds       : ${FOLD_NUMS[*]}"
echo "Sleep       : ${SLEEP_SEC}s between folds"
echo "Extra args  : ${EXTRA_ARGS[*]}"
echo ""


for FOLD in "${FOLD_NUMS[@]}"; do
    echo "========================================="
    echo " Starting Fold $FOLD / $TOTAL"
    echo "========================================="

    python -u scripts/ensemble/ensemble_inference_song_stratified.py \
        --folds "$FOLD" \
        --mel_dir "$MEL_DIR" \
        --pesto_dir "$PESTO_DIR" \
        --midi_dir "$MIDI_DIR" \
        "${EXTRA_ARGS[@]}"

    echo "Fold $FOLD done."

    if [[ "$FOLD" != "${FOLD_NUMS[-1]}" ]]; then
        echo "Sleeping ${SLEEP_SEC}s to clear GPU memory..."
        sleep "$SLEEP_SEC"
    fi
done

echo ""
echo "========================================="
echo " All folds complete. Aggregating results..."
echo "========================================="

python3 - "$OUT_DIR" <<'PYEOF'
import sys, pandas as pd, numpy as np
from pathlib import Path

out_dir = Path(sys.argv[1])

# ── Aggregate per-fold CSVs ───────────────────────────────────────────────────
fold_csvs = sorted(out_dir.glob('fold[0-9]*_fold_metrics.csv'))
seg_csvs  = sorted(out_dir.glob('fold[0-9]*_segment_results.csv'))

if not fold_csvs:
    print("No per-fold CSVs found — skipping aggregation.")
    sys.exit(0)

df_folds = pd.concat([pd.read_csv(f) for f in fold_csvs], ignore_index=True)
df_segs  = pd.concat([pd.read_csv(f) for f in seg_csvs],  ignore_index=True)

df_folds.to_csv(out_dir / 'fold_metrics.csv', index=False)
df_segs.to_csv( out_dir / 'segment_results.csv', index=False)
print(f"fold_metrics.csv    ({len(df_folds)} rows) → {out_dir / 'fold_metrics.csv'}")
print(f"segment_results.csv ({len(df_segs)} rows)  → {out_dir / 'segment_results.csv'}")

# ── Rebuild overall metrics from aggregated segments ─────────────────────────
from sklearn.metrics import f1_score

EVAL_IDX  = [1, 2, 3, 4]
EVAL_NAME = ['우조', '계면조', '아니리', '창조']
CLASS_MAP = {'Unknown': 0, '우조': 1, '계면조': 2, '아니리': 3, '창조': 4}

# Rebuild from per-fold summary text files
txt_files = sorted(out_dir.glob('fold[0-9]*_results_summary.txt'))
combined_txt = []
for tf in txt_files:
    combined_txt.append(tf.read_text(encoding='utf-8').strip())
combined_txt.append('')

per_fold_f1 = df_folds['f1_macro'].tolist()
n_folds = len(df_folds)

# Overall from aggregated segment acc/f1 columns (already computed per segment)
# Use weighted average via n_songs
overall_acc    = (df_folds['acc']      * df_folds['n_songs']).sum() / df_folds['n_songs'].sum()
overall_f1_mac = (df_folds['f1_macro'] * df_folds['n_songs']).sum() / df_folds['n_songs'].sum()

summary_lines = [
    '',
    '=' * 60,
    'Overall Results (all folds aggregated)',
    '=' * 60,
    f"  Acc (weighted)      : {overall_acc:.4f}",
    f"  F1 Macro (weighted) : {overall_f1_mac:.4f}",
]
for c in EVAL_NAME:
    wf1 = (df_folds[f'f1_{c}'] * df_folds['n_songs']).sum() / df_folds['n_songs'].sum()
    summary_lines.append(f"  F1 {c:<6}: {wf1:.4f}")
summary_lines += [
    f"  Folds    : {n_folds}",
    f"  Segments : {len(df_segs)}",
    '',
    f"  Avg per-fold F1 macro : {np.mean(per_fold_f1):.4f} ± {np.std(per_fold_f1):.4f}",
]

# ── Version Test 18곡 통합 frame-level F1 ────────────────────────────────────
vt_npz_files = sorted(out_dir.glob('fold*_vt_frames.npz'))
if vt_npz_files:
    vt_true_all, vt_pred_all = [], []
    for npz in vt_npz_files:
        d = np.load(npz)
        vt_true_all.append(d['y_true'])
        vt_pred_all.append(d['y_pred'])
    yt_vt = np.concatenate(vt_true_all)
    yp_vt = np.concatenate(vt_pred_all)

    mask = yt_vt != 0
    yt_m, yp_m = yt_vt[mask], yp_vt[mask]
    vt_f1_per = f1_score(yt_m, yp_m, average=None,    labels=EVAL_IDX, zero_division=0)
    vt_f1_mac = f1_score(yt_m, yp_m, average='macro', labels=EVAL_IDX, zero_division=0)
    vt_acc    = float((yt_m == yp_m).mean())

    vt_lines = [
        '',
        '=' * 60,
        f'Version Test (18곡) — frame-level F1  [{len(vt_npz_files)} folds, {len(yt_vt):,} frames]',
        '=' * 60,
        f'  Acc      : {vt_acc:.4f}',
        f'  F1 Macro : {vt_f1_mac:.4f}',
        *(f'  F1 {EVAL_NAME[i]:<6}: {vt_f1_per[i]:.4f}' for i in range(len(EVAL_NAME))),
    ]
    combined_txt += vt_lines
    summary_lines += vt_lines
    for line in vt_lines:
        print(line)

combined_txt += summary_lines
full_txt = '\n'.join(combined_txt) + '\n'

txt_out = out_dir / 'results_summary.txt'
txt_out.write_text(full_txt, encoding='utf-8')
print(f"results_summary.txt → {txt_out}")

for line in summary_lines:
    print(line)
PYEOF

echo ""
echo "Done. Results in $OUT_DIR"
