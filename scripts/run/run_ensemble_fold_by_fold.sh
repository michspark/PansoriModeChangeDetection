#!/bin/bash
# Runs each fold as a completely separate conda session.
# Waits for full process exit + sleep before next fold to clear memory.
#
# Usage:
#   ./run_ensemble_fold_by_fold.sh              # folds 1-10
#   ./run_ensemble_fold_by_fold.sh 2 10         # folds 2-10
#   ./run_ensemble_fold_by_fold.sh 5 5          # fold 5 only

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PANSORI_MIDI_REPO="${PANSORI_MIDI_REPO:-$REPO_ROOT/../PansoriMIDIDetection}"
cd "$REPO_ROOT"

MEL_DIR="$REPO_ROOT/weights/frame/Mel_Original_Song_Stratified"
PESTO_DIR="$REPO_ROOT/weights/frame/Pesto_Song_Stratified"
MIDI_DIR="$PANSORI_MIDI_REPO/outputs/MIDI_Song_Stratified"
OUT_DIR="$REPO_ROOT/outputs/ensemble_song_stratified"
SLEEP_SEC=15

START_FOLD="${1:-1}"
END_FOLD="${2:-10}"

echo "Running folds ${START_FOLD}..${END_FOLD} — each as a separate session"
echo ""


for FOLD in $(seq "$START_FOLD" "$END_FOLD"); do
    echo "========================================="
    echo " Fold ${FOLD} / ${END_FOLD}"
    echo "========================================="

    python -u scripts/ensemble/ensemble_inference_song_stratified.py \
        --folds "$FOLD" \
        --mel_dir   "$MEL_DIR" \
        --pesto_dir "$PESTO_DIR" \
        --midi_dir  "$MIDI_DIR"

    echo "Fold ${FOLD} done."
    echo "Sleeping ${SLEEP_SEC}s to let OS reclaim memory..."
    sleep "$SLEEP_SEC"
done

echo ""
echo "========================================="
echo " All folds done. Aggregating results..."
echo "========================================="

python3 - "$OUT_DIR" <<'PYEOF'
import sys, pandas as pd, numpy as np
from pathlib import Path
from sklearn.metrics import f1_score

out_dir = Path(sys.argv[1])

fold_csvs = sorted(out_dir.glob('fold[0-9]*_fold_metrics.csv'))
seg_csvs  = sorted(out_dir.glob('fold[0-9]*_segment_results.csv'))

if not fold_csvs:
    print("No per-fold CSVs found.")
    sys.exit(0)

df_folds = pd.concat([pd.read_csv(f) for f in fold_csvs], ignore_index=True)
df_segs  = pd.concat([pd.read_csv(f) for f in seg_csvs],  ignore_index=True)

df_folds.to_csv(out_dir / 'fold_metrics.csv',    index=False)
df_segs.to_csv( out_dir / 'segment_results.csv', index=False)
print(f"fold_metrics.csv    ({len(df_folds)} rows)")
print(f"segment_results.csv ({len(df_segs)} rows)")

EVAL_IDX  = [1, 2, 3, 4]
EVAL_NAME = ['우조', '계면조', '아니리', '창조']

overall_acc    = (df_folds['acc']      * df_folds['n_songs']).sum() / df_folds['n_songs'].sum()
overall_f1_mac = (df_folds['f1_macro'] * df_folds['n_songs']).sum() / df_folds['n_songs'].sum()
per_fold_f1    = df_folds['f1_macro'].tolist()

summary_lines = [
    '', '='*60, 'Overall Results', '='*60,
    f"  Acc (weighted)      : {overall_acc:.4f}",
    f"  F1 Macro (weighted) : {overall_f1_mac:.4f}",
]
for c in EVAL_NAME:
    wf1 = (df_folds[f'f1_{c}'] * df_folds['n_songs']).sum() / df_folds['n_songs'].sum()
    summary_lines.append(f"  F1 {c:<6}: {wf1:.4f}")
summary_lines += [
    f"  Avg per-fold F1: {np.mean(per_fold_f1):.4f} ± {np.std(per_fold_f1):.4f}",
]

vt_npz_files = sorted(out_dir.glob('fold*_vt_frames.npz'))
if vt_npz_files:
    yt = np.concatenate([np.load(f)['y_true'] for f in vt_npz_files])
    yp = np.concatenate([np.load(f)['y_pred'] for f in vt_npz_files])
    mask = yt != 0
    vt_f1 = f1_score(yt[mask], yp[mask], average='macro', labels=EVAL_IDX, zero_division=0)
    vt_acc = float((yt[mask] == yp[mask]).mean())
    summary_lines += ['', '='*60, f'Version Test (18곡) [{len(vt_npz_files)} folds]', '='*60,
                      f'  Acc      : {vt_acc:.4f}', f'  F1 Macro : {vt_f1:.4f}']

txt = '\n'.join(summary_lines) + '\n'
(out_dir / 'results_summary.txt').write_text(txt, encoding='utf-8')
for line in summary_lines:
    print(line)
PYEOF

echo "Done. Results in $OUT_DIR"
