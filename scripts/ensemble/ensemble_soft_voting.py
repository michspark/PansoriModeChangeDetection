"""
Soft Voting Ensemble: Mel_Separated + Chroma + MIDI_Detection

Segment matching: IoU on (start_sec, end_sec) instead of exact time_range string,
because MIDI segmentation drifts ~0.6s from the frame model's exact 30s boundaries.

Usage (after re-running inference to regenerate CSVs with prob_* columns):
    python scripts/ensemble/ensemble_soft_voting.py

To use custom weights (must sum to 1 across models):
    python scripts/ensemble/ensemble_soft_voting.py --weights 0.5 0.3 0.2
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import MIDI_REPO, REPO_ROOT

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score

# ── Paths ─────────────────────────────────────────────────────────────────────

MODEL_CSVS = {
    'Mel_Separated': REPO_ROOT / 'weights/frame/Verstion_Stratified/0410_Mel_Separated_Version/fold1_posteriorgrams/test_results.csv',
    'Chroma':        REPO_ROOT / 'weights/frame/Verstion_Stratified/0410_Chroma_Version/fold1_posteriorgrams/test_results.csv',
    'MIDI':          MIDI_REPO / 'outputs/version_st/test_results.csv',
}

CLASSES = ['우조', '계면조', '아니리', '창조']
PROB_COLS = [f'prob_{c}' for c in CLASSES]

IOU_THRESHOLD = 0.7  # segments must overlap by at least this fraction to be matched

# ── Helpers ───────────────────────────────────────────────────────────────────

def seg_iou(s1, e1, s2, e2):
    inter = max(0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union > 0 else 0.0


def load_csv(path, is_midi=False):
    df = pd.read_csv(path)
    df = df[~df['time_range'].astype(str).str.contains('full', na=False)].copy()
    df['start_sec'] = df['start_sec'].astype(float)
    df['end_sec'] = df['end_sec'].astype(float)
    if is_midi:
        # Normalize song_name: strip .mid suffix, extract UUID
        df['uuid'] = df['song_name'].str.extract(r'^([a-f0-9\-]+)-\d+')[0]
    else:
        df['uuid'] = df['song_name'].astype(str)
    return df


def check_prob_columns(df, model_name):
    missing = [c for c in PROB_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{model_name}] Missing probability columns: {missing}\n"
            f"  → Re-run inference to regenerate test_results.csv with prob_* columns."
        )


def match_segments(df_frame, df_midi):
    """
    Match frame-model segments to MIDI segments by IoU on (start_sec, end_sec).
    Returns a list of dicts: {uuid, frame_idx, midi_idx, iou, start_frame, end_frame, start_midi, end_midi}
    """
    matched = []
    for uuid in df_frame['uuid'].unique():
        f_rows = df_frame[df_frame['uuid'] == uuid].reset_index(drop=True)
        m_rows = df_midi[df_midi['uuid'] == uuid].reset_index(drop=True)
        if m_rows.empty:
            continue
        for fi, frow in f_rows.iterrows():
            best_iou, best_mi = 0.0, -1
            for mi, mrow in m_rows.iterrows():
                iou = seg_iou(frow['start_sec'], frow['end_sec'],
                              mrow['start_sec'], mrow['end_sec'])
                if iou > best_iou:
                    best_iou, best_mi = iou, mi
            if best_iou >= IOU_THRESHOLD:
                matched.append({
                    'uuid': uuid,
                    'frame_idx': fi,
                    'midi_idx': best_mi,
                    'iou': best_iou,
                    'start_frame': frow['start_sec'],
                    'end_frame': frow['end_sec'],
                    'start_midi': m_rows.loc[best_mi, 'start_sec'],
                    'end_midi': m_rows.loc[best_mi, 'end_sec'],
                })
    return matched


def compute_metrics(y_true, y_pred):
    """y_true / y_pred are integer class arrays (0=우조,1=계면조,2=아니리,3=창조)."""
    f1_per = f1_score(y_true, y_pred, average=None, labels=list(range(len(CLASSES))), zero_division=0)
    f1_mac = f1_score(y_true, y_pred, average='macro', labels=list(range(len(CLASSES))), zero_division=0)
    acc = (y_true == y_pred).mean()
    return acc, f1_mac, f1_per


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs=3, type=float, default=None,
                        metavar=('W_MEL', 'W_CHROMA', 'W_MIDI'),
                        help='Soft voting weights for [Mel_Separated, Chroma, MIDI]. Must sum to 1.')
    parser.add_argument('--iou_threshold', type=float, default=IOU_THRESHOLD)
    parser.add_argument('--out', type=Path, default=REPO_ROOT / 'outputs/ensemble_soft_voting.csv')
    args = parser.parse_args()

    # ── Load CSVs ─────────────────────────────────────────────────────────────
    print("Loading CSVs...")
    df_mel  = load_csv(MODEL_CSVS['Mel_Separated'], is_midi=False)
    df_chr  = load_csv(MODEL_CSVS['Chroma'],        is_midi=False)
    df_midi = load_csv(MODEL_CSVS['MIDI'],           is_midi=True)

    for name, df in [('Mel_Separated', df_mel), ('Chroma', df_chr), ('MIDI', df_midi)]:
        check_prob_columns(df, name)

    # ── Weights ───────────────────────────────────────────────────────────────
    if args.weights:
        w = np.array(args.weights, dtype=float)
        assert abs(w.sum() - 1.0) < 1e-6, "Weights must sum to 1.0"
    else:
        # Equal weights
        w = np.array([1/3, 1/3, 1/3])
    w_mel, w_chr, w_midi = w
    print(f"Weights  Mel={w_mel:.3f}  Chroma={w_chr:.3f}  MIDI={w_midi:.3f}")

    # ── Match frame segments to MIDI segments ─────────────────────────────────
    # First match Mel_Separated ↔ Chroma (both have exact boundaries, should be 1:1)
    # Then match the frame pair ↔ MIDI by IoU

    # Mel & Chroma share exact boundaries → merge on uuid + time_range
    df_frame = df_mel.merge(
        df_chr[['uuid', 'time_range', 'start_sec', 'end_sec'] + PROB_COLS],
        on=['uuid', 'time_range'],
        suffixes=('_mel', '_chr')
    )
    # start_sec / end_sec come from Mel (they're identical for both frame models)
    df_frame['start_sec'] = df_frame['start_sec_mel']
    df_frame['end_sec']   = df_frame['end_sec_mel']
    df_frame = df_frame.reset_index(drop=True)

    print(f"\nFrame model common segments (Mel ∩ Chroma): {len(df_frame)}")

    # Match frame pair ↔ MIDI by IoU
    matched = match_segments(df_frame, df_midi)
    print(f"Matched frame ↔ MIDI (IoU≥{args.iou_threshold}): {len(matched)} / {len(df_frame)} frame segments")

    if len(matched) == 0:
        print("\nERROR: No segments matched. Check that CSVs have prob_* columns and re-run inference.")
        return

    # ── Soft Voting ───────────────────────────────────────────────────────────
    results = []
    for m in matched:
        frow = df_frame.loc[m['frame_idx']]
        mrow = df_midi[df_midi['uuid'] == m['uuid']].reset_index(drop=True).loc[m['midi_idx']]

        p_mel  = np.array([frow[f'prob_{c}_mel'] for c in CLASSES])
        p_chr  = np.array([frow[f'prob_{c}_chr'] for c in CLASSES])
        p_midi = np.array([mrow[f'prob_{c}'] for c in CLASSES])

        # Weighted average of probabilities
        p_ens = w_mel * p_mel + w_chr * p_chr + w_midi * p_midi
        pred_class = int(np.argmax(p_ens))

        results.append({
            'uuid': m['uuid'],
            'start_sec': m['start_frame'],
            'end_sec': m['end_frame'],
            'time_range': f"{m['start_frame']:.0f}-{m['end_frame']:.0f}s",
            'iou_with_midi': round(m['iou'], 4),
            'pred_ensemble': CLASSES[pred_class],
            **{f'p_mel_{c}': round(float(p_mel[i]), 4) for i, c in enumerate(CLASSES)},
            **{f'p_chr_{c}': round(float(p_chr[i]), 4) for i, c in enumerate(CLASSES)},
            **{f'p_midi_{c}': round(float(p_midi[i]), 4) for i, c in enumerate(CLASSES)},
            **{f'p_ens_{c}': round(float(p_ens[i]), 4) for i, c in enumerate(CLASSES)},
        })

    out_df = pd.DataFrame(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"\nEnsemble predictions saved → {args.out}")
    print(f"  (Note: GT labels needed for metric computation — add gt column from original CSVs if available)")


if __name__ == '__main__':
    main()
