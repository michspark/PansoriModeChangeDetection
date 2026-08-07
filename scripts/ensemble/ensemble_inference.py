"""
End-to-end Soft Voting Ensemble Inference
  Mel_Separated + Chroma + MIDI_Detection

Flow per song:
  1. Load audio → compute mel / chroma features (pre-loaded by dataset)
  2. Load MIDI  → generate piano roll (pre-loaded by dataset)
  3. Slide exact 30-second windows through each full-song feature tensor
  4. Run each model → collect frame-level softmax probs
  5. Align audio probs (31.25 fps) → MIDI frame rate (10 fps) via avg-pooling
  6. Weighted average → ensemble prediction
  7. Compute per-segment & overall metrics

Usage:
    cd /path/to/PansoriModeChangeDetection
    python scripts/ensemble/ensemble_inference.py
    python scripts/ensemble/ensemble_inference.py --weights 0.45 0.25 0.30
"""

import os
os.environ['WANDB_MODE'] = 'disabled'

import sys
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import f1_score
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import DATA_ROOT, REPO_ROOT
# NOTE: PansoriMIDIDetection is added to sys.path lazily inside functions
# to avoid shadowing frame models/datasets packages.

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import models as frame_models
import datasets as frame_datasets
import losses as frame_losses
from trainers import plot_posteriorgram

# Paths
REPO_FRAME = REPO_ROOT

MODEL_DIRS = {
    'mel':   REPO_FRAME / 'weights/frame/Verstion_Stratified/0410_Mel_Separated_Version',
    'chroma': REPO_FRAME / 'weights/frame/Verstion_Stratified/0410_Chroma_Version',
}
MIDI_CKPT    = REPO_FRAME / 'weights/midi/MIDI_Version/23-43-24/fold1_best_model.pt'
MIDI_CFG     = REPO_FRAME / 'configs/frame/midi.yaml'
VERSION_SPLIT= DATA_ROOT / 'pansori_version_split'
OUT_DIR      = REPO_FRAME / 'outputs/ensemble'

CLASSES = ['Unknown', '우조', '계면조', '아니리', '창조']
EVAL_CLASSES_IDX = [1, 2, 3, 4]   # ignore Unknown(0) for F1
EVAL_CLASSES_NAME = ['우조', '계면조', '아니리', '창조']

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Frame rates ───────────────────────────────────────────────────────────────
AUDIO_SR         = 16000
AUDIO_HOP        = 512
AUDIO_FPS        = AUDIO_SR / AUDIO_HOP   # ≈ 31.25 fps
MIDI_FPS         = 10                      # fs in MIDI config
WINDOW_SEC       = 30                      # seconds per segment
AUDIO_WIN_FRAMES = int(WINDOW_SEC * AUDIO_FPS)   # ~937
MIDI_WIN_FRAMES  = int(WINDOW_SEC * MIDI_FPS)    # 300


# ── Model loading ─────────────────────────────────────────────────────────────

def load_frame_model(model_dir: Path, device: str):
    cfg = OmegaConf.load(model_dir / 'config.yaml')
    model_class = getattr(frame_models, cfg.model.name)
    model = model_class(cfg.model.params).to(device)
    state = torch.load(model_dir / 'fold1_best_model.pt', map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def load_midi_model(ckpt: Path, cfg_path: Path, device: str):
    """MIDI is a first-class modality: this repo's Conv2DGRU + midi.yaml.

    Checkpoints trained in the old PansoriMIDIDetection repo load unchanged —
    same architecture, same layer names.
    """
    midi_cfg = OmegaConf.load(cfg_path)
    model = frame_models.Conv2DGRU(midi_cfg.model.params).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model, midi_cfg


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_frame_dataset(cfg, test_hash_keys: list):
    """Load frame dataset in test (is_valid) mode for the given hash keys."""
    dataset_class = getattr(frame_datasets, cfg.dataset.name)
    params = OmegaConf.to_container(cfg.dataset.params)
    params['aug'] = False
    params['is_valid'] = True
    ds = dataset_class(**cfg.data, **params)
    # Filter to test keys only
    ds.loaded_hash  = [h for h in ds.loaded_hash if h in set(test_hash_keys)]
    ds.loaded_data  = {h: v for h, v in ds.loaded_data.items()   if h in set(test_hash_keys)}
    ds.loaded_label = {h: v for h, v in ds.loaded_label.items()  if h in set(test_hash_keys)}
    return ds


def load_midi_dataset(midi_cfg, test_hash_keys: list):
    """MidiFrameDataset is hash-keyed like every other modality, so the generic
    load_frame_dataset() above handles it — no name translation needed."""
    fs          = midi_cfg.dataset.params.fs
    window_size = int(midi_cfg.dataset.params.window * fs)
    return load_frame_dataset(midi_cfg, test_hash_keys), fs, window_size


# ── Full-song inference ───────────────────────────────────────────────────────

@torch.no_grad()
def full_song_frame_inference(model, feature_full: torch.Tensor, window_frames: int, device: str):
    """
    feature_full : (n_bins, T_total)
    Returns      : probs (T_total, num_classes) numpy — softmax, at audio frame rate
    """
    n_bins, T = feature_full.shape
    num_classes = None
    all_probs = []

    for start in range(0, T, window_frames):
        end = min(start + window_frames, T)
        chunk = feature_full[:, start:end]                  # (n_bins, chunk_len)
        if chunk.shape[1] < window_frames:
            pad = torch.zeros(n_bins, window_frames - chunk.shape[1])
            chunk = torch.cat([chunk, pad], dim=1)
        x = chunk.unsqueeze(0).to(device)                   # (1, n_bins, window_frames)
        out = model(x)                                       # (1, T_win, C)
        probs = torch.softmax(out, dim=-1)                  # (1, T_win, C)
        probs = probs[0].cpu().numpy()                      # (T_win, C)
        # Only keep the valid (non-padded) portion
        valid_len = end - start
        all_probs.append(probs[:valid_len])
        if num_classes is None:
            num_classes = probs.shape[1]

    return np.concatenate(all_probs, axis=0)  # (T_total, C)


# MIDI needs no inference helper of its own — a (128, T) piano roll is just
# another (n_bins, T) feature, so full_song_frame_inference() above handles it.


# ── Temporal alignment ────────────────────────────────────────────────────────

def align_audio_to_midi(audio_probs: np.ndarray, T_midi: int) -> np.ndarray:
    """
    Resample audio_probs from audio_fps (31.25) to midi_fps (10) via avg-pooling.

    audio_probs : (T_audio, C)
    T_midi      : number of MIDI frames
    Returns     : (T_midi, C)
    """
    T_audio, C = audio_probs.shape
    aligned = np.zeros((T_midi, C), dtype=np.float32)

    ratio = AUDIO_FPS / MIDI_FPS   # ≈ 3.125 audio frames per MIDI frame

    for j in range(T_midi):
        a_start = int(round(j * ratio))
        a_end   = int(round((j + 1) * ratio))
        a_start = max(0, min(a_start, T_audio - 1))
        a_end   = max(a_start + 1, min(a_end, T_audio))
        aligned[j] = audio_probs[a_start:a_end].mean(axis=0)

    return aligned


# ── Test split parsing ────────────────────────────────────────────────────────

def load_version_test_split(version_split_dir: Path):
    """Returns the test hash_keys from test.txt.

    Every modality including MIDI is keyed by hash now, so the old
    hash_key -> midi-filename resolution step is gone.
    """
    lines = [l.strip() for l in (version_split_dir / 'test.txt').read_text('utf-8').splitlines() if l.strip()]
    frame_hashes = [line.split('-')[0] for line in lines]
    print(f"  test.txt: {len(lines)} lines → {len(frame_hashes)} hash keys")
    return frame_hashes


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Ignore Unknown(0) class for F1. y_true/y_pred are integer class arrays."""
    mask = y_true != 0
    if mask.sum() == 0:
        return {'acc': 0.0, 'f1_macro': 0.0,
                **{f'f1_{c}': 0.0 for c in EVAL_CLASSES_NAME}}
    yt, yp = y_true[mask], y_pred[mask]
    acc = (yt == yp).mean()
    f1_per = f1_score(yt, yp, average=None, labels=EVAL_CLASSES_IDX, zero_division=0)
    f1_mac = f1_score(yt, yp, average='macro', labels=EVAL_CLASSES_IDX, zero_division=0)
    return {
        'acc': float(acc),
        'f1_macro': float(f1_mac),
        **{f'f1_{EVAL_CLASSES_NAME[i]}': float(f1_per[i]) for i in range(len(EVAL_CLASSES_NAME))}
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs=3, type=float, default=None,
                        metavar=('W_MEL', 'W_CHROMA', 'W_MIDI'),
                        help='Soft voting weights (must sum to 1). Default: equal 1/3 each.')
    parser.add_argument('--no_posteriors', action='store_true',
                        help='Skip saving per-segment CSV (only print summary)')
    args = parser.parse_args()

    weights = np.array(args.weights if args.weights else [1/3, 1/3, 1/3], dtype=np.float32)
    assert abs(weights.sum() - 1.0) < 1e-5, "Weights must sum to 1"
    w_mel, w_chroma, w_midi = weights
    print(f"Weights  Mel={w_mel:.3f}  Chroma={w_chroma:.3f}  MIDI={w_midi:.3f}")
    print(f"Device: {DEV}\n")

    # ── Load test split ───────────────────────────────────────────────────────
    frame_test_hashes = load_version_test_split(VERSION_SPLIT)

    # ── Load models ───────────────────────────────────────────────────────────
    print("Loading models...")
    mel_model,    mel_cfg    = load_frame_model(MODEL_DIRS['mel'],    DEV)
    chroma_model, chroma_cfg = load_frame_model(MODEL_DIRS['chroma'], DEV)
    midi_model,   midi_cfg   = load_midi_model(MIDI_CKPT, MIDI_CFG, DEV)
    print(f"  Mel    : {mel_cfg.model.name}  ({MODEL_DIRS['mel'].name})")
    print(f"  Chroma : {chroma_cfg.model.name}  ({MODEL_DIRS['chroma'].name})")
    print(f"  MIDI   : Conv2DGRU  ({MIDI_CKPT.name})")

    # ── Load datasets (for feature pre-loading) ───────────────────────────────
    print("\nPre-loading features...")
    mel_ds    = load_frame_dataset(mel_cfg,    frame_test_hashes)
    chroma_ds = load_frame_dataset(chroma_cfg, frame_test_hashes)
    midi_ds, midi_fs, midi_win = load_midi_dataset(midi_cfg, frame_test_hashes)

    # Common hash keys that exist in all 3 datasets
    common_hashes = (
        set(mel_ds.loaded_data.keys())
        & set(chroma_ds.loaded_data.keys())
        & set(midi_ds.loaded_data.keys())
    )
    print(f"\nCommon songs across all 3 modalities: {len(common_hashes)}")

    # ── Posteriorgram output dir ──────────────────────────────────────────────
    post_dir = OUT_DIR / 'posteriorgrams'
    if not args.no_posteriors:
        post_dir.mkdir(parents=True, exist_ok=True)

    # ── Per-song inference ────────────────────────────────────────────────────
    all_segment_results = []
    all_y_true, all_y_pred_ens = [], []

    for hash_key in tqdm(sorted(common_hashes), desc='Songs'):
        # --- Full-song probs at native frame rates ---
        mel_full    = mel_ds.loaded_data[hash_key]    # (n_bins, T_mel)
        chroma_full = chroma_ds.loaded_data[hash_key] # (n_bins, T_chroma)
        piano_roll  = midi_ds.loaded_data[hash_key]    # (128, T_midi)
        gt_label    = midi_ds.loaded_label[hash_key]   # (T_midi, 5) one-hot at midi_fps

        mel_probs    = full_song_frame_inference(mel_model,    mel_full,    AUDIO_WIN_FRAMES, DEV)
        chroma_probs = full_song_frame_inference(chroma_model, chroma_full, AUDIO_WIN_FRAMES, DEV)
        midi_probs   = full_song_frame_inference(midi_model,  piano_roll,  midi_win,         DEV)

        T_midi = midi_probs.shape[0]

        # --- Align audio probs → MIDI frame rate (10 fps) ---
        mel_aligned    = align_audio_to_midi(mel_probs,    T_midi)  # (T_midi, C)
        chroma_aligned = align_audio_to_midi(chroma_probs, T_midi)  # (T_midi, C)

        # --- Weighted soft voting ---
        ens_probs = w_mel * mel_aligned + w_chroma * chroma_aligned + w_midi * midi_probs
        ens_pred  = ens_probs.argmax(axis=1)  # (T_midi,)

        # --- GT labels at MIDI frame rate ---
        gt_label_np = gt_label.numpy()           # (T_midi, 5) one-hot float
        gt_cls      = gt_label_np.argmax(axis=1) # (T_midi,) integer class

        all_y_true.append(gt_cls)
        all_y_pred_ens.append(ens_pred)

        # --- Posteriorgrams & per-segment metrics ---
        song_name = midi_item['song_name']
        stem = f"{song_name.replace('_vocal.mid', '')}_{hash_key}"

        for seg_start in range(0, T_midi, midi_win):
            seg_end   = min(seg_start + midi_win, T_midi)
            start_sec = seg_start / midi_fs
            end_sec   = seg_end   / midi_fs

            gt_seg  = gt_cls[seg_start:seg_end]
            pr_seg  = ens_pred[seg_start:seg_end]

            mel_seg_prob    = mel_aligned[seg_start:seg_end].mean(axis=0)
            chroma_seg_prob = chroma_aligned[seg_start:seg_end].mean(axis=0)
            midi_seg_prob   = midi_probs[seg_start:seg_end].mean(axis=0)
            ens_seg_prob    = ens_probs[seg_start:seg_end].mean(axis=0)

            m = compute_metrics(gt_seg, pr_seg)
            row = {
                'song_name': hash_key,
                'midi_name': song_name,
                'start_sec': round(start_sec, 1),
                'end_sec':   round(end_sec,   1),
                'time_range': f"{int(start_sec)}-{int(end_sec)}s",
                **m,
                **{f'p_mel_{c}':    round(float(mel_seg_prob[i]),    4) for i, c in enumerate(CLASSES)},
                **{f'p_chroma_{c}': round(float(chroma_seg_prob[i]), 4) for i, c in enumerate(CLASSES)},
                **{f'p_midi_{c}':   round(float(midi_seg_prob[i]),   4) for i, c in enumerate(CLASSES)},
                **{f'p_ens_{c}':    round(float(ens_seg_prob[i]),    4) for i, c in enumerate(CLASSES)},
            }
            all_segment_results.append(row)

            # Per-segment posteriorgram
            if not args.no_posteriors:
                seg_label = f"{int(start_sec)}-{int(end_sec)}s"
                fig = plot_posteriorgram(
                    f"{stem}  {seg_label}",
                    gt_label_np[seg_start:seg_end],   # (seg_len, C) one-hot
                    ens_probs[seg_start:seg_end],      # (seg_len, C)
                    CLASSES,
                )
                fig.savefig(post_dir / f"{stem}_{seg_label}.png", dpi=120, bbox_inches='tight')
                plt.close(fig)

        # Full-song posteriorgram
        if not args.no_posteriors:
            fig = plot_posteriorgram(stem, gt_label_np, ens_probs, CLASSES)
            fig.savefig(post_dir / f"{stem}_full.png", dpi=120, bbox_inches='tight')
            plt.close(fig)

    # ── Overall metrics ───────────────────────────────────────────────────────
    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred_ens)
    overall = compute_metrics(y_true_all, y_pred_all)

    print(f"\n{'='*55}")
    print("Ensemble Results (Mel + Chroma + MIDI, soft voting)")
    print(f"{'='*55}")
    print(f"  Acc      : {overall['acc']:.4f}")
    print(f"  F1 Macro : {overall['f1_macro']:.4f}")
    for c in EVAL_CLASSES_NAME:
        print(f"  F1 {c:<6}: {overall[f'f1_{c}']:.4f}")
    print(f"  Segments : {len(all_segment_results)}")
    print(f"  Songs    : {len(common_hashes)}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if not args.no_posteriors:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_csv = OUT_DIR / 'ensemble_results.csv'
        pd.DataFrame(all_segment_results).to_csv(out_csv, index=False)
        print(f"\nSegment-level results saved → {out_csv}")
        png_count = len(list(post_dir.glob('*.png')))
        print(f"Posteriorgrams saved       → {post_dir}  ({png_count} PNGs)")


if __name__ == '__main__':
    main()
