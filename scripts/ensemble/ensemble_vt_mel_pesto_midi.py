"""
Version Test Ensemble: Mel_Original_Version + Pesto_Version + MIDI_Version/23-43-24
  → masked accuracy + per-class accuracy (frame-level, Unknown=0 제외)

Frame rates:
  Mel   : 16000 / 512 ≈ 31.25 fps  (Conv2DGRU)
  Pesto : 20 fps                    (Conv1DGRU)
  MIDI  : 10 fps                    (Conv2DGRU)
  → 모든 prediction을 10fps(MIDI)로 정렬 후 soft voting

Usage:
    cd /path/to/PansoriModeChangeDetection
    python scripts/ensemble/ensemble_vt_mel_pesto_midi.py
    python scripts/ensemble/ensemble_vt_mel_pesto_midi.py --weights 0.4 0.2 0.4
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import DATA_ROOT, REPO_ROOT

import os
os.environ['WANDB_MODE'] = 'disabled'

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

REPO_FRAME = REPO_ROOT

sys.path.insert(0, str(REPO_FRAME))
import models as frame_models
import datasets as frame_datasets

MEL_DIR   = REPO_FRAME / 'weights/frame/Mel_Original_Version'
PESTO_DIR = REPO_FRAME / 'weights/frame/Pesto_Version'
MIDI_DIR  = REPO_FRAME / 'weights/midi/MIDI_Version/23-43-24'

VERSION_SPLIT = DATA_ROOT / 'pansori_version_split'

CLASS_NAMES = {1: '우조', 2: '계면조', 3: '아니리', 4: '창조'}
IGNORE_IDX  = 0

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

AUDIO_SR   = 16000
AUDIO_HOP  = 512
MEL_FPS    = AUDIO_SR / AUDIO_HOP  # ≈ 31.25
PESTO_FPS  = 20.0
MIDI_FPS   = 10.0
WINDOW_SEC = 30


# ── Test split ────────────────────────────────────────────────────────────────

def load_version_test_hashes():
    lines = [l.strip() for l in (VERSION_SPLIT / 'test.txt').read_text('utf-8').splitlines() if l.strip()]
    return [l.split('-')[0] for l in lines]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_frame_model(model_dir: Path, device: str):
    cfg = OmegaConf.load(model_dir / 'config.yaml')
    model_class = getattr(frame_models, cfg.model.name)
    model = model_class(cfg.model.params).to(device)
    state = torch.load(model_dir / 'fold1_best_model.pt', map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def load_midi_model(midi_dir: Path, device: str):
    # Use this repo's midi.yaml, not midi_dir/.hydra/config.yaml: the old
    # snapshot encodes pool_size/dilation as lists, and this repo's Conv2DGRU
    # eval()s them as strings. The weights themselves load either way.
    cfg = OmegaConf.load(REPO_FRAME / 'configs/frame/midi.yaml')
    model = frame_models.Conv2DGRU(cfg.model.params).to(device)
    state = torch.load(midi_dir / 'fold1_best_model.pt', map_location=device, weights_only=True)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model, cfg


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_frame_dataset(cfg, test_hash_keys):
    dataset_class = getattr(frame_datasets, cfg.dataset.name)
    params = OmegaConf.to_container(cfg.dataset.params)
    params['aug']      = False
    params['is_valid'] = True
    ds = dataset_class(**cfg.data, **params)
    test_set = set(test_hash_keys)
    ds.loaded_hash  = [h for h in ds.loaded_hash if h in test_set]
    ds.loaded_data  = {h: v for h, v in ds.loaded_data.items()  if h in test_set}
    ds.loaded_label = {h: v for h, v in ds.loaded_label.items() if h in test_set}
    return ds


def load_midi_dataset(midi_cfg, test_hash_keys):
    """MidiFrameDataset is hash-keyed, so the generic loader above covers it."""
    ds = load_frame_dataset(midi_cfg, test_hash_keys)
    return ds, int(midi_cfg.dataset.params.fs)


# ── Full-song inference ───────────────────────────────────────────────────────

@torch.no_grad()
def infer_full_song_2d(model, feature: torch.Tensor, window: int, device: str) -> np.ndarray:
    """feature: (n_bins, T) → probs: (T, C)  [for Conv2DGRU mel model]"""
    n_bins, T = feature.shape
    all_probs = []
    for start in range(0, T, window):
        end  = min(start + window, T)
        chunk = feature[:, start:end]
        if chunk.shape[1] < window:
            chunk = torch.cat([chunk, torch.zeros(n_bins, window - chunk.shape[1])], dim=1)
        probs = torch.softmax(model(chunk.unsqueeze(0).to(device)), dim=-1)[0].cpu().numpy()
        all_probs.append(probs[:end - start])
    return np.concatenate(all_probs, axis=0)


@torch.no_grad()
def infer_full_song_1d(model, feature: torch.Tensor, window: int, device: str) -> np.ndarray:
    """feature: (2, T) → probs: (T, C)  [for Conv1DGRU pesto model]"""
    C_in, T = feature.shape
    all_probs = []
    for start in range(0, T, window):
        end   = min(start + window, T)
        chunk = feature[:, start:end]
        if chunk.shape[1] < window:
            chunk = torch.cat([chunk, torch.zeros(C_in, window - chunk.shape[1])], dim=1)
        probs = torch.softmax(model(chunk.unsqueeze(0).to(device)), dim=-1)[0].cpu().numpy()
        all_probs.append(probs[:end - start])
    return np.concatenate(all_probs, axis=0)


# ── Temporal alignment ────────────────────────────────────────────────────────

def align_to_midi(probs: np.ndarray, src_fps: float, T_midi: int) -> np.ndarray:
    """Average-pool probs from src_fps down to MIDI_FPS (10fps)."""
    T_src, C = probs.shape
    aligned = np.zeros((T_midi, C), dtype=np.float32)
    ratio = src_fps / MIDI_FPS
    for j in range(T_midi):
        a0 = max(0, min(int(round(j * ratio)),     T_src - 1))
        a1 = max(a0 + 1, min(int(round((j + 1) * ratio)), T_src))
        aligned[j] = probs[a0:a1].mean(axis=0)
    return aligned


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray):
    mask    = y_true != IGNORE_IDX
    overall = float((y_pred[mask] == y_true[mask]).sum()) / max(mask.sum(), 1)
    per_cls = {}
    for idx, name in CLASS_NAMES.items():
        cm = y_true == idx
        per_cls[name] = float((y_pred[cm] == y_true[cm]).sum()) / max(cm.sum(), 1) \
                        if cm.sum() > 0 else float('nan')
    return overall, per_cls


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs=3, type=float, default=None,
                        metavar=('W_MEL', 'W_PESTO', 'W_MIDI'),
                        help='Soft-voting weights (must sum to 1). Default: equal 1/3.')
    args = parser.parse_args()

    weights = np.array(args.weights if args.weights else [1/3, 1/3, 1/3], dtype=np.float32)
    assert abs(weights.sum() - 1.0) < 1e-5, "Weights must sum to 1"
    w_mel, w_pesto, w_midi = weights
    print(f"Weights  Mel={w_mel:.3f}  Pesto={w_pesto:.3f}  MIDI={w_midi:.3f}")
    print(f"Device  : {DEV}")

    # Test split
    test_hashes = load_version_test_hashes()
    print(f"Version test songs: {len(set(test_hashes))}\n")

    # Models
    print("Loading models...")
    mel_model,   mel_cfg   = load_frame_model(MEL_DIR,   DEV)
    pesto_model, pesto_cfg = load_frame_model(PESTO_DIR, DEV)
    midi_model,  midi_cfg  = load_midi_model(MIDI_DIR,   DEV)
    print(f"  Mel   : {mel_cfg.model.name}   ({MEL_DIR.name})")
    print(f"  Pesto : {pesto_cfg.model.name}  ({PESTO_DIR.name})")
    print(f"  MIDI  : Conv2DGRU  ({MIDI_DIR.name})")

    # Datasets
    print("\nPre-loading features...")
    mel_ds   = load_frame_dataset(mel_cfg,   test_hashes)
    pesto_ds = load_frame_dataset(pesto_cfg, test_hashes)
    midi_ds, midi_fs = load_midi_dataset(midi_cfg, test_hashes)

    mel_win   = int(WINDOW_SEC * MEL_FPS)    # ~937 frames
    pesto_win = int(WINDOW_SEC * PESTO_FPS)  # 600  frames
    midi_win  = int(WINDOW_SEC * midi_fs)    # 300  frames

    common = (set(mel_ds.loaded_data.keys())
              & set(pesto_ds.loaded_data.keys())
              & set(midi_ds.loaded_data.keys()))
    print(f"\nCommon songs (all 3 modalities): {len(common)}")

    all_y_true, all_y_pred = [], []

    for hk in tqdm(sorted(common), desc='Ensemble inference'):
        mel_feat   = mel_ds.loaded_data[hk]                       # (40, T_mel)
        pesto_feat = pesto_ds.loaded_data[hk]                     # (2,  T_pesto)
        piano_roll = midi_ds.loaded_data[hk]                       # (128, T_midi)
        gt_label   = midi_ds.loaded_label[hk]                      # (T_midi, 5)

        mel_probs   = infer_full_song_2d(mel_model,   mel_feat,   mel_win,   DEV)
        pesto_probs = infer_full_song_1d(pesto_model, pesto_feat, pesto_win, DEV)
        midi_probs  = infer_full_song_2d(midi_model,  piano_roll, midi_win,  DEV)

        T_midi = midi_probs.shape[0]

        mel_a   = align_to_midi(mel_probs,   MEL_FPS,   T_midi)
        pesto_a = align_to_midi(pesto_probs, PESTO_FPS, T_midi)

        ens   = w_mel * mel_a + w_pesto * pesto_a + w_midi * midi_probs
        pred  = ens.argmax(axis=1)
        gt    = gt_label.numpy().argmax(axis=1)[:T_midi]

        all_y_true.append(gt)
        all_y_pred.append(pred)

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    overall, per_cls = compute_accuracy(y_true, y_pred)

    print(f"\n{'='*60}")
    print("Ensemble Results — Mel + Pesto + MIDI (Version Test)")
    print(f"{'='*60}")
    print(f"Overall masked accuracy : {overall:.4f}  ({overall*100:.2f}%)")
    print()
    print("Per-class accuracy:")
    for name, acc in per_cls.items():
        if np.isnan(acc):
            print(f"  {name:8s}: N/A (no samples)")
        else:
            print(f"  {name:8s}: {acc:.4f}  ({acc*100:.2f}%)")

    evaluated = int((y_true != IGNORE_IDX).sum())
    total     = len(y_true)
    print(f"\nTotal frames : {total:,}  (non-Unknown: {evaluated:,})")
    print(f"Songs        : {len(common)}")


if __name__ == '__main__':
    main()
