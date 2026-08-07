"""
Ensemble Inference — Song-Stratified 10-Fold Cross Validation
  Mel (fold k) + MIDI (fold k) + Pesto (fold k) soft voting

Each fold uses the same song-stratified train/val/test split that was
used during training.  The 10 Mel and 10 Pesto model directories are
discovered automatically from their config.yaml target_folds field.
MIDI fold checkpoints (best_model_fold{k}.pt) are discovered recursively
under --midi_dir.

Usage:
    cd /path/to/PansoriModeChangeDetection
    # Mel + MIDI + Pesto (default)
    python scripts/ensemble/ensemble_inference_song_stratified.py
    # Mel + Pesto only (disable MIDI)
    python scripts/ensemble/ensemble_inference_song_stratified.py --no_midi
    # Custom weights (Mel MIDI Pesto)
    python scripts/ensemble/ensemble_inference_song_stratified.py \\
        --midi_dir $PANSORI_MIDI_REPO/outputs \\
        --weights 0.35 0.35 0.30
    python scripts/ensemble/ensemble_inference_song_stratified.py --no_posteriors
"""

import os
os.environ['WANDB_MODE'] = 'disabled'

import sys
import re
import argparse
import math
from collections import defaultdict
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
from paths import DATA_ROOT, MIDI_REPO, REPO_ROOT

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import models as frame_models
import datasets as frame_datasets
from trainers import plot_posteriorgram

REPO_FRAME = REPO_ROOT
REPO_MIDI  = MIDI_REPO

MEL_DIR   = REPO_FRAME / 'weights/frame/Mel_Original_Song_Stratified'
PESTO_DIR = REPO_FRAME / 'weights/frame/Pesto_Song_Stratified'
OUT_DIR   = REPO_FRAME / 'outputs/ensemble_song_stratified'

CLASSES          = ['Unknown', '우조', '계면조', '아니리', '창조']
EVAL_CLASSES_IDX = [1, 2, 3, 4]
EVAL_CLASSES_NAME = ['우조', '계면조', '아니리', '창조']

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

AUDIO_SR         = 16000
AUDIO_HOP        = 512
AUDIO_FPS        = AUDIO_SR / AUDIO_HOP   # ≈ 31.25
WINDOW_SEC       = 30
AUDIO_WIN_FRAMES = int(WINDOW_SEC * AUDIO_FPS)
PESTO_FPS        = 20
PESTO_WIN_FRAMES = int(WINDOW_SEC * PESTO_FPS)
MIDI_FPS         = 10
MIDI_WIN_FRAMES  = int(WINDOW_SEC * MIDI_FPS)


# ── Model directory discovery ─────────────────────────────────────────────────

def discover_fold_dirs(base_dir: Path) -> dict[int, Path]:
    """Return {fold_number: model_dir} by reading target_folds[0] from config.yaml."""
    result = {}
    for d in sorted(base_dir.iterdir()):
        cfg_path = d / 'config.yaml'
        if not cfg_path.exists():
            continue
        cfg = OmegaConf.load(cfg_path)
        folds = list(cfg.train.get('target_folds', []))
        if len(folds) == 1:
            result[folds[0]] = d
    return dict(sorted(result.items()))


def discover_midi_ckpts(midi_dir: Path) -> dict[int, Path]:
    """
    Recursively find best_model_fold{k}.pt files under midi_dir.
    Returns {fold_number: ckpt_path}.
    """
    result = {}
    for pt in sorted(midi_dir.rglob('best_model_fold*.pt')):
        m = re.search(r'best_model_fold(\d+)\.pt$', pt.name)
        if m:
            result[int(m.group(1))] = pt
    return dict(sorted(result.items()))


# ── Model loading ─────────────────────────────────────────────────────────────

def load_frame_model(model_dir: Path, device: str):
    cfg = OmegaConf.load(model_dir / 'config.yaml')
    fold_num = cfg.train.target_folds[0]
    weight_file = model_dir / f'fold{fold_num}_best_model.pt'
    model_class = getattr(frame_models, cfg.model.name)
    model = model_class(cfg.model.params).to(device)
    state = torch.load(weight_file, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def _load_midi_cfg():
    """Load and merge the MIDI project config (cached after first call)."""
    if not hasattr(_load_midi_cfg, '_cache'):
        midi_cfg = OmegaConf.load(REPO_MIDI / 'configs/config.yaml')
        for key in ['data', 'model', 'train']:
            sub = OmegaConf.load(REPO_MIDI / f'configs/{key}/{key}.yaml')
            OmegaConf.update(midi_cfg, key, sub, merge=True)
        _load_midi_cfg._cache = midi_cfg
    return _load_midi_cfg._cache


def _get_midi_model_class():
    """Import Conv2DGRU from MIDI repo without polluting frame model imports."""
    _evicted = {k: v for k, v in sys.modules.items()
                if k in ('models', 'datasets') or
                   k.startswith('models.') or k.startswith('datasets.')}
    for k in _evicted:
        del sys.modules[k]
    sys.path.insert(0, str(REPO_MIDI))
    try:
        from models.model_zoo import Conv2DGRU as MidiConv2DGRU
        return MidiConv2DGRU
    finally:
        sys.path.pop(0)
        for k in list(sys.modules.keys()):
            if k in ('models', 'datasets') or \
               k.startswith('models.') or k.startswith('datasets.'):
                del sys.modules[k]
        sys.modules.update(_evicted)


def load_midi_model(ckpt: Path, device: str):
    midi_cfg = _load_midi_cfg()
    MidiConv2DGRU = _get_midi_model_class()
    model = MidiConv2DGRU(midi_cfg.model).to(device)
    state = torch.load(ckpt, map_location=device)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model, midi_cfg


# ── Song-stratified split reconstruction ─────────────────────────────────────

def build_song_stratified_splits(song_strat_dir: Path) -> list[dict]:
    """
    Replicate the SongStratified fold generation from trainers.py.
    Returns list of 10 dicts with 'train', 'val', 'test' as FULL LINE lists
    (e.g. '01ed19bb-06-김소희-춘향가_...'), mirroring ensemble_inference.py's
    convention so that both hash-key lookup and MIDI name resolution work.
    """
    halves = defaultdict(dict)
    for txt_file in sorted(song_strat_dir.glob('*.txt')):
        m = re.match(r'^(.+)_([12])$', txt_file.stem)
        if not m:
            continue
        base, idx = m.group(1), int(m.group(2))
        lines = [l.strip() for l in txt_file.read_text(encoding='utf-8').splitlines() if l.strip()]
        halves[base][idx] = lines   # store full lines, not just hash prefixes

    genres = sorted(halves.keys())
    splits = []
    for held_out in genres:
        train_lines = [ln for g in genres if g != held_out
                       for ln in halves[g][1] + halves[g][2]]
        for val_idx, test_idx in [(1, 2), (2, 1)]:
            splits.append({
                'train': train_lines,
                'val':   halves[held_out][val_idx],
                'test':  halves[held_out][test_idx],
            })
    return splits


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_frame_dataset(cfg, test_hash_keys: list):
    dataset_class = getattr(frame_datasets, cfg.dataset.name)
    params = OmegaConf.to_container(cfg.dataset.params)
    params['aug'] = False
    params['is_valid'] = True
    ds = dataset_class(**cfg.data, **params)
    test_set = set(test_hash_keys)
    ds.loaded_hash  = [h for h in ds.loaded_hash  if h in test_set]
    ds.loaded_data  = {h: v for h, v in ds.loaded_data.items()  if h in test_set}
    ds.loaded_label = {h: v for h, v in ds.loaded_label.items() if h in test_set}
    return ds


def load_midi_dataset(midi_cfg, test_midi_names: list):
    _evicted = {k: v for k, v in sys.modules.items()
                if k == 'datasets' or k.startswith('datasets.')}
    for k in _evicted:
        del sys.modules[k]
    sys.path.insert(0, str(REPO_MIDI))
    try:
        from datasets.dataset import BaseDataset
    finally:
        sys.path.pop(0)
        for k in list(sys.modules.keys()):
            if k == 'datasets' or k.startswith('datasets.'):
                del sys.modules[k]
        sys.modules.update(_evicted)

    fs = midi_cfg.data.fs
    window_size = int(midi_cfg.data.window_size * fs)
    ds = BaseDataset(
        midi_cfg.data.dir.midi_dir,
        midi_cfg.data.dir.label_path,
        song_list=test_midi_names,
        fs=fs,
        window_size=window_size,
        is_train=False,
    )
    ds.hash_to_idx = {
        item['song_name'].split('-')[0]: idx
        for idx, item in enumerate(ds.memory_cache)
    }
    return ds, fs, window_size


# ── MIDI test name resolution ─────────────────────────────────────────────────

def build_hash_to_midi_name(midi_cfg) -> dict[str, str]:
    """Build mapping: hash_key prefix → midi filename."""
    import json, unicodedata
    with open(midi_cfg.data.dir.label_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    def _midi_key(name):
        n = name.replace('_vocal.mid', '')
        return re.sub(r'^[0-9a-f]+-\d+-', '', n)

    midi_dir = Path(midi_cfg.data.dir.midi_dir)
    key_to_midi = {}
    for item in raw:
        fu = unicodedata.normalize('NFC', item['file_upload'])
        midi_name = fu.rsplit('.', 1)[0] + '_vocal.mid'
        if (midi_dir / midi_name).exists():
            key_to_midi[_midi_key(midi_name)] = midi_name
    return key_to_midi


def lines_to_midi_names(full_lines: list, key_to_midi: dict) -> list[str]:
    """full_lines: raw txt file entries like '01ed19bb-06-김소희-...' → midi filenames."""
    def _strip_uuid(line):
        return re.sub(r'^[0-9a-f]+-\d+-', '', line)
    return [key_to_midi[_strip_uuid(ln)] for ln in full_lines if _strip_uuid(ln) in key_to_midi]


# ── Full-song inference ───────────────────────────────────────────────────────

@torch.no_grad()
def full_song_frame_inference(model, feature_full: torch.Tensor, window_frames: int, device: str):
    """feature_full: (n_bins, T) → probs (T, C) at audio frame rate"""
    n_bins, T = feature_full.shape
    all_probs = []
    for start in range(0, T, window_frames):
        end   = min(start + window_frames, T)
        chunk = feature_full[:, start:end]
        if chunk.shape[1] < window_frames:
            chunk = torch.cat([chunk, torch.zeros(n_bins, window_frames - chunk.shape[1])], dim=1)
        probs = torch.softmax(model(chunk.unsqueeze(0).to(device)), dim=-1)[0].cpu().numpy()
        all_probs.append(probs[:end - start])
    return np.concatenate(all_probs, axis=0)  # (T, C)


@torch.no_grad()
def full_song_midi_inference(model, piano_roll: torch.Tensor, window_size: int, device: str):
    """piano_roll: (128, T) → probs (T, C) at MIDI frame rate"""
    T = piano_roll.shape[1]
    all_probs = []
    for start in range(0, T, window_size):
        end   = min(start + window_size, T)
        chunk = piano_roll[:, start:end]
        if chunk.shape[1] < window_size:
            chunk = torch.cat([chunk, torch.zeros(128, window_size - chunk.shape[1])], dim=1)
        probs = torch.softmax(model(chunk.unsqueeze(0).to(device)), dim=-1)[0].cpu().numpy()
        all_probs.append(probs[:end - start])
    return np.concatenate(all_probs, axis=0)  # (T, C)


# ── Temporal alignment (audio fps → MIDI fps) ────────────────────────────────

def align_to_fps(src_probs: np.ndarray, src_fps: float, T_dst: int, dst_fps: float) -> np.ndarray:
    """Resample src_probs from src_fps to dst_fps via frame-averaged pooling."""
    T_src, C = src_probs.shape
    aligned = np.zeros((T_dst, C), dtype=np.float32)
    ratio = src_fps / dst_fps
    for j in range(T_dst):
        a_start = max(0, min(int(round(j * ratio)),             T_src - 1))
        a_end   = max(a_start + 1, min(int(round((j + 1) * ratio)), T_src))
        aligned[j] = src_probs[a_start:a_end].mean(axis=0)
    return aligned


# ── Label helpers ─────────────────────────────────────────────────────────────

def ms_label_to_frame_cls(ms_label: torch.Tensor, hop_length: int, sr: int) -> np.ndarray:
    """Convert ms-resolution one-hot label tensor → integer class array at audio fps."""
    ms_per_frame = hop_length / sr * 1000
    frame_width  = int(ms_per_frame)
    num_frames   = int(ms_label.shape[0] / ms_per_frame)
    if num_frames == 0:
        return np.array([ms_label.argmax(dim=-1).item()])
    trimmed = ms_label[:num_frames * frame_width]
    grouped = trimmed.view(num_frames, frame_width, -1).sum(dim=1)
    return grouped.argmax(dim=-1).numpy().astype(np.int64)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = y_true != 0
    if mask.sum() == 0:
        return {'acc': 0.0, 'f1_macro': 0.0,
                **{f'f1_{c}': 0.0 for c in EVAL_CLASSES_NAME}}
    yt, yp = y_true[mask], y_pred[mask]
    acc    = float((yt == yp).mean())
    f1_per = f1_score(yt, yp, average=None,    labels=EVAL_CLASSES_IDX, zero_division=0)
    f1_mac = f1_score(yt, yp, average='macro', labels=EVAL_CLASSES_IDX, zero_division=0)
    return {
        'acc': acc, 'f1_macro': float(f1_mac),
        **{f'f1_{EVAL_CLASSES_NAME[i]}': float(f1_per[i]) for i in range(len(EVAL_CLASSES_NAME))}
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Song-stratified ensemble inference: Mel + MIDI + Pesto (per-fold)')
    parser.add_argument('--mel_dir', type=str, default=str(MEL_DIR),
                        help=f'Directory containing per-fold Mel model subdirs. Default: {MEL_DIR}')
    parser.add_argument('--pesto_dir', type=str, default=str(PESTO_DIR),
                        help=f'Directory containing per-fold Pesto model subdirs. Default: {PESTO_DIR}')
    parser.add_argument('--weights', nargs='+', type=float, default=None,
                        metavar='W',
                        help='Soft voting weights. 2 values for Mel+Pesto, '
                             '3 values for Mel+MIDI+Pesto (must sum to 1).')
    default_midi_dir = str(REPO_MIDI / 'outputs')
    parser.add_argument('--midi_dir', type=str, default=default_midi_dir,
                        help=f'Directory containing best_model_fold{{k}}.pt files '
                             f'(searched recursively). Default: {default_midi_dir}')
    parser.add_argument('--no_midi', action='store_true',
                        help='Disable MIDI modality (Mel + Pesto only).')
    parser.add_argument('--no_posteriors', action='store_true',
                        help='Skip saving per-segment posteriorgram PNGs.')
    parser.add_argument('--folds', nargs='+', type=int, default=None,
                        metavar='K', help='Run only these fold numbers (default: all 10).')
    args = parser.parse_args()

    midi_ckpt_map: dict[int, Path] = {}
    if not args.no_midi and args.midi_dir:
        midi_ckpt_map = discover_midi_ckpts(Path(args.midi_dir))
        if not midi_ckpt_map:
            raise FileNotFoundError(f"No best_model_fold*.pt found under {args.midi_dir}")
        print(f"Found MIDI checkpoints for folds: {sorted(midi_ckpt_map.keys())}")

    use_midi = bool(midi_ckpt_map)
    n_modalities = 3 if use_midi else 2

    if args.weights:
        assert len(args.weights) == n_modalities, \
            f"Expected {n_modalities} weights, got {len(args.weights)}"
        weights = np.array(args.weights, dtype=np.float32)
        assert abs(weights.sum() - 1.0) < 1e-5, "Weights must sum to 1"
    else:
        weights = np.full(n_modalities, 1.0 / n_modalities, dtype=np.float32)

    w_mel, w_pesto = weights[0], weights[-1]
    w_midi = weights[1] if use_midi else 0.0

    print(f"Modalities : Mel" + ("  MIDI" if use_midi else "") + "  Pesto")
    print(f"Weights    : {w_mel:.3f}" + (f"  {w_midi:.3f}" if use_midi else "") + f"  {w_pesto:.3f}")
    print(f"Device     : {DEV}\n")

    # ── Discover fold model directories ──────────────────────────────────────
    mel_fold_dirs   = discover_fold_dirs(Path(args.mel_dir))
    pesto_fold_dirs = discover_fold_dirs(Path(args.pesto_dir))
    assert mel_fold_dirs.keys() == pesto_fold_dirs.keys(), \
        "Mel and Pesto fold sets don't match"

    target_folds = sorted(args.folds) if args.folds else sorted(mel_fold_dirs.keys())
    print(f"Running folds: {target_folds}\n")

    # ── Pre-load shared MIDI config + key→filename mapping ────────────────────
    midi_cfg = key_to_midi = None
    if use_midi:
        midi_cfg    = _load_midi_cfg()
        midi_fs     = midi_cfg.data.fs
        midi_win    = int(midi_cfg.data.window_size * midi_fs)
        key_to_midi = build_hash_to_midi_name(midi_cfg)
        print(f"MIDI config loaded  fs={midi_fs}  window={midi_win} frames\n")

    # ── Load VT hashes (version test 18 songs) ───────────────────────────────
    vt_file = DATA_ROOT / 'pansori_version_split/test.txt'
    vt_hashes = {l.strip().split('-')[0] for l in vt_file.read_text().splitlines() if l.strip()}
    print(f"Version test songs: {len(vt_hashes)}\n")

    # ── Collect results across all folds ──────────────────────────────────────
    all_fold_metrics   = []
    all_segment_rows   = []
    global_y_true      = []
    global_y_pred      = []
    vt_y_true          = []
    vt_y_pred          = []
    log_lines          = []   # accumulated text for summary .txt

    for fold in target_folds:
        print(f"\n{'='*60}")
        print(f"  Fold {fold}")
        print(f"{'='*60}")

        mel_dir   = mel_fold_dirs[fold]
        pesto_dir = pesto_fold_dirs[fold]

        # ── Load models ───────────────────────────────────────────────────────
        mel_model, mel_cfg     = load_frame_model(mel_dir,   DEV)
        pesto_model, pesto_cfg = load_frame_model(pesto_dir, DEV)
        print(f"  Mel   : {mel_dir.name}")
        print(f"  Pesto : {pesto_dir.name}")

        fold_midi_model = None
        if use_midi:
            if fold not in midi_ckpt_map:
                print(f"  WARNING: no MIDI checkpoint for fold {fold} — skipping MIDI for this fold")
            else:
                fold_midi_model, _ = load_midi_model(midi_ckpt_map[fold], DEV)
                print(f"  MIDI: {midi_ckpt_map[fold].relative_to(REPO_MIDI.parent)}")

        fold_use_midi = fold_midi_model is not None

        # ── Reconstruct test split from song_stratified_dir ───────────────────
        song_strat_dir = Path(mel_cfg.train.song_stratified_dir)
        splits      = build_song_stratified_splits(song_strat_dir)
        test_lines  = splits[fold - 1]['test']              # full lines
        test_keys   = [ln.split('-')[0] for ln in test_lines]  # hash prefixes
        print(f"  Test songs: {len(test_keys)}")

        # ── Load datasets filtered to test split ──────────────────────────────
        mel_ds   = load_frame_dataset(mel_cfg,   test_keys)
        pesto_ds = load_frame_dataset(pesto_cfg, test_keys)

        common_keys = sorted(set(mel_ds.loaded_data.keys()) & set(pesto_ds.loaded_data.keys()))
        print(f"  Matched Mel∩Pesto: {len(common_keys)} songs")

        # ── Optional MIDI dataset ─────────────────────────────────────────────
        midi_ds = None
        if fold_use_midi:
            midi_test_names = lines_to_midi_names(test_lines, key_to_midi)
            midi_ds, _midi_fs, _midi_win = load_midi_dataset(midi_cfg, midi_test_names)
            common_keys = [k for k in common_keys if k in midi_ds.hash_to_idx]
            print(f"  With MIDI: {len(common_keys)} songs")

        # ── Per-fold posteriorgram dir ────────────────────────────────────────
        post_dir = OUT_DIR / f'fold{fold}_posteriorgrams'
        if not args.no_posteriors:
            post_dir.mkdir(parents=True, exist_ok=True)

        fold_y_true = []
        fold_y_pred = []

        for hash_key in tqdm(common_keys, desc=f'Fold {fold} inference'):
            mel_full   = mel_ds.loaded_data[hash_key]    # (n_bins, T_audio)
            pesto_full = pesto_ds.loaded_data[hash_key]  # (2, T_pesto) at PESTO_FPS

            mel_probs   = full_song_frame_inference(mel_model,   mel_full,   AUDIO_WIN_FRAMES, DEV)
            pesto_probs = full_song_frame_inference(pesto_model, pesto_full, PESTO_WIN_FRAMES, DEV)

            if fold_use_midi:
                cache_idx  = midi_ds.hash_to_idx[hash_key]
                midi_item  = midi_ds.memory_cache[cache_idx]
                piano_roll = midi_item['piano_roll']                      # (128, T_midi)
                midi_probs = full_song_midi_inference(fold_midi_model, piano_roll, midi_win, DEV)
                T_midi     = midi_probs.shape[0]

                mel_aligned   = align_to_fps(mel_probs,   AUDIO_FPS,  T_midi, MIDI_FPS)
                pesto_aligned = align_to_fps(pesto_probs, PESTO_FPS,  T_midi, MIDI_FPS)
                ens_probs     = w_mel * mel_aligned + w_midi * midi_probs + w_pesto * pesto_aligned

                gt_label_np = midi_item['frame_label'].numpy()            # (T_midi, C) one-hot
                gt_cls      = gt_label_np.argmax(axis=1)
                fps_ref     = MIDI_FPS
                win_ref     = MIDI_WIN_FRAMES
            else:
                # Use Pesto fps as reference; align Mel to Pesto fps
                T_pesto     = pesto_probs.shape[0]
                mel_aligned = align_to_fps(mel_probs, AUDIO_FPS, T_pesto, PESTO_FPS)
                ens_probs   = w_mel * mel_aligned + w_pesto * pesto_probs
                # Pesto loaded_label is already frame-level (T_pesto, C)
                pesto_label = pesto_ds.loaded_label[hash_key]  # tensor (T_pesto, C)
                gt_cls = pesto_label.argmax(dim=-1).numpy()
                T_ref = ens_probs.shape[0]
                if gt_cls.shape[0] > T_ref:
                    gt_cls = gt_cls[:T_ref]
                elif gt_cls.shape[0] < T_ref:
                    gt_cls = np.pad(gt_cls, (0, T_ref - gt_cls.shape[0]))
                fps_ref = PESTO_FPS
                win_ref = PESTO_WIN_FRAMES

            ens_pred = ens_probs.argmax(axis=1)
            fold_y_true.append(gt_cls)
            fold_y_pred.append(ens_pred)
            global_y_true.append(gt_cls)
            global_y_pred.append(ens_pred)
            if hash_key in vt_hashes:
                vt_y_true.append(gt_cls)
                vt_y_pred.append(ens_pred)

            T_ref = ens_probs.shape[0]
            for seg_start in range(0, T_ref, win_ref):
                seg_end   = min(seg_start + win_ref, T_ref)
                start_sec = seg_start / fps_ref
                end_sec   = seg_end   / fps_ref

                gt_seg  = gt_cls[seg_start:seg_end]
                pr_seg  = ens_pred[seg_start:seg_end]
                m       = compute_metrics(gt_seg, pr_seg)

                mel_seg_prob   = mel_aligned[seg_start:seg_end].mean(axis=0)
                pesto_seg_prob = (pesto_probs if not fold_use_midi else pesto_aligned)[seg_start:seg_end].mean(axis=0)
                ens_seg_prob   = ens_probs[seg_start:seg_end].mean(axis=0)

                row = {
                    'fold':       fold,
                    'song_name':  hash_key,
                    'start_sec':  round(start_sec, 1),
                    'end_sec':    round(end_sec,   1),
                    'time_range': f"{int(start_sec)}-{int(end_sec)}s",
                    **m,
                    **{f'p_mel_{c}':   round(float(mel_seg_prob[i]),   4) for i, c in enumerate(CLASSES)},
                    **{f'p_pesto_{c}': round(float(pesto_seg_prob[i]), 4) for i, c in enumerate(CLASSES)},
                    **{f'p_ens_{c}':   round(float(ens_seg_prob[i]),   4) for i, c in enumerate(CLASSES)},
                }
                if fold_use_midi:
                    midi_seg_prob = midi_probs[seg_start:seg_end].mean(axis=0)
                    row.update({f'p_midi_{c}': round(float(midi_seg_prob[i]), 4)
                                for i, c in enumerate(CLASSES)})
                all_segment_rows.append(row)

                if not args.no_posteriors:
                    gt_onehot = np.eye(len(CLASSES))[gt_cls[seg_start:seg_end]]
                    fig = plot_posteriorgram(
                        f"{hash_key}  {int(start_sec)}-{int(end_sec)}s  fold{fold}",
                        gt_onehot, ens_probs[seg_start:seg_end], CLASSES)
                    fig.savefig(
                        post_dir / f"{hash_key}_{int(start_sec)}-{int(end_sec)}s.png",
                        dpi=100, bbox_inches='tight')
                    plt.close(fig)

        # ── Per-fold summary ──────────────────────────────────────────────────
        yt_fold = np.concatenate(fold_y_true)
        yp_fold = np.concatenate(fold_y_pred)
        fm      = compute_metrics(yt_fold, yp_fold)

        fold_header = f"\n  Fold {fold} results ({len(common_keys)} songs):"
        fold_lines = [
            fold_header,
            f"    Acc      : {fm['acc']:.4f}",
            f"    F1 Macro : {fm['f1_macro']:.4f}",
            *(f"    F1 {c:<6}: {fm[f'f1_{c}']:.4f}" for c in EVAL_CLASSES_NAME),
        ]
        for line in fold_lines:
            print(line)
        log_lines.extend(fold_lines)

        all_fold_metrics.append({'fold': fold, **fm, 'n_songs': len(common_keys)})

        # Free VRAM between folds
        del mel_model, pesto_model
        if fold_midi_model is not None:
            del fold_midi_model
        torch.cuda.empty_cache()

    # ── Aggregate across all folds ────────────────────────────────────────────
    yt_all = np.concatenate(global_y_true)
    yp_all = np.concatenate(global_y_pred)
    overall = compute_metrics(yt_all, yp_all)

    per_fold_f1  = [row['f1_macro'] for row in all_fold_metrics]
    overall_lines = [
        f"\n{'='*60}",
        "Overall Results (all folds concatenated)",
        f"{'='*60}",
        f"  Acc      : {overall['acc']:.4f}",
        f"  F1 Macro : {overall['f1_macro']:.4f}",
        *(f"  F1 {c:<6}: {overall[f'f1_{c}']:.4f}" for c in EVAL_CLASSES_NAME),
        f"  Segments : {len(all_segment_rows)}",
        f"\n  Avg per-fold F1 macro: {np.mean(per_fold_f1):.4f} ± {np.std(per_fold_f1):.4f}",
    ]
    for line in overall_lines:
        print(line)
    log_lines.extend(overall_lines)

    # ── Version Test 18곡 frame-level F1 ─────────────────────────────────────
    if vt_y_true:
        yt_vt = np.concatenate(vt_y_true)
        yp_vt = np.concatenate(vt_y_pred)
        vt_m  = compute_metrics(yt_vt, yp_vt)
        vt_lines = [
            f"\n{'='*60}",
            f"Version Test (18곡) — frame-level F1  [{len(vt_y_true)} songs covered]",
            f"{'='*60}",
            f"  Acc      : {vt_m['acc']:.4f}",
            f"  F1 Macro : {vt_m['f1_macro']:.4f}",
            *(f"  F1 {c:<6}: {vt_m[f'f1_{c}']:.4f}" for c in EVAL_CLASSES_NAME),
        ]
        for line in vt_lines:
            print(line)
        log_lines.extend(vt_lines)

    # ── Save outputs ──────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # When running a single fold (called from shell script per-fold), use
    # fold-scoped filenames so parallel/sequential runs don't overwrite each other.
    single_fold = len(target_folds) == 1
    prefix = f'fold{target_folds[0]}_' if single_fold else ''

    # Save VT frame predictions for cross-fold aggregation
    if vt_y_true:
        vt_npz = OUT_DIR / f'{prefix}vt_frames.npz'
        np.savez(vt_npz,
                 y_true=np.concatenate(vt_y_true),
                 y_pred=np.concatenate(vt_y_pred))
        print(f"VT frame preds    → {vt_npz}")

    fold_csv = OUT_DIR / f'{prefix}fold_metrics.csv'
    pd.DataFrame(all_fold_metrics).to_csv(fold_csv, index=False)
    print(f"\nPer-fold metrics  → {fold_csv}")

    seg_csv = OUT_DIR / f'{prefix}segment_results.csv'
    pd.DataFrame(all_segment_rows).to_csv(seg_csv, index=False)
    print(f"Segment results   → {seg_csv}")

    overall_csv = OUT_DIR / f'{prefix}overall_metrics.csv'
    pd.DataFrame([{'n_folds': len(target_folds), **overall}]).to_csv(overall_csv, index=False)
    print(f"Overall metrics   → {overall_csv}")

    txt_path = OUT_DIR / f'{prefix}results_summary.txt'
    txt_path.write_text('\n'.join(log_lines) + '\n', encoding='utf-8')
    print(f"Text summary      → {txt_path}")

    if not args.no_posteriors:
        png_total = sum(len(list((OUT_DIR / f'fold{f}_posteriorgrams').glob('*.png')))
                        for f in target_folds)
        print(f"Posteriorgrams    → {OUT_DIR}  ({png_total} total PNGs)")


if __name__ == '__main__':
    main()
