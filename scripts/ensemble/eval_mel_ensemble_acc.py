"""
Masked accuracy (Unknown excluded) for:
  1. Mel Original Song Stratified
  2. Ensemble (from pre-saved fold*_overall_metrics.csv)

Mel dataset is loaded ONCE and reused across all folds.
"""

import sys

import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
import models as frame_models
import datasets as frame_datasets

REPO_FRAME = REPO_ROOT
MEL_DIR    = REPO_FRAME / 'weights/frame/Mel_Original_Song_Stratified'
ENS_DIR    = REPO_FRAME / 'outputs/ensemble_song_stratified'
DEV        = 'cuda' if torch.cuda.is_available() else 'cpu'

AUDIO_SR   = 16000
AUDIO_HOP  = 512
AUDIO_FPS  = AUDIO_SR / AUDIO_HOP
WIN_SEC    = 30
AUDIO_WIN  = int(WIN_SEC * AUDIO_FPS)


def discover_fold_dirs(base):
    out = {}
    for d in sorted(base.iterdir()):
        cfg_path = d / 'config.yaml'
        if not cfg_path.exists(): continue
        cfg = OmegaConf.load(cfg_path)
        folds = list(cfg.train.get('target_folds', []))
        if len(folds) == 1:
            out[folds[0]] = d
    return dict(sorted(out.items()))


def build_splits(song_strat_dir):
    halves = defaultdict(dict)
    for txt in sorted(Path(song_strat_dir).glob('*.txt')):
        m = re.match(r'^(.+)_([12])$', txt.stem)
        if not m: continue
        base, idx = m.group(1), int(m.group(2))
        lines = [l.strip() for l in txt.read_text(encoding='utf-8').splitlines() if l.strip()]
        halves[base][idx] = lines
    splits = []
    for held in sorted(halves.keys()):
        for val_i, test_i in [(1, 2), (2, 1)]:
            splits.append([ln.split('-')[0] for ln in halves[held][test_i]])
    return splits


@torch.no_grad()
def infer_mel(model, feat, win_frames):
    n_bins, T = feat.shape
    out = []
    for s in range(0, T, win_frames):
        e = min(s + win_frames, T)
        chunk = feat[:, s:e]
        if chunk.shape[1] < win_frames:
            chunk = torch.cat([chunk, torch.zeros(n_bins, win_frames - chunk.shape[1])], dim=1)
        p = torch.softmax(model(chunk.unsqueeze(0).to(DEV)), dim=-1)[0].cpu().numpy()
        out.append(p[:e - s])
    return np.concatenate(out, axis=0)


def masked_acc(y_true, y_pred):
    mask = y_true != 0
    if mask.sum() == 0: return float('nan')
    return float((y_true[mask] == y_pred[mask]).mean())


def main():
    mel_dirs = discover_fold_dirs(MEL_DIR)
    # Use fold 1 config to load the dataset (all folds share same data)
    ref_cfg = OmegaConf.load(mel_dirs[1] / 'config.yaml')

    print("Loading Mel dataset (once)...")
    ds_cls = getattr(frame_datasets, ref_cfg.dataset.name)
    params = OmegaConf.to_container(ref_cfg.dataset.params)
    params['aug'] = False
    params['is_valid'] = True
    mel_ds = ds_cls(**ref_cfg.data, **params)
    print(f"  Loaded {len(mel_ds.loaded_hash)} songs\n")

    song_strat_dir = ref_cfg.train.song_stratified_dir
    splits = build_splits(song_strat_dir)

    mel_accs, ens_accs = [], []

    for fold in sorted(mel_dirs):
        test_keys = set(splits[fold - 1])
        common = [h for h in mel_ds.loaded_hash if h in test_keys]
        print(f"Fold {fold}: {len(common)} test songs", end='  ')

        # Load mel model
        model_cls = getattr(frame_models, ref_cfg.model.name)
        model = model_cls(ref_cfg.model.params).to(DEV)
        model.load_state_dict(torch.load(
            mel_dirs[fold] / f'fold{fold}_best_model.pt',
            map_location=DEV, weights_only=True))
        model.eval()

        yt_list, yp_list = [], []
        for hk in common:
            feat  = mel_ds.loaded_data[hk]    # (n_bins, T_audio)
            label = mel_ds.loaded_label[hk]   # (T_ms, C) — ms resolution
            probs = infer_mel(model, feat, AUDIO_WIN)  # (T_audio, C)
            T = probs.shape[0]

            # Convert ms-level label to frame-level
            frame_label = mel_ds.ms_to_frame_label(label)  # (T_frames, C)
            gt = frame_label.argmax(dim=-1).numpy()
            gt = gt[:T] if len(gt) >= T else np.pad(gt, (0, T - len(gt)))

            yt_list.append(gt)
            yp_list.append(probs.argmax(axis=1))

        del model
        torch.cuda.empty_cache()

        yt = np.concatenate(yt_list)
        yp = np.concatenate(yp_list)
        acc_mel = masked_acc(yt, yp)
        mel_accs.append(acc_mel)

        # Ensemble: read from pre-saved CSV
        ens_csv = ENS_DIR / f'fold{fold}_overall_metrics.csv'
        acc_ens = float(pd.read_csv(ens_csv)['acc'].values[0])
        ens_accs.append(acc_ens)

        print(f"Mel={acc_mel:.4f}  Ens={acc_ens:.4f}")

    print(f"\n{'='*42}")
    print(f"{'Fold':<6} {'Mel Acc':>10} {'Ens Acc':>10}")
    print(f"{'-'*28}")
    for i, (m, e) in enumerate(zip(mel_accs, ens_accs), 1):
        print(f"{i:<6} {m:>10.4f} {e:>10.4f}")
    print(f"{'-'*28}")
    print(f"{'Mean':<6} {np.mean(mel_accs):>10.4f} {np.mean(ens_accs):>10.4f}")
    print(f"{'Std':<6} {np.std(mel_accs):>10.4f} {np.std(ens_accs):>10.4f}")
    print(f"{'='*42}")


if __name__ == '__main__':
    main()
