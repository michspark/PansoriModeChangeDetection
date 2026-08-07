"""
Masked accuracy on version-split test 18곡:
  1. Mel Original Song Stratified (per-fold model on each VT song)
  2. Ensemble (from pre-saved fold*_vt_frames.npz)
"""

import sys

import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import DATA_ROOT, REPO_ROOT
import models as frame_models
import datasets as frame_datasets

REPO_FRAME = REPO_ROOT
MEL_DIR    = REPO_FRAME / 'weights/frame/Mel_Original_Song_Stratified'
ENS_DIR    = REPO_FRAME / 'outputs/ensemble_song_stratified'
VT_FILE    = DATA_ROOT / 'pansori_version_split/test.txt'

DEV       = 'cuda' if torch.cuda.is_available() else 'cpu'
AUDIO_WIN = int(30 * 16000 / 512)


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
        for _, test_i in [(1, 2), (2, 1)]:
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
    vt_hashes = {l.strip().split('-')[0] for l in VT_FILE.read_text().splitlines() if l.strip()}
    print(f"VT songs: {len(vt_hashes)}\n")

    mel_dirs = discover_fold_dirs(MEL_DIR)
    ref_cfg  = OmegaConf.load(mel_dirs[1] / 'config.yaml')

    print("Loading Mel dataset (once)...")
    ds_cls = getattr(frame_datasets, ref_cfg.dataset.name)
    params = OmegaConf.to_container(ref_cfg.dataset.params)
    params['aug'] = False
    params['is_valid'] = True
    mel_ds = ds_cls(**ref_cfg.data, **params)
    print(f"  Loaded {len(mel_ds.loaded_hash)} songs\n")

    splits = build_splits(ref_cfg.train.song_stratified_dir)

    # Ensemble: aggregate pre-saved vt_frames.npz across folds
    ens_true_all, ens_pred_all = [], []
    for fold in sorted(mel_dirs):
        npz = ENS_DIR / f'fold{fold}_vt_frames.npz'
        if npz.exists():
            d = np.load(npz)
            ens_true_all.append(d['y_true'])
            ens_pred_all.append(d['y_pred'])
    ens_true = np.concatenate(ens_true_all)
    ens_pred = np.concatenate(ens_pred_all)
    acc_ens = masked_acc(ens_true, ens_pred)

    # Mel: for each fold, run on VT songs in that fold's test set
    mel_true_all, mel_pred_all = [], []

    for fold in sorted(mel_dirs):
        test_keys = set(splits[fold - 1])
        vt_in_fold = [h for h in mel_ds.loaded_hash if h in test_keys and h in vt_hashes]
        if not vt_in_fold:
            continue

        model_cls = getattr(frame_models, ref_cfg.model.name)
        model = model_cls(ref_cfg.model.params).to(DEV)
        model.load_state_dict(torch.load(
            mel_dirs[fold] / f'fold{fold}_best_model.pt',
            map_location=DEV, weights_only=True))
        model.eval()

        for hk in vt_in_fold:
            feat  = mel_ds.loaded_data[hk]
            label = mel_ds.loaded_label[hk]
            probs = infer_mel(model, feat, AUDIO_WIN)
            T = probs.shape[0]
            frame_label = mel_ds.ms_to_frame_label(label)
            gt = frame_label.argmax(dim=-1).numpy()
            gt = gt[:T] if len(gt) >= T else np.pad(gt, (0, T - len(gt)))
            mel_true_all.append(gt)
            mel_pred_all.append(probs.argmax(axis=1))

        del model
        torch.cuda.empty_cache()
        print(f"  Fold {fold}: {len(vt_in_fold)} VT songs")

    mel_true = np.concatenate(mel_true_all)
    mel_pred = np.concatenate(mel_pred_all)
    acc_mel  = masked_acc(mel_true, mel_pred)

    print(f"\n{'='*40}")
    print(f"Version Split Test (18곡) — Masked Accuracy")
    print(f"{'='*40}")
    print(f"  Mel      : {acc_mel:.4f}")
    print(f"  Ensemble : {acc_ens:.4f}")
    print(f"  Diff     : {acc_ens - acc_mel:+.4f}")
    print(f"{'='*40}")
    print(f"  Mel frames  : {len(mel_true)}")
    print(f"  Ens frames  : {len(ens_true)}")


if __name__ == '__main__':
    main()
