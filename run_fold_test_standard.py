"""
Re-evaluate all 10 fold models using:
  - song_stratified splits (same as Mel/CMERT)
  - standard 30-second boundaries (no note snapping)

This produces test_results.csv files with uniform time_ranges like 0-30s, 30-60s, 60-90s
directly comparable to Mel/Pesto/CMERT fold test results.

Usage (run from PansoriMIDIDetection/):
    # After training with song_stratified split:
    python run_fold_test_standard.py --model_dir outputs/2026-04-21/XX-XX-XX

    # Or specify the config path separately:
    python run_fold_test_standard.py --model_dir <path> --config <path>

Output:
    outputs/fold_test_standard/fold{N}/test_results.csv  (per-fold)
    outputs/fold_test_standard/all_folds_test_results.csv (combined)
"""

import os
os.environ['WANDB_MODE'] = 'disabled'

import sys
import csv
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))

from datasets.dataset import BaseDataset
from losses import FocalLoss
from models.model_zoo import Conv2DGRU
from trainer.trainer import run_test_epoch
from utils import load_song_stratified_folds

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / 'outputs/fold_test_standard'

FIELDNAMES = ['fold', 'song_name', 'filename', 'time_range', 'start_sec', 'end_sec',
              'loss', 'acc', 'f1_ujoh', 'f1_gyemyeon', 'f1_aniri', 'f1_changjo', 'f1_macro']


def _write_csv(rows, path):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_dir', type=str, required=True,
                   help='Directory containing best_model_fold1.pt ~ best_model_fold10.pt')
    p.add_argument('--config', type=str, default=None,
                   help='Path to config.yaml (default: <model_dir>/.hydra/config.yaml)')
    return p.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)

    cfg_path = Path(args.config) if args.config else model_dir / '.hydra' / 'config.yaml'
    assert cfg_path.exists(), f"Config not found: {cfg_path}"

    cfg = OmegaConf.load(cfg_path)
    fs = cfg.data.fs                              # 10
    window_size = int(cfg.data.window_size * fs)  # 300 frames = 30s

    loss_cfg = cfg.train.loss
    alpha = torch.tensor(loss_cfg.weights).to(DEV)
    criterion = FocalLoss(alpha=alpha, gamma=loss_cfg.gamma,
                          ignore_index=loss_cfg.ignore_index, reduction='mean')

    # Load song_stratified folds (same splits as Mel/CMERT)
    print("Loading song_stratified folds...")
    folds = load_song_stratified_folds(
        cfg.data.dir.song_stratified_dir,
        cfg.data.dir.midi_dir,
        cfg.data.dir.label_path,
    )
    print(f"  {len(folds)} folds loaded")

    all_segments = []

    for fold_idx, fold in enumerate(folds):
        fold_num = fold_idx + 1
        ckpt = model_dir / f'best_model_fold{fold_num}.pt'

        if not ckpt.exists():
            print(f'[Fold {fold_num}] model not found: {ckpt} — skipping')
            continue

        test_songs = fold['test']
        print(f'\n{"="*60}')
        print(f'Fold {fold_num} | {len(test_songs)} test songs | {ckpt}')

        # snap_boundaries=False → standard 30s windows (no note snapping)
        test_dataset = BaseDataset(
            cfg.data.dir.midi_dir,
            cfg.data.dir.label_path,
            song_list=test_songs,
            fs=fs,
            window_size=cfg.data.window_size,
            is_train=False,
            snap_boundaries=False,
        )
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        print(f'  Segments: {len(test_dataset)}')

        model = Conv2DGRU(cfg.model).to(DEV)
        model.load_state_dict(torch.load(ckpt, map_location=DEV))

        test_loss, test_acc, _, test_f1, _, segment_results = run_test_epoch(
            test_loader, model, criterion, DEV,
            fs=fs, window_size=window_size,
        )

        print(f'  loss={test_loss:.4f}  acc={test_acc:.4f}  '
              f'f1_macro={test_f1["f1_macro"]:.4f}  '
              f'[우조={test_f1["f1_ujoh"]:.4f} / 계면조={test_f1["f1_gyemyeon"]:.4f} / '
              f'아니리={test_f1["f1_aniri"]:.4f} / 창조={test_f1["f1_changjo"]:.4f}]')

        # Add fold column and parse hash + filename from full MIDI song_name
        for r in segment_results:
            r['fold'] = f'fold{fold_num}'
            full_name = r['song_name']  # e.g. "abc12345-02-가수-곡명_vocal.mid"
            r['song_name'] = full_name[:8]
            r['filename'] = full_name[9:].replace('_vocal.mid', '') if len(full_name) > 9 else ''

        fold_out = OUT_DIR / f'fold{fold_num}'
        fold_out.mkdir(parents=True, exist_ok=True)
        _write_csv(segment_results, fold_out / 'test_results.csv')
        print(f'  Saved → {fold_out}/test_results.csv')

        all_segments.extend(segment_results)

    # Combined CSV
    if all_segments:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        _write_csv(all_segments, OUT_DIR / 'all_folds_test_results.csv')
        print(f'\nAll folds combined → {OUT_DIR}/all_folds_test_results.csv')
        print(f'Total segments: {len(all_segments)}')

        import re
        n_std = sum(
            1 for r in all_segments
            if re.match(r'^(\d+)-(\d+)s$', r['time_range']) and
               int(re.match(r'^(\d+)', r['time_range']).group()) % 30 == 0
        )
        print(f'Standard 30s boundaries: {n_std}/{len(all_segments)} ({n_std/len(all_segments)*100:.1f}%)')


if __name__ == '__main__':
    main()
