"""
Test-only script for MIDI Detection model.
Loads saved config + best model weights, runs run_test_epoch(), saves test_results.csv
with prob_* columns (requires trainer/trainer.py to have the prob_ patch applied).

Usage:
    cd /home/sangheon/Desktop/PansoriMIDIDetection
    python run_test_only.py --ckpt outputs/2026-04-16/14-18-10/best_model_fold1.pt

Output is saved to: outputs/version_st/test_results.csv (overwrites)
"""

import os
os.environ['WANDB_MODE'] = 'disabled'

import sys
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))

from utils import load_version_split, save_test_csv
from models.model_zoo import Conv2DGRU
from losses import FocalLoss
from trainer.trainer import run_test_epoch
from datasets.dataset import BaseDataset

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_DIR = Path(__file__).parent / 'outputs' / 'version_st'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        default='outputs/2026-04-16/14-18-10/best_model_fold1.pt',
                        help='Path to best_model_fold1.pt')
    parser.add_argument('--config', type=str,
                        default='configs/config.yaml',
                        help='Path to config.yaml (default: configs/config.yaml)')
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    # ── Load config ───────────────────────────────────────────────────────────
    cfg = OmegaConf.load(args.config)
    # Load sub-configs
    for key in ['data', 'model', 'train']:
        sub = OmegaConf.load(f'configs/{key}/{key}.yaml')
        OmegaConf.update(cfg, key, sub, merge=True)

    print(f"Checkpoint: {ckpt_path}")
    print(f"Device    : {DEV}")

    # ── Data split ────────────────────────────────────────────────────────────
    folds = load_version_split(
        cfg.data.dir.version_split_dir,
        cfg.data.dir.midi_dir,
        cfg.data.dir.label_path,
    )
    fold = folds[0]  # version split returns a single-element list
    fs          = cfg.data.fs
    window_size = int(cfg.data.window_size * fs)

    test_dataset = BaseDataset(
        cfg.data.dir.midi_dir,
        cfg.data.dir.label_path,
        song_list=fold['test'],
        fs=fs,
        window_size=window_size,
        is_train=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"Test set: {len(fold['test'])} songs, {len(test_dataset)} segments")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = Conv2DGRU(cfg.model).to(DEV)
    state_dict = torch.load(ckpt_path, map_location=DEV)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded from {ckpt_path}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_cfg = cfg.train.loss
    if loss_cfg.name == 'FocalLoss':
        alpha = torch.tensor(loss_cfg.weights).to(DEV) if loss_cfg.weights else 1
        criterion = FocalLoss(alpha=alpha, gamma=loss_cfg.gamma,
                              ignore_index=loss_cfg.ignore_index, reduction='mean')
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=loss_cfg.ignore_index)

    # ── Run test ──────────────────────────────────────────────────────────────
    print("\nRunning run_test_epoch()...")
    test_loss, test_acc, test_acc_per_cls, test_f1, song_data, segment_results = run_test_epoch(
        test_loader, model, criterion, DEV,
        fs=fs, window_size=window_size,
    )

    print(f"Test Loss: {test_loss:.4f}  Acc: {test_acc:.4f}")
    print(f"F1: macro={test_f1['f1_macro']:.4f}  "
          f"우조={test_f1['f1_ujoh']:.4f}  계면조={test_f1['f1_gyemyeon']:.4f}  "
          f"아니리={test_f1['f1_aniri']:.4f}  창조={test_f1['f1_changjo']:.4f}")
    print(f"Segments: {len(segment_results)}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / 'test_results.csv'
    save_test_csv(segment_results, csv_path)
    print(f"\nSaved → {csv_path}")

    # Verify prob_* columns
    import csv
    with open(csv_path, newline='', encoding='utf-8') as f:
        header = next(csv.reader(f))
    prob_cols = [c for c in header if c.startswith('prob_')]
    if prob_cols:
        print(f"prob_* columns: {prob_cols}")
    else:
        print("WARNING: no prob_* columns found — check trainer/trainer.py patch")


if __name__ == '__main__':
    main()
