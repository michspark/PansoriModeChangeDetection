"""
Test-only script for frame-based models (Mel_Separated, Chroma, CQT, etc.)
Loads saved config.yaml + best model weights, runs evaluate_test(), saves test_results.csv
with prob_* columns (requires trainers.py to have the prob_ patch applied).

Usage:
    cd /path/to/PansoriModeChangeDetection
    python scripts/eval/run_test_only.py --model_dir weights/frame/Verstion_Stratified/0410_Mel_Separated_Version
    python scripts/eval/run_test_only.py --model_dir weights/frame/Verstion_Stratified/0410_Chroma_Version
    python scripts/eval/run_test_only.py --model_dir weights/frame/Verstion_Stratified/0410_CQT_Version
"""

import os
os.environ['WANDB_MODE'] = 'disabled'   # must be set before wandb is imported

import sys
import argparse
import re
from pathlib import Path
from copy import deepcopy

import torch
import pandas as pd
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# Add repo root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import models
import losses
import trainers
import datasets


DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_version_test_split(version_split_dir, loaded_hash):
    split_dir = Path(version_split_dir)
    loaded = set(loaded_hash)

    def _load_keys(fname):
        lines = [l.strip() for l in (split_dir / fname).read_text(encoding='utf-8').splitlines() if l.strip()]
        keys = [line.split('-')[0] for line in lines]
        return [hk for hk in keys if hk in loaded]

    train_keys = _load_keys('train.txt')
    val_keys   = _load_keys('val.txt')
    test_keys  = _load_keys('test.txt')
    print(f"  Version split — train: {len(train_keys)}, val: {len(val_keys)}, test: {len(test_keys)}")
    return train_keys, val_keys, test_keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to model directory containing config.yaml and fold1_best_model.pt')
    parser.add_argument('--fold', type=int, default=1, help='Fold number (default: 1)')
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    fold = args.fold
    config_path = model_dir / 'config.yaml'
    weight_path = model_dir / f'fold{fold}_best_model.pt'
    out_dir = model_dir / f'fold{fold}_posteriorgrams'

    assert config_path.exists(), f"config.yaml not found: {config_path}"
    assert weight_path.exists(), f"Model weights not found: {weight_path}"

    print(f"Model dir : {model_dir}")
    print(f"Weights   : {weight_path}")
    print(f"Output    : {out_dir}")

    # ── Load config ───────────────────────────────────────────────────────────
    cfg = OmegaConf.load(config_path)

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset_class = getattr(datasets, cfg.dataset.name)
    dataset_params = OmegaConf.to_container(cfg.dataset.params)
    dataset = dataset_class(**cfg.data, **dataset_params)
    print(f"Dataset loaded: {len(dataset)} items")

    # ── Version split ─────────────────────────────────────────────────────────
    train_keys, val_keys, test_keys = load_version_test_split(
        cfg.train.version_split_dir, dataset.loaded_hash)
    testset = dataset.get_split(test_keys, split='test')
    print(f"Test set: {len(test_keys)} songs")

    # ── Model ─────────────────────────────────────────────────────────────────
    model_class = getattr(models, cfg.model.name)
    model = model_class(cfg.model.params).to(DEV)
    state_dict = torch.load(weight_path, map_location=DEV, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded: {cfg.model.name}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion_class = getattr(losses, cfg.loss.name)
    criterion_params = OmegaConf.to_container(cfg.loss.params)
    criterion = criterion_class(**criterion_params)

    # ── Trainer (for evaluate_test + save_posteriorgrams) ─────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    trainer = trainers.FrameTrainer(
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        criterion=criterion,
        device=DEV,
        save_dir=model_dir,
        config=cfg,
    )

    # ── Run test evaluation ───────────────────────────────────────────────────
    print("\nRunning evaluate_test()...")
    (test_loss, test_acc, test_acc_masked,
     per_class_acc, per_class_f1, macro_f1,
     cm, song_data, segment_results) = trainer.evaluate_test(testset)

    print(f"Test Loss: {test_loss:.4f}  Acc: {test_acc:.4f}  Masked Acc: {test_acc_masked:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Per-class F1: { {k: f'{v:.4f}' for k, v in per_class_f1.items()} }")
    print(f"Segments: {len(segment_results)}")

    # ── Save test_results.csv ─────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'test_results.csv'
    pd.DataFrame(segment_results).to_csv(csv_path, index=False)
    print(f"\nSaved → {csv_path}")

    # Verify prob_* columns are present
    df = pd.read_csv(csv_path)
    prob_cols = [c for c in df.columns if c.startswith('prob_')]
    if prob_cols:
        print(f"prob_* columns: {prob_cols}")
    else:
        print("WARNING: no prob_* columns found — check trainers.py patch")


if __name__ == '__main__':
    main()
