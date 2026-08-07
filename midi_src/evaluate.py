"""
Evaluate saved best-model checkpoints on the test set for each fold.

Usage (run from PansoriMIDIDetection/):
    python evaluate.py --run_dir outputs/2026-03-06/13-48-42
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from datasets.dataset import BaseDataset
from losses import FocalLoss
from models.model_zoo import Conv2DGRU
from trainer.trainer import run_test_epoch
from utils import create_kfold_splits, get_all_song_names, plot_posteriorgram


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run_dir",
        type=str,
        default="outputs/2026-03-06/13-48-42",
        help="Path to the Hydra run directory that contains best_model_fold*.pt",
    )
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    cfg_path = run_dir / ".hydra" / "config.yaml"
    assert cfg_path.exists(), f"Config not found: {cfg_path}"

    cfg = OmegaConf.load(cfg_path)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── recreate the same folds ───────────────────────────────────────────────
    all_songs = get_all_song_names(cfg.data.dir.midi_dir, cfg.data.dir.label_path)
    folds = create_kfold_splits(all_songs, k=cfg.train.k_folds, seed=cfg.random_seed)

    # ── loss (same as training) ───────────────────────────────────────────────
    alpha_weights = torch.tensor([0.5, 3.0, 1.0, 1.5]).to(device)
    criterion = FocalLoss(alpha=alpha_weights, gamma=2, ignore_index=-100, reduction="mean")

    # ── results accumulator for cross-fold summary ───────────────────────────
    fold_metrics = []

    for fold_idx, fold in enumerate(folds):
        ckpt_path = run_dir / f"best_model_fold{fold_idx + 1}.pt"
        if not ckpt_path.exists():
            print(f"[Fold {fold_idx + 1}] checkpoint not found, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1} / {cfg.train.k_folds}  — loading {ckpt_path}")

        # ── dataset & loader ─────────────────────────────────────────────────
        test_dataset = BaseDataset(
            cfg.data.dir.midi_dir,
            cfg.data.dir.label_path,
            song_list=fold["test"],
            fs=cfg.data.fs,
            window_size=cfg.data.window_size,
            is_train=False,
        )
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        print(f"  Test songs : {len(test_dataset.memory_cache)}")
        print(f"  Test segs  : {len(test_dataset)}")

        # ── model ─────────────────────────────────────────────────────────────
        model = Conv2DGRU(cfg.model).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        # ── inference ─────────────────────────────────────────────────────────
        test_loss, test_acc, test_f1, song_data = run_test_epoch(
            test_loader, model, criterion, device
        )

        print(
            f"  loss={test_loss:.4f}  acc={test_acc:.4f}  "
            f"f1_macro={test_f1['f1_macro']:.4f}  "
            f"[우조={test_f1['f1_ujoh']:.4f} / "
            f"계면조={test_f1['f1_gyemyeon']:.4f} / "
            f"아니리={test_f1['f1_aniri']:.4f}]"
        )

        fold_metrics.append(
            {
                "fold": fold_idx + 1,
                "loss": test_loss,
                "acc": test_acc,
                **test_f1,
            }
        )

        # ── save posteriorgrams ───────────────────────────────────────────────
        pg_dir = run_dir / f"fold{fold_idx + 1}_test" / "posteriorgrams"
        pg_dir.mkdir(parents=True, exist_ok=True)

        for sname, data in song_data.items():
            fig = plot_posteriorgram(sname, data["gt"], data["pred_probs"])
            out_path = pg_dir / f"{Path(sname).stem}.png"
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"    saved → {out_path}")

    # ── cross-fold summary ────────────────────────────────────────────────────
    if not fold_metrics:
        print("No folds evaluated.")
        return

    print(f"\n{'='*60}")
    print("Cross-fold Summary")
    print(f"{'Fold':>5}  {'loss':>7}  {'acc':>7}  {'macro':>7}  {'우조':>7}  {'계면조':>7}  {'아니리':>7}")
    for m in fold_metrics:
        print(
            f"{m['fold']:>5}  {m['loss']:>7.4f}  {m['acc']:>7.4f}  "
            f"{m['f1_macro']:>7.4f}  {m['f1_ujoh']:>7.4f}  "
            f"{m['f1_gyemyeon']:>7.4f}  {m['f1_aniri']:>7.4f}"
        )

    keys = ["loss", "acc", "f1_macro", "f1_ujoh", "f1_gyemyeon", "f1_aniri"]
    import numpy as np
    means = {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}
    stds  = {k: float(np.std([m[k]  for m in fold_metrics])) for k in keys}
    print(
        f"\n{'mean':>5}  {means['loss']:>7.4f}  {means['acc']:>7.4f}  "
        f"{means['f1_macro']:>7.4f}  {means['f1_ujoh']:>7.4f}  "
        f"{means['f1_gyemyeon']:>7.4f}  {means['f1_aniri']:>7.4f}"
    )
    print(
        f"{'std':>5}  {stds['loss']:>7.4f}  {stds['acc']:>7.4f}  "
        f"{stds['f1_macro']:>7.4f}  {stds['f1_ujoh']:>7.4f}  "
        f"{stds['f1_gyemyeon']:>7.4f}  {stds['f1_aniri']:>7.4f}"
    )

    # ── save summary CSV ──────────────────────────────────────────────────────
    csv_path = run_dir / "test_results.csv"
    with open(csv_path, "w") as f:
        f.write("fold,loss,acc,f1_macro,f1_ujoh,f1_gyemyeon,f1_aniri\n")
        for m in fold_metrics:
            f.write(
                f"{m['fold']},{m['loss']:.6f},{m['acc']:.6f},"
                f"{m['f1_macro']:.6f},{m['f1_ujoh']:.6f},"
                f"{m['f1_gyemyeon']:.6f},{m['f1_aniri']:.6f}\n"
            )
        f.write(
            f"mean,{means['loss']:.6f},{means['acc']:.6f},"
            f"{means['f1_macro']:.6f},{means['f1_ujoh']:.6f},"
            f"{means['f1_gyemyeon']:.6f},{means['f1_aniri']:.6f}\n"
        )
        f.write(
            f"std,{stds['loss']:.6f},{stds['acc']:.6f},"
            f"{stds['f1_macro']:.6f},{stds['f1_ujoh']:.6f},"
            f"{stds['f1_gyemyeon']:.6f},{stds['f1_aniri']:.6f}\n"
        )
    print(f"\nSummary saved → {csv_path}")


if __name__ == "__main__":
    main()
