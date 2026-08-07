"""
Version Test set에 대한 frame-level masked accuracy 계산.

전체 masked accuracy (Unknown 제외) + 각 클래스별 accuracy를 출력합니다.

Usage:
    python scripts/eval/run_vt_accuracy.py --strat_dir weights/frame/Mel_Original_Song_Stratified
"""

import os
os.environ['WANDB_MODE'] = 'disabled'

import sys
import argparse
import re
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import DATA_ROOT, REPO_ROOT
import models
import datasets

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

VT_FILE = DATA_ROOT / 'pansori_version_split/test.txt'

# label_map 상 실제 존재하는 class index → 이름
# Unknown=0, 우조계열=1, 계면조=2, 아니리=3, 창조=4
CLASS_NAMES = {1: "우조", 2: "계면조", 3: "아니리", 4: "창조"}


def load_vt_hashes():
    hashes = set()
    for line in VT_FILE.read_text().splitlines():
        line = line.strip()
        if line:
            hashes.add(line.split("-")[0])
    return hashes


def get_fold_number(run_dir: Path):
    pts = list(run_dir.glob("fold*_best_model.pt"))
    if not pts:
        return None
    m = re.search(r'fold(\d+)_best_model', pts[0].name)
    return int(m.group(1)) if m else None


def get_fold_test_keys(run_dir: Path, fold: int, vt_hashes: set):
    csv = run_dir / f"fold{fold}_posteriorgrams" / "test_results.csv"
    if not csv.exists():
        return []
    df = pd.read_csv(csv)
    song_names = df["song_name"].unique()
    keys = list({str(s).split("-")[0] for s in song_names})
    return [k for k in keys if k in vt_hashes]


def run_inference_on_songs(model, dataset, test_keys, device):
    testset = dataset.get_split(test_keys, split='test')
    if testset is None or len(testset) == 0:
        return [], []

    model.eval()
    dataloader = DataLoader(testset, batch_size=1, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            _, x, y = batch
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            if outputs.shape[1] != y.shape[1]:
                outputs = outputs[:, :y.shape[1]]
            preds = outputs.argmax(dim=-1).view(-1).cpu()
            labels = y.argmax(dim=-1).view(-1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    return all_preds, all_labels


def compute_accuracy(all_preds, all_labels, ignore_idx=0):
    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    # Overall masked accuracy (Unknown 제외)
    mask = labels != ignore_idx
    masked_preds  = preds[mask]
    masked_labels = labels[mask]
    overall_acc = float((masked_preds == masked_labels).sum()) / max(len(masked_labels), 1)

    # Per-class accuracy
    per_class = {}
    for cls_idx, cls_name in CLASS_NAMES.items():
        cls_mask = labels == cls_idx
        if cls_mask.sum() == 0:
            per_class[cls_name] = float('nan')
        else:
            cls_preds  = preds[cls_mask]
            cls_labels = labels[cls_mask]
            per_class[cls_name] = float((cls_preds == cls_labels).sum()) / len(cls_labels)

    return overall_acc, per_class


def process_strat_dir(strat_dir: Path, vt_hashes: set):
    run_dirs = sorted([
        d for d in strat_dir.iterdir()
        if d.is_dir() and (d / "config.yaml").exists()
    ])
    if not run_dirs:
        print(f"  No run dirs found in {strat_dir}")
        return None

    cfg = OmegaConf.load(run_dirs[0] / "config.yaml")
    dataset_class = getattr(datasets, cfg.dataset.name)
    dataset_params = OmegaConf.to_container(cfg.dataset.params)
    dataset = dataset_class(**cfg.data, **dataset_params)
    print(f"  Dataset loaded ({cfg.dataset.name}): {len(dataset)} total items")

    label_map = dataset.label_map
    ignore_idx = label_map["Unknown"]

    all_preds, all_labels = [], []
    covered_vt = set()

    for run_dir in run_dirs:
        fold = get_fold_number(run_dir)
        if fold is None:
            continue

        weight_path = run_dir / f"fold{fold}_best_model.pt"
        if not weight_path.exists():
            continue

        vt_keys = get_fold_test_keys(run_dir, fold, vt_hashes)
        if not vt_keys:
            print(f"    fold{fold}: no VT songs in test set, skipping")
            continue

        model_class = getattr(models, cfg.model.name)
        model = model_class(cfg.model.params).to(DEV)
        state = torch.load(weight_path, map_location=DEV, weights_only=True)
        model.load_state_dict(state)

        preds, labels = run_inference_on_songs(model, dataset, vt_keys, DEV)
        all_preds.extend(preds)
        all_labels.extend(labels)
        covered_vt.update(vt_keys)
        total_frames = sum(p.shape[0] for p in preds)
        print(f"    fold{fold}: {len(vt_keys)} VT songs → {total_frames:,} frames")

    if not all_preds:
        print("  No predictions collected.")
        return None

    print(f"  Covered VT songs: {len(covered_vt)}/{len(vt_hashes)}")
    overall_acc, per_class = compute_accuracy(all_preds, all_labels, ignore_idx)
    return overall_acc, per_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strat_dir', type=str,
                        default='weights/frame/Mel_Original_Song_Stratified')
    args = parser.parse_args()

    vt_hashes = load_vt_hashes()
    print(f"Version test songs: {len(vt_hashes)}\n")

    base = REPO_ROOT
    strat_dir = base / args.strat_dir

    print(f"{'='*60}")
    print(f"Model dir: {strat_dir}")
    print(f"Device: {DEV}")
    print(f"{'='*60}")

    ret = process_strat_dir(strat_dir, vt_hashes)
    if ret is None:
        return

    overall_acc, per_class = ret

    print(f"\n{'='*60}")
    print("RESULTS — frame-level masked accuracy (Version Test, Unknown 제외)")
    print(f"{'='*60}")
    print(f"Overall masked accuracy : {overall_acc:.4f}  ({overall_acc*100:.2f}%)")
    print()
    print("Per-class accuracy:")
    for cls_name, acc in per_class.items():
        if np.isnan(acc):
            print(f"  {cls_name:8s}: N/A (no samples)")
        else:
            print(f"  {cls_name:8s}: {acc:.4f}  ({acc*100:.2f}%)")


if __name__ == '__main__':
    main()
