"""
Song Stratified 모델들에서 Version Test 18곡의 frame-level F1을 계산합니다.

각 fold 모델에서 해당 fold의 test set에 속하는 VT 곡들만 inference 후,
전체 fold 예측을 concat해 frame-level F1 계산 (wandb 방식과 동일).

Usage:
    cd /path/to/PansoriModeChangeDetection

    # 특정 feature type
    python scripts/eval/run_song_strat_vt_f1.py --strat_dir weights/frame/0411MelSongStrat
    python scripts/eval/run_song_strat_vt_f1.py --strat_dir weights/frame/0411MelOriginalSongStrat
    python scripts/eval/run_song_strat_vt_f1.py --strat_dir weights/frame/0411_CQT_SongStrat
    python scripts/eval/run_song_strat_vt_f1.py --strat_dir weights/frame/0411ChromaSongStrat
    python scripts/eval/run_song_strat_vt_f1.py --strat_dir weights/frame/0411_Pesto_SongStrat
    python scripts/eval/run_song_strat_vt_f1.py --strat_dir weights/cmert

    # 전체 한 번에
    python scripts/eval/run_song_strat_vt_f1.py --all
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
from sklearn.metrics import f1_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import DATA_ROOT, REPO_ROOT
import models
import losses
import trainers
import datasets

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

VT_FILE = DATA_ROOT / 'pansori_version_split/test.txt'

STRAT_DIRS = {
    "Mel (sep)":  "weights/frame/0411MelSongStrat",
    "Mel (orig)": "weights/frame/0411MelOriginalSongStrat",
    "CQT":        "weights/frame/0411_CQT_SongStrat",
    "Chroma":     "weights/frame/0411ChromaSongStrat",
    "Pesto":      "weights/frame/0411_Pesto_SongStrat",
    "CMERT":      "weights/cmert",
}

CLASSES = ["우조", "계면조", "아니리", "창조"]


def load_vt_hashes():
    hashes = set()
    for line in VT_FILE.read_text().splitlines():
        line = line.strip()
        if line:
            hashes.add(line.split("-")[0])
    return hashes


def get_fold_number(run_dir: Path):
    """fold{N}_best_model.pt에서 N 추출."""
    pts = list(run_dir.glob("fold*_best_model.pt"))
    if not pts:
        return None
    m = re.search(r'fold(\d+)_best_model', pts[0].name)
    return int(m.group(1)) if m else None


def get_fold_test_keys(run_dir: Path, fold: int, vt_hashes: set):
    """기존 test_results.csv에서 이 fold의 test 곡 중 VT 곡 추출."""
    csv = run_dir / f"fold{fold}_posteriorgrams" / "test_results.csv"
    if not csv.exists():
        return []
    df = pd.read_csv(csv)
    song_names = df["song_name"].unique()
    keys = list({str(s).split("-")[0] for s in song_names})
    return [k for k in keys if k in vt_hashes]


def run_inference_on_songs(model, dataset, test_keys, criterion, device):
    """지정된 test_keys에 대해 frame-level 예측 수집."""
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
            probs = torch.softmax(outputs, dim=-1)
            preds = probs.argmax(dim=-1).view(-1).cpu()
            labels = y.argmax(dim=-1).view(-1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    return all_preds, all_labels


def compute_f1(all_preds, all_labels, label_map):
    """Frame-level F1 (wandb 방식: Unknown 제외, 나머지 클래스 전체)."""
    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    ignore_idx = label_map["Unknown"]
    mask = labels != ignore_idx
    preds  = preds[mask]
    labels = labels[mask]

    inv_map = {v: k for k, v in label_map.items() if v != ignore_idx}
    label_indices = sorted(inv_map.keys())

    f1s = f1_score(labels, preds, labels=label_indices, average=None, zero_division=0)
    macro = f1_score(labels, preds, labels=label_indices, average='macro', zero_division=0)

    per_class = {inv_map[i]: float(f1s[j]) for j, i in enumerate(label_indices)}
    return per_class, float(macro)


def process_strat_dir(strat_dir: Path, vt_hashes: set):
    run_dirs = sorted([
        d for d in strat_dir.iterdir()
        if d.is_dir() and (d / "config.yaml").exists()
    ])
    if not run_dirs:
        print(f"  No run dirs found in {strat_dir}")
        return None

    # 첫 번째 run dir에서 config/dataset 로드 (모든 fold 동일한 dataset config)
    cfg = OmegaConf.load(run_dirs[0] / "config.yaml")
    dataset_class = getattr(datasets, cfg.dataset.name)
    dataset_params = OmegaConf.to_container(cfg.dataset.params)
    dataset = dataset_class(**cfg.data, **dataset_params)
    print(f"  Dataset loaded ({cfg.dataset.name}): {len(dataset)} total items")

    criterion_class = getattr(losses, cfg.loss.name)
    criterion = criterion_class(**OmegaConf.to_container(cfg.loss.params))

    label_map = dataset.label_map

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
            continue

        model_class = getattr(models, cfg.model.name)
        model = model_class(cfg.model.params).to(DEV)
        state = torch.load(weight_path, map_location=DEV, weights_only=True)
        model.load_state_dict(state)

        preds, labels = run_inference_on_songs(model, dataset, vt_keys, criterion, DEV)
        all_preds.extend(preds)
        all_labels.extend(labels)
        covered_vt.update(vt_keys)
        print(f"    fold{fold}: {len(vt_keys)} VT songs → {sum(p.shape[0] for p in preds):,} frames")

    if not all_preds:
        print("  No predictions collected.")
        return None

    print(f"  Covered VT songs: {len(covered_vt)}/{len(vt_hashes)}")
    per_class, macro = compute_f1(all_preds, all_labels, label_map)
    return per_class, macro


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strat_dir', type=str, default=None)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    vt_hashes = load_vt_hashes()
    print(f"Version test songs: {len(vt_hashes)}\n")

    base = REPO_ROOT

    if args.all:
        targets = {name: base / path for name, path in STRAT_DIRS.items()}
    else:
        # Find feature name from path
        strat_path = Path(args.strat_dir)
        name = next((k for k, v in STRAT_DIRS.items() if Path(v).name == strat_path.name), strat_path.name)
        targets = {name: strat_path}

    results = {}
    for feat, strat_dir in targets.items():
        print(f"\n{'='*60}")
        print(f"Feature: {feat}  ({strat_dir})")
        ret = process_strat_dir(strat_dir, vt_hashes)
        if ret:
            per_class, macro = ret
            results[feat] = {"macro": macro, **per_class}
            print(f"  >> Macro F1: {macro:.4f}  |  " +
                  "  ".join(f"{c}: {per_class[c]:.4f}" for c in CLASSES if c in per_class))

    if results:
        print(f"\n{'='*60}")
        print("SUMMARY (Song Stratified — frame-level, VT 18곡)")
        print(f"{'Feature':<14} {'Macro':>7} {'우조':>7} {'계면조':>7} {'아니리':>7} {'창조':>7}")
        print("-" * 55)
        for feat, r in results.items():
            print(f"{feat:<14} {r['macro']:>7.4f} {r.get('우조',0):>7.4f} "
                  f"{r.get('계면조',0):>7.4f} {r.get('아니리',0):>7.4f} {r.get('창조',0):>7.4f}")


if __name__ == '__main__':
    main()


