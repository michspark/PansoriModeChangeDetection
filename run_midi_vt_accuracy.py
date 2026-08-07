"""
MIDI_Song_Stratified 모델들에서 Version Test 18곡의 frame-level masked accuracy 계산.

Usage:
    cd /home/sangheon/Desktop/PansoriMIDIDetection
    python run_midi_vt_accuracy.py --strat_dir outputs/MIDI_Song_Stratified
"""

import os
os.environ['WANDB_MODE'] = 'disabled'

import sys
import re
import json
import argparse
from pathlib import Path

import torch
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from models import Conv2DGRU
from datasets.dataset import BaseDataset

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

VT_FILE = Path("/home/sangheon/Desktop/Pansori_Data/pansori_version_split/test.txt")

IGNORE_IDX = 0
CLASS_NAMES = {1: "우조", 2: "계면조", 3: "아니리", 4: "창조"}


def load_vt_hashes():
    hashes = set()
    for line in VT_FILE.read_text().splitlines():
        line = line.strip()
        if line:
            hashes.add(line.split("-")[0])
    return hashes


def find_fold_pairs(strat_dir: Path):
    """fold{N}_* 디렉토리와 대응하는 model weight 디렉토리를 매칭.
    timestamp는 1초 오차가 있을 수 있으므로 best_model_fold{N}.pt 파일로 탐색."""
    # HH-MM-SS 형식 디렉토리에서 weight 파일 인덱스 구성
    weight_index = {}  # fold_n → (hydra_dir, weight_path, hydra_cfg)
    for d in strat_dir.iterdir():
        if not re.match(r'^\d{2}-\d{2}-\d{2}$', d.name):
            continue
        for wp in d.glob("best_model_fold*.pt"):
            m = re.search(r'best_model_fold(\d+)\.pt', wp.name)
            if m:
                fold_n = int(m.group(1))
                weight_index[fold_n] = (d, wp, d / ".hydra" / "config.yaml")

    pairs = []
    for fold_out in sorted(strat_dir.iterdir()):
        m = re.match(r'fold(\d+)_(\d{4})_(\d{6})', fold_out.name)
        if not m:
            continue
        fold_n = int(m.group(1))
        split_json = fold_out / "split_info.json"
        if not split_json.exists() or fold_n not in weight_index:
            continue
        hydra_dir, weight_path, hydra_cfg = weight_index[fold_n]
        pairs.append((fold_n, fold_out, hydra_dir, weight_path, hydra_cfg))

    return sorted(pairs, key=lambda x: x[0])


def get_vt_songs_for_fold(fold_out: Path, vt_hashes: set):
    split_info = json.loads((fold_out / "split_info.json").read_text())
    test_songs = split_info.get("test", [])
    return [s for s in test_songs if s.split("-")[0] in vt_hashes]


def run_inference(model, vt_songs, cfg_data, device):
    ds = BaseDataset(
        data_dir=cfg_data.dir.midi_dir,
        label_json=cfg_data.dir.label_path,
        song_list=vt_songs,
        fs=cfg_data.fs,
        window_size=cfg_data.window_size,
        is_train=False,
        snap_boundaries=True,
    )
    if len(ds) == 0:
        return [], []

    loader = DataLoader(ds, batch_size=1, shuffle=False)
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for _, _, piano, label in loader:
            piano = piano.to(device)
            out = model(piano)
            preds  = out.argmax(dim=-1).view(-1).cpu()
            labels = label.argmax(dim=-1).view(-1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    return all_preds, all_labels


def compute_accuracy(all_preds, all_labels):
    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    mask = labels != IGNORE_IDX
    overall_acc = float((preds[mask] == labels[mask]).sum()) / max(mask.sum(), 1)

    per_class = {}
    for cls_idx, cls_name in CLASS_NAMES.items():
        cls_mask = labels == cls_idx
        if cls_mask.sum() == 0:
            per_class[cls_name] = float('nan')
        else:
            per_class[cls_name] = float((preds[cls_mask] == labels[cls_mask]).sum()) / cls_mask.sum()

    return overall_acc, per_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strat_dir', type=str, default='outputs/MIDI_Song_Stratified')
    args = parser.parse_args()

    base = Path(__file__).parent
    strat_dir = base / args.strat_dir

    vt_hashes = load_vt_hashes()
    print(f"Version test songs: {len(vt_hashes)}")
    print(f"Model dir: {strat_dir}")
    print(f"Device: {DEV}")
    print("=" * 60)

    pairs = find_fold_pairs(strat_dir)
    if not pairs:
        print("No valid fold pairs found.")
        return

    # config는 첫 번째 fold의 hydra config 사용
    cfg = OmegaConf.load(pairs[0][4])

    all_preds, all_labels = [], []
    covered_vt = set()

    for fold_n, fold_out, hydra_dir, weight_path, _ in pairs:
        vt_songs = get_vt_songs_for_fold(fold_out, vt_hashes)
        if not vt_songs:
            print(f"  fold{fold_n:>2}: no VT songs in test set, skipping")
            continue

        model = Conv2DGRU(cfg.model).to(DEV)
        state = torch.load(weight_path, map_location=DEV, weights_only=True)
        model.load_state_dict(state)

        preds, labels = run_inference(model, vt_songs, cfg.data, DEV)
        all_preds.extend(preds)
        all_labels.extend(labels)
        covered_vt.update(s.split("-")[0] for s in vt_songs)

        n_frames = sum(p.shape[0] for p in preds)
        print(f"  fold{fold_n:>2}: {len(vt_songs)} VT songs → {n_frames:,} frames")

    print(f"\nCovered VT songs: {len(covered_vt)}/{len(vt_hashes)}")

    if not all_preds:
        print("No predictions collected.")
        return

    overall_acc, per_class = compute_accuracy(all_preds, all_labels)

    print(f"\n{'='*60}")
    print("RESULTS — frame-level masked accuracy (Version Test, Unknown 제외)")
    print(f"{'='*60}")
    print(f"Overall masked accuracy : {overall_acc:.4f}  ({overall_acc*100:.2f}%)")
    print()
    print("Per-class accuracy:")
    for cls_name, acc in per_class.items():
        if np.isnan(acc):
            print(f"  {cls_name:8s}: N/A")
        else:
            print(f"  {cls_name:8s}: {acc:.4f}  ({acc*100:.2f}%)")


if __name__ == '__main__':
    main()
