"""
MIDI Song Stratified 모델들에서 Version Test 18곡의 frame-level F1을 계산합니다.

각 fold 모델에서 해당 fold의 test set에 속하는 VT 곡들만 inference 후,
전체 fold 예측을 concat해 frame-level F1 계산 (wandb 방식과 동일).

Label map: 0=background(ignored), 1=우조계열, 2=계면조, 3=아니리, 4=창조

Usage:
    cd /home/sangheon/Desktop/PansoriMIDIDetection
    python run_midi_vt_f1.py
"""

import os
os.environ['WANDB_MODE'] = 'disabled'

import sys
import json
from pathlib import Path

import torch
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent))
from models import Conv2DGRU
from datasets.dataset import BaseDataset

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

VT_FILE = Path("/home/sangheon/Desktop/Pansori_Data/pansori_version_split/test.txt")

# Maps fold number → (fold_output_dir, hydra_run_dir)
FOLD_MAP = {
    1:  ("outputs/fold1_0421_211302",  "outputs/2026-04-21/21-13-02"),
    2:  ("outputs/fold2_0421_214358",  "outputs/2026-04-21/21-43-58"),
    3:  ("outputs/fold3_0421_220648",  "outputs/2026-04-21/22-06-48"),
    4:  ("outputs/fold4_0421_223703",  "outputs/2026-04-21/22-37-03"),
    5:  ("outputs/fold5_0421_230411",  "outputs/2026-04-21/23-04-11"),
    6:  ("outputs/fold6_0421_232943",  "outputs/2026-04-21/23-29-43"),
    7:  ("outputs/fold7_0421_235812",  "outputs/2026-04-21/23-58-12"),
    8:  ("outputs/fold8_0422_003452",  "outputs/2026-04-22/00-34-52"),
    9:  ("outputs/fold9_0422_010826",  "outputs/2026-04-22/01-08-26"),
    10: ("outputs/fold10_0422_013305", "outputs/2026-04-22/01-33-05"),
}

# Label index → display name
LABEL_NAMES = {1: "우조계열", 2: "계면조", 3: "아니리", 4: "창조"}
IGNORE_IDX = 0  # background / no label


def load_vt_hashes():
    hashes = set()
    for line in VT_FILE.read_text().splitlines():
        line = line.strip()
        if line:
            hashes.add(line.split("-")[0])
    return hashes


def get_vt_songs_for_fold(fold_output_dir: Path, vt_hashes: set):
    """Returns full midi filenames that belong to this fold's test set AND are VT songs."""
    split_info = json.loads((fold_output_dir / "split_info.json").read_text())
    test_songs = split_info.get("test", [])
    vt_songs = [s for s in test_songs if s.split("-")[0] in vt_hashes]
    return vt_songs


def run_inference(model, dataset_items, cfg_data, device):
    """
    dataset_items: list of full midi filenames to run inference on.
    Returns (all_preds, all_labels) as flat tensors.
    """
    if not dataset_items:
        return [], []

    ds = BaseDataset(
        data_dir=cfg_data.dir.midi_dir,
        label_json=cfg_data.dir.label_path,
        song_list=dataset_items,
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
            out = model(piano)  # (1, T, num_classes)
            preds = out.argmax(dim=-1).view(-1).cpu()
            labels = label.argmax(dim=-1).view(-1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    return all_preds, all_labels


def compute_f1(all_preds, all_labels):
    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    mask = labels != IGNORE_IDX
    preds  = preds[mask]
    labels = labels[mask]

    label_indices = sorted(LABEL_NAMES.keys())
    f1s   = f1_score(labels, preds, labels=label_indices, average=None, zero_division=0)
    macro = f1_score(labels, preds, labels=label_indices, average='macro', zero_division=0)

    per_class = {LABEL_NAMES[i]: float(f1s[j]) for j, i in enumerate(label_indices)}
    return per_class, float(macro)


def main():
    base = Path(__file__).parent
    vt_hashes = load_vt_hashes()
    print(f"Version test songs: {len(vt_hashes)}\n")

    # Load config once (all folds share the same data/model config)
    cfg = OmegaConf.load(base / FOLD_MAP[1][1] / ".hydra" / "config.yaml")

    all_preds, all_labels = [], []
    covered_vt = set()

    for fold, (fold_out_rel, hydra_rel) in sorted(FOLD_MAP.items()):
        fold_out_dir = base / fold_out_rel
        hydra_dir    = base / hydra_rel
        weight_path  = hydra_dir / f"best_model_fold{fold}.pt"

        if not weight_path.exists():
            print(f"  fold{fold}: weights not found, skipping")
            continue

        vt_songs = get_vt_songs_for_fold(fold_out_dir, vt_hashes)
        if not vt_songs:
            print(f"  fold{fold}: no VT songs in test set, skipping")
            continue

        model = Conv2DGRU(cfg.model).to(DEV)
        state = torch.load(weight_path, map_location=DEV, weights_only=True)
        model.load_state_dict(state)

        preds, labels = run_inference(model, vt_songs, cfg.data, DEV)
        all_preds.extend(preds)
        all_labels.extend(labels)
        covered_vt.update(s.split("-")[0] for s in vt_songs)

        n_frames = sum(p.shape[0] for p in preds)
        print(f"  fold{fold:>2}: {len(vt_songs)} VT songs → {n_frames:,} frames")

    print(f"\nCovered VT songs: {len(covered_vt)}/{len(vt_hashes)}")

    if not all_preds:
        print("No predictions collected.")
        return

    per_class, macro = compute_f1(all_preds, all_labels)

    print(f"\n{'='*55}")
    print("SUMMARY (MIDI Song Stratified — frame-level, VT 18곡)")
    print(f"{'Macro':>8}  " + "  ".join(f"{n}: {per_class[n]:.4f}" for n in LABEL_NAMES.values()))
    print(f"{macro:>8.4f}")

    print(f"\n{'Class':<12} {'F1':>7}")
    print("-" * 22)
    for name, f1 in per_class.items():
        print(f"{name:<12} {f1:>7.4f}")
    print(f"{'Macro':<12} {macro:>7.4f}")


if __name__ == '__main__':
    main()
