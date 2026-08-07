"""
Song-Stratified 10-fold 결과를 fold 평균이 아니라 '전체 데이터셋 pooled'로 계산합니다.

각 fold의 best checkpoint로 그 fold의 test set을 다시 추론하고, 10개 fold의
frame 단위 예측/정답을 하나로 concat한 뒤 accuracy / macro F1 / per-class F1을
한 번만 계산합니다. (10개 fold의 test set은 서로 겹치지 않고 합치면 전체 곡을 커버)

Usage:
    cd /path/to/PansoriModeChangeDetection
    python scripts/eval/pooled_eval_song_stratified.py \
        --strat_dir weights/cmert/layer08_song_stratified
"""

import os
os.environ['WANDB_MODE'] = 'disabled'

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import pandas as pd
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import models
import datasets

DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_fold_run_dirs(strat_dir: Path) -> Dict[int, Path]:
    """test_metrics/fold{N}_test_metrics.csv의 run_dir 컬럼으로 fold별 최종 run 디렉토리를 특정한다.

    같은 fold가 여러 번 재시작된 경우 timestamp 디렉토리가 여러 개 남아 있으므로,
    학습이 실제로 끝나면서 metric을 남긴 run_dir만 골라야 한다.
    """
    metrics_dir = strat_dir / 'test_metrics'
    if not metrics_dir.exists():
        raise FileNotFoundError(f"test_metrics 디렉토리가 없습니다: {metrics_dir}")

    fold_to_run_dir: Dict[int, Path] = {}
    for metrics_file in sorted(metrics_dir.glob('fold*_test_metrics.csv')):
        metrics_row = pd.read_csv(metrics_file).iloc[0]
        fold_number = int(metrics_row['fold'])
        fold_to_run_dir[fold_number] = strat_dir / str(metrics_row['run_dir'])

    return dict(sorted(fold_to_run_dir.items()))


def load_fold_test_keys(run_dir: Path, fold_number: int) -> List[str]:
    """학습 때 저장된 test_results.csv에서 그 fold의 test 곡 hash key 목록을 읽는다.

    fold 구성을 다시 계산하지 않고 저장된 결과에서 읽으므로, 학습 때 평가한
    test set과 정확히 동일한 곡 집합이 보장된다.
    """
    results_file = run_dir / f'fold{fold_number}_posteriorgrams' / 'test_results.csv'
    if not results_file.exists():
        raise FileNotFoundError(f"test_results.csv가 없습니다: {results_file}")

    results_df = pd.read_csv(results_file)
    unique_song_names = results_df['song_name'].astype(str).unique()
    # song_name은 이미 hash key지만, 혹시 'hash-파일명' 형태여도 안전하게 자른다
    return sorted({name.split('-')[0] for name in unique_song_names})


def build_dataset(config) -> torch.utils.data.Dataset:
    """config.yaml에 기록된 dataset 설정 그대로 전체 데이터셋을 한 번 만든다."""
    dataset_class = getattr(datasets, config.dataset.name)
    dataset_params = OmegaConf.to_container(config.dataset.params)
    return dataset_class(**config.data, **dataset_params)


def build_model(config, weight_path: Path, device: str) -> torch.nn.Module:
    """config의 모델을 만들고 해당 fold의 best checkpoint를 올린다."""
    model_class = getattr(models, config.model.name)
    model = model_class(config.model.params).to(device)
    state_dict = torch.load(weight_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_fold_inference(model: torch.nn.Module,
                       dataset: torch.utils.data.Dataset,
                       test_keys: List[str],
                       device: str,
                       progress_desc: str) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """한 fold의 test set을 추론해 frame 단위 예측/정답을 flat tensor로 돌려준다."""
    # split='test'는 곡을 30초 non-overlap segment로 잘라 valid 형태의 dataset을 만든다
    testset = dataset.get_split(test_keys, split='test')

    # trainers.evaluate_test와 동일하게 batch_size=1 (segment 길이가 곡 끝에서 달라질 수 있음)
    dataloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    fold_preds: List[torch.Tensor] = []
    fold_labels: List[torch.Tensor] = []
    num_segments = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=progress_desc):
            _, input_audio, frame_label = batch
            input_audio = input_audio.to(device)
            frame_label = frame_label.to(device)

            outputs = model(input_audio)                      # (1, T, C) logits
            # 모델 출력 frame 수가 label보다 길면 잘라서 맞춘다 (학습 때와 동일한 처리)
            if outputs.shape[1] != frame_label.shape[1]:
                outputs = outputs[:, :frame_label.shape[1]]

            # softmax는 단조 증가 함수라 argmax 결과가 바뀌지 않으므로 logit에서 바로 argmax
            fold_preds.append(outputs.argmax(dim=-1).view(-1).cpu())
            fold_labels.append(frame_label.argmax(dim=-1).view(-1).cpu())
            num_segments += 1

    return torch.cat(fold_preds), torch.cat(fold_labels), num_segments


def compute_pooled_metrics(all_preds: torch.Tensor,
                           all_labels: torch.Tensor,
                           label_map: Dict[str, int]) -> Dict[str, float]:
    """모든 fold의 frame을 합친 상태에서 accuracy / macro F1 / per-class F1을 계산한다."""
    unknown_index = label_map['Unknown']
    # label_map은 여러 이름이 같은 인덱스를 가리키므로(경드름/평조/우조 → 1) 역매핑은 마지막 이름을 쓴다
    index_to_class_name = {index: name for name, index in label_map.items() if index != unknown_index}
    evaluated_indices = sorted(index_to_class_name.keys())

    preds_numpy = all_preds.numpy()
    labels_numpy = all_labels.numpy()

    # Unknown 프레임을 포함한 전체 정확도
    accuracy_all = float((preds_numpy == labels_numpy).mean())

    # Unknown(정답이 0)인 프레임을 제외한 정확도 — 학습 로그의 masked acc와 같은 정의
    valid_mask = labels_numpy != unknown_index
    valid_preds = preds_numpy[valid_mask]
    valid_labels = labels_numpy[valid_mask]
    accuracy_masked = float((valid_preds == valid_labels).mean())

    per_class_f1_values = f1_score(valid_labels, valid_preds,
                                   labels=evaluated_indices, average=None, zero_division=0)
    macro_f1 = float(f1_score(valid_labels, valid_preds,
                              labels=evaluated_indices, average='macro', zero_division=0))

    metrics: Dict[str, float] = {
        'pooled_acc': accuracy_all,
        'pooled_masked_acc': accuracy_masked,
        'pooled_macro_f1': macro_f1,
    }
    for position, class_index in enumerate(evaluated_indices):
        metrics[f'pooled_f1_{index_to_class_name[class_index]}'] = float(per_class_f1_values[position])

    metrics['n_frames_total'] = int(labels_numpy.shape[0])
    metrics['n_frames_valid'] = int(valid_mask.sum())
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Song-stratified 10-fold의 전체 데이터셋 pooled 성능 계산')
    parser.add_argument('--strat_dir', type=str,
                        default='weights/cmert/layer08_song_stratified',
                        help='fold별 run 디렉토리와 test_metrics/가 들어있는 상위 디렉토리')
    parser.add_argument('--out_name', type=str, default='pooled_test_metrics.csv',
                        help='--strat_dir 아래에 저장할 결과 CSV 파일명')
    args = parser.parse_args()

    strat_dir = Path(args.strat_dir).resolve()
    fold_to_run_dir = load_fold_run_dirs(strat_dir)
    print(f"Strat dir : {strat_dir}")
    print(f"Device    : {DEVICE}")
    print(f"Folds     : {sorted(fold_to_run_dir.keys())}\n")

    # dataset config는 모든 fold가 동일하므로 첫 fold의 config로 한 번만 만든다
    first_fold = min(fold_to_run_dir.keys())
    base_config = OmegaConf.load(fold_to_run_dir[first_fold] / 'config.yaml')
    dataset = build_dataset(base_config)
    print(f"Dataset   : {base_config.dataset.name}, {len(dataset.loaded_hash)} songs indexed\n")

    pooled_preds: List[torch.Tensor] = []
    pooled_labels: List[torch.Tensor] = []
    covered_song_keys: set = set()
    total_segments = 0

    for fold_number, run_dir in fold_to_run_dir.items():
        fold_config = OmegaConf.load(run_dir / 'config.yaml')
        weight_path = run_dir / f'fold{fold_number}_best_model.pt'
        test_keys = load_fold_test_keys(run_dir, fold_number)

        # 같은 곡이 두 fold에 중복으로 들어가면 pooled 계산이 왜곡되므로 확인만 해둔다
        overlapping_keys = covered_song_keys & set(test_keys)
        if overlapping_keys:
            print(f"  WARNING: fold{fold_number}의 test 곡 {len(overlapping_keys)}개가 이전 fold와 중복됩니다")

        model = build_model(fold_config, weight_path, DEVICE)
        fold_preds, fold_labels, num_segments = run_fold_inference(
            model, dataset, test_keys, DEVICE, progress_desc=f'fold{fold_number} ({len(test_keys)} songs)')

        pooled_preds.append(fold_preds)
        pooled_labels.append(fold_labels)
        covered_song_keys.update(test_keys)
        total_segments += num_segments

        print(f"  fold{fold_number}: {len(test_keys)} songs, {num_segments} segments, "
              f"{fold_preds.shape[0]:,} frames  ({run_dir.name})")

        # 다음 fold 모델을 올리기 전에 VRAM을 비운다
        del model
        torch.cuda.empty_cache()

    all_preds = torch.cat(pooled_preds)
    all_labels = torch.cat(pooled_labels)
    metrics = compute_pooled_metrics(all_preds, all_labels, dataset.label_map)
    metrics['n_folds'] = len(fold_to_run_dir)
    metrics['n_songs'] = len(covered_song_keys)
    metrics['n_segments'] = total_segments

    output_path = strat_dir / args.out_name
    pd.DataFrame([metrics]).to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print("POOLED (전체 데이터셋 frame을 한 번에 계산)")
    print(f"{'='*60}")
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            print(f"  {metric_name:<24} {metric_value:.4f}")
        else:
            print(f"  {metric_name:<24} {metric_value:,}")
    print(f"\nSaved → {output_path}")


if __name__ == '__main__':
    main()
