"""
CultureMERT layer-wise ablation (Version split) 결과를 하나의 표로 모읍니다.

Test/Macro F1 과 per-class 지표는 콘솔에 찍히지 않고 wandb 로만 로깅되지만
(trainers.py 의 wandb.log(...)), wandb 는 그 값들을 로컬 디스크에도 저장합니다.
그래서 추가 추론 없이 wandb/run-*/files/ 만 읽어서 집계할 수 있습니다.

Usage:
    python scripts/eval/aggregate_layer_ablation.py
    python scripts/eval/aggregate_layer_ablation.py --wandb_dir wandb --log_dir outputs/layer_ablation_version
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

CLASS_NAMES: List[str] = ["우조", "계면조", "아니리", "창조"]


def load_wandb_run(run_dir: Path, iters_filter: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """wandb run 디렉터리 하나에서 layer ablation row 를 뽑습니다. 해당 없으면 None."""
    config_path = run_dir / "files" / "config.yaml"
    summary_path = run_dir / "files" / "wandb-summary.json"
    if not config_path.exists() or not summary_path.exists():
        return None

    # wandb config.yaml is nested as {key: {desc: ..., value: <actual value>}}
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict) or "model" not in config or "train" not in config:
        return None

    model_config = config["model"].get("value", {})
    train_config = config["train"].get("value", {})

    # Keep only CultureMERT runs on the Version split that carry a layer_index
    if model_config.get("name") != "CMERTClassifier":
        return None
    if train_config.get("selection") != "Version":
        return None
    layer_index = model_config.get("params", {}).get("layer_index")
    if layer_index is None:
        return None
    # Keep runs of one training budget together — a 10k-iter rerun of a single
    # layer must not silently overwrite that layer's row in a 5k sweep table.
    if iters_filter is not None and train_config.get("num_iterations") != iters_filter:
        return None

    summary = json.loads(summary_path.read_text())
    # A run that crashed before the test phase has no Test/* keys — skip it
    if "Test/Masked Acc" not in summary:
        return None

    row: Dict[str, Any] = {
        "layer": int(layer_index),
        "test_masked_acc": summary.get("Test/Masked Acc"),
        "test_macro_f1": summary.get("Test/Macro F1"),
        "test_acc": summary.get("Test/Acc"),
        "test_loss": summary.get("Test/Loss"),
    }
    for class_name in CLASS_NAMES:
        row[f"acc_{class_name}"] = summary.get(f"Test/Acc_{class_name}")
    for class_name in CLASS_NAMES:
        row[f"f1_{class_name}"] = summary.get(f"Test/F1_{class_name}")

    row["iters"] = summary.get("_step")
    row["hours"] = round(summary["_runtime"] / 3600, 2) if "_runtime" in summary else None
    row["source"] = run_dir.name
    return row


def load_log_fallback(log_path: Path) -> Optional[Dict[str, Any]]:
    """wandb 기록이 없을 때 tee 로그에서 masked accuracy 만 건져냅니다."""
    match = re.search(r"layer(\d+)", log_path.stem)
    if match is None:
        return None

    # Trainer prints: "Test Loss: x, Test Acc: y, Test Masked Acc: z"
    hits = re.findall(
        r"Test Loss: ([\d.]+), Test Acc: ([\d.]+), Test Masked Acc: ([\d.]+)",
        log_path.read_text(errors="replace"),
    )
    if not hits:
        return None

    test_loss, test_acc, test_masked_acc = hits[-1]
    return {
        "layer": int(match.group(1)),
        "test_masked_acc": float(test_masked_acc),
        "test_macro_f1": None,
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
        "source": log_path.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_dir", type=str, default="wandb")
    parser.add_argument("--log_dir", type=str, default="outputs/layer_ablation_version")
    parser.add_argument("--out_csv", type=str, default=None,
                        help="default: <log_dir>/layer_ablation_summary.csv")
    parser.add_argument("--iters", type=int, default=None,
                        help="keep only runs trained for this many iterations (e.g. 5000)")
    args = parser.parse_args()

    base_dir = REPO_ROOT
    wandb_dir = base_dir / args.wandb_dir
    log_dir = base_dir / args.log_dir
    out_csv = Path(args.out_csv) if args.out_csv else log_dir / "layer_ablation_summary.csv"

    # Sorted by directory name = chronological, so a later run overwrites an earlier
    # run with the same layer index (we keep the newest attempt per layer).
    rows_by_layer: Dict[int, Dict[str, Any]] = {}
    for run_dir in sorted(wandb_dir.glob("run-*")):
        row = load_wandb_run(run_dir, iters_filter=args.iters)
        if row is not None:
            rows_by_layer[row["layer"]] = row
    scope = f" (num_iterations={args.iters})" if args.iters else ""
    print(f"wandb 에서 찾은 layer ablation run{scope}: {len(rows_by_layer)}개")

    # Fill in any layer that has a log but no usable wandb record. Skipped when
    # --iters is set: the tee'd logs do not record the training budget, so they
    # cannot be filtered and would mix runs of different budgets into one table.
    if log_dir.exists() and args.iters is None:
        for log_path in sorted(log_dir.glob("layer*.log")):
            fallback = load_log_fallback(log_path)
            if fallback is not None and fallback["layer"] not in rows_by_layer:
                rows_by_layer[fallback["layer"]] = fallback
                print(f"  layer {fallback['layer']}: 로그에서 fallback 집계")

    if not rows_by_layer:
        print("집계할 결과가 없습니다. run_layer_ablation.sh 를 먼저 실행하세요.")
        return

    df = pd.DataFrame([rows_by_layer[layer] for layer in sorted(rows_by_layer)])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # Console tables: overall + per-class accuracy, then per-class F1
    overall_columns = ["layer", "test_masked_acc", "test_macro_f1"] \
        + [f"acc_{name}" for name in CLASS_NAMES] + ["hours"]
    f1_columns = ["layer", "test_macro_f1"] + [f"f1_{name}" for name in CLASS_NAMES]

    def show(title: str, columns: List[str]) -> None:
        columns = [c for c in columns if c in df.columns]
        print()
        print("=" * 78)
        print(title)
        print("=" * 78)
        print(df[columns].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    show("CultureMERT layer ablation — Version split test set (accuracy)", overall_columns)
    show("Per-class F1", f1_columns)
    print()

    best = df.loc[df["test_masked_acc"].idxmax()]
    print(f"Best masked acc : layer {int(best['layer'])}  ({best['test_masked_acc']:.4f})")
    if df["test_macro_f1"].notna().any():
        best_f1 = df.loc[df["test_macro_f1"].idxmax()]
        print(f"Best macro F1   : layer {int(best_f1['layer'])}  ({best_f1['test_macro_f1']:.4f})")
    print(f"\nSaved → {out_csv}")


if __name__ == "__main__":
    main()
