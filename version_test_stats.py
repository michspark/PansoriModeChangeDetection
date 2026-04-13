"""
version_test_stats.py

주어진 version test CSV 파일의 GT-present 기반 통계를 출력합니다.

사용법:
    python version_test_stats.py <csv_path> [<csv_path2> ...]
    python version_test_stats.py weights/frame/0411MelVersionTest/test_results.csv
    python version_test_stats.py results_a.csv results_b.csv results_c.csv

지원 포맷:
    - 한글 컬럼: f1_우조, f1_계면조, f1_아니리, f1_창조  (PansoriModeChangeDetection)
    - 영문 컬럼: f1_ujoh, f1_gyemyeon, f1_aniri, f1_changjo  (PansoriMIDIDetection)

song_name 컬럼:
    - 8자리 hash 그대로 (audio)
    - hash-filename_vocal.mid 형식 (MIDI) → hash 자동 추출
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

LABEL_CSV = Path("/home/sangheon/Desktop/PansoriModeChangeDetection/data/Label/label.csv")

# 한글 클래스명 → 각 포맷의 컬럼명 매핑
COL_VARIANTS = {
    "한글": {"우조": "f1_우조",     "계면조": "f1_계면조",    "아니리": "f1_아니리",  "창조": "f1_창조"},
    "영문": {"우조": "f1_ujoh",    "계면조": "f1_gyemyeon", "아니리": "f1_aniri", "창조": "f1_changjo"},
}
CLASSES = ["우조", "계면조", "아니리", "창조"]


def detect_col_format(df: pd.DataFrame) -> dict:
    for fmt, mapping in COL_VARIANTS.items():
        if all(col in df.columns for col in mapping.values()):
            return mapping
    raise ValueError(
        f"F1 컬럼을 찾을 수 없습니다. 확인된 컬럼: {list(df.columns)}\n"
        f"필요한 컬럼 (한글): {list(COL_VARIANTS['한글'].values())}\n"
        f"필요한 컬럼 (영문): {list(COL_VARIANTS['영문'].values())}"
    )


def extract_hash(song_name: str) -> str:
    """hash-filename_vocal.mid 또는 hash 그대로인 경우 모두 처리"""
    return song_name.split("-")[0]


def load_label_df() -> pd.DataFrame:
    if not LABEL_CSV.exists():
        raise FileNotFoundError(f"레이블 파일을 찾을 수 없습니다: {LABEL_CSV}")
    return pd.read_csv(LABEL_CSV)


def get_present_classes(hash_key: str, start_sec: float, end_sec: float,
                        label_df: pd.DataFrame) -> set:
    seg_start_ms = start_sec * 1000
    seg_end_ms   = end_sec   * 1000
    song_labels  = label_df[label_df["hash_key"] == hash_key]
    overlap = song_labels[
        (song_labels["start"] < seg_end_ms) &
        (song_labels["end"]   > seg_start_ms)
    ]
    return set(overlap["label"].unique())


def compute_stats(df: pd.DataFrame, label_df: pd.DataFrame) -> dict:
    col_map = detect_col_format(df)

    # hash 추출
    df = df.copy()
    df["_hash"] = df["song_name"].apply(extract_hash)

    # 각 구간에 GT 클래스 존재 여부 계산
    present = {cls: [] for cls in CLASSES}
    for _, row in df.iterrows():
        pc = get_present_classes(row["_hash"], row["start_sec"], row["end_sec"], label_df)
        for cls in CLASSES:
            present[cls].append(cls in pc)

    result = {
        "n_segments": len(df),
        "n_songs":    df["_hash"].nunique(),
        "loss":       df["loss"].mean(),
        "acc":        df["acc"].mean(),
    }

    gt_f1s = []
    for cls in CLASSES:
        mask  = pd.Series(present[cls])
        n_gt  = int(mask.sum())
        f1_gt = df[col_map[cls]][mask].mean() if mask.any() else float("nan")
        result[f"f1_{cls}"]        = f1_gt
        result[f"n_gt_{cls}"]      = n_gt
        gt_f1s.append(f1_gt)

    result["f1_macro_gt"] = sum(gt_f1s) / len(gt_f1s)
    return result


def print_stats(stats: dict, label: str = "") -> None:
    header = f"=== {label} ===" if label else "=== 결과 ==="
    print(header)
    print(f"  Segments      : {stats['n_segments']}  |  Songs: {stats['n_songs']}")
    print(f"  Avg Loss      : {stats['loss']:.4f}")
    print(f"  Avg Accuracy  : {stats['acc']:.4f}")
    print(f"  Macro F1 (GT) : {stats['f1_macro_gt']:.4f}")
    print()
    print(f"  {'클래스':8s}  {'GT구간':>6s}  {'F1 (GT-present)':>16s}")
    print(f"  {'-'*36}")
    for cls in CLASSES:
        n   = stats[f"n_gt_{cls}"]
        f1  = stats[f"f1_{cls}"]
        print(f"  {cls:8s}  {n:>6d}  {f1:>16.4f}")
    print()


def print_comparison_table(all_stats: dict) -> None:
    names   = list(all_stats.keys())
    metrics = [
        ("loss",        "Avg Loss"),
        ("acc",         "Avg Accuracy"),
        ("f1_macro_gt", "Macro F1 (GT-present)"),
        ("f1_우조",      "F1 우조  (GT-present)"),
        ("f1_계면조",     "F1 계면조 (GT-present)"),
        ("f1_아니리",     "F1 아니리 (GT-present)"),
        ("f1_창조",      "F1 창조  (GT-present)"),
    ]

    col_w   = 16
    label_w = 28
    header  = f"{'':>{label_w}}" + "".join(f"{n:>{col_w}}" for n in names)
    print(header)
    print("-" * (label_w + col_w * len(names)))
    for key, label in metrics:
        vals = [all_stats[n][key] for n in names]
        best = min(vals) if key == "loss" else max(vals)
        row  = f"{label:<{label_w}}"
        for v in vals:
            marker = " *" if v == best else "  "
            row += f"{v:>{col_w - 2}.4f}{marker}"
        print(row)
    print()


def main():
    parser = argparse.ArgumentParser(description="Version test CSV GT-present 통계 출력")
    parser.add_argument("csv_paths", nargs="+", help="분석할 CSV 파일 경로 (여러 개 가능)")
    parser.add_argument("--names", nargs="*", help="각 CSV의 표시 이름 (순서대로, 생략 시 파일명 사용)")
    args = parser.parse_args()

    label_df = load_label_df()

    csv_paths = [Path(p) for p in args.csv_paths]
    names = args.names if args.names else [p.parent.name for p in csv_paths]

    if len(names) != len(csv_paths):
        parser.error("--names 개수가 csv_paths 개수와 맞지 않습니다.")

    all_stats = {}
    for path, name in zip(csv_paths, names):
        if not path.exists():
            print(f"[경고] 파일 없음: {path}", file=sys.stderr)
            continue
        df = pd.read_csv(path)
        stats = compute_stats(df, label_df)
        all_stats[name] = stats

    if not all_stats:
        print("분석할 파일이 없습니다.", file=sys.stderr)
        sys.exit(1)

    if len(all_stats) == 1:
        name, stats = next(iter(all_stats.items()))
        print_stats(stats, label=name)
    else:
        for name, stats in all_stats.items():
            print_stats(stats, label=name)
        print("=== 비교표 ===")
        print_comparison_table(all_stats)


if __name__ == "__main__":
    main()
