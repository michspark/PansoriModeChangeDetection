"""
combine_version_tests.py

Concatenates per-fold version_test_results.csv files for each feature type
(Chroma, CQT, Mel) from the complete 0411 10-fold runs and saves the combined
results so they can be compared against the 0410 baseline test results.

Output directories (created under weights/frame/):
  0411ChromaVersionTest/test_results.csv
  0411CQTVersionTest/test_results.csv
  0411MelVersionTest/test_results.csv
"""

import pandas as pd
from pathlib import Path

BASE = Path("/home/sangheon/Desktop/PansoriModeChangeDetection/weights/frame")

# Complete 10-fold runs for each feature type.
# Each entry is (directory, fold_label).
# fold9 is missing for all three types (no version_test_summary was saved).
GROUPS = {
    "0411ChromaVersionTest": [
        (BASE / "0411_1748_Audio_Conv2DGRU_ChromaFrameDataset", "fold1"),
        (BASE / "0411_1756_Audio_Conv2DGRU_ChromaFrameDataset", "fold2"),
        (BASE / "0411_1803_Audio_Conv2DGRU_ChromaFrameDataset", "fold3"),
        (BASE / "0411_1809_Audio_Conv2DGRU_ChromaFrameDataset", "fold4"),
        (BASE / "0411_1815_Audio_Conv2DGRU_ChromaFrameDataset", "fold5"),
        (BASE / "0411_1821_Audio_Conv2DGRU_ChromaFrameDataset", "fold6"),
        (BASE / "0411_1827_Audio_Conv2DGRU_ChromaFrameDataset", "fold7"),
        (BASE / "0411_1836_Audio_Conv2DGRU_ChromaFrameDataset", "fold8"),
        # fold9 (0411_1845) – no version_test_summary
        (BASE / "0411_1851_Audio_Conv2DGRU_ChromaFrameDataset", "fold10"),
    ],
    "0411CQTVersionTest": [
        (BASE / "0411_1856_Audio_Conv2DGRU_CQTFrameDataset", "fold1"),
        (BASE / "0411_1912_Audio_Conv2DGRU_CQTFrameDataset", "fold2"),
        (BASE / "0411_1928_Audio_Conv2DGRU_CQTFrameDataset", "fold3"),
        (BASE / "0411_1943_Audio_Conv2DGRU_CQTFrameDataset", "fold4"),
        (BASE / "0411_1957_Audio_Conv2DGRU_CQTFrameDataset", "fold5"),
        (BASE / "0411_2012_Audio_Conv2DGRU_CQTFrameDataset", "fold6"),
        (BASE / "0411_2027_Audio_Conv2DGRU_CQTFrameDataset", "fold7"),
        (BASE / "0411_2044_Audio_Conv2DGRU_CQTFrameDataset", "fold8"),
        # fold9 (0411_2102) – no version_test_summary
        (BASE / "0411_2116_Audio_Conv2DGRU_CQTFrameDataset", "fold10"),
    ],
    "0411MelVersionTest": [
        (BASE / "0411_2129_Audio_Conv2DGRU_MelFrameDataset", "fold1"),
        (BASE / "0411_2140_Audio_Conv2DGRU_MelFrameDataset", "fold2"),
        (BASE / "0411_2150_Audio_Conv2DGRU_MelFrameDataset", "fold3"),
        (BASE / "0411_2159_Audio_Conv2DGRU_MelFrameDataset", "fold4"),
        (BASE / "0411_2208_Audio_Conv2DGRU_MelFrameDataset", "fold5"),
        (BASE / "0411_2217_Audio_Conv2DGRU_MelFrameDataset", "fold6"),
        (BASE / "0411_2226_Audio_Conv2DGRU_MelFrameDataset", "fold7"),
        (BASE / "0411_2238_Audio_Conv2DGRU_MelFrameDataset", "fold8"),
        # fold9 (0411_2250) – no version_test_summary
        (BASE / "0411_2258_Audio_Conv2DGRU_MelFrameDataset", "fold10"),
    ],
}


def combine_and_save(group_name: str, entries: list[tuple[Path, str]]) -> None:
    dfs = []
    missing = []

    for vdir, fold_label in entries:
        csv_path = vdir / "version_test_summary" / "version_test_results.csv"
        if not csv_path.exists():
            missing.append(f"{fold_label} ({vdir.name})")
            continue
        df = pd.read_csv(csv_path)
        df.insert(0, "fold", fold_label)          # track which fold each row came from
        dfs.append(df)

    if not dfs:
        print(f"[{group_name}] No data found – skipping.")
        return

    combined = pd.concat(dfs, ignore_index=True)

    out_dir = BASE / group_name
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "test_results.csv"
    combined.to_csv(out_path, index=False)

    songs = combined["song_name"].nunique()
    segs  = len(combined)
    print(f"[{group_name}]")
    print(f"  Folds combined : {len(dfs)}")
    print(f"  Unique songs   : {songs}")
    print(f"  Total segments : {segs}")
    print(f"  Saved to       : {out_path}")
    if missing:
        print(f"  ⚠ Skipped (no version_test_summary): {', '.join(missing)}")
    print()


if __name__ == "__main__":
    for group_name, entries in GROUPS.items():
        combine_and_save(group_name, entries)

    print("Done. You can now compare these files against the 0410 baselines:")
    for name in GROUPS:
        print(f"  {BASE / name / 'test_results.csv'}")
    print()
    print("0410 baselines:")
    for v in ["0410_Chroma_Version", "0410_CQT_Version",
              "0410_Mel_Original_Version", "0410_Mel_Separated_Version", "0410_Pesto_Version"]:
        p = BASE / v / "fold1_posteriorgrams" / "test_results.csv"
        if p.exists():
            print(f"  {p}")
