"""
Deletes segment posteriorgrams NOT in fold_loss_summary.csv,
then copies the kept ones into outputs/{today}_posteriorgrams/.
"""

import os
import shutil
import glob
from datetime import date
import pandas as pd

OUTPUTS_DIR = "/home/sangheon/Desktop/PansoriMIDIDetection/outputs"
SUMMARY_CSV = "/home/sangheon/Desktop/PansoriMIDIDetection/analysis/fold_loss_summary.csv"
TODAY = date.today().strftime("%Y-%m-%d")
DEST_DIR = os.path.join(OUTPUTS_DIR, f"{TODAY}_posteriorgrams")

# ── Build the set of PNG paths that should be kept ───────────────────────────
df = pd.read_csv(SUMMARY_CSV)

kept_paths = set()
for _, row in df.iterrows():
    fold = row["fold"]                          # e.g. fold10_0314_215039
    song_stem = row["song_name"].replace(".mid", "")  # drop .mid
    time_range = row["time_range"]              # e.g. 300-330s
    png_name = f"{song_stem}_{time_range}.png"
    png_path = os.path.join(OUTPUTS_DIR, fold, "test", song_stem, png_name)
    kept_paths.add(os.path.normpath(png_path))

print(f"CSV contains {len(kept_paths)} unique segment posteriorgrams to keep.")

# ── Walk all segment PNGs and delete those not in kept_paths ─────────────────
all_pngs = glob.glob(os.path.join(OUTPUTS_DIR, "fold*_*/test/**/*.png"), recursive=True)
print(f"Found {len(all_pngs)} total segment posteriorgram PNGs.")

deleted = 0
for png in all_pngs:
    if os.path.normpath(png) not in kept_paths:
        os.remove(png)
        deleted += 1

print(f"Deleted {deleted} PNGs not in the summary CSV.")

# ── Copy kept PNGs (those that still exist) into the destination folder ──────
os.makedirs(os.path.join(DEST_DIR, "highest"), exist_ok=True)
os.makedirs(os.path.join(DEST_DIR, "lowest"), exist_ok=True)

# Build a map from png_path -> type for routing to subfolders
path_to_type = {}
for _, row in df.iterrows():
    fold = row["fold"]
    song_stem = row["song_name"].replace(".mid", "")
    time_range = row["time_range"]
    png_name = f"{song_stem}_{time_range}.png"
    png_path = os.path.normpath(os.path.join(OUTPUTS_DIR, fold, "test", song_stem, png_name))
    path_to_type[png_path] = row["type"]  # "highest" or "lowest"

copied = 0
missing = 0
for png_path, loss_type in sorted(path_to_type.items()):
    if not os.path.exists(png_path):
        print(f"  WARNING: expected file not found: {png_path}")
        missing += 1
        continue
    rel = os.path.relpath(png_path, OUTPUTS_DIR)
    fold_name = rel.split(os.sep)[0]
    dest_name = f"{fold_name}_{os.path.basename(png_path)}"
    shutil.move(png_path, os.path.join(DEST_DIR, loss_type, dest_name))
    copied += 1

print(f"Copied {copied} PNGs to {DEST_DIR}")
if missing:
    print(f"  {missing} expected files were missing (check warnings above).")
