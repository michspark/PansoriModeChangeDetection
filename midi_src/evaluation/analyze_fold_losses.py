import pandas as pd
import glob

# Find all fold test_results.csv files (fold* pattern only)
csv_paths = sorted(glob.glob(
    "/home/sangheon/Desktop/PansoriMIDIDetection/outputs/fold*/test/test_results.csv"
))

if not csv_paths:
    print("No fold test_results.csv files found.")
    exit(1)

DISPLAY_COLS = ["song_name", "time_range", "loss", "acc", "f1_macro"]
TOP_N = 10
OUTPUT_PATH = "/home/sangheon/Desktop/PansoriMIDIDetection/analysis/fold_loss_summary.csv"

rows = []
for csv_path in csv_paths:
    fold_name = csv_path.split("/outputs/")[1].split("/")[0]
    df = pd.read_csv(csv_path)

    for rank, (_, row) in enumerate(df.nlargest(TOP_N, "loss").iterrows(), start=1):
        rows.append({"fold": fold_name, "rank": rank, "type": "highest", **{c: row[c] for c in DISPLAY_COLS}})

    for rank, (_, row) in enumerate(df.nsmallest(TOP_N, "loss").iterrows(), start=1):
        rows.append({"fold": fold_name, "rank": rank, "type": "lowest", **{c: row[c] for c in DISPLAY_COLS}})

out = pd.DataFrame(rows, columns=["fold", "type", "rank"] + DISPLAY_COLS)
out.to_csv(OUTPUT_PATH, index=False)
print(f"Saved to {OUTPUT_PATH}")


