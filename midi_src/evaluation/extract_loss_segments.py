import pandas as pd
from pydub import AudioSegment
from pathlib import Path

AUDIO_DIR  = Path("/home/sangheon/Desktop/Pansori_Data/Audio")
OUTPUT_DIR = Path("/home/sangheon/Desktop/PansoriMIDIDetection/outputs")
SUMMARY    = Path("/home/sangheon/Desktop/PansoriMIDIDetection/analysis/fold_loss_summary.csv")

df = pd.read_csv(SUMMARY)

missing = []
for _, row in df.iterrows():
    fold      = row["fold"]           # e.g. fold1_0312_153800
    loss_type = row["type"]           # highest / lowest
    rank      = int(row["rank"])
    song_name = row["song_name"]      # e.g. xxx_vocal.mid
    time_range = row["time_range"]    # e.g. 150-180s

    # Parse start/end from time_range
    parts = time_range.rstrip("s").split("-")
    start_ms = int(parts[0]) * 1000
    end_ms   = int(parts[1]) * 1000

    # Locate wav file
    wav_name = song_name.replace(".mid", ".wav")
    wav_path = AUDIO_DIR / wav_name
    if not wav_path.exists():
        missing.append(str(wav_path))
        continue

    # Build output directory: outputs/<fold>/highest|lowest/
    out_dir = OUTPUT_DIR / fold / loss_type
    out_dir.mkdir(parents=True, exist_ok=True)

    # Output filename: rank_songname_timerange.wav
    stem = Path(wav_name).stem
    out_filename = f"{rank:02d}_{stem}_{time_range}.wav"
    out_path = out_dir / out_filename

    if out_path.exists():
        print(f"  [skip] {out_path.name}")
        continue

    audio   = AudioSegment.from_wav(wav_path)
    segment = audio[start_ms:end_ms]
    segment.export(out_path, format="wav")
    print(f"  [ok]   {fold}/{loss_type}/{out_filename}  (loss={row['loss']:.4f})")

if missing:
    print(f"\nMissing audio files ({len(missing)}):")
    for m in missing:
        print(f"  {m}")
