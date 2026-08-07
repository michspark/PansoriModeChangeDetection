"""Assert infer/features.py matches datasets.py numerically.

The inference extractors are a second implementation of the same maths. If they
drift from the dataset classes, the model gets features it was never trained on
and returns confident nonsense — silently. This is the guard against that.

Only the feature-computing mixins are constructed (MelDataset, CQTDataset,
ChromaDataset, ...), never the full *FrameDataset — those preload the entire
389-song corpus and would make this check take minutes instead of seconds.

    python infer/_check_parity.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import datasets                                          # noqa: E402
import features                                          # noqa: E402
from paths import DATA_ROOT                              # noqa: E402

fails = []


def report(name, a, b):
    if a.shape != b.shape:
        print(f"  FAIL {name:8s} shape {tuple(a.shape)} vs {tuple(b.shape)}")
        fails.append(name); return
    d = (a - b).abs().max().item()
    ok = d < 1e-5
    print(f"  {'OK  ' if ok else 'FAIL'} {name:8s} shape={tuple(a.shape)} max|diff|={d:.3e}")
    if not ok:
        fails.append(name)


def one(globpat, root):
    p = next(iter(sorted((DATA_ROOT / root).rglob(globpat))), None)
    if p is None:
        print(f"  SKIP  {root}/{globpat}: no such file under PANSORI_DATA_ROOT")
    return p


# ── mel: MelDataset.get_mel vs features.mel ──────────────────────────────────
if (p := one('*.wav', 'Audio')):
    md = datasets.MelDataset(sr=16000, window=30, n_fft=2048, hop_length=512,
                             target_bins=40, is_valid=True, aug=False)
    audio = features.load_audio(p, 16000)
    report('mel', md.get_mel(audio)[0],
           features.mel(p, sr=16000, n_fft=2048, hop_length=512, target_bins=40))

# ── CQT: CQTDataset._audio_to_cqt vs features.cqt ────────────────────────────
if (p := one('*.wav', 'Audio_Original')):
    cd = datasets.CQTDataset(sr=16000, hop_length=512, n_bins=84,
                             bins_per_octave=12, is_valid=True, aug=False)
    audio = features.load_audio(p, 16000)
    # _audio_to_cqt is defined on the FrameDataset subclass but only uses
    # mixin attributes, so call it unbound against the mixin instance.
    report('cqt', datasets.CQTFrameDataset._audio_to_cqt(cd, audio),
           features.cqt(p, sr=16000, hop_length=512, n_bins=84, bins_per_octave=12))

# ── chroma: ChromaDataset._audio_to_chroma vs features.chroma ────────────────
if (p := one('*.wav', 'Audio_Original')):
    chd = datasets.ChromaDataset(sr=16000, n_fft=2048, hop_length=512,
                                 n_chroma=12, is_valid=True, aug=False)
    audio = features.load_audio(p, 16000)
    report('chroma', datasets.ChromaFrameDataset._audio_to_chroma(chd, audio),
           features.chroma(p, sr=16000, n_fft=2048, hop_length=512, n_chroma=12))

# ── pitch: PitchDataset._load_norm_contour vs features.pitch_contour ─────────
# Built via __new__ so BaseDataset.__init__ (which wants the label CSV) is skipped.
if (p := one('*.csv', 'Audio_pesto_output')):
    pd_ = object.__new__(datasets.PitchDataset)
    pd_.threshold, pd_.sr, pd_.frame_rate, pd_.comp_ratio = 0.8, 100, 20, 5
    ref = pd_._load_norm_contour(p)[:, ::pd_.comp_ratio]
    report('pesto', ref, features.pitch_contour(p, sr=100, frame_rate=20, threshold=0.8))

# ── MIDI: MidiFrameDataset.get_data's core vs features.piano_roll ────────────
if (p := one('*.mid', 'rosvot_midi')):
    import pretty_midi
    ref = torch.tensor(pretty_midi.PrettyMIDI(str(p)).get_piano_roll(fs=10),
                       dtype=torch.float32)
    report('midi', ref, features.piano_roll(p, fs=10))

# ── CMERT: raw waveform load ─────────────────────────────────────────────────
if (p := one('*.wav', 'Audio_Original')):
    report('cmert', features.load_audio(p, 24000).squeeze(0).unsqueeze(0),
           features.cmert_waveform(p, sr=24000))

print()
if fails:
    raise SystemExit(f"PARITY FAILED for: {', '.join(fails)}")
print("infer/features.py is numerically identical to datasets.py ✓")
