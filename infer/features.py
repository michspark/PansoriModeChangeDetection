"""Standalone feature extractors for inference on arbitrary files.

These mirror the corresponding classes in `datasets.py` exactly, but take a
single file path instead of the labeled corpus. The dataset classes cannot be
reused here: they key everything off `hash_key` and require `label.csv`, so they
only work on annotated corpus files.

Every parameter is read from the checkpoint's own `config.yaml` rather than
hardcoded — a feature computed with different settings than the model was
trained on produces confident nonsense rather than an error.

`tests/` equivalent: `infer/_check_parity.py` asserts each extractor here is
numerically identical to its `datasets.py` counterpart.
"""

import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch


# ── audio loading (mirrors AudioDataset._load_audio) ──────────────────────────

def load_audio(path, sr, mono=True):
    """(1, samples) float32 at `sr`, mono-mixed."""
    import soundfile as sf
    import torchaudio
    data, file_sr = sf.read(str(path), dtype='float32', always_2d=True)
    audio = torch.from_numpy(data.T)
    if file_sr != sr:
        audio = torchaudio.functional.resample(audio, orig_freq=file_sr, new_freq=sr)
    if mono:
        audio = audio.mean(dim=0, keepdim=True)
    return audio


# ── mel (mirrors MelDataset.get_mel) ──────────────────────────────────────────

def mel(path, sr=16000, n_fft=2048, hop_length=512, target_bins=40, **_):
    from torchaudio.transforms import Spectrogram, MelScale, AmplitudeToDB
    audio = load_audio(path, sr)
    spec = Spectrogram(n_fft=n_fft, hop_length=hop_length, power=1.0)(audio)
    m = MelScale(n_stft=n_fft // 2 + 1, n_mels=target_bins,
                 sample_rate=sr, f_min=80, f_max=2000)(spec).squeeze(0)
    return AmplitudeToDB()(m) / 100                      # (target_bins, T)


# ── CQT (mirrors CQTDataset._audio_to_cqt) ────────────────────────────────────

def cqt(path, sr=16000, hop_length=512, n_bins=84, bins_per_octave=12, **_):
    from nnAudio.features.cqt import CQT
    audio = load_audio(path, sr)
    tf = CQT(sr=sr, hop_length=hop_length, fmin=32.7, n_bins=n_bins,
             bins_per_octave=bins_per_octave, filter_scale=1, norm=1,
             window='hann', center=True, pad_mode='reflect',
             trainable=False, output_format='Magnitude', verbose=False)
    return torch.log1p(tf(audio).squeeze(0) * 1000)      # (n_bins, T)


# ── chroma (mirrors ChromaDataset._audio_to_chroma) ───────────────────────────

def chroma(path, sr=16000, n_fft=2048, hop_length=512, n_chroma=12, **_):
    from torchaudio.transforms import Spectrogram
    from nnAudio.librosa_functions import chroma as chroma_filterbank
    audio = load_audio(path, sr)
    spec = Spectrogram(n_fft=n_fft, hop_length=hop_length, power=1.0)(audio).squeeze(0)
    w = torch.from_numpy(chroma_filterbank(sr=sr, n_fft=n_fft, n_chroma=n_chroma))
    return torch.log1p(torch.matmul(w, spec) * 1000)     # (n_chroma, T)


# ── pitch contour (mirrors PitchDataset._load_norm_contour) ───────────────────

def pitch_contour(path, sr=100, frame_rate=20, threshold=0.8, **_):
    """f0 CSV -> (2, T) = [tonic-normalised MIDI, confidence], subsampled to
    `frame_rate`. The CSV must have `frequency` (Hz) and `confidence` columns at
    `sr` Hz (100 Hz for both PESTO and CREPE output).

    Tonic normalisation is why a raw CSV cannot be fed to the model directly:
    the tonic is the most common rounded MIDI value among confident frames, and
    the contour is expressed in octaves relative to it.
    """
    import pandas as pd
    assert 100 % frame_rate == 0, f"frame_rate must divide 100, got {frame_rate}"
    df = pd.read_csv(path)
    freq = df['frequency'].values.astype(np.float64)
    conf = df['confidence'].values.astype(np.float32)

    midi = [69 + 12 * math.log2(max(f, 1e-8) / 440) for f in freq]
    confident = np.round(midi)[conf >= threshold]
    if confident.size == 0:
        raise ValueError(
            f"{path}: no frame reaches confidence >= {threshold}, so the tonic is "
            f"undefined. Lower --threshold or check the contour.")
    tonic = float(Counter(confident).most_common(1)[0][0])

    norm = [(m - tonic) / 12 for m in midi]
    feat = torch.tensor(np.stack([norm, conf], axis=0), dtype=torch.float32)
    return feat[:, ::(sr // frame_rate)]                 # (2, T)


# ── MIDI piano roll (mirrors MidiFrameDataset.get_data) ───────────────────────

def piano_roll(path, fs=10, **_):
    import pretty_midi
    roll = pretty_midi.PrettyMIDI(str(path)).get_piano_roll(fs=fs)
    return torch.tensor(roll, dtype=torch.float32)       # (128, T) raw velocities


# ── CultureMERT (raw waveform; the model holds the feature extractor) ─────────

def cmert_waveform(path, sr=24000, **_):
    """CMERTClassifier runs Wav2Vec2FeatureExtractor internally, so the model
    input is the raw 24 kHz waveform, not a spectrogram."""
    return load_audio(path, sr).squeeze(0).unsqueeze(0)  # (1, samples)


# ── registry: representation -> (extractor, expected input suffix) ────────────

EXTRACTORS = {
    'mel':    (mel,            ('.wav', '.mp3', '.flac')),
    'cqt':    (cqt,            ('.wav', '.mp3', '.flac')),
    'chroma': (chroma,         ('.wav', '.mp3', '.flac')),
    'pesto':  (pitch_contour,  ('.csv',)),
    'crepe':  (pitch_contour,  ('.csv',)),
    'midi':   (piano_roll,     ('.mid', '.midi')),
    'cmert':  (cmert_waveform, ('.wav', '.mp3', '.flac')),
}
