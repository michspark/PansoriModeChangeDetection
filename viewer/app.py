"""
Pansori Mode Change Detection - Ensemble Inference Viewer
Mel (original) + PESTO (pitch) + MIDI (ROSVOT) soft-voting ensemble.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA_ROOT, MIDI_REPO
import os, json, sys, math
import numpy as np
from pathlib import Path
from collections import Counter
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import uvicorn

ROOT      = Path(__file__).parent.parent
AUDIO_DIR = DATA_ROOT / 'pansori_inf_separated'
F0_DIR    = DATA_ROOT / 'pansori_inf_f0'
MIDI_DIR  = DATA_ROOT / 'pansori_inf_midi/midi'
CACHE_DIR = Path(__file__).parent / 'cache'
CACHE_DIR.mkdir(exist_ok=True)

MEL_CKPT  = ROOT / 'weights/frame/Mel_Original_Version/fold1_best_model.pt'
PESTO_CKPT= ROOT / 'weights/frame/Pesto_Version/fold1_best_model.pt'
MIDI_CKPT = MIDI_REPO / 'outputs/MIDI_Version/23-43-24/best_model_fold1.pt'
REPO_MIDI = MIDI_REPO

CLASS_NAMES = ['Unknown', '우조', '계면조', '아니리', '창조']

SR         = 16000
HOP        = 512
N_FFT      = 2048
MEL_BINS   = 40
AUDIO_FPS  = SR / HOP        # 31.25
AUDIO_WIN  = 30 * SR // HOP  # ~937

PESTO_FPS  = 20
PESTO_WIN  = 30 * PESTO_FPS  # 600
PESTO_COMP = 5               # 100fps raw → 20fps

MIDI_FPS   = 10
MIDI_WIN   = 30 * MIDI_FPS   # 300

_models = {}

sys.path.insert(0, str(ROOT))


# ── Model loaders ──────────────────────────────────────────────────────────────

def get_mel_model():
    if 'mel' in _models:
        return _models['mel']
    import torch
    from omegaconf import OmegaConf
    from models.Conv2DGRU import Conv2DGRU
    cfg = OmegaConf.create({
        'num_layers': 3, 'in_channels': 1, 'num_bins': MEL_BINS,
        'last_hidden_dim': 64, 'use_gradual_size': False,
        'kernel_size': 3, 'pool_size': '(2,1)', 'dilation': '(1,2)',
        'cnn_dropout': 0, 'num_gru': 1, 'hidden_dim': 64,
        'dropout': 0.3, 'num_classes': 5,
    })
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Conv2DGRU(cfg).to(device)
    state = torch.load(MEL_CKPT, map_location=device, weights_only=True)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()})
    model.eval()
    print(f'[viewer] mel loaded → {device}')
    _models['mel'] = (model, device)
    return _models['mel']


def get_pesto_model():
    if 'pesto' in _models:
        return _models['pesto']
    import torch
    from omegaconf import OmegaConf
    from models.Conv1DGRU import Conv1DGRU
    cfg = OmegaConf.create({
        'num_layers': 3, 'in_channels': 2,
        'last_hidden_dim': 128, 'use_gradual_size': False,
        'kernel_size': 3, 'pool_size': False, 'dilation': 1,
        'cnn_dropout': 0, 'num_gru': 3, 'hidden_dim': 128,
        'dropout': 0.0, 'num_frames': PESTO_WIN,
        'num_classes': 5,
    })
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Conv1DGRU(cfg).to(device)
    state = torch.load(PESTO_CKPT, map_location=device, weights_only=True)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()})
    model.eval()
    print(f'[viewer] pesto loaded → {device}')
    _models['pesto'] = (model, device)
    return _models['pesto']


def get_midi_model():
    if 'midi' in _models:
        return _models['midi']
    import torch
    from omegaconf import OmegaConf

    _evicted = {k: v for k, v in sys.modules.items()
                if k in ('models', 'datasets') or k.startswith(('models.', 'datasets.'))}
    for k in _evicted:
        del sys.modules[k]
    sys.path.insert(0, str(REPO_MIDI))
    try:
        from models.model_zoo import Conv2DGRU as MidiModel
    finally:
        sys.path.pop(0)
        for k in list(sys.modules.keys()):
            if k in ('models', 'datasets') or k.startswith(('models.', 'datasets.')):
                del sys.modules[k]
        sys.modules.update(_evicted)

    midi_cfg = OmegaConf.load(REPO_MIDI / 'configs/config.yaml')
    for key in ['data', 'model', 'train']:
        OmegaConf.update(midi_cfg, key,
                         OmegaConf.load(REPO_MIDI / f'configs/{key}/{key}.yaml'), merge=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MidiModel(midi_cfg.model).to(device)
    state = torch.load(MIDI_CKPT, map_location=device)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()})
    model.eval()
    print(f'[viewer] midi loaded → {device}')
    _models['midi'] = (model, device)
    return _models['midi']


# ── Feature extraction ─────────────────────────────────────────────────────────

def audio_to_mel(wav_path: Path) -> np.ndarray:
    import torch, soundfile as sf
    from torchaudio.transforms import Spectrogram, MelScale, AmplitudeToDB
    data, sr = sf.read(str(wav_path), dtype='float32', always_2d=True)
    audio = torch.from_numpy(data.T)
    if sr != SR:
        import torchaudio
        audio = torchaudio.functional.resample(audio, sr, SR)
    audio = audio.mean(dim=0, keepdim=True)
    spec = Spectrogram(n_fft=N_FFT, hop_length=HOP, power=1.0)(audio)
    mel  = MelScale(n_stft=N_FFT//2+1, n_mels=MEL_BINS, sample_rate=SR, f_min=80, f_max=2000)(spec).squeeze(0)
    return (AmplitudeToDB()(mel) / 100).numpy()  # (MEL_BINS, T)


def pesto_to_contour(csv_path: Path) -> np.ndarray:
    import pandas as pd
    df = pd.read_csv(csv_path)
    freq = df['frequency'].values.astype(np.float64)
    conf = df['confidence'].values.astype(np.float32)
    midi = np.array([69 + 12 * math.log2(max(f, 1e-8) / 440) for f in freq])
    mask = conf >= 0.0
    tonic = float(Counter(np.round(midi[mask]).tolist()).most_common(1)[0][0]) if mask.any() else 69.0
    norm  = ((midi - tonic) / 12).astype(np.float32)
    feat  = np.stack([norm, conf], axis=0)          # (2, T_100fps)
    return feat[:, ::PESTO_COMP]                    # (2, T_20fps)


def midi_to_piano_roll(midi_path: Path) -> np.ndarray:
    import pretty_midi
    pm   = pretty_midi.PrettyMIDI(str(midi_path))
    roll = pm.get_piano_roll(fs=MIDI_FPS)           # (128, T_10fps)
    return roll.astype(np.float32)


# ── Sliding-window inference ───────────────────────────────────────────────────

def sliding_inference(model, device, feat: np.ndarray, win: int) -> np.ndarray:
    """feat: (n_bins, T) → probs: (T, 5)"""
    import torch
    n_bins, T = feat.shape
    all_probs = []
    with torch.no_grad():
        for start in range(0, T, win):
            end   = min(start + win, T)
            chunk = feat[:, start:end]
            if chunk.shape[1] < win:
                chunk = np.concatenate(
                    [chunk, np.zeros((n_bins, win - chunk.shape[1]), dtype=np.float32)], axis=1)
            x     = torch.tensor(chunk[np.newaxis], dtype=torch.float32).to(device)
            probs = torch.softmax(model(x), dim=-1)[0].cpu().numpy()   # (win, 5)
            all_probs.append(probs[:end - start])
    return np.concatenate(all_probs, axis=0)  # (T, 5)


def align_to_fps(probs: np.ndarray, src_fps: float, T_target: int) -> np.ndarray:
    T_src, C = probs.shape
    ratio     = src_fps / MIDI_FPS
    out       = np.zeros((T_target, C), dtype=np.float32)
    for j in range(T_target):
        a = max(0, min(int(round(j * ratio)), T_src - 1))
        b = max(a + 1, min(int(round((j + 1) * ratio)), T_src))
        out[j] = probs[a:b].mean(axis=0)
    return out


# ── Ensemble ───────────────────────────────────────────────────────────────────

def run_ensemble(song_name: str) -> dict:
    wav_path  = AUDIO_DIR / f'{song_name}.wav'
    f0_path   = F0_DIR    / f'{song_name}.f0.csv'
    midi_path = MIDI_DIR  / f'{song_name}.mid'

    if not wav_path.exists():
        raise FileNotFoundError(f'Audio not found: {wav_path}')

    # Mel
    mel_feat  = audio_to_mel(wav_path)
    mel_model, mel_dev = get_mel_model()
    mel_probs = sliding_inference(mel_model, mel_dev, mel_feat, AUDIO_WIN)

    # MIDI — determines reference length T_midi
    if midi_path.exists():
        roll = midi_to_piano_roll(midi_path)
        T_midi = roll.shape[1]
        midi_model, midi_dev = get_midi_model()
        midi_probs = sliding_inference(midi_model, midi_dev, roll, MIDI_WIN)
    else:
        T_midi     = int(mel_feat.shape[1] * MIDI_FPS / AUDIO_FPS)
        midi_probs = None

    # PESTO
    if f0_path.exists():
        pesto_feat  = pesto_to_contour(f0_path)
        pm, pd_     = get_pesto_model()
        pesto_probs = sliding_inference(pm, pd_, pesto_feat, PESTO_WIN)
    else:
        pesto_probs = None

    mel_a   = align_to_fps(mel_probs, AUDIO_FPS, T_midi)
    pesto_a = align_to_fps(pesto_probs, PESTO_FPS, T_midi) if pesto_probs is not None else mel_a
    midi_a  = midi_probs if midi_probs is not None else mel_a

    ens = (mel_a + pesto_a + midi_a) / 3.0

    return {
        'ensemble': ens.tolist(),
        'mel':      mel_a.tolist(),
        'pesto':    pesto_a.tolist(),
        'midi':     midi_a.tolist(),
        'fs':       float(MIDI_FPS),
        'classes':  CLASS_NAMES,
        'song':     song_name,
    }


# ── FastAPI ────────────────────────────────────────────────────────────────────

app = FastAPI()


@app.get('/')
def index():
    return HTMLResponse((Path(__file__).parent / 'index.html').read_text(encoding='utf-8'))


@app.get('/api/songs')
def list_songs():
    songs = []
    for f in sorted(AUDIO_DIR.iterdir()):
        if f.suffix == '.wav':
            name = f.stem
            songs.append({
                'name':   name,
                'cached': (CACHE_DIR / f'{name}__ensemble.json').exists(),
            })
    return songs


@app.get('/api/inference/{song_name}')
def get_inference(song_name: str):
    cache = CACHE_DIR / f'{song_name}__ensemble.json'
    if cache.exists():
        return JSONResponse(json.loads(cache.read_text()))

    wav = AUDIO_DIR / f'{song_name}.wav'
    if not wav.exists():
        raise HTTPException(404, f'Audio not found: {song_name}')

    print(f'[viewer] running ensemble for {song_name} …')
    result = run_ensemble(song_name)
    cache.write_text(json.dumps(result))
    return JSONResponse(result)


@app.get('/audio/{song_name}')
def serve_audio(song_name: str):
    f = AUDIO_DIR / f'{song_name}.wav'
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(str(f), media_type='audio/wav')


@app.delete('/api/cache/{song_name}')
def clear_cache(song_name: str):
    (CACHE_DIR / f'{song_name}__ensemble.json').unlink(missing_ok=True)
    return {'status': 'ok'}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=7861)
    p.add_argument('--host', default='0.0.0.0')
    a = p.parse_args()
    uvicorn.run('app:app', host=a.host, port=a.port, reload=False)
