"""
Pansori Mode Change Detection - Interactive Inference Viewer
FastAPI server: serves audio, runs mel-spectrogram inference on demand.
"""
import os, json, sys
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import uvicorn

# ── paths ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent          # PansoriModeChangeDetection/
AUDIO_DIR = Path('/home/sangheon/Desktop/Pansori_Data/pansori_inf_separated')
CACHE_DIR = Path(__file__).parent / 'cache'
CACHE_DIR.mkdir(exist_ok=True)

WEIGHTS_BASE = ROOT / 'weights/frame'
DEFAULT_CKPT = ROOT / '/home/sangheon/Desktop/PansoriModeChangeDetection/weights/frame/0402_0251_Audio_Conv2DGRU_MelFrameDataset/fold7_best_model.pt'
DEFAULT_FOLD = 7

# fold → best_model.pt path (auto-discovered)
FOLD_CKPTS = {}
for d in sorted(WEIGHTS_BASE.iterdir()):
    for pt in sorted(d.glob('fold*_best_model.pt')):
        fold_num = int(pt.stem.split('_')[0].replace('fold', ''))
        if fold_num not in FOLD_CKPTS:          # keep first (earliest timestamp)
            FOLD_CKPTS[fold_num] = pt
# override fold7 with the preferred checkpoint
FOLD_CKPTS[7] = DEFAULT_CKPT

CLASS_NAMES = ['Unknown', '우조', '계면조', '아니리', '창조']

# mel spectrogram params (must match mel_base.yaml)
SR           = 16000
HOP_LENGTH   = 512
N_FFT        = 2048
TARGET_BINS  = 40
FS           = SR / HOP_LENGTH          # ≈ 31.25 frames/sec
WINDOW_FRAMES = 30 * SR // HOP_LENGTH  # 937
HOP_FRAMES   = WINDOW_FRAMES // 3      # ≈ 312

# ── model (lazy-loaded per fold) ───────────────────────────────────────────
_models = {}

def get_model(fold: int):
    if fold in _models:
        return _models[fold]

    if fold not in FOLD_CKPTS:
        raise RuntimeError(f'No checkpoint found for fold {fold}. Available: {sorted(FOLD_CKPTS.keys())}')

    sys.path.insert(0, str(ROOT))
    import torch
    from omegaconf import OmegaConf
    from models.Conv2DGRU import Conv2DGRU

    cfg = OmegaConf.create({
        'num_layers': 3, 'in_channels': 1, 'num_bins': TARGET_BINS,
        'last_hidden_dim': 64, 'use_gradual_size': False,
        'kernel_size': 3, 'pool_size': '(2,1)', 'dilation': '(1,2)',
        'cnn_dropout': 0, 'num_gru': 1, 'hidden_dim': 64,
        'dropout': 0.3, 'num_classes': 5,
    })

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = Conv2DGRU(cfg).to(device)
    ckpt_path = FOLD_CKPTS[fold]
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    print(f'[viewer] fold{fold} loaded from {ckpt_path.name} on {device}')

    _models[fold] = (model, device)
    return model, device


def audio_to_mel(wav_path: Path) -> np.ndarray:
    """Load wav → mel spectrogram (TARGET_BINS, T), normalised /100."""
    import torch, soundfile as sf
    from torchaudio.transforms import Spectrogram, MelScale, AmplitudeToDB

    data, sr = sf.read(str(wav_path), dtype='float32', always_2d=True)
    audio = torch.from_numpy(data.T)                     # (ch, samples)
    if sr != SR:
        import torchaudio
        audio = torchaudio.functional.resample(audio, sr, SR)
    audio = audio.mean(dim=0, keepdim=True)              # mono

    spec_cvt  = Spectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH, power=1.0)
    spec2mel  = MelScale(n_stft=N_FFT//2+1, n_mels=TARGET_BINS,
                         sample_rate=SR, f_min=80, f_max=2000)
    db_cvt    = AmplitudeToDB()

    spec = spec_cvt(audio)                               # (1, n_stft, T)
    mel  = spec2mel(spec).squeeze(0)                     # (TARGET_BINS, T)
    mel  = db_cvt(mel) / 100                             # normalised
    return mel.numpy()


def run_inference(wav_path: Path, fold: int) -> list:
    """
    Returns (T, 5) probability matrix as a Python list.
    Window strategy mirrors PansoriMIDIDetection viewer:
      first chunk  → leftmost  HOP_FRAMES
      middle chunks → center   HOP_FRAMES
      last chunk   → remaining frames only
    """
    import torch

    model, device = get_model(fold)
    mel   = audio_to_mel(wav_path)          # (TARGET_BINS, total_frames)
    total = mel.shape[1]
    all_probs = []

    with torch.no_grad():
        for chunk_start in range(0, total, HOP_FRAMES):
            win_start = max(0, chunk_start - HOP_FRAMES)
            clip = mel[:, win_start: win_start + WINDOW_FRAMES]

            if clip.shape[1] < WINDOW_FRAMES:
                clip = np.concatenate(
                    [clip, np.zeros((TARGET_BINS, WINDOW_FRAMES - clip.shape[1]))],
                    axis=1)

            x      = torch.tensor(clip, dtype=torch.float32).unsqueeze(0).to(device)
            logits = model(x)                          # (1, WINDOW_FRAMES, 5)
            probs  = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

            offset   = chunk_start - win_start
            is_last  = (chunk_start + HOP_FRAMES >= total)
            keep_len = (total - chunk_start) if is_last else HOP_FRAMES
            all_probs.append(probs[offset: offset + keep_len])

    return np.concatenate(all_probs, axis=0)[:total].tolist()


app = FastAPI()

@app.get('/')
def index():
    html_path = Path(__file__).parent / 'index.html'
    return HTMLResponse(html_path.read_text(encoding='utf-8'))


@app.get('/api/songs')
def list_songs():
    songs = []
    for f in sorted(AUDIO_DIR.iterdir()):
        if f.suffix == '.wav':
            name = f.stem
            songs.append({
                'name':   name,
                'cached': (CACHE_DIR / (name + '.json')).exists(),
            })
    return songs


@app.get('/api/folds')
def list_folds():
    return sorted(FOLD_CKPTS.keys())


@app.get('/api/inference/{song_name}')
def get_inference(song_name: str, fold: int = DEFAULT_FOLD):
    cache_file = CACHE_DIR / f'{song_name}__fold{fold}.json'

    if cache_file.exists():
        return JSONResponse(json.loads(cache_file.read_text()))

    wav_file = AUDIO_DIR / (song_name + '.wav')
    if not wav_file.exists():
        raise HTTPException(404, f'Audio not found: {song_name}')

    print(f'[viewer] running inference (fold{fold}) for {song_name} …')
    probs  = run_inference(wav_file, fold)
    result = {'song': song_name, 'fold': fold, 'fs': FS,
               'classes': CLASS_NAMES, 'probs': probs}
    cache_file.write_text(json.dumps(result))
    return JSONResponse(result)


@app.get('/audio/{song_name}')
def serve_audio(song_name: str):
    audio_file = AUDIO_DIR / (song_name + '.wav')
    if not audio_file.exists():
        raise HTTPException(404, f'Audio not found: {song_name}')
    return FileResponse(str(audio_file), media_type='audio/wav')


@app.delete('/api/cache/{song_name}')
def delete_cache(song_name: str, fold: int = DEFAULT_FOLD):
    cache_file = CACHE_DIR / f'{song_name}__fold{fold}.json'
    if cache_file.exists():
        cache_file.unlink()
    return {'status': 'ok'}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7861)
    parser.add_argument('--host', default='0.0.0.0')
    args = parser.parse_args()
    uvicorn.run('app:app', host=args.host, port=args.port, reload=False)
