"""
2-column pattern comparison figure
  Pattern A (col 0): Gmj → Ujo  — 8abdaa3f, 360-390s, fold7
  Pattern B (col 1): Ujo → Gmj  — b43b56d6, 150-180s, fold2

Rows per column:
  0. F0 Contour   (scatter, coloured by ensemble argmax)
  1. GT bar
  2. Mel bar
  3. MIDI bar
  4. PESTO bar
  5. CMERT bar
  6. Ensemble bar
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import DATA_ROOT, MIDI_REPO, REPO_ROOT

import sys

import math, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import Spectrogram, MelScale, AmplitudeToDB
from omegaconf import OmegaConf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import matplotlib.font_manager as fm
from pathlib import Path

# ── repo paths ─────────────────────────────────────────────────────────────────
REPO_MODE = REPO_ROOT
REPO_MIDI = MIDI_REPO

# Mode Detection repo first — must stay first so 'models' resolves to REPO_MODE
sys.path.insert(0, str(REPO_MODE))
import models as mode_models

# MIDI repo: import via importlib to avoid polluting sys.path 'models' namespace
import importlib.util, types

def _import_from(repo: Path, rel: str, as_name: str):
    spec = importlib.util.spec_from_file_location(as_name, repo / rel)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[as_name] = mod
    spec.loader.exec_module(mod)
    return mod

# Load MIDI models package as 'midi_models' to avoid collision with 'models'
import importlib
_midi_pkg = types.ModuleType('midi_models')
_midi_pkg.__path__ = [str(REPO_MIDI / 'models')]
_midi_pkg.__package__ = 'midi_models'
sys.modules['midi_models'] = _midi_pkg

for _sub in ('modules', 'model_utils', 'model_zoo'):
    _spec = importlib.util.spec_from_file_location(
        f'midi_models.{_sub}', REPO_MIDI / f'models/{_sub}.py',
        submodule_search_locations=[]
    )
    _mod = importlib.util.module_from_spec(_spec)
    _mod.__package__ = 'midi_models'
    sys.modules[f'midi_models.{_sub}'] = _mod
    _spec.loader.exec_module(_mod)

MidiConv2DGRU = sys.modules['midi_models.model_zoo'].Conv2DGRU

# PianoRollGenerator (only needs pretty_midi)
import pretty_midi

class PianoRollGenerator:
    def __init__(self, midi_path, fs=100):
        self.midi = pretty_midi.PrettyMIDI(midi_path)
        self.fs   = fs
    def generate_piano_roll(self):
        return self.midi.get_piano_roll(fs=self.fs)

# ── style ──────────────────────────────────────────────────────────────────────
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
plt.rcParams['font.family'] = fm.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams['axes.unicode_minus'] = False

CLASS_NAMES  = ['Unknown', 'Ujo', 'Gmj', 'Anr', 'CJo']
CLASS_COLORS = ['#cccccc', '#4C72B0', '#C44E52', '#999999', '#DD8452']
LABEL_MAP    = {'Unknown': 0, '경드름': 1, '설렁제': 1, '평조': 1, '우조': 1,
                '계면조': 2, '아니리': 3, '창조': 4}
KOR_TO_ENG   = {'Unknown': 'Unknown', '우조': 'Ujo', '경드름': 'Ujo', '설렁제': 'Ujo',
                '평조': 'Ujo', '계면조': 'Gmj', '아니리': 'Anr', '창조': 'CJo'}
N_CLASSES    = 5
CONF_THRESH  = 0.5
DATA_DIR     = DATA_ROOT
LABEL_CSV    = REPO_MODE / 'data/Label/label.csv'
DEV          = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── segment definitions ────────────────────────────────────────────────────────
SEGS = [
    dict(
        title   = 'Pattern A:  Gmj → Ujo',
        hash    = '8abdaa3f',
        start_s = 360.0,
        end_s   = 390.0,
        fold    = 7,
        audio   = DATA_DIR / 'Audio_Original/8abdaa3f-03-이일주-심청가_심청_탄생.wav',
        pesto   = DATA_DIR / 'Audio_pesto_output/8abdaa3f-03-이일주-심청가_심청_탄생_vocal.f0.csv',
        midi    = DATA_DIR / 'rosvot_midi/8abdaa3f-03-이일주-심청가_심청_탄생_vocal.mid',
    ),
    dict(
        title   = 'Pattern B:  Ujo → Gmj',
        hash    = 'b43b56d6',
        start_s = 150.0,
        end_s   = 180.0,
        fold    = 2,
        audio   = DATA_DIR / 'Audio_Original/b43b56d6-03-성우향-춘향가_이도령이_광한루에서_구경하는_데적성가춘향이_등장하는_데.wav',
        pesto   = DATA_DIR / 'Audio_pesto_output/b43b56d6-03-성우향-춘향가_이도령이_광한루에서_구경하는_데적성가춘향이_등장하는_데_vocal.f0.csv',
        midi    = DATA_DIR / 'rosvot_midi/b43b56d6-03-성우향-춘향가_이도령이_광한루에서_구경하는_데적성가춘향이_등장하는_데_vocal.mid',
    ),
]

# ── model checkpoint paths per fold ────────────────────────────────────────────
def mel_ckpt(fold):
    dirs = {
        7: REPO_MODE / 'weights/frame/Mel_Original_Song_Stratified/0424_1937_Audio_Original_Conv2DGRU_MelFrameDataset',
        2: REPO_MODE / 'weights/frame/Mel_Original_Song_Stratified/0424_1851_Audio_Original_Conv2DGRU_MelFrameDataset',
    }
    return dirs[fold] / f'fold{fold}_best_model.pt', dirs[fold] / 'config.yaml'

def pesto_ckpt(fold):
    dirs = {
        7: REPO_MODE / 'weights/frame/Pesto_Song_Stratified/0424_2116_Audio_pesto_output_Conv1DGRU_PitchFrameDataset',
        2: REPO_MODE / 'weights/frame/Pesto_Song_Stratified/0424_2050_Audio_pesto_output_Conv1DGRU_PitchFrameDataset',
    }
    return dirs[fold] / f'fold{fold}_best_model.pt', dirs[fold] / 'config.yaml'

def cmert_ckpt(fold):
    dirs = {
        7: REPO_MODE / 'weights/cmert/cmert_song_stratified/0426_0206_Audio_Original_CMERTClassifier_CMERTFrameDataset',
        2: REPO_MODE / 'weights/cmert/cmert_song_stratified/0425_0758_Audio_Original_CMERTClassifier_CMERTFrameDataset',
    }
    return dirs[fold] / f'fold{fold}_best_model.pt', dirs[fold] / 'config.yaml'

def midi_ckpt(fold):
    dirs = {
        7: REPO_MIDI / 'outputs/MIDI_Song_Stratified/22-43-41',
        2: REPO_MIDI / 'outputs/MIDI_Song_Stratified/22-03-32',
    }
    return dirs[fold] / f'best_model_fold{fold}.pt'

# ── GT loader ─────────────────────────────────────────────────────────────────
df_label = pd.read_csv(LABEL_CSV)

def get_gt_segments(hash_key, start_s, end_s):
    song = df_label[df_label['hash_key'] == hash_key].copy()
    song['start_s'] = song['start'] / 1000.0
    song['end_s']   = song['end']   / 1000.0
    song['cls']     = song['label'].map(LABEL_MAP)
    overlapping = song[(song['end_s'] > start_s) & (song['start_s'] < end_s)].copy()
    overlapping['start_s'] = overlapping['start_s'].clip(lower=start_s)
    overlapping['end_s']   = overlapping['end_s'].clip(upper=end_s)
    return overlapping

# ── PESTO pitch loader ─────────────────────────────────────────────────────────
def load_pesto(path, start_s, end_s, conf_thresh=CONF_THRESH):
    df = pd.read_csv(path)
    seg = df[(df['time'] >= start_s * 1000) & (df['time'] < end_s * 1000)].copy()
    seg['time_s'] = seg['time'] / 1000.0
    seg['midi']   = seg['frequency'].apply(lambda f: 69 + 12 * math.log2(max(f, 1e-6) / 440.0))
    return seg[seg['confidence'] >= conf_thresh].copy()

# ── audio loader ───────────────────────────────────────────────────────────────
def load_audio_slice(path, start_s, end_s, target_sr):
    audio, sr = sf.read(path, dtype='float32', always_2d=True)
    audio = torch.from_numpy(audio.T).mean(0, keepdim=True)
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    s, e = int(start_s * target_sr), int(end_s * target_sr)
    return audio[:, s:e]

# ── inference: Mel ─────────────────────────────────────────────────────────────
def infer_mel(seg):
    pt, cfg_path = mel_ckpt(seg['fold'])
    cfg = OmegaConf.load(cfg_path)
    p = cfg.dataset.params
    SR, HOP, N_FFT, N_MELS = p.sr, p.hop_length, p.n_fft, p.target_bins

    audio = load_audio_slice(seg['audio'], seg['start_s'], seg['end_s'], SR)
    spec  = Spectrogram(n_fft=N_FFT, hop_length=HOP, power=1.0)(audio)
    mel   = MelScale(n_stft=N_FFT//2+1, n_mels=N_MELS, sample_rate=SR, f_min=80, f_max=2000)(spec).squeeze(0)
    mel   = AmplitudeToDB()(mel) / 100

    model = mode_models.Conv2DGRU(cfg.model.params)
    model.load_state_dict(torch.load(pt, map_location=DEV, weights_only=True))
    model.eval().to(DEV)

    with torch.no_grad():
        probs = torch.softmax(model(mel.unsqueeze(0).to(DEV)), dim=-1)[0].cpu().numpy()
    fps = SR / HOP
    return probs, fps   # (T, 5)

# ── inference: PESTO pitch model ───────────────────────────────────────────────
def infer_pesto_model(seg):
    pt, cfg_path = pesto_ckpt(seg['fold'])
    cfg = OmegaConf.load(cfg_path)
    p   = cfg.dataset.params
    SR_PESTO, FRAME_RATE = 100, p.frame_rate   # PESTO SR = 100 Hz (10ms)
    COMP = SR_PESTO // FRAME_RATE

    df = pd.read_csv(seg['pesto'])
    s_ms, e_ms = seg['start_s'] * 1000, seg['end_s'] * 1000
    slice_df = df[(df['time'] >= s_ms) & (df['time'] < e_ms)].copy()
    freq  = torch.tensor(slice_df['frequency'].values, dtype=torch.float32)
    conf  = torch.tensor(slice_df['confidence'].values, dtype=torch.float32)
    midi  = 69 + 12 * torch.log2((freq / 440.0).clamp(min=1e-6))
    tonic_conf_mask = conf >= p.threshold
    if tonic_conf_mask.sum() > 0:
        tonic = torch.mode(midi[tonic_conf_mask].round().long())[0].float()
    else:
        tonic = midi.mean()
    norm_midi = (midi - tonic) / 12.0
    freq_conf = torch.stack([norm_midi, conf], dim=0)        # (2, T_100fps)
    freq_conf = freq_conf[:, ::COMP]                         # (2, T_20fps)

    model = mode_models.Conv1DGRU(cfg.model.params)
    model.load_state_dict(torch.load(pt, map_location=DEV, weights_only=True))
    model.eval().to(DEV)

    with torch.no_grad():
        probs = torch.softmax(model(freq_conf.unsqueeze(0).to(DEV)), dim=-1)[0].cpu().numpy()
    return probs, float(FRAME_RATE)   # (T, 5)

# ── inference: CMERT ───────────────────────────────────────────────────────────
def infer_cmert(seg):
    pt, cfg_path = cmert_ckpt(seg['fold'])
    cfg = OmegaConf.load(cfg_path)
    SR  = cfg.dataset.params.sr   # 24000

    audio = load_audio_slice(seg['audio'], seg['start_s'], seg['end_s'], SR).squeeze(0)

    model = mode_models.CMERTClassifier(cfg.model.params)
    model.load_state_dict(torch.load(pt, map_location=DEV, weights_only=True))
    model.eval().to(DEV)

    with torch.no_grad():
        probs = torch.softmax(model(audio.unsqueeze(0).to(DEV)), dim=-1)[0].cpu().numpy()
    HOP_CMERT = 320
    fps = SR / HOP_CMERT
    return probs, fps   # (T, 5)

# ── inference: MIDI ────────────────────────────────────────────────────────────
def infer_midi(seg):
    pt = midi_ckpt(seg['fold'])
    midi_cfg_base = OmegaConf.load(REPO_MIDI / 'configs/config.yaml')
    for key in ['data', 'model', 'train']:
        sub = OmegaConf.load(REPO_MIDI / f'configs/{key}/{key}.yaml')
        OmegaConf.update(midi_cfg_base, key, sub, merge=True)
    mcfg = midi_cfg_base

    FS         = int(mcfg.data.fs)
    WIN_FRAMES = int(mcfg.data.window_size * FS)

    gen  = PianoRollGenerator(str(seg['midi']), fs=FS)
    roll = torch.tensor(gen.generate_piano_roll(), dtype=torch.float32)  # (128, T_total)

    s_fr = int(seg['start_s'] * FS)
    e_fr = int(seg['end_s']   * FS)
    slice_roll = roll[:, s_fr:e_fr]
    if slice_roll.shape[1] < WIN_FRAMES:
        pad = WIN_FRAMES - slice_roll.shape[1]
        slice_roll = F.pad(slice_roll, (0, pad))
    else:
        slice_roll = slice_roll[:, :WIN_FRAMES]

    model_cfg = mcfg.model
    model = MidiConv2DGRU(model_cfg)
    model.load_state_dict(torch.load(pt, map_location=DEV, weights_only=True))
    model.eval().to(DEV)

    x = slice_roll.unsqueeze(0).to(DEV)   # (1, 128, 300)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=-1)[0].cpu().numpy()
    return probs, float(FS)   # (T, 5)

# ── helper: resample to common fps grid ───────────────────────────────────────
def to_common(probs, fps, target_frames):
    """Resample (T, 5) probs to target_frames via linear interp."""
    t = torch.tensor(probs).T.unsqueeze(0)        # (1, 5, T)
    out = F.interpolate(t, size=target_frames, mode='linear', align_corners=False)
    return out.squeeze(0).T.numpy()               # (target_frames, 5)

# ── colour array for imshow bar ────────────────────────────────────────────────
cmap5 = ListedColormap(CLASS_COLORS)

def to_rgba(cls_array):
    norm = cls_array / (N_CLASSES - 1)
    rgba = cmap5(norm)
    return rgba[np.newaxis, :, :]   # (1, T, 4)

# ══════════════════════════════════════════════════════════════════════════════
# Main: run all inference
# ══════════════════════════════════════════════════════════════════════════════
TARGET_FRAMES = 600   # 30s × 20fps common grid

results = []
for seg in SEGS:
    print(f'\n── {seg["title"]} ({seg["hash"]}) ──')
    r = {}

    r['voiced']  = load_pesto(seg['pesto'], seg['start_s'], seg['end_s'])
    r['gt_segs'] = get_gt_segments(seg['hash'], seg['start_s'], seg['end_s'])

    print('  Mel …', end=' ', flush=True)
    p_mel, fps_mel = infer_mel(seg)
    r['mel'] = to_common(p_mel, fps_mel, TARGET_FRAMES)
    print('done')

    print('  PESTO model …', end=' ', flush=True)
    p_pesto, fps_pesto = infer_pesto_model(seg)
    r['pesto'] = to_common(p_pesto, fps_pesto, TARGET_FRAMES)
    print('done')

    print('  CMERT …', end=' ', flush=True)
    p_cmert, fps_cmert = infer_cmert(seg)
    r['cmert'] = to_common(p_cmert, fps_cmert, TARGET_FRAMES)
    print('done')

    print('  MIDI …', end=' ', flush=True)
    p_midi, fps_midi = infer_midi(seg)
    r['midi'] = to_common(p_midi, fps_midi, TARGET_FRAMES)
    print('done')

    r['ens'] = (r['mel'] + r['pesto'] + r['cmert'] + r['midi']) / 4.0
    results.append(r)

# ══════════════════════════════════════════════════════════════════════════════
# Figure
# ══════════════════════════════════════════════════════════════════════════════
N_ROWS = 7   # F0 | GT | Mel | MIDI | PESTO | CMERT | Ens
ROW_HEIGHTS = [4.5, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55]
BAR_LABELS  = ['GT', 'Mel', 'MIDI', 'PESTO', 'CMERT', 'Ensemble']
BAR_KEYS    = ['gt',  'mel', 'midi', 'pesto', 'cmert', 'ens']

fig, axes = plt.subplots(
    N_ROWS, 2,
    figsize=(14, 7),
    gridspec_kw={'height_ratios': ROW_HEIGHTS, 'hspace': 0.08, 'wspace': 0.12},
)

for col, (seg, r) in enumerate(zip(SEGS, results)):
    voiced   = r['voiced']
    gt_segs  = r['gt_segs']
    start_s, end_s = seg['start_s'], seg['end_s']
    XLIM = (start_s, end_s)
    time_axis = start_s + np.linspace(0, 30, TARGET_FRAMES, endpoint=False)

    ens_argmax = r['ens'].argmax(axis=1)   # (TARGET_FRAMES,)

    # ── row 0: F0 contour ────────────────────────────────────────────────────
    ax = axes[0, col]
    # colour each pitch dot by ensemble argmax at that time
    pesto_times  = voiced['time_s'].values
    pesto_midis  = voiced['midi'].values
    frame_idx    = np.clip(
        ((pesto_times - start_s) / 30.0 * TARGET_FRAMES).astype(int),
        0, TARGET_FRAMES - 1
    )
    dot_cls = ens_argmax[frame_idx]

    for cls_idx in range(N_CLASSES):
        mask = dot_cls == cls_idx
        if not mask.any(): continue
        ax.scatter(pesto_times[mask], pesto_midis[mask],
                   s=4, color=CLASS_COLORS[cls_idx], alpha=0.75,
                   linewidths=0, rasterized=True, zorder=2)

    for _, row in gt_segs.iterrows():
        ax.axvline(row['start_s'], color='#444', lw=0.9, ls='--', alpha=0.5, zorder=3)

    ax.set_xlim(*XLIM)
    ax.set_ylim(48, 72)
    ax.set_yticks([48, 52, 55, 60, 64, 67, 72])
    ax.set_yticklabels(['C3','E3','G3','C4','E4','G4','C5'], fontsize=6)
    ax.grid(axis='y', color='#ddd', lw=0.5, zorder=0)
    ax.tick_params(labelbottom=False)
    ax.set_title(seg['title'], fontsize=10, pad=5)
    ax.text(0.01, 0.96, 'F0 Contour', transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='top', ha='left')

    if col == 0:
        ax.set_ylabel('MIDI note', fontsize=9)
        handles = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
                   for i in range(1, N_CLASSES)]
        ax.legend(handles=handles, loc='upper right', fontsize=7,
                  ncol=4, framealpha=0.85, handlelength=1.0,
                  columnspacing=0.6, handletextpad=0.4)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel('')

    # ── rows 1-6: GT + model prediction bars ─────────────────────────────────
    for row_i, (blabel, bkey) in enumerate(zip(BAR_LABELS, BAR_KEYS)):
        ax = axes[row_i + 1, col]

        if bkey == 'gt':
            # build frame-level GT from gt_segs
            gt_frames = np.zeros(TARGET_FRAMES, dtype=int)
            for _, grow in gt_segs.iterrows():
                fs = max(0, int((grow['start_s'] - start_s) / 30.0 * TARGET_FRAMES))
                fe = min(TARGET_FRAMES, int((grow['end_s'] - start_s) / 30.0 * TARGET_FRAMES))
                gt_frames[fs:fe] = int(grow['cls'])
            rgba = to_rgba(gt_frames)
            label_src = gt_segs
        else:
            pred_cls = r[bkey].argmax(axis=1)
            rgba = to_rgba(pred_cls)
            label_src = None

        ax.imshow(rgba, aspect='auto', extent=[*XLIM, 0, 1],
                  interpolation='nearest', origin='upper')
        ax.set_xlim(*XLIM)
        ax.set_ylim(0, 1)
        ax.set_yticks([])

        lpad = 30 if col == 0 else 30
        ax.set_ylabel(blabel, fontsize=8, rotation=0, labelpad=lpad, va='center')

        # GT text labels
        if bkey == 'gt':
            for _, grow in gt_segs.iterrows():
                cx = (grow['start_s'] + grow['end_s']) / 2
                ax.text(cx, 0.5, KOR_TO_ENG.get(grow['label'], grow['label']),
                        ha='center', va='center', fontsize=8,
                        color='white', fontweight='bold',
                        transform=ax.get_xaxis_transform())
            for _, grow in gt_segs.iterrows():
                ax.axvline(grow['start_s'], color='white', lw=0.8, alpha=0.6)
        else:
            for _, grow in gt_segs.iterrows():
                ax.axvline(grow['start_s'], color='white', lw=0.8, alpha=0.5, ls='--')

        is_last_row = (row_i == len(BAR_LABELS) - 1)
        if is_last_row:
            ax.set_xlabel('Time (s)', fontsize=9)
            ax.set_xticks(np.arange(start_s, end_s + 1, 10))
        else:
            ax.tick_params(labelbottom=False)

# ── save ──────────────────────────────────────────────────────────────────────
out_png = REPO_MODE / 'pattern_figure.png'
out_pdf = REPO_MODE / 'pattern_figure.pdf'
fig.savefig(out_png, dpi=150, bbox_inches='tight')
fig.savefig(out_pdf, bbox_inches='tight')
print(f'\nSaved → {out_png}')
print(f'Saved → {out_pdf}')
