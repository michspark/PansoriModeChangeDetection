"""
Example figure: 04-이일주-심청가_곽씨_죽음 (5be7b760), 60-90s
Three panels:
  1. PESTO pitch contour (vocal melody)
  2. Ground truth annotation bar
  3. Model softmax prediction
"""

import sys

import math
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torchaudio.transforms import Spectrogram, MelScale, AmplitudeToDB
from omegaconf import OmegaConf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib.colors import ListedColormap
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import DATA_ROOT, REPO_ROOT
import models

# ── paths ──────────────────────────────────────────────────────────────────────
AUDIO_PATH  = str(DATA_ROOT / 'Audio_Original/5be7b760-04-이일주-심청가_곽씨_죽음.wav')
PESTO_PATH  = str(DATA_ROOT / 'Audio_pesto_output/5be7b760-04-이일주-심청가_곽씨_죽음_vocal.f0.csv')
LABEL_CSV   = str(REPO_ROOT / 'data/Label/label.csv')
CKPT_PATH   = str(REPO_ROOT / 'weights/frame/Mel_Original_Version/fold1_best_model.pt')
CONFIG_PATH = str(REPO_ROOT / 'weights/frame/Mel_Original_Version/config.yaml')
OUT_PNG     = str(REPO_ROOT / 'example_figure.png')
OUT_PDF     = str(REPO_ROOT / 'example_figure.pdf')

# ── segment ────────────────────────────────────────────────────────────────────
SEG_START_S = 60.0
SEG_END_S   = 90.0

# ── style ──────────────────────────────────────────────────────────────────────
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
_fp = fm.FontProperties(fname=FONT_PATH)
plt.rcParams['font.family'] = _fp.get_name()
plt.rcParams['axes.unicode_minus'] = False

CLASS_NAMES  = ['Unknown', 'Ujo', 'Gmj', 'Anr', 'CJo']
CLASS_COLORS = ['#cccccc', '#4C72B0', '#C44E52', '#999999', '#DD8452']
LABEL_MAP    = {
    'Unknown': 0, '경드름': 1, '설렁제': 1, '평조': 1, '우조': 1,
    '계면조': 2, '아니리': 3, '창조': 4,
}
KOR_TO_ENG = {'Unknown': 'Unknown', '우조': 'Ujo', '경드름': 'Ujo', '설렁제': 'Ujo',
              '평조': 'Ujo', '계면조': 'Gmj', '아니리': 'Anr', '창조': 'CJo'}
CONF_THRESH = 0.5

# ── 1. PESTO pitch for 60-90s ──────────────────────────────────────────────────
pesto_df = pd.read_csv(PESTO_PATH)
seg_ms = pesto_df[
    (pesto_df['time'] >= SEG_START_S * 1000) &
    (pesto_df['time'] <  SEG_END_S   * 1000)
].copy()
seg_ms['time_s'] = seg_ms['time'] / 1000.0

def hz_to_midi(f):
    return 69 + 12 * math.log2(max(f, 1e-6) / 440.0)

seg_ms['midi'] = seg_ms['frequency'].apply(hz_to_midi)
voiced = seg_ms[seg_ms['confidence'] >= CONF_THRESH].copy()

# ── 2. GT annotations ──────────────────────────────────────────────────────────
df_label = pd.read_csv(LABEL_CSV)
song_df = df_label[df_label['hash_key'] == '5be7b760'].copy()
song_df['start_s'] = song_df['start'] / 1000.0
song_df['end_s']   = song_df['end']   / 1000.0
song_df['class']   = song_df['label'].map(LABEL_MAP)

gt_seg = song_df[
    (song_df['end_s'] > SEG_START_S) & (song_df['start_s'] < SEG_END_S)
].copy()
gt_seg = gt_seg.copy()
gt_seg['start_s'] = gt_seg['start_s'].clip(lower=SEG_START_S)
gt_seg['end_s']   = gt_seg['end_s'].clip(upper=SEG_END_S)

# ── 3. audio → mel → model inference ──────────────────────────────────────────
cfg = OmegaConf.load(CONFIG_PATH)
SR         = cfg.dataset.params.sr
HOP        = cfg.dataset.params.hop_length
N_FFT      = cfg.dataset.params.n_fft
N_MELS     = cfg.dataset.params.target_bins
N_CLASSES  = cfg.dataset.params.num_classes

audio_full, orig_sr = sf.read(AUDIO_PATH, dtype='float32', always_2d=True)
audio_full = torch.from_numpy(audio_full.T)
if orig_sr != SR:
    audio_full = torchaudio.functional.resample(audio_full, orig_sr, SR)
audio_full = audio_full.mean(dim=0, keepdim=True)

start_sample = int(SEG_START_S * SR)
end_sample   = int(SEG_END_S   * SR)
audio_slice  = audio_full[:, start_sample:end_sample]

spec_cvt  = Spectrogram(n_fft=N_FFT, hop_length=HOP, power=1.0)
spec2mel  = MelScale(n_stft=N_FFT//2+1, n_mels=N_MELS, sample_rate=SR, f_min=80, f_max=2000)
db_cvt    = AmplitudeToDB()

spec = spec_cvt(audio_slice)
mel  = spec2mel(spec).squeeze(0)
mel  = db_cvt(mel) / 100

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_cfg = cfg.model.params
model = models.Conv2DGRU(model_cfg)
state  = torch.load(CKPT_PATH, map_location=device, weights_only=True)
model.load_state_dict(state)
model.eval().to(device)

with torch.no_grad():
    x = mel.unsqueeze(0).to(device)
    probs = torch.softmax(model(x), dim=-1)
    pred  = probs.argmax(dim=-1)

probs_np = probs[0].cpu().numpy()   # (T, 5)
pred_np  = pred[0].cpu().numpy()    # (T,)
T_frames = probs_np.shape[0]
print(f'Frames: {T_frames}')

frame_times = SEG_START_S + np.arange(T_frames) * HOP / SR

# ── 4. frame-level GT array ────────────────────────────────────────────────────
gt_frames = np.zeros(T_frames, dtype=int)
for _, row in gt_seg.iterrows():
    f_s = max(0, int((row['start_s'] - SEG_START_S) * SR / HOP))
    f_e = min(T_frames, int((row['end_s'] - SEG_START_S) * SR / HOP))
    gt_frames[f_s:f_e] = int(row['class'])

# ── 5. colour arrays for annotation bars (imshow) ─────────────────────────────
cmap5 = ListedColormap(CLASS_COLORS)

def frames_to_rgba(class_array, cmap, n_classes=5):
    """Convert 1D class index array to (1, T, 4) RGBA for imshow."""
    norm = class_array / (n_classes - 1)   # normalise to [0,1]
    rgba = cmap(norm)                       # (T, 4)
    return rgba[np.newaxis, :, :]           # (1, T, 4)

gt_rgba   = frames_to_rgba(gt_frames, cmap5)
pred_rgba = frames_to_rgba(pred_np,   cmap5)

# ── 6. figure ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 5))
gs  = fig.add_gridspec(3, 1, height_ratios=[4.5, 0.6, 0.6], hspace=0.10)

ax_pitch = fig.add_subplot(gs[0])
ax_gt    = fig.add_subplot(gs[1])
ax_pred  = fig.add_subplot(gs[2])

XLIM = (SEG_START_S, SEG_END_S)

# ─── panel 1: pitch contour ───────────────────────────────────────────────────
gt_at_pesto = np.zeros(len(voiced), dtype=int)
for _, row in gt_seg.iterrows():
    mask = (voiced['time_s'].values >= row['start_s']) & \
           (voiced['time_s'].values <  row['end_s'])
    gt_at_pesto[mask] = int(row['class'])

for cls_idx in range(N_CLASSES):
    mask = gt_at_pesto == cls_idx
    if not mask.any():
        continue
    ax_pitch.scatter(
        voiced['time_s'].values[mask],
        voiced['midi'].values[mask],
        s=4, alpha=0.75,
        color=CLASS_COLORS[cls_idx],
        label=CLASS_NAMES[cls_idx],
        linewidths=0,
        rasterized=True,
        zorder=2,
    )

# GT boundary vertical lines
for _, row in gt_seg.iterrows():
    ax_pitch.axvline(row['start_s'], color='#444444', lw=0.9, ls='--', alpha=0.55, zorder=3)

ax_pitch.set_xlim(*XLIM)
ax_pitch.set_ylabel('MIDI note', fontsize=20)
ax_pitch.set_ylim(48, 72)
yticks_midi = [48, 52, 55, 60, 64, 67, 72]
ytick_names = ['C3','E3','G3','C4','E4','G4','C5']
ax_pitch.set_yticks(yticks_midi)
ax_pitch.set_yticklabels(ytick_names, fontsize=14)
ax_pitch.grid(axis='y', color='#dddddd', lw=0.5, zorder=0)
ax_pitch.tick_params(labelbottom=False)

handles = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
           for i in range(1, N_CLASSES)]
ax_pitch.legend(handles=handles, loc='upper right', fontsize=16,
                framealpha=0.85, ncol=4, handlelength=1.2,
                columnspacing=0.8, handletextpad=0.5)

# ─── panel 2: GT bar ──────────────────────────────────────────────────────────
ax_gt.imshow(gt_rgba, aspect='auto', extent=[*XLIM, 0, 1],
             interpolation='nearest', origin='upper')
ax_gt.set_xlim(*XLIM)
ax_gt.set_ylim(0, 1)
ax_gt.set_yticks([])
ax_gt.set_ylabel('GT', fontsize=18, rotation=0, labelpad=24, va='center')
ax_gt.tick_params(labelbottom=False)
# mode label text
for _, row in gt_seg.iterrows():
    cx = (row['start_s'] + row['end_s']) / 2
    ax_gt.text(cx, 0.5, KOR_TO_ENG.get(row['label'], row['label']),
               ha='center', va='center',
               fontsize=17, color='white', fontweight='bold',
               transform=ax_gt.get_xaxis_transform())
for _, row in gt_seg.iterrows():
    ax_gt.axvline(row['start_s'], color='white', lw=0.8, alpha=0.6)

# ─── panel 3: prediction bar ───────────────────────────────────────────────────
ax_pred.imshow(pred_rgba, aspect='auto', extent=[*XLIM, 0, 1],
               interpolation='nearest', origin='upper')
ax_pred.set_xlim(*XLIM)
ax_pred.set_ylim(0, 1)
ax_pred.set_yticks([])
ax_pred.set_ylabel('Pred', fontsize=18, rotation=0, labelpad=24, va='center')
for _, row in gt_seg.iterrows():
    ax_pred.axvline(row['start_s'], color='white', lw=0.8, alpha=0.6)
ax_pred.set_xlabel('Time (s)', fontsize=20)
ax_pred.set_xticks(np.arange(60, 91, 10))
ax_pred.set_xticklabels([0, 10, 20, 30])
ax_pred.tick_params(axis='x', labelsize=16)

# ── save ──────────────────────────────────────────────────────────────────────
fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
fig.savefig(OUT_PDF, bbox_inches='tight')
print(f'Saved → {OUT_PNG}')
print(f'Saved → {OUT_PDF}')
