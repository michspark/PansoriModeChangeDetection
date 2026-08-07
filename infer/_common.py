"""Shared machinery for the per-representation inference scripts.

Each `infer_<rep>.py` is a thin wrapper: it names its representation and its
default checkpoint directory, and everything else lives here.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import models                                            # noqa: E402
from paths import REPO_ROOT                              # noqa: E402

from features import EXTRACTORS                          # noqa: E402

CLASSES = ['Unknown', '우조', '계면조', '아니리', '창조']

#: Nominal output frame rate per representation. Used only as a fallback —
#: fps_for() below derives the real value from the checkpoint's own config,
#: because a run trained with a different hop_length or frame_rate would
#: otherwise get correct predictions with wrong timestamps.
FPS = {
    'mel':    16000 / 512,      # 31.25
    'cqt':    16000 / 512,
    'chroma': 16000 / 512,
    'pesto':  20.0,
    'crepe':  20.0,
    'midi':   10.0,
    'cmert':  24000 / 320,      # 75.0
}

#: Dataset-class defaults, applied when a saved config omits a parameter.
#: Must stay in sync with the signatures in datasets.py.
_DEFAULTS = {
    'mel':    dict(sr=16000, n_fft=2048, hop_length=512, target_bins=40),
    'cqt':    dict(sr=16000, hop_length=512, n_bins=84, bins_per_octave=12),
    'chroma': dict(sr=16000, n_fft=2048, hop_length=512, n_chroma=12),
    'pesto':  dict(sr=100, frame_rate=20, threshold=0.8),
    'crepe':  dict(sr=100, frame_rate=20, threshold=0.8),
    'midi':   dict(fs=10),
    'cmert':  dict(sr=24000),
}


def feature_kwargs(rep, params):
    """Feature settings for `rep`, taken from the checkpoint's config where
    present and from the dataset-class default otherwise."""
    return {k: params.get(k, d) for k, d in _DEFAULTS[rep].items()}


def window_frames_for(rep, params):
    """Sliding-window length, mirroring datasets.py exactly.

    The audio datasets use integer floor division (`window * sr // hop_length`),
    NOT rounding: at 16 kHz / hop 512 a 30 s window is 937 frames, and
    round(937.5) would give 938. One frame of drift shifts every subsequent
    window and quietly changes the predictions, so this must match the training
    code character for character.

    CMERT is the exception: its model input is the raw waveform, so the window is
    measured in samples.
    """
    kw = feature_kwargs(rep, params)
    window = params.window
    if rep in ('mel', 'cqt', 'chroma'):
        return window * kw['sr'] // kw['hop_length']
    if rep in ('pesto', 'crepe'):
        return window * kw['frame_rate']
    if rep == 'midi':
        return window * kw['fs']
    if rep == 'cmert':
        return window * kw['sr']                  # samples, not frames
    raise KeyError(rep)


def fps_for(rep, params):
    """Model output frame rate implied by the checkpoint's own config."""
    kw = feature_kwargs(rep, params)
    if rep in ('mel', 'cqt', 'chroma'):
        return kw['sr'] / kw['hop_length']
    if rep in ('pesto', 'crepe'):
        return float(kw['frame_rate'])
    if rep == 'midi':
        return float(kw['fs'])
    if rep == 'cmert':
        return kw['sr'] / 320          # MERT CNN hop is fixed at 320 samples
    return FPS[rep]


def default_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def load_run(run_dir, device, fold=None, fallback_config=None, config=None, rep=None):
    """Load config.yaml + fold{N}_best_model.pt from a training run directory.

    Reading the config that sits next to the weights is what keeps inference
    honest: the feature settings and the architecture both come from the run
    that produced the checkpoint, never from defaults in this file. This matters
    more than it looks — e.g. every PESTO run here was trained with
    `threshold: 0.0`, not the PitchDataset default of 0.8, so hardcoding the
    class default would silently feed the model a differently-normalised
    contour.

    Checkpoints inherited from the old PansoriMIDIDetection repo have no
    config.yaml next to them; `fallback_config` (this repo's config for that
    representation) covers those. `config` overrides everything.
    """
    run_dir = Path(run_dir)
    cfg_path = Path(config) if config else run_dir / 'config.yaml'
    if not cfg_path.exists():
        if fallback_config and Path(fallback_config).exists():
            cfg_path = Path(fallback_config)
            print(f"  note    : no config.yaml in {run_dir}, falling back to {cfg_path}")
        else:
            raise SystemExit(
                f"\nNo config.yaml in {run_dir} and no fallback. Point --run_dir at a "
                f"training run directory, or pass --config explicitly.\n")
    cfg = OmegaConf.load(cfg_path)

    ckpts = sorted(run_dir.glob('fold*_best_model.pt'))
    if fold is not None:
        ckpts = [p for p in ckpts if p.name == f'fold{fold}_best_model.pt']
    if not ckpts:
        nested = sorted(run_dir.glob('*/fold*_best_model.pt'))
        hint = ""
        if nested:
            # A common mistake: pointing at the experiment folder rather than one
            # of the run folders inside it.
            hint = ("\n  It looks like a parent of several run directories. Pick one:\n"
                    + "\n".join(f"    {d}" for d in sorted({q.parent for q in nested})))
        elif not run_dir.exists():
            cfg_name = Path(FALLBACK_CONFIG[rep]).stem if rep in FALLBACK_CONFIG else '<config>'
            hint = (f"\n  {run_dir} does not exist. No {rep or 'such'} checkpoint ships with "
                    f"this repo — train one first:\n"
                    f"    python train.py --config-name {cfg_name}\n"
                    f"  then pass its run directory with --run_dir.")
        raise SystemExit(
            f"\nNo fold*_best_model.pt in {run_dir}"
            + (f" for fold {fold}" if fold else "") + hint + "\n")
    ckpt = ckpts[0]

    model = getattr(models, cfg.model.name)(cfg.model.params).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()})
    model.eval()
    return model, cfg, ckpt


@torch.no_grad()
def sliding_inference(model, feat, window_frames, device, sample_domain=False):
    """(n_bins, T) feature -> (T, n_classes) softmax probabilities.

    Windows are exact and non-overlapping, matching how the models were trained
    and evaluated. The tail window is zero-padded and then trimmed back, so the
    output length always equals the input length.
    """
    n_bins, T = feat.shape
    out = []
    for start in range(0, T, window_frames):
        end = min(start + window_frames, T)
        chunk = feat[:, start:end]
        if chunk.shape[1] < window_frames:
            chunk = torch.cat(
                [chunk, torch.zeros(n_bins, window_frames - chunk.shape[1])], dim=1)
        # Spectrogram-like features are (n_bins, T) and the model wants
        # (batch, n_bins, T), so add a batch dim. CMERT's input is a raw waveform:
        # CMERTFrameDataset yields `audio_slice.squeeze(0)`, i.e. (samples,), so a
        # batch is (batch, samples) — and `chunk` is already exactly that shape.
        x = chunk.to(device) if sample_domain else chunk.unsqueeze(0).to(device)
        probs = torch.softmax(model(x), dim=-1)[0].cpu().numpy()
        # In the sample domain the model emits far fewer frames than the window
        # has samples, so `end - start` is not a frame count — keep the whole
        # window's output and let the caller trim the total.
        out.append(probs if sample_domain else probs[:end - start])
    return np.concatenate(out, axis=0)


def align_to_fps(probs, src_fps, dst_len, dst_fps):
    """Resample (T_src, C) probabilities onto `dst_len` frames at `dst_fps` by
    frame-averaged pooling. Used by the ensemble to bring every stream onto the
    slowest modality's rate before voting."""
    T_src, C = probs.shape
    ratio = src_fps / dst_fps
    out = np.zeros((dst_len, C), dtype=np.float32)
    for j in range(dst_len):
        a = max(0, min(int(round(j * ratio)), T_src - 1))
        b = max(a + 1, min(int(round((j + 1) * ratio)), T_src))
        out[j] = probs[a:b].mean(axis=0)
    return out


def write_predictions(probs, fps, out_csv, source):
    """One row per frame: time, predicted class, and every class probability.

    The schema is identical across representations so infer_ensemble.py can read
    any of them, and so two representations can be diffed directly.
    """
    import pandas as pd
    pred = probs.argmax(axis=1)
    df = pd.DataFrame({
        'frame': np.arange(len(pred)),
        'time_sec': np.round(np.arange(len(pred)) / fps, 4),
        'pred_id': pred,
        'pred_label': [CLASSES[i] for i in pred],
    })
    for i, c in enumerate(CLASSES):
        df[f'prob_{c}'] = np.round(probs[:, i], 6)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def summarise(df, fps, name):
    """Collapse the frame sequence into contiguous mode segments — the thing the
    model is actually for, since mode *changes* are the boundaries between runs
    of a constant label."""
    seg, cur, start = [], None, 0
    ids = df['pred_id'].tolist()
    for i, p in enumerate(ids + [None]):
        if p != cur:
            if cur is not None:
                seg.append((start / fps, i / fps, CLASSES[cur]))
            cur, start = p, i
    print(f"\n  {name}: {len(df)} frames, {len(seg)} mode segments")
    for s, e, lab in seg[:12]:
        print(f"    {s:7.2f}s - {e:7.2f}s   {lab}")
    if len(seg) > 12:
        print(f"    ... (+{len(seg) - 12} more)")
    return seg


def clean_stem(path):
    """`song.f0.csv` -> `song`. PESTO/CREPE outputs carry a double extension, so
    Path.stem alone would leave a stray `.f0` in every output filename."""
    s = Path(path).stem
    return s[:-3] if s.endswith('.f0') else s


def collect_inputs(paths, suffixes):
    """Accept files or directories; recurse into directories."""
    out = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            out += [q for q in sorted(p.rglob('*')) if q.suffix.lower() in suffixes]
        elif p.suffix.lower() in suffixes:
            out.append(p)
        else:
            print(f"  skipping {p} (expected one of {suffixes})")
    if not out:
        raise SystemExit(f"No input files with suffix {suffixes} found.")
    return out


#: Repo config used when a checkpoint directory has no config.yaml of its own
#: (checkpoints inherited from the standalone MIDI repo, mainly).
FALLBACK_CONFIG = {
    'mel':    'configs/frame/mel_base.yaml',
    'cqt':    'configs/frame/cqt_base.yaml',
    'chroma': 'configs/frame/chroma_base.yaml',
    'pesto':  'configs/frame/pesto_best.yaml',
    'crepe':  'configs/frame/crepe_best.yaml',
    'midi':   'configs/frame/midi.yaml',
    'cmert':  'configs/frame/cmert.yaml',
}


def build_parser(rep, default_run_dir, extra_help=''):
    p = argparse.ArgumentParser(
        description=f"Frame-level pansori mode inference from the '{rep}' representation. {extra_help}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('input', nargs='+',
                   help=f'input file(s) or directory; expects {EXTRACTORS[rep][1]}')
    p.add_argument('--run_dir', default=str(default_run_dir),
                   help='training run directory (config.yaml + fold*_best_model.pt)')
    p.add_argument('--fold', type=int, default=None,
                   help='which fold checkpoint to use (default: lowest available)')
    p.add_argument('--out', default='outputs/infer',
                   help='directory for the per-frame prediction CSVs')
    p.add_argument('--device', default=default_device())
    p.add_argument('--config', default=None,
                   help='config.yaml to use instead of the one in --run_dir')
    p.add_argument('--quiet', action='store_true', help='suppress the segment summary')
    return p


def run_single_representation(rep, args, _unused=None):
    """Shared main() body for the single-representation scripts."""
    extractor, suffixes = EXTRACTORS[rep]
    model, cfg, ckpt = load_run(args.run_dir, args.device, args.fold,
                                fallback_config=REPO_ROOT / FALLBACK_CONFIG[rep],
                                config=args.config, rep=rep)
    print(f"  model   : {cfg.model.name}  ({ckpt.relative_to(REPO_ROOT) if ckpt.is_relative_to(REPO_ROOT) else ckpt})")
    print(f"  dataset : {cfg.dataset.name}")

    kw = feature_kwargs(rep, cfg.dataset.params)
    fps = fps_for(rep, cfg.dataset.params)
    window_frames = window_frames_for(rep, cfg.dataset.params)
    unit = 'samples' if rep == 'cmert' else 'frames'
    print(f"  window  : {cfg.dataset.params.window}s = {window_frames} {unit} @ {fps:.4g} fps")
    print(f"  features: {kw}")

    out_dir = Path(args.out)
    for path in collect_inputs(args.input, suffixes):
        feat = extractor(path, **kw)
        probs = sliding_inference(model, feat, window_frames, args.device,
                                  sample_domain=(rep == 'cmert'))
        if rep == 'cmert':
            probs = probs[:int(feat.shape[1] / 320)]
        out_csv = out_dir / f'{clean_stem(path)}_{rep}.csv'
        df = write_predictions(probs, fps, out_csv, path)
        print(f"\n  {path.name} -> {out_csv}")
        if not args.quiet:
            summarise(df, fps, rep)
