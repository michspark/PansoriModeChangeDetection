"""Soft-voting ensemble inference across representations.

This is the scriptable form of what viewer/app.py's run_ensemble() does: given
one song's audio / f0 contour / MIDI, run each modality, align every stream onto
the slowest frame rate, and average the probabilities.

Each modality reads a different file, so inputs are given per modality rather
than as one path:

    python infer/infer_ensemble.py \\
        --mel   song.wav \\
        --pesto song.f0.csv \\
        --midi  song.mid

Any subset works — pass only what you have, and the weights renormalise over the
modalities actually present:

    python infer/infer_ensemble.py --pesto song.f0.csv --midi song.mid

Alignment target is the slowest modality present (MIDI at 10 fps if included,
otherwise PESTO/CREPE at 20 fps, otherwise the audio rate). Everything is pooled
onto that rate before voting, which is why the per-modality models are all
trained on exact 30 s windows.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (CLASSES, FALLBACK_CONFIG, align_to_fps, default_device,
                     clean_stem, feature_kwargs, fps_for, load_run, sliding_inference,
                     window_frames_for,
                     summarise, write_predictions)
from features import EXTRACTORS

#: modality -> default checkpoint directory. Feature settings and frame rates
#: come from each checkpoint's own config via _common.feature_kwargs/fps_for.
DEFAULT_RUN_DIR = {
    'mel':    'weights/frame/Mel_Original_Version',
    'pesto':  'weights/frame/Pesto_Version',
    'midi':   'weights/midi/MIDI_Version/23-43-24',
    'cmert':  'weights/cmert/layer08_10k_version/layer08',
    'cqt':    'weights/frame/CQT_Version',
    'chroma': 'weights/frame/Chroma_Version',
    'crepe':  'weights/frame/Crepe_Version',
}


def main():
    import argparse
    p = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    for rep in DEFAULT_RUN_DIR:
        p.add_argument(f'--{rep}', metavar='FILE', default=None,
                       help=f'input file for the {rep} modality')
        p.add_argument(f'--{rep}_dir', metavar='DIR', default=DEFAULT_RUN_DIR[rep],
                       help=f'{rep} run directory')
    p.add_argument('--weights', nargs='+', type=float, default=None,
                   help='one weight per active modality, in the order listed by '
                        '--help (default: equal weights). Renormalised to sum to 1.')
    p.add_argument('--out', default='outputs/infer')
    p.add_argument('--name', default=None, help='output stem (default: from the first input)')
    p.add_argument('--fold', type=int, default=None)
    p.add_argument('--device', default=default_device())
    p.add_argument('--quiet', action='store_true')
    p.add_argument('--keep_individual', action='store_true',
                   help='also write each modality\'s own per-frame CSV')
    args = p.parse_args()

    active = [r for r in DEFAULT_RUN_DIR if getattr(args, r) is not None]
    if not active:
        raise SystemExit("No modality given. Pass at least one, e.g. --pesto song.f0.csv")

    if args.weights:
        if len(args.weights) != len(active):
            raise SystemExit(f"--weights needs {len(active)} values for {active}, "
                             f"got {len(args.weights)}")
        w = np.array(args.weights, dtype=np.float32)
    else:
        w = np.ones(len(active), dtype=np.float32)
    w = w / w.sum()

    stem = args.name or clean_stem(getattr(args, active[0]))
    out_dir = Path(args.out)
    print(f"  modalities : {', '.join(f'{r}({wi:.3f})' for r, wi in zip(active, w))}")

    # ── run each modality at its native rate ─────────────────────────────────
    per_rep = {}
    for rep in active:
        path = Path(getattr(args, rep))
        run_dir = getattr(args, f'{rep}_dir')
        extractor, suffixes = EXTRACTORS[rep]
        if path.suffix.lower() not in suffixes:
            raise SystemExit(f"--{rep} expects {suffixes}, got {path.name}")
        model, cfg, ckpt = load_run(
            run_dir, args.device, args.fold,
            fallback_config=Path(__file__).resolve().parents[1] / FALLBACK_CONFIG[rep])
        feat = extractor(path, **feature_kwargs(rep, cfg.dataset.params))
        fps = fps_for(rep, cfg.dataset.params)
        win = window_frames_for(rep, cfg.dataset.params)
        probs = sliding_inference(model, feat, win, args.device,
                                  sample_domain=(rep == 'cmert'))
        if rep == 'cmert':
            probs = probs[:int(feat.shape[1] / 320)]
        per_rep[rep] = (probs, fps)
        print(f"  {rep:7s} {path.name:36s} -> {probs.shape[0]:6d} frames @ {fps:.4g} fps")
        if args.keep_individual:
            write_predictions(probs, fps, out_dir / f'{stem}_{rep}.csv', path)
        del model

    # ── align every stream onto the slowest present, then vote ───────────────
    target = min(active, key=lambda r: per_rep[r][1])
    dst_fps, dst_len = per_rep[target][1], per_rep[target][0].shape[0]
    print(f"  aligning to {target} @ {dst_fps:.4g} fps ({dst_len} frames)")

    ens = np.zeros((dst_len, len(CLASSES)), dtype=np.float32)
    for rep, wi in zip(active, w):
        pr, src_fps = per_rep[rep]
        a = pr if rep == target else align_to_fps(pr, src_fps, dst_len, dst_fps)
        ens += wi * a

    out_csv = out_dir / f'{stem}_ensemble.csv'
    df = write_predictions(ens, dst_fps, out_csv, stem)
    print(f"\n  ensemble -> {out_csv}")
    if not args.quiet:
        summarise(df, dst_fps, 'ensemble')


if __name__ == '__main__':
    main()
