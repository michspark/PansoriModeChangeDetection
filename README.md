# Frame-Level Pansori Mode Classification with Complementary Audio Representations

**Sangheon Park, Seonguk Ju, Suin Chung, Danbinaerin Han, Dasaem Jeong**

The official implementation of the paper *"Frame-Level Pansori Mode Classification with Complementary Audio Representations"*.

[**Paper**](#) &nbsp;|&nbsp;
[**Demo**](#) &nbsp;|&nbsp;
[**Data & Weights**](https://github.com/michspark/Pansori-Mode-Classification/releases)
<!-- TODO(camera-ready): paper / arXiv / demo links -->

[![Code License: MIT](https://img.shields.io/badge/code%20license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)](requirements.txt)
<!-- TODO(camera-ready): arXiv badge -->

The model labels every frame of a pansori recording with the mode (조) being
sung, so mode-change boundaries fall out of the frame sequence rather than being
predicted directly:

* **One pipeline, seven input representations** — waveform-derived (mel, CQT,
  chroma), pitch-contour (PESTO, CREPE), symbolic (transcribed-MIDI piano roll),
  and self-supervised (CultureMERT-95M). Each is a single Hydra config over the
  same `datasets.py` / `trainers.py` / `models/` stack, so representations are
  compared under identical training, splits, and metrics.
* **Cross-modality soft voting** — Streams run at different frame rates (mel
  31.25 fps, PESTO 20 fps, MIDI 10 fps). The ensemble aligns each onto the
  slowest by frame-averaged pooling before a weighted vote, which is why every
  modality uses exact 30 s windows rather than content-aware boundaries.

| Config | Dataset | Model | Input |
|---|---|---|---|
| `mel_base` | `MelFrameDataset` | `Conv2DGRU` | 40-bin mel spectrogram, 31.25 fps |
| `cqt_base` | `CQTFrameDataset` | `Conv2DGRU` | CQT (nnAudio) |
| `chroma_base` | `ChromaFrameDataset` | `Conv2DGRU` | chroma |
| `pesto_best` | `PitchFrameDataset` | `Conv1DGRU` | PESTO f0 contour, 20 fps |
| `crepe_best` | `PitchFrameDataset` | `Conv1DGRU` | CREPE f0 contour |
| `midi` | `MidiFrameDataset` | `Conv2DGRU` | transcribed-MIDI piano roll, 128 bins @ 10 fps |
| `cmert` | `CMERTFrameDataset` | `CMERTClassifier` | CultureMERT-95M features, 75 fps |

`configs/segment/` holds segment-level variants (one label per 30 s window)
trained with `SegmentTrainer`.

**Classes.** Five frame labels; 경드름 / 설렁제 / 평조 are folded into 우조, and
`Unknown` is ignored by the loss (`ignore_index: 0`).

| id | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| | Unknown | 우조 (ujo) | 계면조 (gyemyeonjo) | 아니리 (aniri) | 창조 (changjo) |

The bundled annotations (`data/Label/`) cover **395 recordings / 3,667 labeled
segments**.

---

## Install

```bash
git clone https://github.com/michspark/Pansori-Mode-Classification.git
cd Pansori-Mode-Classification
python3 -m venv .venv && source .venv/bin/activate

# torch is pinned to a CUDA 11.8 build, so the torch index is required --
# a bare `pip install -r requirements.txt` will NOT resolve it
pip install -r requirements.txt \
    --index-url https://download.pytorch.org/whl/cu118 \
    --extra-index-url https://pypi.org/simple
```

CPU-only: drop the `+cu118` suffixes from the `torch` / `torchaudio` /
`torchvision` pins and install from PyPI normally.

Tested with Python 3.10 and 3.13, PyTorch 2.5.0 (CUDA 11.8) on an NVIDIA RTX
4090. `--config-name cmert` downloads `ntua-slp/CultureMERT-95M` from the Hugging
Face Hub on first run.

---

## Download data & weights

**The audio corpus is not distributed with this repo** — only the annotations
(`data/Label/label.{csv,json}`). Point `PANSORI_DATA_ROOT` at your copy:

```bash
export PANSORI_DATA_ROOT=/path/to/Pansori_Data
```

It defaults to `../Pansori_Data`, a sibling of this checkout.

<!-- TODO(camera-ready): attach these as GitHub Release assets and fill in sizes -->

| Asset | Contents | Size |
|---|---|---|
| `checkpoints.tar.gz` | pretrained models per modality and split | TODO |
| `pansori_contours.tar.gz` | PESTO / CREPE f0 CSVs, transcribed MIDI | TODO |
| `pansori_splits.tar.gz` | version / song-stratified / random fold index files | TODO |

Expected corpus layout:

```
Pansori_Data/
├── Audio/                  wav, source-separated vocal, 16 kHz   (mel)
├── Audio_Original/         wav, unseparated                      (cqt, chroma, cmert)
├── Audio_pesto_output/     PESTO f0 CSVs, one per recording      (pesto)
├── crepe/                  CREPE f0 CSVs, one per recording      (crepe)
├── rosvot_midi/            transcribed .mid, one per recording   (midi)
├── pansori_version_split/  train.txt / val.txt / test.txt        (selection: Version)
├── song_stratified/        ch_1.txt, ch_2.txt, hb_1.txt, ...     (selection: SongStratified)
└── pansori_random_fold/    fold_01.txt .. fold_10.txt            (selection: SharedFold)
```

**Filename convention matters.** Every audio, contour, and MIDI file must begin
with the `hash_key` from `data/Label/label.csv`, followed by `-`:

```
608e81d1-01-김일구-적벽가_군사들이_싸움타령_하는데.f0.csv
^^^^^^^^ hash_key
```

Every loader keys off `filename.split("-")[0]`, so a file whose name does not
start with a known hash is **silently skipped** — a whole directory named the
wrong way loads as zero rows rather than raising.
`preproc/add_hash_to_filenames.py` assigns these prefixes;
`preproc/downsample.py` resamples to 16 kHz and `preproc/make_pitch_shift.py`
builds the augmentation corpus.

Checkpoint ↔ split mapping. Every checkpoint is named
`fold{N}_best_model.pt` and sits next to the `config.yaml` it was trained with,
so a directory is self-describing:

| Checkpoint dir | Representation | Folds |
|---|---|---|
| `weights/frame/Mel_Original_Version` | mel (unseparated) | 1 (version split) |
| `weights/frame/Mel_Original_Song_Stratified` | mel (unseparated) | 10 |
| `weights/frame/Mel_Sep_Version`, `Mel_Sep_Stratified` | mel (source-separated) | 1 / 10 |
| `weights/frame/Pesto_Version`, `Pesto_Song_Stratified` | PESTO f0 | 1 / 10 |
| `weights/midi/MIDI_Version`, `MIDI_Song_Stratified` | transcribed MIDI | 1 / 10 |
| `weights/cmert/layer08_song_stratified` | CultureMERT (layer 8) | 10 |
| `weights/cmert/layer08_10k_version` | CultureMERT (layer 8) | 1 (version split) |

All ten folds are kept for every song-stratified representation:
`pooled_eval_song_stratified.py`, `run_vt_accuracy.py` and the ensemble scripts
each load fold *k*'s model to score fold *k*'s test set, so a partial set would
silently change the reported numbers.

Periodic mid-training snapshots (`*_iter.pt`, `step*.pt`) and posteriorgram PNGs
are not distributed — they are regenerable from the checkpoints. The
`fold{N}_posteriorgrams/test_results.csv` files are kept, because
`pooled_eval_song_stratified.py` reads the test-song membership from them.

---

## Reproducing the paper

### 1 · Train

Run from the repo root; `hydra.run.dir` is `.`, so relative paths resolve.

```bash
python train.py --config-name mel_base        # mel
python train.py --config-name cqt_base        # CQT
python train.py --config-name chroma_base     # chroma
python train.py --config-name pesto_best      # PESTO pitch
python train.py --config-name crepe_best      # CREPE pitch
python train.py --config-name midi            # transcribed MIDI piano roll
python train.py --config-name cmert           # CultureMERT

# segment-level instead of frame-level
python train.py --config-path configs/segment --config-name mel_base
```

- CLI dotted overrides take precedence over the YAML files; see `configs/` for
  the full set of knobs, e.g.
  `python train.py --config-name mel_base train.num_iterations=5000 train.target_folds=[1,2]`.
- Checkpoints and the resolved `config.yaml` land in
  `weights/frame/<MMDD_HHMM>_<data>_<model>_<dataset>/`.
- W&B is on by default; `WANDB_MODE=disabled` turns it off.

**Cross-validation strategies** — set `train.selection`:

| Value | Split source |
|---|---|
| `Version` | `version_split_dir` — held-out **Version Test (VT)** set: different recordings of the same pieces |
| `SongStratified` | `song_stratified_dir` — 10 folds stratified by song; no song spans train and test |
| `SharedFold` | `shared_fold_dir` — fixed 10-fold split shared across modalities so results are comparable |
| `KFold` | k-fold over hash keys, computed in-process |
| `RandomSplit` | single random train/val/test split |
| `Artist` | needs `data/Stratify/stratify.csv`, **not included** (path hardcoded at `trainers.py:242`) |

Batch drivers:

```bash
bash scripts/run/run_folds.sh              # loop folds for one config
bash scripts/run/run_song_stratified.sh    # full song-stratified sweep
bash scripts/run/run_all_modalities.sh     # every modality end to end
bash scripts/run/run_layer_ablation.sh     # CultureMERT layer 1..12 ablation
```

### 2 · Evaluate a single model

"Test" throughout this repo means the **Version Test data split**, not software
tests; there is no test suite.

```bash
# re-run evaluate_test() from a saved checkpoint + config
python scripts/eval/run_test_only.py --model_dir weights/frame/Mel_Original_Version --fold 1

# frame-level masked accuracy on VT (Unknown excluded), overall + per class
python scripts/eval/run_vt_accuracy.py --strat_dir weights/frame/Mel_Original_Song_Stratified

# pool all 10 folds' frames, then score once (not a mean of fold means)
python scripts/eval/pooled_eval_song_stratified.py --strat_dir weights/frame/Mel_Original_Song_Stratified
```

`scripts/eval/` also holds `run_song_strat_vt_f1.py` (VT F1 across folds),
`version_test_stats.py` (GT-present summary stats), and
`aggregate_layer_ablation.py` (collects layer-ablation runs from `wandb/`).

### 3 · Ensembles

```bash
# song-stratified 10-fold, Mel + PESTO + MIDI
python scripts/ensemble/ensemble_inference_song_stratified.py
python scripts/ensemble/ensemble_inference_song_stratified.py --folds 1 2 3 --no_posteriors
python scripts/ensemble/ensemble_inference_song_stratified.py --no_midi
python scripts/ensemble/ensemble_inference_song_stratified.py --weights 0.35 0.35 0.30

# version-test split, Mel + PESTO + MIDI
python scripts/ensemble/ensemble_vt_mel_pesto_midi.py --weights 0.4 0.2 0.4
```

Results are written to `outputs/ensemble_song_stratified/` as per-fold metrics,
segment CSVs, VT frame predictions, and a text summary.

### 4 · Inference on your own recordings

`infer/` runs a trained model on arbitrary files — no corpus, label CSV, or split
index needed. One script per representation, plus an ensemble:

```bash
python infer/infer_mel.py    song.wav          # mel
python infer/infer_pesto.py  song.f0.csv       # PESTO f0 contour
python infer/infer_midi.py   song.mid          # transcribed MIDI
python infer/infer_cmert.py  song.wav          # CultureMERT
python infer/infer_cqt.py  /  infer_chroma.py  /  infer_crepe.py

# soft-voting ensemble; pass whichever modalities you have
python infer/infer_ensemble.py \
    --mel song.wav --pesto song.f0.csv --midi song.mid \
    --weights 0.4 0.2 0.4
```

Each writes one row per frame — `frame, time_sec, pred_id, pred_label,` and a
probability per class — and prints the contiguous mode segments. Directories are
accepted and recursed. `--run_dir` selects a different checkpoint, `--fold` a
different fold.

**Working from pitch contours?** `infer_pesto.py` takes a PESTO/CREPE CSV
(`frequency` + `confidence` at 100 Hz) directly, and the checkpoint is 3.4 MB.
Tonic normalisation is applied inside, so a raw contour file works as-is.

Two properties make this safe to point at a checkpoint you did not train:

- Every feature setting is read from the checkpoint's own `config.yaml`, never
  from a default in the inference code. This is not cosmetic — the PESTO runs
  here were trained with `threshold: 0.0` while `PitchDataset` defaults to `0.8`,
  and hardcoding the default would silently feed the model a differently
  normalised contour.
- `infer/_check_parity.py` asserts the standalone feature extractors are
  numerically identical to `datasets.py` (all six currently match to 0.0):

```bash
python infer/_check_parity.py
```

### 5 · Figures and analysis

```bash
python scripts/figures/make_example_figure.py       # example_figure.{png,pdf}
python scripts/figures/make_pattern_figure.py       # pattern_figure.{png,pdf}
python scripts/analysis/run_analysis_categorize.py  # which modality got which segment right
python scripts/analysis/run_analysis_simple.py      # condensed view of the above
```

Figure scripts need trained checkpoints under `weights/`. The two clips in
`assets/audio/` are inputs to `scripts/analysis/contour_visualization.py`.

### 6 · Viewer

A FastAPI posteriorgram browser for ensemble predictions:

```bash
bash viewer/run.sh              # http://localhost:7861
TUNNEL=1 bash viewer/run.sh     # also expose publicly via Cloudflare Tunnel
```

`TUNNEL` is opt-in — it publishes the viewer and the audio it serves to a public
URL.
---

## Directory layout

```
train.py                  # single Hydra training entry point
datasets.py               # one dataset class per input representation
trainers.py               # FrameTrainer / SegmentTrainer: CV splits, eval, W&B
losses.py                 # FocalLoss (alpha=1, gamma=0 reduces to cross-entropy)
paths.py                  # resolves PANSORI_DATA_ROOT
models/
    Conv2DGRU.py          # Conv2DGRU / SegConv2DGRU  (mel, CQT, chroma, MIDI)
    Conv1DGRU.py          # Conv1DGRU / SegConv1DGRU  (PESTO, CREPE)
    CMERT_classifier.py   # CultureMERT-95M classifier heads
    modules.py            # Conv1DBlock / Conv2DBlock
    model_utils.py        # conv parameter planning
configs/
    frame/                # frame-level configs, one per representation
    segment/              # segment-level variants
preproc/                  # corpus preparation (downsample, pitch shift, hash naming)
scripts/
    eval/                 # evaluation on saved checkpoints
    ensemble/             # cross-modality soft voting
    analysis/             # error categorization, contour plots
    figures/              # paper figure generation
    run/                  # batch drivers for the above
viewer/                   # FastAPI posteriorgram viewer
notebooks/                # exploratory notebooks
data/Label/               # annotations (the only data shipped here)
assets/audio/             # two clips used by the contour figure

weights/                  # checkpoints (gitignored)
    frame/                # per-modality runs
    midi/                 # MIDI runs inherited from the merged repo
outputs/                  # per-run CSV / NPZ / summaries (gitignored)
wandb/ analysis/ analysis_simple/   # experiment artifacts (gitignored)

requirements.txt          # pip environment
```

---

## Citation

```bibtex
% TODO(camera-ready): replace with the final BibTeX entry
@inproceedings{TODO,
  title     = {TODO(camera-ready): paper title},
  author    = {TODO(camera-ready): author list},
  booktitle = {TODO(camera-ready): venue},
  year      = {2026},
}
```

## License

[MIT](LICENSE). The license covers the code only; the pansori recordings are not
distributed here and carry their own terms.
