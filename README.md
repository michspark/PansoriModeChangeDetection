# Pansori Mode Change Detection

Frame-level detection of **mode (조) changes** in Korean *pansori* singing.

Given a recording, the model labels every frame with the mode being sung, so the
boundaries where a singer shifts between modes fall out of the frame sequence.
The repo covers six input representations over the same training pipeline, plus
soft-voting ensembles across them.

| Class | id | Notes |
|---|---|---|
| Unknown  | 0 | unlabeled / ignored by the loss (`ignore_index: 0`) |
| 우조 (ujo)        | 1 | 경드름, 설렁제, 평조 are folded into this class |
| 계면조 (gyemyeonjo) | 2 | |
| 아니리 (aniri)     | 3 | spoken narration |
| 창조 (changjo)     | 4 | |

The bundled annotations (`data/Label/`) cover **395 recordings / 3,667 labeled
segments**.

---

## Input representations

Each modality is one Hydra config. All frame configs share `FrameTrainer`.

| Config | Dataset | Model | Input |
|---|---|---|---|
| `configs/frame/mel_base.yaml`   | `MelFrameDataset`    | `Conv2DGRU`      | 40-bin mel spectrogram, 31.25 fps |
| `configs/frame/cqt_base.yaml`   | `CQTFrameDataset`    | `Conv2DGRU`      | CQT (nnAudio) |
| `configs/frame/chroma_base.yaml`| `ChromaFrameDataset` | `Conv2DGRU`      | chroma |
| `configs/frame/pesto_best.yaml` | `PitchFrameDataset`  | `Conv1DGRU`      | PESTO f0 contour, 20 fps |
| `configs/frame/crepe_best.yaml` | `PitchFrameDataset`  | `Conv1DGRU`      | CREPE f0 contour |
| `configs/frame/cmert.yaml`      | `CMERTFrameDataset`  | `CMERTClassifier`| CultureMERT-95M features, 75 fps |

`configs/segment/` holds the segment-level (one label per 30 s window) variants,
trained with `SegmentTrainer`.

---

## Install

```bash
git clone <this-repo> && cd PansoriModeChangeDetection
python -m venv .venv && source .venv/bin/activate

# torch is pinned to a CUDA 11.8 build, so the torch index is required
pip install -r requirements.txt \
    --index-url https://download.pytorch.org/whl/cu118 \
    --extra-index-url https://pypi.org/simple
```

CPU-only: drop the `+cu118` suffixes from the `torch` / `torchaudio` /
`torchvision` pins and install from PyPI normally.

`configs/frame/cmert.yaml` downloads `ntua-slp/CultureMERT-95M` from the Hugging
Face Hub on first run.

---

## Expected data layout

**The audio corpus is not distributed with this repo.** Only the annotations
(`data/Label/label.{csv,json}`) are included. Point `PANSORI_DATA_ROOT` at your
corpus:

```bash
export PANSORI_DATA_ROOT=/path/to/Pansori_Data
```

It defaults to `../Pansori_Data` (a sibling of this checkout) and is expected to
contain:

```
Pansori_Data/
├── Audio/                  wav, source-separated vocal, 16 kHz   (mel)
├── Audio_Original/         wav, unseparated                      (cqt, chroma, cmert)
├── Audio_pesto_output/     PESTO f0 CSVs, one per recording      (pesto)
├── crepe/                  CREPE f0 CSVs, one per recording      (crepe)
├── pansori_version_split/  train.txt / val.txt / test.txt        (selection: Version)
├── song_stratified/        ch_1.txt, ch_2.txt, hb_1.txt, ...     (selection: SongStratified)
└── pansori_random_fold/    fold_01.txt .. fold_10.txt            (selection: SharedFold)
```

**Filename convention matters.** Every audio and contour file must begin with the
`hash_key` that appears in `data/Label/label.csv`, followed by `-`:

```
608e81d1-01-김일구-적벽가_군사들이_싸움타령_하는데.f0.csv
^^^^^^^^ hash_key
```

Both dataset loaders key off `filename.split("-")[0]`, so a file whose name does
not start with a known hash is silently skipped — a whole directory named the
wrong way loads as zero rows rather than raising. `preproc/add_hash_to_filenames.py`
assigns these prefixes.

`preproc/` holds the corpus-preparation helpers: `downsample.py` (to 16 kHz),
`make_pitch_shift.py` (augmentation), `add_hash_to_filenames.py` (assigns the
`hash_key` that every split file and the label CSV join on).

---

## Train

Run from the repo root; `hydra.run.dir` is `.`, so relative paths resolve.

```bash
python train.py --config-name mel_base        # mel
python train.py --config-name cqt_base        # CQT
python train.py --config-name chroma_base     # chroma
python train.py --config-name pesto_best      # PESTO pitch
python train.py --config-name crepe_best      # CREPE pitch
python train.py --config-name cmert           # CultureMERT

# segment-level instead of frame-level
python train.py --config-path configs/segment --config-name mel_base
```

Any config key can be overridden inline:

```bash
python train.py --config-name mel_base train.num_iterations=5000 train.target_folds=[1,2]
```

Checkpoints land in `weights/frame/<MMDD_HHMM>_<data>_<model>_<dataset>/`
(gitignored). Metrics go to Weights & Biases; `WANDB_MODE=disabled` turns that
off.

### Cross-validation strategies

Set `train.selection`:

| Value | Split source |
|---|---|
| `Version`        | `version_split_dir` — held-out **Version Test (VT)** set: different recordings of the same pieces |
| `SongStratified` | `song_stratified_dir` — 10 folds stratified by song, no song spans train and test |
| `SharedFold`     | `shared_fold_dir` — a fixed 10-fold split shared across modalities so results are comparable |
| `KFold`          | k-fold over hash keys, split in-process |
| `RandomSplit`    | single random train/val/test split |
| `Artist`         | needs `data/Stratify/stratify.csv`, **not included** in this repo (path is hardcoded at `trainers.py:242`) |

Batch drivers are in `scripts/run/`:

```bash
bash scripts/run/run_folds.sh              # loop folds for one config
bash scripts/run/run_song_stratified.sh    # full song-stratified sweep
bash scripts/run/run_all_modalities.sh     # every modality end to end
bash scripts/run/run_layer_ablation.sh     # CultureMERT layer 1..12 ablation
```

---

## Evaluate

`scripts/eval/` — note that "test" here means the **Version Test data split**,
not software tests. This repo has no unit tests.

| Script | Purpose |
|---|---|
| `run_test_only.py` | Re-run `evaluate_test()` from a saved checkpoint + config |
| `run_vt_accuracy.py` | Frame-level masked accuracy on VT (Unknown excluded), overall + per class |
| `run_song_strat_vt_f1.py` | VT frame-level F1 across song-stratified folds |
| `pooled_eval_song_stratified.py` | Pools all 10 folds' frames, then scores once (not a mean of fold means) |
| `version_test_stats.py` | GT-present summary stats for a version-test CSV |
| `aggregate_layer_ablation.py` | Collects the layer-ablation runs from `wandb/` into one table |

---

## Ensembles

`scripts/ensemble/` soft-votes across modalities, aligning each stream's frame
rate (mel 31.25 fps, PESTO 20 fps, MIDI 10 fps) onto the slowest before
averaging.

> **Requires a second repository.** The MIDI modality's checkpoints live in a
> sibling `PansoriMIDIDetection` checkout. Point `PANSORI_MIDI_REPO` at it
> (defaults to `../PansoriMIDIDetection`). Without it, only the `--no_midi`
> paths will run.

```bash
export PANSORI_MIDI_REPO=/path/to/PansoriMIDIDetection
bash scripts/run/run_ensemble_song_stratified.sh     # 10-fold, mel+pesto+midi
```

| Script | Split |
|---|---|
| `ensemble_inference_song_stratified.py` | song-stratified 10-fold; wrote `outputs/ensemble_song_stratified/` |
| `ensemble_inference.py` | version split, end-to-end from audio |
| `ensemble_vt_mel_pesto_midi.py` | version test, mel + PESTO + MIDI |
| `ensemble_soft_voting.py` | offline, votes over already-saved `prob_*` CSV columns |
| `eval_mel_ensemble_acc.py`, `eval_vt_mel_ensemble.py` | masked accuracy from saved fold metrics |

These accumulated across experiments and overlap heavily —
`ensemble_soft_voting.py` is the earliest (offline CSV) and
`ensemble_inference_song_stratified.py` the most complete.

---

## Figures and analysis

```bash
python scripts/figures/make_example_figure.py    # example_figure.{png,pdf}
python scripts/figures/make_pattern_figure.py    # pattern_figure.{png,pdf}
python scripts/analysis/run_analysis_categorize.py  # which modality got which segment right
python scripts/analysis/run_analysis_simple.py      # condensed view of the above
```

Figure scripts need trained checkpoints under `weights/`. The two clips in
`assets/audio/` are inputs to `scripts/analysis/contour_visualization.py`.

## Viewer

A FastAPI posteriorgram browser for ensemble predictions:

```bash
bash viewer/run.sh              # http://localhost:7861
TUNNEL=1 bash viewer/run.sh     # also expose publicly via Cloudflare Tunnel
```

`TUNNEL` is opt-in — it publishes the viewer and the audio it serves to a public
URL.

---

## Repo map

```
train.py            single training entrypoint (Hydra)
datasets.py         all dataset classes, one per input representation
trainers.py         FrameTrainer / SegmentTrainer, CV splits, eval, W&B logging
losses.py           FocalLoss (alpha=1, gamma=0 reduces to cross-entropy)
paths.py            resolves PANSORI_DATA_ROOT / PANSORI_MIDI_REPO
models/             Conv1DGRU, Conv2DGRU, CMERTClassifier + segment variants
configs/frame/      frame-level configs      configs/segment/  segment-level
preproc/            corpus preparation
scripts/eval/       evaluation on saved checkpoints
scripts/ensemble/   cross-modality soft voting
scripts/analysis/   error categorization, contour plots
scripts/figures/    paper figure generation
scripts/run/        batch drivers for the above
viewer/             FastAPI posteriorgram viewer
notebooks/          exploratory notebooks
data/Label/         annotations (the only data shipped here)
```

`weights/`, `wandb/`, `outputs/`, `analysis/`, `analysis_simple/` are experiment
artifacts and are gitignored.

## License

MIT — see [LICENSE](LICENSE). The license covers the code only; the pansori
recordings are not distributed here and carry their own terms.
