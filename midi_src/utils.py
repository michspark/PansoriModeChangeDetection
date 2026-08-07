import numpy as np
import random
import unicodedata
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import csv

def load_song_stratified_folds(genre_dir, midi_dir, label_json):
    """
    Reads song_stratified files (ch_1.txt, ch_2.txt, hb_1.txt, ...) from genre_dir.
    Groups files by base genre name (strip trailing _1/_2).
    For each genre two folds are generated (swapping val and test):
      - Fold A: val=genre_X_1, test=genre_X_2, train=all other 8 halves
      - Fold B: val=genre_X_2, test=genre_X_1, train=all other 8 halves
    Returns 10 folds (2 per genre).
    """
    import re

    genre_dir = Path(genre_dir)
    midi_dir = Path(midi_dir)

    with open(label_json, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    def _midi_key(name):
        n = name.replace('_vocal.mid', '')
        n = re.sub(r'^[0-9a-f]+-\d+-', '', n)
        return n

    key_to_midi = {}
    for item in raw:
        fu = unicodedata.normalize('NFC', item['file_upload'])
        midi_name = fu.rsplit('.', 1)[0] + '_vocal.mid'
        if (midi_dir / midi_name).exists():
            key_to_midi[_midi_key(midi_name)] = midi_name

    def _strip_uuid(key):
        return re.sub(r'^[0-9a-f]+-\d+-', '', key)

    def _load_half(path):
        keys = [l.strip() for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]
        return [key_to_midi[_strip_uuid(k)] for k in keys if _strip_uuid(k) in key_to_midi]

    # Group by base genre: {genre: [half1_songs, half2_songs]}
    from collections import defaultdict
    halves = defaultdict(dict)
    for f in sorted(genre_dir.glob('*.txt')):
        m = re.match(r'^(.+)_([12])$', f.stem)
        if not m:
            continue
        base, idx = m.group(1), int(m.group(2))
        songs = _load_half(f)
        halves[base][idx] = songs
        print(f"  {f.name}: {len(songs)} MIDI files")

    genres = sorted(halves.keys())
    folds = []
    fold_num = 1
    for held_out in genres:
        train_songs = [s for g in genres if g != held_out for s in halves[g][1] + halves[g][2]]
        for val_idx, test_idx in [(1, 2), (2, 1)]:
            val_songs  = halves[held_out][val_idx]
            test_songs = halves[held_out][test_idx]
            folds.append({'train': train_songs, 'val': val_songs, 'test': test_songs,
                          'split_info': {'train': train_songs, 'val': val_songs, 'test': test_songs}})
            print(f"  Fold {fold_num} (test={held_out}_{test_idx}, val={held_out}_{val_idx}): "
                  f"train={len(train_songs)}, val={len(val_songs)}, test={len(test_songs)}")
            fold_num += 1
    return folds


def load_folds_from_files(fold_dir, midi_dir, label_json, k=10):
    """
    fold_dir 의 fold_01.txt ~ fold_10.txt 를 읽어
    각 fold에 해당하는 MIDI 파일명 리스트로 변환하고
    train/val/test 분할 딕셔너리 리스트를 반환한다.
    """
    import re

    fold_dir = Path(fold_dir)
    midi_dir = Path(midi_dir)

    with open(label_json, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    def _midi_key(name):
        n = name.replace('_vocal.mid', '')
        n = re.sub(r'^[0-9a-f]+-\d+-', '', n)
        return n

    key_to_midi = {}
    for item in raw:
        fu = unicodedata.normalize('NFC', item['file_upload'])
        midi_name = fu.rsplit('.', 1)[0] + '_vocal.mid'
        if (midi_dir / midi_name).exists():
            key_to_midi[_midi_key(midi_name)] = midi_name

    chunks = []
    for i in range(1, k + 1):
        fold_file = fold_dir / f'fold_{i:02d}.txt'
        keys = [line.strip() for line in fold_file.read_text(encoding='utf-8').splitlines() if line.strip()]
        midi_names = [key_to_midi[key] for key in keys if key in key_to_midi]
        chunks.append(midi_names)
        print(f"  Fold {i:2d}: {len(keys)} keys -> {len(midi_names)} MIDI files")

    folds = []
    for i in range(k):
        test_songs = chunks[i]
        val_songs = chunks[(i + 1) % k]
        train_songs = []
        for j in range(k):
            if j != i and j != (i + 1) % k:
                train_songs.extend(chunks[j])
        folds.append({'train': train_songs, 'val': val_songs, 'test': test_songs})
    return folds


def load_version_split(split_dir, midi_dir, label_json):
    """
    pansori_version_split/ 의 train.txt / val.txt / test.txt 를 읽어
    단일 fold 딕셔너리 {'train': [...], 'val': [...], 'test': [...]} 를 반환한다.
    각 파일의 한 줄은 '{hash}-{track}-{singer}-{desc}' 형식이며
    hash 앞부분과 track 번호를 제거한 키로 MIDI 파일명을 조회한다.
    """
    import re

    split_dir = Path(split_dir)
    midi_dir  = Path(midi_dir)

    with open(label_json, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    def _midi_key(name):
        n = name.replace('_vocal.mid', '')
        n = re.sub(r'^[0-9a-f]+-\d+-', '', n)
        return n

    key_to_midi = {}
    for item in raw:
        fu = unicodedata.normalize('NFC', item['file_upload'])
        midi_name = fu.rsplit('.', 1)[0] + '_vocal.mid'
        if (midi_dir / midi_name).exists():
            key_to_midi[_midi_key(midi_name)] = midi_name

    def _strip_uuid(key):
        return re.sub(r'^[0-9a-f]+-\d+-', '', key)

    def _load_split(fname):
        lines = [l.strip() for l in (split_dir / fname).read_text(encoding='utf-8').splitlines() if l.strip()]
        midi_names = [key_to_midi[_strip_uuid(k)] for k in lines if _strip_uuid(k) in key_to_midi]
        print(f"  {fname}: {len(lines)} keys -> {len(midi_names)} MIDI files")
        return midi_names

    train_songs = _load_split('train.txt')
    val_songs   = _load_split('val.txt')
    test_songs  = _load_split('test.txt')

    return [{'train': train_songs, 'val': val_songs, 'test': test_songs,
             'split_info': {'train': train_songs, 'val': val_songs, 'test': test_songs}}]


def create_kfold_splits(song_list, k=10, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    song_list_shuffled = song_list.copy()
    random.shuffle(song_list_shuffled)

    num_chunks = k
    chunk_size = len(song_list_shuffled) // num_chunks
    chunks = []

    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_chunks - 1 else len(song_list_shuffled)
        chunks.append(song_list_shuffled[start:end])

    folds = []
    for i in range(k):

        test_idx = i
        test_songs = chunks[test_idx]


        val_idx = (i + 1) % k
        val_songs = chunks[val_idx]


        train_songs = []
        for j in range(num_chunks):
            if j != test_idx and j != val_idx:
                train_songs.extend(chunks[j])

        folds.append({
            'train': train_songs,
            'val': val_songs,
            'test': test_songs
        })

    for idx, fold in enumerate(folds):
        print(f"  Fold {idx + 1}: Train={len(fold['train'])}, Val={len(fold['val'])}, Test={len(fold['test'])}")

    return folds

def get_all_song_names(data_dir, label_json):
    data_dir = Path(data_dir)
    with open(label_json, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    names = []
    for item in raw:
        fu = unicodedata.normalize('NFC', item['file_upload'])
        midi_name = fu.rsplit('.', 1)[0] + '_vocal.mid'
        if (data_dir / midi_name).exists():
            names.append(midi_name)
    return names


def plot_posteriorgram(song_name, gt, pred_probs):
    """
    gt         : (T, 5) numpy array, one-hot ground truth
    pred_probs : (T, 5) numpy array, softmax probabilities
    Returns a matplotlib Figure.
    """
    CLASS_NAMES = ['no label', 'Ujo', 'GMjo', 'ANR', 'CJO']
    _CMAP = plt.cm.get_cmap('tab10')
    _CLASS_COLORS = [_CMAP(7), _CMAP(0), _CMAP(3), _CMAP(2), _CMAP(4)]

    T = gt.shape[0]
    n = len(CLASS_NAMES)

    fig, axes = plt.subplots(2, 1, figsize=(16, 5), sharex=True,
                             constrained_layout=True)
    fig.suptitle(song_name, fontsize=10)

    gt_labels = np.argmax(gt, axis=1)
    import matplotlib.colors as mcolors
    gt_cmap = mcolors.ListedColormap([_CLASS_COLORS[i] for i in range(n)])
    axes[0].imshow(gt_labels[np.newaxis, :], aspect='auto', origin='lower',
                   cmap=gt_cmap, vmin=-0.5, vmax=n - 0.5, interpolation='nearest',
                   extent=[0, T, -0.5, 0.5])
    axes[0].set_yticks([0])
    axes[0].set_yticklabels(['class'])
    axes[0].set_title('Ground Truth')

    legend_handles = [Patch(color=_CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(n)]
    axes[0].legend(handles=legend_handles, loc='upper right', fontsize=8, framealpha=0.7)

    im = axes[1].imshow(np.flipud(pred_probs.T), aspect='auto', origin='lower',
                        vmin=0, vmax=1, cmap='gray_r', interpolation='nearest',
                        extent=[0, T, -0.5, n - 0.5])
    axes[1].set_yticks(list(range(n)))
    axes[1].set_yticklabels(CLASS_NAMES[::-1])
    axes[1].set_title('Predicted Posteriorgram')

    axes[-1].set_xlabel('Frame')
    fig.colorbar(im, ax=axes[1], label='Probability', shrink=0.8)

    return fig

def plot_confusion_matrix(song_data):
    """Build a confusion matrix from all songs in song_data and return as numpy array."""
    from io import BytesIO
    from PIL import Image
    CLASS_NAMES = ['no label', 'Ujo', 'GMjo', 'ANR', 'CJO']
    all_gt, all_pred = [], []
    for data in song_data.values():
        all_gt.append(np.argmax(data['gt'], axis=1))
        all_pred.append(np.argmax(data['pred_probs'], axis=1))
    all_gt = np.concatenate(all_gt)
    all_pred = np.concatenate(all_pred)

    n = len(CLASS_NAMES)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(all_gt, all_pred):
        cm[t, p] += 1

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap='Blues')
    fig.colorbar(im, ax=ax, label='Recall')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right')
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix (normalised by true class)')
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.2f})",
                    ha='center', va='center', fontsize=8,
                    color='white' if cm_norm[i, j] > 0.6 else 'black')
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    image_np = np.array(Image.open(buf).convert('RGB'))
    buf.close()
    plt.close(fig)
    return image_np


def save_test_csv(segment_results, csv_path):
    """Save per-segment test results to CSV."""
    if not segment_results:
        return
    fieldnames = ['song_name', 'time_range', 'start_sec', 'end_sec',
                  'loss', 'acc', 'f1_ujoh', 'f1_gyemyeon', 'f1_aniri', 'f1_changjo', 'f1_macro',
                  'prob_우조', 'prob_계면조', 'prob_아니리', 'prob_창조']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(segment_results)


