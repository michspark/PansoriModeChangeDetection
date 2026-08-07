from .dataset_utils import PianoRollGenerator
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm
import torch
import json
import unicodedata
import random

class BaseDataset(Dataset):
    label_map = {
        "우조": 1, "평조": 1, "경드름": 1, "설렁제": 1,
        "계면조": 2,
        "아니리": 3,
        "창조": 4
    }  # 0: no label

    def __init__(self, data_dir, label_json, song_list=None, fs=100, window_size=30.0, is_train=True, snap_boundaries=True):
        self.data_dir = Path(data_dir)
        self.song_list = set(song_list) if song_list is not None else None
        self.fs = fs
        self.window_size = int(window_size * fs) # convert seconds to frames
        self.is_train = is_train
        self.snap_boundaries = snap_boundaries

        with open(label_json, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        self.memory_cache = []
        self.training_instances = []

        for song_idx, item in enumerate(tqdm(raw)):
            fu = unicodedata.normalize('NFC', item['file_upload'])
            midi_name = fu.rsplit('.', 1)[0] + '_vocal.mid'

            if self.song_list is not None and midi_name not in self.song_list:
                continue

            midi_path = self.data_dir / midi_name
            if not midi_path.exists():
                continue

            result = item['annotations'][0]['result']
            if not result:
                continue

            original_length = result[0]['original_length']
            gen = PianoRollGenerator(str(midi_path), fs=self.fs)
            piano_roll = torch.tensor(gen.generate_piano_roll(), dtype=torch.float32)
            frame_label = self.build_frame_label(original_length, result)

            T = min(piano_roll.shape[1], frame_label.shape[0])

            cache_idx = len(self.memory_cache)
            self.memory_cache.append({
                'song_name': midi_name,
                'piano_roll': piano_roll[:, :T],
                'frame_label': frame_label[:T, :],
                'total_frames': T,
            })

            if self.is_train:
                num_repeats = max(1, T // self.window_size)
                for _ in range(num_repeats):
                    self.training_instances.append(cache_idx)

        if not self.is_train:
            self.val_segments = self.prepare_val_segments()

    def build_frame_label(self, original_length: float, result: list) -> torch.Tensor:
        num_frames = int(original_length * self.fs)
        frame_label = torch.zeros(num_frames, 5)
        frame_label[:, 0] = 1  # default: no label
        for ann in result:
            val = ann['value']
            s = int(val['start'] * self.fs)
            e = min(int(val['end'] * self.fs), num_frames)
            name = val['labels'][0]
            if name in self.label_map:
                cls_idx = self.label_map[name]
                frame_label[s:e, :] = 0
                frame_label[s:e, cls_idx] = 1
        return frame_label

    def prepare_val_segments(self):
        """Creates non-overlapping segments.

        When snap_boundaries=True (default): snaps boundaries to silence so
        that notes active at a window boundary are not cut mid-note.
        Searches within ±1 second of the target boundary for a silent frame.
        Falls back to the original boundary if no silence is found.

        When snap_boundaries=False: uses exact 30-second (window_size) boundaries,
        producing standard time ranges like 0-30s, 30-60s, 60-90s, ...
        """
        segments = []
        tolerance = self.fs  # 1 second in frames
        for idx, item in enumerate(self.memory_cache):
            piano_roll = item['piano_roll']  # [128, T]
            total_frames = item['total_frames']
            start = 0
            while start < total_frames:
                segments.append((idx, start))
                target_end = start + self.window_size
                if target_end >= total_frames:
                    break
                if self.snap_boundaries and piano_roll[:, target_end].any():
                    # Search forward first (note ends soon after boundary)
                    snapped = None
                    for f in range(target_end + 1, min(total_frames, target_end + tolerance)):
                        if not piano_roll[:, f].any():
                            snapped = f
                            break
                    # If not found forward, search backward
                    if snapped is None:
                        for f in range(target_end - 1, max(start, target_end - tolerance) - 1, -1):
                            if not piano_roll[:, f].any():
                                snapped = f + 1
                                break
                    start = snapped if snapped is not None else target_end
                else:
                    start = target_end
        return segments

    def pitch_shift(self, piano_roll, shift_range=(-3, 3)):
        shift = random.randint(*shift_range)
        if shift == 0:
            return piano_roll.clone()
        shifted = torch.roll(piano_roll, shift, dims=0)
        if shift > 0:
            shifted[:shift, :] = 0
        else:
            shifted[shift:, :] = 0
        return shifted

    def time_masking(self, piano_roll, total_mask_sec=5.0, num_masks=3):
        pr = piano_roll.clone()
        T = pr.shape[1]
        max_per_mask = int(total_mask_sec * self.fs) // num_masks

        for _ in range(num_masks):
            mask_len = random.randint(1, max_per_mask)
            if T <= mask_len:
                continue
            start = random.randint(0, T - mask_len)
            pr[:, start:start + mask_len] = 0
        return pr

    def __len__(self):
        return len(self.training_instances) if self.is_train else len(self.val_segments)

    def __getitem__(self, idx):
        if self.is_train:
            song_idx = self.training_instances[idx]
            item = self.memory_cache[song_idx]
            if item['total_frames'] > self.window_size:
                start_frame = random.randint(0, item['total_frames'] - self.window_size)
            else:
                start_frame = 0
        else:
            song_idx, start_frame = self.val_segments[idx]
            item = self.memory_cache[song_idx]

        end_frame = start_frame + self.window_size
        slice_piano = item['piano_roll'][:, start_frame:end_frame]
        slice_label = item['frame_label'][start_frame:end_frame, :]

        if self.is_train:
            if random.random() < 0.5:
                slice_piano = self.pitch_shift(slice_piano)
            if random.random() < 0.5:
                slice_piano = self.time_masking(slice_piano)

        curr_w = slice_piano.shape[1]
        if curr_w < self.window_size:
            pad_w = self.window_size - curr_w
            slice_piano = torch.nn.functional.pad(slice_piano, (0, pad_w), value=0)
            pad_label = torch.zeros((pad_w, 5))
            pad_label[:, 0] = 1
            slice_label = torch.cat([slice_label, pad_label], dim=0)
        elif curr_w > self.window_size:
            slice_piano = slice_piano[:, :self.window_size]
            slice_label = slice_label[:self.window_size, :]

        return item['song_name'], start_frame, slice_piano, slice_label




