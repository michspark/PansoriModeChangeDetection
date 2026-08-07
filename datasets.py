import os
import math
import json
import random
import unicodedata
from pathlib import Path
from copy import copy, deepcopy
from abc import abstractmethod
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import TimeStretch, FrequencyMasking
from torchaudio.transforms import Spectrogram, MelScale, AmplitudeToDB
from nnAudio.features.cqt import CQT
from nnAudio.librosa_functions import chroma

class BaseDataset(Dataset):
    def __init__(self, data_dir, label_dir, num_classes=4, sr=16000, window=20, margin_ratio=1, is_valid=False, aug=False):
        self.sr = sr
        self.window = window

        self.num_classes = num_classes
        self.label_map = {"Unknown": 0, "경드름": 1, "설렁제": 1, "평조": 1, "우조": 1, "계면조": 2, "아니리": 3, "창조": 4}
        self.data_dir = Path(data_dir)

        label_csv = os.path.join(label_dir, 'label.csv')
        self.df = pd.read_csv(label_csv) if os.path.exists(label_csv) else self.get_df(label_dir)
        self.loaded_hash = list(set(self.df['hash_key'].tolist()))

        self.aug = aug
        self.margin_ratio = margin_ratio
        self.is_valid = is_valid


    def get_df(self, label_dir):
        label_json = f'{label_dir}/label.json'
        with open(label_json, 'r', encoding='utf-8') as file: label_data = json.load(file)
        df = pd.DataFrame(columns=['hash_key', 'filename', 'duration', 'start', 'end', 'label'], index=None)

        cnt = 0
        for item in tqdm(label_data, desc='Save DF'):
            file_upload = unicodedata.normalize('NFC', item["file_upload"])
            hash_key = file_upload.split("-")[0]
            filename = '-'.join(file_upload.split("-")[1:])[:-4]

            for result in item['annotations'][0]['result']:
                duration = int(result['original_length'] * 1000)
                start =  int(result['value']['start'] * 1000)
                end =  int(result['value']['end'] * 1000)
                label = result['value']['labels'][0]
                df.loc[cnt] = [hash_key, filename, duration, start, end, label]
                cnt += 1

        df.to_csv(f'{label_dir}/label.csv', index=None)

        return df

    def get_frame_label(self):
        loaded_label = {}
        for hash_key in tqdm(self.loaded_hash, desc='Load Label'):
            target_df = self.df[self.df['hash_key']==hash_key]
            sample_duration = target_df['duration'].values[0]
            label = torch.zeros(sample_duration, self.num_classes)
            label[:,self.label_map['Unknown']] = 1

            for row in target_df[['start', 'end', 'label']].to_numpy():
                label[row[0]:row[1], self.label_map[row[2]]] = 1
                label[row[0]:row[1], self.label_map['Unknown']] = 0

            loaded_label[hash_key] = label

        return loaded_label

    @abstractmethod
    def get_data(self): return None

    @abstractmethod
    def __len__(self): return None

    @abstractmethod
    def __getitem__(self, idx): return None



class AudioDataset(BaseDataset):
    # One-entry decode cache, used only on validation/test splits. prepare_val_segments
    # emits every 30s window of a song back to back and the eval loaders use
    # shuffle=False, so the same file was being fully decoded once per window
    # (~20x per song). Holding the last decode costs one song of memory (~35MB).
    _cached_audio_file = None
    _cached_audio = None

    def __init__(self, data_dir, label_dir, num_classes=4, sr=16000, channels='mono', window=20, margin_ratio=1.0, is_valid=False, aug=False):
        super().__init__(data_dir, label_dir, num_classes, sr, window, margin_ratio, is_valid, aug)
        self.channels = channels

        self.pitch_shift_dir = os.path.join(os.path.dirname(data_dir), "PitchShiftedAudio")
        if self.aug and not self.is_valid and os.path.exists(self.pitch_shift_dir): self.pitch_shifted_audio = self.load_pitch_shifted_audio()


    def _load_audio(self, audio_file):
        # Training draws a random song each time, so caching there would only cost
        # memory in every worker. Validation/test are sequential — cache those.
        if self.is_valid and self._cached_audio_file == audio_file:
            return self._cached_audio

        data, sr = sf.read(audio_file, dtype='float32', always_2d=True)
        audio = torch.from_numpy(data.T)  # (channels, samples)
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.sr) if self.sr!=sr else audio
        audio = audio.mean(dim=0, keepdim=True) if self.channels=='mono' else audio

        if self.is_valid:
            self._cached_audio_file = audio_file
            self._cached_audio = audio
        return audio


    def load_pitch_shifted_audio(self):
        pitch_shifted_audio = {}
        if not os.path.exists(self.pitch_shift_dir):
            print(f"Warning: Pitch shift directory {self.pitch_shift_dir} not found.")
            return None

        for audio_file in tqdm(os.listdir(self.pitch_shift_dir), desc="Load Pitch-Shifted Audio"):
            if not audio_file.endswith(".wav"): continue

            parts = audio_file.split("_ps_")
            if len(parts) != 2: continue

            original_name = parts[0]
            hash_key = unicodedata.normalize('NFC', original_name.split("-")[0])

            if hash_key not in self.loaded_hash: continue

            audio = self._load_audio(os.path.join(self.pitch_shift_dir, audio_file))
            if hash_key not in pitch_shifted_audio: pitch_shifted_audio[hash_key] = []
            pitch_shifted_audio[hash_key].append(audio)

        return pitch_shifted_audio


    def apply_audio_augmentation(self, audio):
        if not self.aug or random.random() > 0.5: return audio
        # Noise
        if random.random() < 0.3:
            noise_level = random.uniform(0.001, 0.005)
            noise = torch.randn_like(audio) * noise_level
            audio = audio + noise
        # Gain
        if random.random() < 0.3:
            gain = random.uniform(0.7, 1.3)
            audio = audio * gain
        return audio


    @abstractmethod
    def get_data(self): return None

    # @abstractmethod
    # def get_split(self): return None

    @abstractmethod
    def __len__(self): return None

    @abstractmethod
    def __getitem__(self, idx): return None



class AudioFrameDataset(AudioDataset):
    def __init__(self, data_dir, label_dir, num_classes=4, sr=16000, channels='mono', window=20, margin_ratio=1.0, is_valid=False, aug=False):
        super().__init__(data_dir, label_dir, num_classes, sr, channels, window, margin_ratio, is_valid, aug)
        self.loaded_data = self.get_data()
        self.loaded_label = self.get_frame_label()

        self.training_instances = []
        self.val_segments = []
        self.prepare_training_instances()


    def get_data(self):
        loaded_data = {}
        for audio_file in tqdm(list(self.data_dir.rglob('*.wav')), desc='Index Audio'):
            hash_key = unicodedata.normalize('NFC', audio_file.name.split("-")[0])
            if hash_key not in self.loaded_hash: continue
            loaded_data[hash_key] = audio_file  # store path only, load on-demand

        return loaded_data


    def prepare_training_instances(self, target_hash_keys=None):
        self.training_instances = []
        window_samples = self.window * self.sr

        for hash_key in list(self.loaded_hash):
            if hash_key not in self.loaded_data:
                self.loaded_hash.remove(hash_key)
                continue
            if target_hash_keys is not None and hash_key not in target_hash_keys:
                continue

            duration_ms = self.df[self.df['hash_key'] == hash_key]['duration'].values[0]
            audio_length = int(duration_ms * self.sr / 1000)
            num_repeats = max(1, audio_length // window_samples)
            self.training_instances.extend([hash_key] * num_repeats)

    def update_training_instances(self, target_hash_keys):
        self.prepare_training_instances(target_hash_keys)

    def prepare_val_segments(self, target_hash_keys):
        val_segments = []
        window_samples = self.window * self.sr
        for hash_key in target_hash_keys:
            if hash_key not in self.loaded_data:
                continue
            duration_ms = self.df[self.df['hash_key'] == hash_key]['duration'].values[0]
            audio_length = int(duration_ms * self.sr / 1000)
            for start in range(0, audio_length, window_samples):
                val_segments.append((hash_key, start, min(start + window_samples, audio_length)))
        return val_segments

    def compose_validset(self, target_hash_keys):
        validset = copy(self)  # shallow copy — shares loaded_data/label references, no duplication
        validset.is_valid = True
        validset.loaded_hash = [k for k in target_hash_keys if k in self.loaded_data]
        validset.loaded_data = {k: self.loaded_data[k] for k in validset.loaded_hash}
        validset.loaded_label = {k: self.loaded_label[k] for k in validset.loaded_hash}
        validset.val_segments = self.prepare_val_segments(validset.loaded_hash)
        return validset


    def get_split(self, target_hash_keys, split='train'):
        if split == 'train':
            self.update_training_instances(target_hash_keys)
        else:
            return self.compose_validset(target_hash_keys)


    def __len__(self):
        if self.is_valid: return len(self.val_segments)
        return len(self.training_instances)


    def __getitem__(self, idx):
        window_samples = self.window * self.sr

        if self.is_valid:
            hash_key, start, end = self.val_segments[idx]
            audio = self._load_audio(self.loaded_data[hash_key])
            audio_slice = audio[:, start:end]
            if audio_slice.shape[1] < window_samples:
                pad = window_samples - audio_slice.shape[1]
                audio_slice = torch.nn.functional.pad(audio_slice, (0, pad))
            start_ms, end_ms = int(start / self.sr * 1000), int(end / self.sr * 1000)
            label = self.loaded_label[hash_key][start_ms:end_ms]
            return hash_key, audio_slice, label

        hash_key = self.training_instances[idx]

        if self.aug and hasattr(self, 'pitch_shifted_audio') and hash_key in self.pitch_shifted_audio and random.random() < 0.5:
            audio = random.choice(self.pitch_shifted_audio[hash_key])
        else:
            audio = self._load_audio(self.loaded_data[hash_key])

        audio_length = audio.shape[1]
        if audio_length > window_samples:
            start = random.randint(0, audio_length - window_samples)
        else:
            start = 0
        end = start + window_samples

        margin_length = int(self.margin_ratio * window_samples)
        if start - margin_length // 2 < 0:
            end = margin_length
            start = 0
        elif end + margin_length // 2 > audio_length:
            end = audio_length
            start = max(0, audio_length - margin_length)
        else:
            start, end = start - margin_length // 2, end + margin_length // 2
        audio = audio[:, start:end]

        start_ms, end_ms = int(start / self.sr * 1000), int(end / self.sr * 1000)
        label = self.loaded_label[hash_key][start_ms:end_ms]

        if self.aug: audio = self.apply_audio_augmentation(audio)

        return hash_key, audio, label



class AudioSegmentDataset(AudioDataset):
    def __init__(self, data_dir, label_dir, num_classes=3, sr=16000, channels='mono', window=20, margin_ratio=1.0, is_valid=False, aug=False):
        super().__init__(data_dir, label_dir, num_classes, sr, channels, window, margin_ratio, is_valid, aug)
        self.loaded_all = self.get_data()
        self.loaded_data, self.loaded_label, self.loaded_meta = self.loaded_all

    def get_data(self):
        loaded_data, loaded_label, loaded_meta = [], [], []
        for audio_file in tqdm(list(Path(self.data_dir).glob('*.wav')), desc="Index Audio Segments"):
            hash_key = unicodedata.normalize('NFC', audio_file.name.split("-")[0])
            if hash_key not in self.loaded_hash: continue

            hash_df = self.df[self.df['hash_key']==hash_key]
            segment_meta = hash_df[['start', 'end','label']].to_numpy()

            for start_ms, end_ms, label in segment_meta:
                start, end = int(start_ms * self.sr /1000), int(end_ms * self.sr /1000)
                label = self.label_map[label]

                loaded_data.append(audio_file)  # store path only
                loaded_label.append(label)
                loaded_meta.append((hash_key, start, end))

        return loaded_data, loaded_label, loaded_meta


    def compose_validset(self, target_hash_keys):
        loaded_data, loaded_label, loaded_meta = [], [], []

        for audio, label, (hash_key, start, end) in zip(self.loaded_all[0], self.loaded_all[1], self.loaded_all[2]):
            if hash_key not in target_hash_keys: continue
            seg_len, window_len = (end - start), self.window * self.sr

            if seg_len > window_len:
                current_pos = 0
                while current_pos + window_len <= seg_len:
                    start_seg, end_seg = current_pos, current_pos+window_len
                    segment = audio[:, start_seg:end_seg]
                    current_pos += window_len

                    loaded_data.append(segment)
                    loaded_label.append(label)
                    loaded_meta.append((hash_key, start+start_seg, start+end_seg))

                if current_pos < seg_len:
                    final_start = seg_len - window_len
                    segment = audio[:, final_start:seg_len]

                    loaded_data.append(segment)
                    loaded_label.append(label)
                    loaded_meta.append((hash_key, final_start, seg_len))

            else:
                segment = audio[:, :]
                loaded_data.append(segment)
                loaded_label.append(label)
                loaded_meta.append((hash_key, start, end))

        return loaded_data, loaded_label, loaded_meta


    def get_split(self, target_hash_keys, split='train'):
        if split == 'train':
            target_indices = [idx for idx, tup in enumerate(self.loaded_all[2]) if tup[0] in target_hash_keys]
            self.loaded_data = [self.loaded_all[0][idx] for idx in target_indices]
            self.loaded_label = [self.loaded_all[1][idx] for idx in target_indices]
            self.loaded_meta = [self.loaded_all[2][idx] for idx in target_indices]
        else:
            validset = copy(self)  # shallow copy — no data duplication
            validset.is_valid = True
            validset.loaded_data, validset.loaded_label, validset.loaded_meta = self.compose_validset(target_hash_keys)
            return validset


    def __len__(self):
        return len(self.loaded_data)


    def __getitem__(self, idx):
        audio_path, label = self.loaded_data[idx], self.loaded_label[idx]
        hash_key, start, end = self.loaded_meta[idx]

        full_audio = self._load_audio(audio_path)
        if self.aug and not self.is_valid and hash_key in self.pitch_shifted_audio and random.random() < 0.5:
            full_audio = random.choice(self.pitch_shifted_audio[hash_key])
        audio = full_audio[:, start:end]

        if self.is_valid:
            return hash_key, audio, label

        if self.aug and not self.is_valid: audio = self.apply_audio_augmentation(audio)

        audio_len, window_len = audio.shape[-1], self.window * self.sr

        if audio_len > window_len:
            start = random.randint(0, min(audio_len, audio_len - window_len))
            audio = audio[:,start:start+window_len]

        return hash_key, audio, label



class MelDataset():
    def __init__(self, sr=16000, window=20, n_fft=2048, hop_length=512, target_bins=40, margin_ratio=1.6, is_valid=False, aug=False):
        self.sr = sr
        self.window = window

        self.hop_length = hop_length
        self.target_bins = target_bins
        self.window_frame = window * sr // hop_length

        self.margin_ratio = margin_ratio
        self.is_valid = is_valid
        self.aug = aug

        self.spec_cvt = Spectrogram(n_fft=n_fft, hop_length=hop_length, power=1.0)
        self.spec2mel = MelScale(n_stft=n_fft//2+1, n_mels=target_bins, sample_rate=sr, f_min=80, f_max=2000)
        self.db_cvt = AmplitudeToDB()

        if self.aug:
            self.time_stretch = TimeStretch(hop_length=hop_length, n_freq=n_fft//2+1)
            self.freq_mask = FrequencyMasking(freq_mask_param=10)


    def get_mel(self, audio):
        spec = self.spec_cvt(audio)
        if self.aug and not self.is_valid: spec, stretch_factor = self.apply_spec_time_stretch(spec)
        else: stretch_factor = 1.0

        mel = self.spec2mel(spec).squeeze(0)
        mel = self.db_cvt(mel) / 100

        return mel, stretch_factor


    def apply_spec_time_stretch(self, spec):
        stretch_factor = 1.0
        if random.random() < 0.3:
            stretch_factor = random.uniform(0.7, 1.5)
            if not torch.is_complex(spec): spec = torch.complex(spec, torch.zeros_like(spec))
            spec = self.time_stretch(spec, stretch_factor)
            spec = spec.abs()

        return spec, stretch_factor


    def apply_spec_pitch_shift(self, spec):
        if random.random() < 0.3:
            shift_steps = random.randint(-4, 4)
            if shift_steps == 0: return spec

            spec_shifted = torch.roll(spec, shifts=shift_steps, dims=-2)
            if shift_steps > 0: spec_shifted[..., :shift_steps, :] = 0
            else:
                shift_steps = abs(shift_steps)
                spec_shifted[..., -shift_steps:, :] = 0
            return spec_shifted

        return spec


    def apply_spec_augmentation(self, mel):
        """Apply spectrogram augmentations"""
        if not self.aug or random.random() > 0.8: return mel
        if random.random() < 0.5: mel = self.apply_spec_pitch_shift(mel)
        if random.random() < 0.5: mel = self.freq_mask(mel)
        return mel


    def apply_time_stretch(self, label, stretch_factor):
        if stretch_factor == 1.0: return label
        orig_length = label.shape[0]
        new_length = int(orig_length / stretch_factor)
        stretched_label = torch.zeros((new_length, label.shape[1]))

        if new_length > 0:
            orig_indices = torch.linspace(0, orig_length - 1, new_length).long()
            stretched_label[:new_length] = label[orig_indices]

        return stretched_label


    def ms_to_frame_label(self, ms_label):
        ms_per_frame = self.hop_length / self.sr * 1000
        frame_width = int(ms_per_frame)
        num_frames = int(ms_label.shape[0] / ms_per_frame)

        if num_frames == 0:
            # Segment shorter than one frame: pad to one frame and return
            pad_len = frame_width - ms_label.shape[0]
            ms_label = torch.nn.functional.pad(ms_label, (0, 0, 0, pad_len))
            num_frames = 1

        trimmed = ms_label[:num_frames * frame_width]           # (num_frames * frame_width, num_classes)
        grouped = trimmed.view(num_frames, frame_width, -1)     # (num_frames, frame_width, num_classes)
        class_sums = grouped.sum(dim=1)                         # (num_frames, num_classes)
        frame_label = (class_sums == class_sums.max(dim=-1, keepdim=True).values).float()

        return frame_label

class MelFrameDataset(AudioFrameDataset, MelDataset):
    def __init__(self, data_dir, label_dir, num_classes=4, sr=16000, channels='mono', window=20, n_fft=2048, hop_length=512, target_bins=40, margin_ratio=1.6, is_valid=False, aug=False):
        AudioFrameDataset.__init__(self, data_dir, label_dir, num_classes, sr, channels, window, margin_ratio, is_valid, aug)
        MelDataset.__init__(self, sr=sr, window=window, n_fft=n_fft, hop_length=hop_length, target_bins=target_bins, margin_ratio=margin_ratio, is_valid=is_valid, aug=aug)
        self.loaded_data = self._preload_mel(self.loaded_data)
        if self.aug and hasattr(self, 'pitch_shifted_audio') and self.pitch_shifted_audio:
            self.pitch_shifted_mel = self._preload_pitch_shifted_mel()

    def _audio_to_mel(self, audio):
        """Compute mel spectrogram without augmentation."""
        spec = self.spec_cvt(audio)
        mel = self.spec2mel(spec).squeeze(0)
        return self.db_cvt(mel) / 100

    def _preload_mel(self, path_dict):
        mel_dict = {}
        for hash_key, audio_path in tqdm(path_dict.items(), desc='Preload Mel Spectrograms'):
            audio = self._load_audio(audio_path)
            mel_dict[hash_key] = self._audio_to_mel(audio)
        return mel_dict

    def _preload_pitch_shifted_mel(self):
        mel_dict = {}
        for hash_key, audio_list in tqdm(self.pitch_shifted_audio.items(), desc='Preload Pitch-Shifted Mel'):
            mel_dict[hash_key] = [self._audio_to_mel(a) for a in audio_list]
        del self.pitch_shifted_audio
        return mel_dict

    def _apply_mel_time_stretch(self, mel):
        """Approximate time stretch via bilinear interpolation on the mel spectrogram."""
        stretch_factor = 1.0
        if random.random() < 0.3:
            stretch_factor = random.uniform(0.7, 1.5)
            new_len = int(mel.shape[1] / stretch_factor)
            if new_len > 0:
                mel = torch.nn.functional.interpolate(
                    mel.unsqueeze(0).unsqueeze(0),
                    scale_factor=(1.0, 1.0 / stretch_factor),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)
        return mel, stretch_factor

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        if self.is_valid:
            hash_key, start_sample, end_sample = self.val_segments[idx]
            mel_full = self.loaded_data[hash_key]
            start_frame = start_sample // self.hop_length
            end_frame = end_sample // self.hop_length
            mel = mel_full[:, start_frame:end_frame]
            if mel.shape[-1] < self.window_frame:
                mel = torch.nn.functional.pad(mel, (0, self.window_frame - mel.shape[-1]))
            start_ms = int(start_sample / self.sr * 1000)
            end_ms = int(end_sample / self.sr * 1000)
            label = self.loaded_label[hash_key][start_ms:end_ms]
            frame_label = self.ms_to_frame_label(label)
            if mel.shape[-1] != frame_label.shape[0]: mel = mel[:, :frame_label.shape[0]]
            return hash_key, mel, frame_label

        hash_key = self.training_instances[idx]

        if self.aug and hasattr(self, 'pitch_shifted_mel') and hash_key in self.pitch_shifted_mel and random.random() < 0.5:
            mel_full = random.choice(self.pitch_shifted_mel[hash_key])
        else:
            mel_full = self.loaded_data[hash_key]

        total_frames = mel_full.shape[1]
        window_frames = self.window_frame
        margin_frames = int(self.margin_ratio * window_frames)

        if total_frames > window_frames:
            start_frame = random.randint(0, total_frames - window_frames)
        else:
            start_frame = 0
        end_frame = start_frame + window_frames

        if start_frame - margin_frames // 2 < 0:
            end_frame = margin_frames
            start_frame = 0
        elif end_frame + margin_frames // 2 > total_frames:
            end_frame = total_frames
            start_frame = max(0, total_frames - margin_frames)
        else:
            start_frame = start_frame - margin_frames // 2
            end_frame = end_frame + margin_frames // 2

        mel = mel_full[:, start_frame:end_frame].clone()

        if self.aug:
            mel, stretch_factor = self._apply_mel_time_stretch(mel)
        else:
            stretch_factor = 1.0

        start_ms = int(start_frame * self.hop_length / self.sr * 1000)
        end_ms = int(end_frame * self.hop_length / self.sr * 1000)
        label = self.loaded_label[hash_key][start_ms:end_ms]

        center_start = (mel.shape[1] - window_frames) // 2
        mel = mel[:, center_start:center_start + window_frames]
        if self.aug: mel = self.apply_spec_augmentation(mel)

        label = self.apply_time_stretch(label, stretch_factor)
        frame_label = self.ms_to_frame_label(label)
        frame_label = frame_label[center_start:center_start + window_frames]
        assert frame_label.shape[0] == mel.shape[1], f"frame_label.shape[0] != mel.shape[1]: {frame_label.shape[0]} != {mel.shape[1]}, time stretch factor: {stretch_factor}"
        return hash_key, mel, frame_label



class MelSegmentDataset(AudioSegmentDataset, MelDataset):
    def __init__(self, data_dir, label_dir, num_classes=3, sr=16000, channels='mono', window=20, n_fft=2048, hop_length=512, target_bins=40, margin_ratio=1.0, is_valid=False, aug=False):
        AudioSegmentDataset.__init__(self, data_dir, label_dir, num_classes, sr, channels, window, margin_ratio, is_valid, aug)
        MelDataset.__init__(self, sr=sr, window=window, n_fft=n_fft, hop_length=hop_length, target_bins=target_bins, margin_ratio=margin_ratio, is_valid=is_valid, aug=aug)

    def __len__(self):
        return super().__len__()


    def __getitem__(self, idx):
        hash_key, audio, label = super().__getitem__(idx)

        mel, _ = self.get_mel(audio)

        if mel.shape[1] > self.window_frame:
            start_idx = (mel.shape[1] - self.window_frame) // 2
            mel = mel[:, start_idx:start_idx+self.window_frame]
        else:
            num_pad = self.window_frame - mel.shape[1]
            l_pad = num_pad // 2
            r_pad = num_pad - l_pad
            mel = torch.nn.functional.pad(mel, (l_pad, r_pad), mode='constant', value=0)

        if self.aug and not self.is_valid: mel = self.apply_spec_augmentation(mel)
        assert mel.shape[1] == self.window_frame, f"mel.shape[1] != self.window_frame: {mel.shape[1]} != {self.window_frame}"
        return hash_key, mel, label


class CQTDataset():
    def __init__(self, sr=16000, hop_length=512, n_bins=84, bins_per_octave=12, is_valid=False, aug=False):
        self.sr = sr
        self.hop_length = hop_length
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.window_frame = None
        self.is_valid = is_valid
        self.aug = aug

        self.cqt_transform = CQT(
            sr=sr, hop_length=hop_length, fmin=32.7,
            n_bins=n_bins, bins_per_octave=bins_per_octave,
            filter_scale=1, norm=1, window='hann',
            center=True, pad_mode='reflect',
            trainable=False, output_format='Magnitude', verbose=False
        )

        if self.aug:
            self.freq_mask = FrequencyMasking(freq_mask_param=10)


    def _apply_cqt_time_stretch(self, cqt):
        """Approximate time stretch on magnitude CQT via bilinear interpolation."""
        stretch_factor = 1.0
        if random.random() < 0.3:
            stretch_factor = random.uniform(0.7, 1.5)
            new_len = int(cqt.shape[-1] / stretch_factor)
            if new_len > 0:
                cqt = torch.nn.functional.interpolate(
                    cqt.unsqueeze(0).unsqueeze(0),
                    scale_factor=(1.0, 1.0 / stretch_factor),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)
        return cqt, stretch_factor


    def apply_spec_pitch_shift(self, cqt):
        if random.random() < 0.3:
            shift_steps = random.randint(-4, 4)
            if shift_steps == 0: return cqt

            cqt_shifted = torch.roll(cqt, shifts=shift_steps, dims=-2)
            if shift_steps > 0: cqt_shifted[..., :shift_steps, :] = 0
            else:
                shift_steps = abs(shift_steps)
                cqt_shifted[..., -shift_steps:, :] = 0
            return cqt_shifted

        return cqt


    def apply_spec_augmentation(self, cqt):
        if not self.aug or random.random() > 0.8: return cqt
        if random.random() < 0.5: cqt = self.apply_spec_pitch_shift(cqt)
        if random.random() < 0.5: cqt = self.freq_mask(cqt)
        return cqt


    def apply_time_stretch(self, label, stretch_factor):
        if stretch_factor == 1.0: return label
        orig_length = label.shape[0]
        new_length = int(orig_length / stretch_factor)
        stretched_label = torch.zeros((new_length, label.shape[1]))

        if new_length > 0:
            orig_indices = torch.linspace(0, orig_length - 1, new_length).long()
            stretched_label[:new_length] = label[orig_indices]

        return stretched_label


    def ms_to_frame_label(self, ms_label):
        ms_per_frame = self.hop_length / self.sr * 1000
        frame_width = int(ms_per_frame)
        num_frames = int(ms_label.shape[0] / ms_per_frame)

        if num_frames == 0:
            pad_len = frame_width - ms_label.shape[0]
            ms_label = torch.nn.functional.pad(ms_label, (0, 0, 0, pad_len))
            num_frames = 1

        trimmed = ms_label[:num_frames * frame_width]
        grouped = trimmed.view(num_frames, frame_width, -1)
        class_sums = grouped.sum(dim=1)
        frame_label = (class_sums == class_sums.max(dim=-1, keepdim=True).values).float()

        return frame_label


class CQTFrameDataset(AudioFrameDataset, CQTDataset):
    def __init__(self, data_dir, label_dir, num_classes=4, sr=16000, channels='mono', window=20,
                 hop_length=512, n_bins=84, bins_per_octave=12, margin_ratio=1.6, is_valid=False, aug=False):
        AudioFrameDataset.__init__(self, data_dir, label_dir, num_classes, sr, channels, window, margin_ratio, is_valid, aug)
        CQTDataset.__init__(self, sr=sr, hop_length=hop_length, n_bins=n_bins, bins_per_octave=bins_per_octave, is_valid=is_valid, aug=aug)
        self.window_frame = window * sr // hop_length
        self.loaded_data = self._preload_cqt(self.loaded_data)
        if self.aug and hasattr(self, 'pitch_shifted_audio') and self.pitch_shifted_audio:
            self.pitch_shifted_cqt = self._preload_pitch_shifted_cqt()

    def _audio_to_cqt(self, audio):
        """Compute CQT magnitude without augmentation."""
        cqt = self.cqt_transform(audio)  # (batch, n_bins, frames)
        cqt = cqt.squeeze(0)             # (n_bins, frames)
        cqt = torch.log1p(cqt * 1000)   # log compression
        return cqt

    def _preload_cqt(self, path_dict):
        cqt_dict = {}
        for hash_key, audio_path in tqdm(path_dict.items(), desc='Preload CQT Spectrograms'):
            audio = self._load_audio(audio_path)
            cqt_dict[hash_key] = self._audio_to_cqt(audio)
        return cqt_dict

    def _preload_pitch_shifted_cqt(self):
        cqt_dict = {}
        for hash_key, audio_list in tqdm(self.pitch_shifted_audio.items(), desc='Preload Pitch-Shifted CQT'):
            cqt_dict[hash_key] = [self._audio_to_cqt(a) for a in audio_list]
        del self.pitch_shifted_audio
        return cqt_dict

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        if self.is_valid:
            hash_key, start_sample, end_sample = self.val_segments[idx]
            cqt_full = self.loaded_data[hash_key]
            start_frame = start_sample // self.hop_length
            end_frame = end_sample // self.hop_length
            cqt = cqt_full[:, start_frame:end_frame]
            if cqt.shape[-1] < self.window_frame:
                cqt = torch.nn.functional.pad(cqt, (0, self.window_frame - cqt.shape[-1]))
            start_ms = int(start_sample / self.sr * 1000)
            end_ms = int(end_sample / self.sr * 1000)
            label = self.loaded_label[hash_key][start_ms:end_ms]
            frame_label = self.ms_to_frame_label(label)
            if cqt.shape[-1] != frame_label.shape[0]: cqt = cqt[:, :frame_label.shape[0]]
            return hash_key, cqt, frame_label

        hash_key = self.training_instances[idx]

        if self.aug and hasattr(self, 'pitch_shifted_cqt') and hash_key in self.pitch_shifted_cqt and random.random() < 0.5:
            cqt_full = random.choice(self.pitch_shifted_cqt[hash_key])
        else:
            cqt_full = self.loaded_data[hash_key]

        total_frames = cqt_full.shape[1]
        window_frames = self.window_frame
        margin_frames = int(self.margin_ratio * window_frames)

        if total_frames > window_frames:
            start_frame = random.randint(0, total_frames - window_frames)
        else:
            start_frame = 0
        end_frame = start_frame + window_frames

        if start_frame - margin_frames // 2 < 0:
            end_frame = margin_frames
            start_frame = 0
        elif end_frame + margin_frames // 2 > total_frames:
            end_frame = total_frames
            start_frame = max(0, total_frames - margin_frames)
        else:
            start_frame = start_frame - margin_frames // 2
            end_frame = end_frame + margin_frames // 2

        cqt = cqt_full[:, start_frame:end_frame].clone()

        if self.aug:
            cqt, stretch_factor = self._apply_cqt_time_stretch(cqt)
        else:
            stretch_factor = 1.0

        start_ms = int(start_frame * self.hop_length / self.sr * 1000)
        end_ms = int(end_frame * self.hop_length / self.sr * 1000)
        label = self.loaded_label[hash_key][start_ms:end_ms]

        center_start = (cqt.shape[1] - window_frames) // 2
        cqt = cqt[:, center_start:center_start + window_frames]
        if self.aug: cqt = self.apply_spec_augmentation(cqt)

        label = self.apply_time_stretch(label, stretch_factor)
        frame_label = self.ms_to_frame_label(label)
        frame_label = frame_label[center_start:center_start + window_frames]
        assert frame_label.shape[0] == cqt.shape[1], f"frame_label.shape[0] != cqt.shape[1]: {frame_label.shape[0]} != {cqt.shape[1]}, time stretch factor: {stretch_factor}"
        return hash_key, cqt, frame_label

class ChromaDataset():
    def __init__(self, sr=16000, n_fft=2048, hop_length=512, n_chroma=12, is_valid=False, aug=False):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_chroma = n_chroma
        self.window_frame = None  # set by subclass via window param
        self.is_valid = is_valid
        self.aug = aug

        self.spec_cvt = Spectrogram(n_fft=n_fft, hop_length=hop_length, power=1.0)

        # Build chroma filter bank and register as a buffer-style tensor
        chroma_weights = chroma(sr=sr, n_fft=n_fft, n_chroma=n_chroma)  # (n_chroma, n_fft//2+1)
        self.chroma_weights = torch.from_numpy(chroma_weights)  # keep on CPU; move in _audio_to_chroma

        if self.aug:
            self.freq_mask = FrequencyMasking(freq_mask_param=4)  # n_chroma=12 so small param


    def _apply_chroma_time_stretch(self, chroma_gram):
        """Approximate time stretch on chromagram via bilinear interpolation."""
        stretch_factor = 1.0
        if random.random() < 0.3:
            stretch_factor = random.uniform(0.7, 1.5)
            new_len = int(chroma_gram.shape[-1] / stretch_factor)
            if new_len > 0:
                chroma_gram = torch.nn.functional.interpolate(
                    chroma_gram.unsqueeze(0).unsqueeze(0),
                    scale_factor=(1.0, 1.0 / stretch_factor),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)
        return chroma_gram, stretch_factor


    def apply_spec_pitch_shift(self, chroma_gram):
        """Roll along the chroma axis — musically meaningful (circular pitch class shift)."""
        if random.random() < 0.3:
            shift_steps = random.randint(-3, 3)
            if shift_steps == 0: return chroma_gram
            chroma_gram = torch.roll(chroma_gram, shifts=shift_steps, dims=-2)
        return chroma_gram


    def apply_spec_augmentation(self, chroma_gram):
        if not self.aug or random.random() > 0.8: return chroma_gram
        if random.random() < 0.5: chroma_gram = self.apply_spec_pitch_shift(chroma_gram)
        if random.random() < 0.5: chroma_gram = self.freq_mask(chroma_gram)
        return chroma_gram


    def apply_time_stretch(self, label, stretch_factor):
        if stretch_factor == 1.0: return label
        orig_length = label.shape[0]
        new_length = int(orig_length / stretch_factor)
        stretched_label = torch.zeros((new_length, label.shape[1]))
        if new_length > 0:
            orig_indices = torch.linspace(0, orig_length - 1, new_length).long()
            stretched_label[:new_length] = label[orig_indices]
        return stretched_label


    def ms_to_frame_label(self, ms_label):
        ms_per_frame = self.hop_length / self.sr * 1000
        frame_width = int(ms_per_frame)
        num_frames = int(ms_label.shape[0] / ms_per_frame)

        if num_frames == 0:
            pad_len = frame_width - ms_label.shape[0]
            ms_label = torch.nn.functional.pad(ms_label, (0, 0, 0, pad_len))
            num_frames = 1

        trimmed = ms_label[:num_frames * frame_width]
        grouped = trimmed.view(num_frames, frame_width, -1)
        class_sums = grouped.sum(dim=1)
        frame_label = (class_sums == class_sums.max(dim=-1, keepdim=True).values).float()
        return frame_label


class ChromaFrameDataset(AudioFrameDataset, ChromaDataset):
    def __init__(self, data_dir, label_dir, num_classes=4, sr=16000, channels='mono', window=20,
                 n_fft=2048, hop_length=512, n_chroma=12, margin_ratio=1.6, is_valid=False, aug=False):
        AudioFrameDataset.__init__(self, data_dir, label_dir, num_classes, sr, channels, window, margin_ratio, is_valid, aug)
        ChromaDataset.__init__(self, sr=sr, n_fft=n_fft, hop_length=hop_length, n_chroma=n_chroma, is_valid=is_valid, aug=aug)
        self.window_frame = window * sr // hop_length
        self.loaded_data = self._preload_chroma(self.loaded_data)
        if self.aug and hasattr(self, 'pitch_shifted_audio') and self.pitch_shifted_audio:
            self.pitch_shifted_chroma = self._preload_pitch_shifted_chroma()

    def _audio_to_chroma(self, audio):
        """Compute chromagram: STFT magnitude → chroma filter → log compression."""
        spec = self.spec_cvt(audio)                                   # (1, n_fft//2+1, frames)
        spec = spec.squeeze(0)                                        # (n_fft//2+1, frames)
        weights = self.chroma_weights.to(spec.device)                 # (n_chroma, n_fft//2+1)
        chroma_gram = torch.matmul(weights, spec)                     # (n_chroma, frames)
        chroma_gram = torch.log1p(chroma_gram * 1000)
        return chroma_gram

    def _preload_chroma(self, path_dict):
        chroma_dict = {}
        for hash_key, audio_path in tqdm(path_dict.items(), desc='Preload Chromagrams'):
            audio = self._load_audio(audio_path)
            chroma_dict[hash_key] = self._audio_to_chroma(audio)
        return chroma_dict

    def _preload_pitch_shifted_chroma(self):
        chroma_dict = {}
        for hash_key, audio_list in tqdm(self.pitch_shifted_audio.items(), desc='Preload Pitch-Shifted Chroma'):
            chroma_dict[hash_key] = [self._audio_to_chroma(a) for a in audio_list]
        del self.pitch_shifted_audio
        return chroma_dict

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        if self.is_valid:
            hash_key, start_sample, end_sample = self.val_segments[idx]
            chroma_full = self.loaded_data[hash_key]
            start_frame = start_sample // self.hop_length
            end_frame = end_sample // self.hop_length
            chroma_gram = chroma_full[:, start_frame:end_frame]
            if chroma_gram.shape[-1] < self.window_frame:
                chroma_gram = torch.nn.functional.pad(chroma_gram, (0, self.window_frame - chroma_gram.shape[-1]))
            start_ms = int(start_sample / self.sr * 1000)
            end_ms = int(end_sample / self.sr * 1000)
            label = self.loaded_label[hash_key][start_ms:end_ms]
            frame_label = self.ms_to_frame_label(label)
            if chroma_gram.shape[-1] != frame_label.shape[0]: chroma_gram = chroma_gram[:, :frame_label.shape[0]]
            return hash_key, chroma_gram, frame_label

        hash_key = self.training_instances[idx]

        if self.aug and hasattr(self, 'pitch_shifted_chroma') and hash_key in self.pitch_shifted_chroma and random.random() < 0.5:
            chroma_full = random.choice(self.pitch_shifted_chroma[hash_key])
        else:
            chroma_full = self.loaded_data[hash_key]

        total_frames = chroma_full.shape[1]
        window_frames = self.window_frame
        margin_frames = int(self.margin_ratio * window_frames)

        if total_frames > window_frames:
            start_frame = random.randint(0, total_frames - window_frames)
        else:
            start_frame = 0
        end_frame = start_frame + window_frames

        if start_frame - margin_frames // 2 < 0:
            end_frame = margin_frames
            start_frame = 0
        elif end_frame + margin_frames // 2 > total_frames:
            end_frame = total_frames
            start_frame = max(0, total_frames - margin_frames)
        else:
            start_frame = start_frame - margin_frames // 2
            end_frame = end_frame + margin_frames // 2

        chroma_gram = chroma_full[:, start_frame:end_frame].clone()

        if self.aug:
            chroma_gram, stretch_factor = self._apply_chroma_time_stretch(chroma_gram)
        else:
            stretch_factor = 1.0

        start_ms = int(start_frame * self.hop_length / self.sr * 1000)
        end_ms = int(end_frame * self.hop_length / self.sr * 1000)
        label = self.loaded_label[hash_key][start_ms:end_ms]

        center_start = (chroma_gram.shape[1] - window_frames) // 2
        chroma_gram = chroma_gram[:, center_start:center_start + window_frames]
        if self.aug: chroma_gram = self.apply_spec_augmentation(chroma_gram)

        label = self.apply_time_stretch(label, stretch_factor)
        frame_label = self.ms_to_frame_label(label)
        frame_label = frame_label[center_start:center_start + window_frames]
        assert frame_label.shape[0] == chroma_gram.shape[1], f"frame_label.shape[0] != chroma_gram.shape[1]: {frame_label.shape[0]} != {chroma_gram.shape[1]}, time stretch factor: {stretch_factor}"
        return hash_key, chroma_gram, frame_label


class PitchDataset(BaseDataset):
    def __init__(self, data_dir, label_dir, num_classes=4, sr=100, frame_rate=20, threshold=0.8, window=20, margin_ratio=1, is_valid=False, aug=False):
        super().__init__(data_dir, label_dir, num_classes, sr, window, margin_ratio, is_valid, aug)
        self.threshold = threshold
        self.frame_rate = frame_rate
        assert 100 % self.frame_rate == 0
        self.comp_ratio = self.sr // self.frame_rate
        self.window_frame = self.window * frame_rate

    def frequency_to_midi(self, frequency):
        return 69 + 12 * math.log2(frequency / 440)

    def shift_contour(self, contour):
        shift = (random.random() * 12 - 6) / 12
        contour[0,:] += shift

        return contour

    def _load_norm_contour(self, csv_file):
        contour_df = pd.read_csv(csv_file)
        frequency, confidence = contour_df['frequency'].values, contour_df['confidence'].values
        midi = [self.frequency_to_midi(freq) for freq in frequency]

        tonic_counter = Counter(np.round(midi)[confidence >= self.threshold]).most_common(1)
        tonic = tonic_counter[0][0]

        norm_midi = [(mi-float(tonic))/12 for mi in midi]
        freq_conf = torch.tensor(np.stack([norm_midi, confidence], axis=0), dtype=torch.float32)

        return freq_conf


    @abstractmethod
    def get_data(self): return None

    # @abstractmethod
    # def get_split(self): return None

    @abstractmethod
    def __len__(self): return None

    @abstractmethod
    def __getitem__(self, idx): return None


class PitchFrameDataset(PitchDataset):
    def __init__(self, data_dir, label_dir, num_classes=4, sr=100, frame_rate=20, threshold=0.8, window=20, margin_ratio=1, is_valid=False, aug=False):
        super().__init__(data_dir, label_dir, num_classes, sr, frame_rate, threshold, window, margin_ratio, is_valid, aug)

        self.loaded_data = self.get_data()
        self.loaded_label = self.get_frame_label()

        for hash_key in set(self.loaded_label.keys()) - set(self.loaded_data.keys()): self.loaded_hash.remove(hash_key)
        self.loaded_label = {key:self.ms_to_frame_label(val) for key, val in self.loaded_label.items()}

        self.training_instances = []
        self.val_segments = []
        self.prepare_training_instances()


    def get_data(self):
        loaded_data = {}
        for csv_file in tqdm(list(self.data_dir.rglob('*.csv')), desc="Load Contours"):
            hash_key = unicodedata.normalize('NFC', csv_file.name.split("-")[0])
            if hash_key not in self.loaded_hash: continue

            freq_conf = self._load_norm_contour(csv_file)

            loaded_data[hash_key] = freq_conf[:,::self.comp_ratio]

        return loaded_data


    def prepare_training_instances(self, target_hash_keys=None):
        self.training_instances = []
        window_frames = self.window * self.frame_rate

        for hash_key in list(self.loaded_hash):
            if hash_key not in self.loaded_data:
                self.loaded_hash.remove(hash_key)
                continue
            if target_hash_keys is not None and hash_key not in target_hash_keys:
                continue

            contour_length = self.loaded_data[hash_key].shape[1]
            num_repeats = max(1, contour_length // window_frames)
            self.training_instances.extend([hash_key] * num_repeats)

    def update_training_instances(self, target_hash_keys):
        self.prepare_training_instances(target_hash_keys)

    def prepare_val_segments(self, target_hash_keys):
        val_segments = []
        window_frames = self.window * self.frame_rate
        for hash_key in target_hash_keys:
            if hash_key not in self.loaded_data:
                continue
            contour_length = self.loaded_data[hash_key].shape[1]
            for start in range(0, contour_length, window_frames):
                val_segments.append((hash_key, start, min(start + window_frames, contour_length)))
        return val_segments

    def compose_validset(self, target_hash_keys):
        validset = copy(self)
        validset.is_valid = True
        validset.loaded_hash = [k for k in target_hash_keys if k in self.loaded_data]
        validset.loaded_data = {k: self.loaded_data[k] for k in validset.loaded_hash}
        validset.loaded_label = {k: self.loaded_label[k] for k in validset.loaded_hash}
        validset.val_segments = self.prepare_val_segments(validset.loaded_hash)
        return validset


    def get_split(self, target_hash_keys, split='train'):
        if split == 'train':
            self.update_training_instances(target_hash_keys)
        else:
            return self.compose_validset(target_hash_keys)


    def ms_to_frame_label(self, ms_label):
        ms_per_frame = self.comp_ratio * 10
        num_frames = int(ms_label.shape[0]/ms_per_frame)
        frame_label = torch.zeros((num_frames, self.num_classes))

        for frame_idx in range(num_frames):
            frame_start_ms = int(frame_idx * ms_per_frame)
            frame_end_ms = int((frame_idx + 1) * ms_per_frame)

            if frame_start_ms >= ms_label.shape[0]: frame_label[frame_idx, 0] = 1
            else:
                frame_end_ms = min(frame_end_ms, ms_label.shape[0])
                ms_segment = ms_label[frame_start_ms:frame_end_ms]

                if ms_segment.shape[0] > 0:
                    class_sums = ms_segment.sum(dim=0)
                    frame_label[frame_idx] = (class_sums == class_sums.max()).float()

                else: frame_label[frame_idx, 0] = 1

        return frame_label


    def __len__(self):
        if self.is_valid: return len(self.val_segments)
        return len(self.training_instances)


    def __getitem__(self, idx):
        window_frames = self.window * self.frame_rate

        if self.is_valid:
            hash_key, start, end = self.val_segments[idx]
            contour, label = self.loaded_data[hash_key], self.loaded_label[hash_key]
            contour_slice = contour[:, start:end]
            label_slice = label[start:end]
            if contour_slice.shape[1] < window_frames:
                pad = window_frames - contour_slice.shape[1]
                contour_slice = torch.nn.functional.pad(contour_slice, (0, pad))
                pad_label = torch.zeros((pad, label_slice.shape[1]))
                pad_label[:, 0] = 1
                label_slice = torch.cat([label_slice, pad_label], dim=0)
            return hash_key, contour_slice, label_slice

        hash_key = self.training_instances[idx]
        contour, label = self.loaded_data[hash_key], self.loaded_label[hash_key]
        if contour.shape[1] > label.shape[0]: contour = contour[:, :label.shape[0]]
        elif contour.shape[1] < label.shape[0]: label = label[:contour.shape[1]]

        contour_length = contour.shape[1]
        if contour_length > window_frames:
            start = random.randint(0, contour_length - window_frames)
        else:
            start = 0
        end = start + window_frames

        margin_length = int(self.margin_ratio * window_frames)
        if start - margin_length // 2 < 0:
            end = margin_length
            start = 0
        elif end + margin_length // 2 > contour_length:
            end = contour_length
            start = max(0, contour_length - margin_length)
        else:
            start, end = start - margin_length // 2, end + margin_length // 2

        contour = contour[:, start:end]
        label = label[start:end]

        start_idx = (contour.shape[1] - window_frames) // 2
        contour = contour[:, start_idx:start_idx + window_frames]
        label = label[start_idx:start_idx + window_frames]

        if self.aug and random.random() < 0.5:
            contour = self.shift_contour(contour)

        assert label.shape[0] == contour.shape[1], f"label.shape[0] != contour.shape[1]: {label.shape[0]} != {contour.shape[1]}"

        return hash_key, contour, label


class PitchSegmentDataset(PitchDataset):
    def __init__(self, data_dir, label_dir, num_classes=3, sr=100, frame_rate=20, threshold=0.8, window=20, margin_ratio=1, is_valid=False, aug=False):
        super().__init__(data_dir, label_dir, num_classes, sr, frame_rate, threshold, window, margin_ratio, is_valid, aug)
        self.loaded_all = self.get_data()
        self.loaded_data, self.loaded_label, self.loaded_meta = self.loaded_all
        for hash_key in set(self.loaded_hash) - set([m[0] for m in self.loaded_meta]): self.loaded_hash.remove(hash_key)

    def frequency_to_midi(self, frequency):
        # Convert frequency to MIDI note
        return 69 + 12 * math.log2(frequency / 440)


    def get_data(self):
        loaded_data, loaded_label, loaded_meta = [], [], []
        for csv_file in tqdm(list(self.data_dir.rglob('*.csv')), desc="Load Contours"):
            hash_key = unicodedata.normalize('NFC', csv_file.name.split("-")[0])
            if hash_key not in self.loaded_hash: continue

            hash_df = self.df[self.df['hash_key']==hash_key]
            segment_meta = hash_df[['start', 'end','label']].to_numpy()

            freq_conf = self._load_norm_contour(csv_file)

            for start_ms, end_ms, label in segment_meta:
                start, end = int(start_ms*self.sr/1000), int(end_ms*self.sr/1000)
                segment = freq_conf[:, start:end]
                segment = segment[:,::self.comp_ratio]
                label = self.label_map[label]

                loaded_data.append(segment)
                loaded_label.append(label)
                loaded_meta.append((hash_key, start, end))

        return loaded_data, loaded_label, loaded_meta


    def compose_validset(self, target_indices):
        loaded_data, loaded_label, loaded_meta = [], [], []

        for contour, label, (hash_key, start, end) in zip(self.loaded_all[0], self.loaded_all[1], self.loaded_all[2]):
            if hash_key not in target_indices: continue
            seg_len, window_len = contour.shape[-1], self.window_frame

            if seg_len > window_len:
                current_pos = 0
                while current_pos + window_len <= seg_len:
                    start_seg, end_seg = current_pos, current_pos+window_len
                    segment = contour[:, start_seg:end_seg]
                    current_pos += window_len

                    loaded_data.append(segment)
                    loaded_label.append(label)
                    loaded_meta.append((hash_key, start+start_seg, start+end_seg))

                if current_pos < seg_len:
                    final_start = seg_len - window_len
                    segment = contour[:, final_start:seg_len]

                    loaded_data.append(segment)
                    loaded_label.append(label)
                    loaded_meta.append((hash_key, final_start, seg_len))

            else:
                segment = contour[:, :]
                loaded_data.append(segment)
                loaded_label.append(label)
                loaded_meta.append((hash_key, start, end))

        return loaded_data, loaded_label, loaded_meta


    def get_split(self, target_hash_keys, split='train'):
        if split == 'train':
            target_indices = [idx for idx, tup in enumerate(self.loaded_all[2]) if tup[0] in target_hash_keys]
            self.loaded_data = [self.loaded_all[0][idx] for idx in target_indices]
            self.loaded_label = [self.loaded_all[1][idx] for idx in target_indices]
            self.loaded_meta = [self.loaded_all[2][idx] for idx in target_indices]
        else:
            validset = copy(self)  # shallow copy — no data duplication
            validset.is_valid = True
            validset.loaded_data, validset.loaded_label, validset.loaded_meta = self.compose_validset(target_hash_keys)
            return validset


    def __len__(self):
        return len(self.loaded_data)


    def __getitem__(self, idx):
        contour, label = self.loaded_data[idx], self.loaded_label[idx]
        hash_key, start, end = self.loaded_meta[idx]

        contour_len = contour.shape[-1]

        if contour_len > self.window_frame:
            start = random.randint(0, min(contour_len, contour_len - self.window_frame))
            contour = contour[:,start:start+self.window_frame]

        else:
            num_pad = self.window_frame - contour_len
            l_pad = num_pad // 2
            r_pad = num_pad - l_pad
            contour = torch.nn.functional.pad(contour, (l_pad, r_pad), mode='constant', value=0)

        if self.aug and not self.is_valid and random.random() < 0.5:
            contour = self.shift_contour(contour)

        return hash_key, contour, label


class CMERTFrameDataset(AudioFrameDataset):
    def __init__(self,
                 data_dir,
                 label_dir,
                 num_classes=4,
                 sr=24000,
                 channels='mono',
                 window=30,
                 margin_ratio=1,
                 is_valid=False,
                 aug=False):
        super().__init__(data_dir, label_dir, num_classes, sr, channels, window, margin_ratio, is_valid, aug)
        self.hop_length = 320  # MERT CNN hop (fixed at 320 samples regardless of sr → 75fps at 24kHz)
        self.len_window = self.sr * self.window
        self.len_frames = int(self.window / (self.hop_length / self.sr) - 1)  # 30/(320/24000)-1 = 2249

    def ms_to_frame(self, ms_label):
        ms_per_frame = self.hop_length / self.sr * 1000
        num_frames = int(ms_label.shape[0] / ms_per_frame) - 1

        frame_label = torch.zeros((num_frames, self.num_classes))
        for frame_idx in range(num_frames):
            frame_start_ms = int(frame_idx * ms_per_frame)
            frame_end_ms = int((frame_idx + 1) * ms_per_frame)

            if frame_start_ms >= ms_label.shape[0]: frame_label[frame_idx, 0] = 1
            else:
                frame_end_ms = min(frame_end_ms, ms_label.shape[0])
                ms_segment = ms_label[frame_start_ms:frame_end_ms]

                if ms_segment.shape[0] > 0:
                    class_sums = ms_segment.sum(dim=0)
                    frame_label[frame_idx] = (class_sums == class_sums.max()).float()
                else: frame_label[frame_idx, 0] = 1

        return frame_label

    def __getitem__(self, idx):
        window_samples = self.len_window

        if self.is_valid:
            hash_key, start, end = self.val_segments[idx]
            audio = self._load_audio(self.loaded_data[hash_key])
            audio_slice = audio[:, start:end]
            if audio_slice.shape[1] < window_samples:
                pad = window_samples - audio_slice.shape[1]
                audio_slice = torch.nn.functional.pad(audio_slice, (0, pad))
            start_ms, end_ms = int(start / self.sr * 1000), int(end / self.sr * 1000)
            ms_label = self.loaded_label[hash_key][start_ms:end_ms]
            window_ms = self.window * 1000
            if ms_label.shape[0] < window_ms:
                pad_ms = window_ms - ms_label.shape[0]
                pad = torch.zeros(pad_ms, self.num_classes)
                pad[:, 0] = 1  # Unknown
                ms_label = torch.cat([ms_label, pad], dim=0)
            frame_label = self.ms_to_frame(ms_label)
            return hash_key, audio_slice.squeeze(0), frame_label

        hash_key = self.training_instances[idx]

        if self.aug and hasattr(self, 'pitch_shifted_audio') and hash_key in self.pitch_shifted_audio and random.random() < 0.5:
            audio = random.choice(self.pitch_shifted_audio[hash_key])
        else:
            audio = self._load_audio(self.loaded_data[hash_key])

        audio_length = audio.shape[1]
        if audio_length > window_samples:
            start = random.randint(0, audio_length - window_samples)
        else:
            start = 0
        end = start + window_samples

        audio_slice = audio[:, start:end]
        if audio_slice.shape[1] < window_samples:
            audio_slice = torch.nn.functional.pad(audio_slice, (0, window_samples - audio_slice.shape[1]))

        if self.aug: audio_slice = self.apply_audio_augmentation(audio_slice)

        start_ms, end_ms = int(start / self.sr * 1000), int(end / self.sr * 1000)
        ms_label = self.loaded_label[hash_key][start_ms:end_ms]
        window_ms = self.window * 1000
        if ms_label.shape[0] < window_ms:
            label_pad = torch.zeros(window_ms - ms_label.shape[0], self.num_classes)
            label_pad[:, 0] = 1  # Unknown
            ms_label = torch.cat([ms_label, label_pad], dim=0)
        frame_label = self.ms_to_frame(ms_label)

        return hash_key, audio_slice.squeeze(0), frame_label



class MidiFrameDataset(BaseDataset):
    """Frame-level dataset over transcribed-MIDI piano rolls.

    Ported from the PansoriMIDIDetection repo so MIDI is a peer of the audio
    modalities. Structurally this mirrors PitchFrameDataset — a non-audio,
    preloaded-array dataset keyed by hash_key — with a [128, T] piano roll in
    place of the [2, T] pitch contour.

    Two things are deliberately preserved from the original implementation so
    that checkpoints trained there keep loading and scoring the same:

      * Piano roll values stay raw pretty_midi velocities (0-127). They are
        NOT normalised; the trained weights expect that input range.
      * Training crops are a plain uniform random window (no margin_ratio
        expansion), matching the original sampler.

    Eval windows are exact `window` seconds. The original snapped window
    boundaries to silence (+/-1s) to avoid cutting notes; that is intentionally
    dropped here so MIDI frames line up with the other modalities for
    soft-voting ensembles.
    """

    def __init__(self, data_dir, label_dir, num_classes=5, fs=10, window=30,
                 margin_ratio=1, is_valid=False, aug=False):
        # BaseDataset's `sr` is the label sampling rate concept; for MIDI the
        # piano roll frame rate `fs` plays that role.
        super().__init__(data_dir, label_dir, num_classes, fs, window, margin_ratio, is_valid, aug)
        self.fs = fs
        self.window_frame = self.window * self.fs

        self.loaded_data = self.get_data()
        self.loaded_label = self.get_frame_label()

        for hash_key in set(self.loaded_label.keys()) - set(self.loaded_data.keys()):
            self.loaded_hash.remove(hash_key)
        self.loaded_label = {key: self.ms_to_frame_label(val) for key, val in self.loaded_label.items()}

        # Piano roll length (ceil(duration*fs)) and label length (duration_ms //
        # ms_per_frame) disagree by a few frames on almost every song. Trim both
        # to the shorter here, at load time -- not just in __getitem__ -- because
        # the ensemble scripts read loaded_data/loaded_label directly and would
        # otherwise concatenate mismatched prediction/GT arrays.
        for hash_key in list(self.loaded_data.keys()):
            if hash_key not in self.loaded_label:
                continue
            T = min(self.loaded_data[hash_key].shape[1], self.loaded_label[hash_key].shape[0])
            self.loaded_data[hash_key]  = self.loaded_data[hash_key][:, :T]
            self.loaded_label[hash_key] = self.loaded_label[hash_key][:T]

        self.training_instances = []
        self.val_segments = []
        self.prepare_training_instances()


    def get_data(self):
        # Imported lazily so pretty_midi stays optional for the audio modalities.
        import pretty_midi

        loaded_data = {}
        for midi_file in tqdm(sorted(self.data_dir.rglob('*.mid')), desc='Load Piano Rolls'):
            hash_key = unicodedata.normalize('NFC', midi_file.name.split("-")[0])
            if hash_key not in self.loaded_hash: continue

            roll = pretty_midi.PrettyMIDI(str(midi_file)).get_piano_roll(fs=self.fs)
            loaded_data[hash_key] = torch.tensor(roll, dtype=torch.float32)  # [128, T], raw velocity

        return loaded_data


    def pitch_shift(self, piano_roll, shift_range=(-3, 3)):
        shift = random.randint(*shift_range)
        if shift == 0: return piano_roll
        shifted = torch.roll(piano_roll, shift, dims=0)
        if shift > 0: shifted[:shift, :] = 0
        else: shifted[shift:, :] = 0
        return shifted


    def time_masking(self, piano_roll, total_mask_sec=5.0, num_masks=3):
        pr = piano_roll.clone()
        T = pr.shape[1]
        max_per_mask = int(total_mask_sec * self.fs) // num_masks
        if max_per_mask < 1: return pr

        for _ in range(num_masks):
            mask_len = random.randint(1, max_per_mask)
            if T <= mask_len: continue
            start = random.randint(0, T - mask_len)
            pr[:, start:start + mask_len] = 0
        return pr


    def prepare_training_instances(self, target_hash_keys=None):
        self.training_instances = []

        for hash_key in list(self.loaded_hash):
            if hash_key not in self.loaded_data:
                self.loaded_hash.remove(hash_key)
                continue
            if target_hash_keys is not None and hash_key not in target_hash_keys:
                continue

            roll_length = self.loaded_data[hash_key].shape[1]
            num_repeats = max(1, roll_length // self.window_frame)
            self.training_instances.extend([hash_key] * num_repeats)

    def update_training_instances(self, target_hash_keys):
        self.prepare_training_instances(target_hash_keys)

    def prepare_val_segments(self, target_hash_keys):
        val_segments = []
        for hash_key in target_hash_keys:
            if hash_key not in self.loaded_data: continue
            roll_length = self.loaded_data[hash_key].shape[1]
            for start in range(0, roll_length, self.window_frame):
                val_segments.append((hash_key, start, min(start + self.window_frame, roll_length)))
        return val_segments

    def compose_validset(self, target_hash_keys):
        validset = copy(self)
        validset.is_valid = True
        validset.loaded_hash = [k for k in target_hash_keys if k in self.loaded_data]
        validset.loaded_data = {k: self.loaded_data[k] for k in validset.loaded_hash}
        validset.loaded_label = {k: self.loaded_label[k] for k in validset.loaded_hash}
        validset.val_segments = self.prepare_val_segments(validset.loaded_hash)
        return validset

    def get_split(self, target_hash_keys, split='train'):
        if split == 'train':
            self.update_training_instances(target_hash_keys)
        else:
            return self.compose_validset(target_hash_keys)


    def ms_to_frame_label(self, ms_label):
        # 1000/fs ms per frame (100ms at fs=10). Vectorised majority vote,
        # same construction as MelDataset.ms_to_frame_label.
        ms_per_frame = 1000 // self.fs
        num_frames = int(ms_label.shape[0] / ms_per_frame)

        if num_frames == 0:
            pad_len = ms_per_frame - ms_label.shape[0]
            ms_label = torch.nn.functional.pad(ms_label, (0, 0, 0, pad_len))
            num_frames = 1

        trimmed = ms_label[:num_frames * ms_per_frame]
        grouped = trimmed.view(num_frames, ms_per_frame, -1)
        class_sums = grouped.sum(dim=1)
        return (class_sums == class_sums.max(dim=-1, keepdim=True).values).float()


    def __len__(self):
        if self.is_valid: return len(self.val_segments)
        return len(self.training_instances)


    def _pad_to_window(self, roll_slice, label_slice):
        if roll_slice.shape[1] < self.window_frame:
            pad = self.window_frame - roll_slice.shape[1]
            roll_slice = torch.nn.functional.pad(roll_slice, (0, pad))
            pad_label = torch.zeros((pad, label_slice.shape[1]))
            pad_label[:, self.label_map['Unknown']] = 1
            label_slice = torch.cat([label_slice, pad_label], dim=0)
        return roll_slice, label_slice


    def __getitem__(self, idx):
        if self.is_valid:
            hash_key, start, end = self.val_segments[idx]
            roll, label = self.loaded_data[hash_key], self.loaded_label[hash_key]
            roll_slice, label_slice = self._pad_to_window(roll[:, start:end], label[start:end])
            return hash_key, roll_slice, label_slice

        hash_key = self.training_instances[idx]
        roll, label = self.loaded_data[hash_key], self.loaded_label[hash_key]

        # Piano roll length and label length can differ by a frame or two after
        # the ms->frame majority vote; trim both to the shorter.
        T = min(roll.shape[1], label.shape[0])
        roll, label = roll[:, :T], label[:T]

        start = random.randint(0, T - self.window_frame) if T > self.window_frame else 0
        roll_slice, label_slice = self._pad_to_window(
            roll[:, start:start + self.window_frame], label[start:start + self.window_frame])

        if self.aug:
            if random.random() < 0.5: roll_slice = self.pitch_shift(roll_slice)
            if random.random() < 0.5: roll_slice = self.time_masking(roll_slice)

        assert label_slice.shape[0] == roll_slice.shape[1], \
            f"label {label_slice.shape[0]} != roll {roll_slice.shape[1]}"

        return hash_key, roll_slice, label_slice
