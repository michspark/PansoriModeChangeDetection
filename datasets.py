import os
import math
import json
import random
import unicodedata
from pathlib import Path
from copy import deepcopy
from abc import abstractmethod
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import TimeStretch, FrequencyMasking
from torchaudio.transforms import Spectrogram, MelScale, AmplitudeToDB



class BaseDataset(Dataset):
    def __init__(self, data_dir, label_dir, num_classes=4, sr=16000, window=20, margin_ratio=1, is_valid=False, aug=False):
        self.sr = sr
        self.window = window

        self.num_classes = num_classes
        self.label_map = {"창조":0, "아니리":0, "설렁제":1, "경드름":1, "우조":1, "평조":1, "계면조": 2, "Unknown":3}

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
    def __init__(self, data_dir, label_dir, num_classes=4, sr=16000, channels='mono', window=20, margin_ratio=1.0, is_valid=False, aug=False):
        super().__init__(data_dir, label_dir, num_classes, sr, window, margin_ratio, is_valid, aug)
        self.channels = channels

        self.pitch_shift_dir = os.path.join(os.path.dirname(data_dir), "PitchShiftedAudio")
        if self.aug and not self.is_valid and os.path.exists(self.pitch_shift_dir): self.pitch_shifted_audio = self.load_pitch_shifted_audio()


    def _load_audio(self, audio_file):
        audio, sr = torchaudio.load(audio_file)
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.sr) if self.sr!=sr else audio
        audio = audio.mean(dim=0, keepdim=True) if self.channels=='mono' else audio
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

        self.slice_indices = []
        self.prepare_slice_indices()


    def get_data(self):
        loaded_data = {}
        for audio_file in tqdm(list(self.data_dir.rglob('*.wav')), desc='Load Audio'):
            hash_key = unicodedata.normalize('NFC', audio_file.name.split("-")[0])
            if hash_key not in self.loaded_hash: continue

            audio = self._load_audio(audio_file)
            loaded_data[hash_key] = audio
        
        return loaded_data


    def prepare_slice_indices(self, target_hash_keys=None, random_offset=True):
        self.slice_indices = []
        window_samples = self.window * self.sr
        
        for hash_key in self.loaded_hash:
            if hash_key not in self.loaded_data.keys():
                self.loaded_hash.remove(hash_key)
                continue

            if target_hash_keys is not None and hash_key not in target_hash_keys: continue

            audio = self.loaded_data[hash_key]
            audio_length = audio.shape[1]
            
            if audio_length < window_samples: continue
            offset = random.randint(0, min(window_samples, audio_length - window_samples)) if random_offset else 0
            
            current_pos = offset
            while current_pos + window_samples <= audio_length:
                self.slice_indices.append((hash_key, current_pos, current_pos + window_samples))
                current_pos += window_samples

            if current_pos < audio_length and audio_length - window_samples >= 0:
                final_start = audio_length - window_samples
                self.slice_indices.append((hash_key, final_start, audio_length))

    def update_slice_indices(self, target_hash_keys):
        self.prepare_slice_indices(target_hash_keys, random_offset=True)

    def compose_validset(self, target_hash_keys):
        validset = deepcopy(self)
        validset.is_valid = True
        validset.loaded_hash = target_hash_keys
        validset.loaded_data = {hash_key:self.loaded_data[hash_key] for hash_key in target_hash_keys}
        validset.loaded_label = {hash_key:self.loaded_label[hash_key] for hash_key in target_hash_keys}
        return validset


    def get_split(self, target_hash_keys, split='train'):
        if split == 'train':
            self.update_slice_indices(target_hash_keys)
        else:
            return self.compose_validset(target_hash_keys)


    def __len__(self):
        if self.is_valid: return len(self.loaded_hash)
        return len(self.slice_indices)
    

    def __getitem__(self, idx):
        if self.is_valid:
            hash_key = self.loaded_hash[idx]
            audio, label = self.loaded_data[hash_key], self.loaded_label[hash_key]
            return hash_key, audio, label
        
        hash_key, start, end = self.slice_indices[idx]

        if self.aug and hash_key in self.pitch_shifted_audio and random.random() < 0.5: audio = random.choice(self.pitch_shifted_audio[hash_key])
        else: audio = self.loaded_data[hash_key]

        margin_length = int(self.margin_ratio * (end-start))
        if start - margin_length//2 < 0: end = start + margin_length
        elif end + margin_length//2 > audio.shape[1]:
            end = audio.shape[1]
            start = end - margin_length
        else: start, end = (start - margin_length//2), (end + margin_length//2)
        audio = audio[:, start:end]

        start_ms, end_ms = int(start/self.sr*1000), int(end/self.sr*1000)
        label = self.loaded_label[hash_key][start_ms:end_ms]

        if self.aug: audio = self.apply_audio_augmentation(audio)

        return hash_key, audio, label



class AudioSegmentDataset(AudioDataset):
    def __init__(self, data_dir, label_dir, num_classes=3, sr=16000, channels='mono', window=20, margin_ratio=1.0, is_valid=False, aug=False, **kwargs):
        super().__init__(data_dir, label_dir, num_classes, sr, channels, window, margin_ratio, is_valid, aug)
        self.loaded_all = self.get_data()
        self.loaded_data, self.loaded_label, self.loaded_meta = self.loaded_all

    def get_data(self):
        loaded_data, loaded_label, loaded_meta = [], [], []
        for audio_file in tqdm(list(Path(self.data_dir).glob('*.wav')), desc="Load Audio Segment & Label"):
            hash_key = unicodedata.normalize('NFC', audio_file.name.split("-")[0])
            if hash_key not in self.loaded_hash: continue

            audio = self._load_audio(audio_file)
            hash_df = self.df[self.df['hash_key']==hash_key]
            segment_meta = hash_df[['start', 'end','label']].to_numpy()

            for start_ms, end_ms, label in segment_meta:
                start, end = int(start_ms * self.sr /1000), int(end_ms * self.sr /1000)

                segment = audio[:, start:end]
                label = self.label_map[label]

                loaded_data.append(segment)
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
            validset = deepcopy(self)
            validset.is_valid = True
            validset.loaded_data, validset.loaded_label, validset.loaded_meta = self.compose_validset(target_hash_keys)
            return validset


    def __len__(self):
        return len(self.loaded_data)
    

    def __getitem__(self, idx):
        if self.is_valid:
            audio, label = self.loaded_data[idx], self.loaded_label[idx]
            hash_key, _, _ = self.loaded_meta[idx]
            return hash_key, audio, label

        audio, label = self.loaded_data[idx], self.loaded_label[idx]
        hash_key, start, end = self.loaded_meta[idx]

        if self.aug and not self.is_valid and random.random() < 0.5:  audio = random.choice(self.pitch_shifted_audio[hash_key])[:,start:end]
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
        num_frames = int(ms_label.shape[0] / ms_per_frame)
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



class MelFrameDataset(AudioFrameDataset, MelDataset):
    def __init__(self, data_dir, label_dir, num_classes=4, sr=16000, channels='mono', window=20, n_fft=2048, hop_length=512, target_bins=40, margin_ratio=1.6, is_valid=False, aug=False):
        AudioFrameDataset.__init__(self, data_dir, label_dir, num_classes, sr, channels, window, margin_ratio, is_valid, aug)
        MelDataset.__init__(self, sr=sr, window=window, n_fft=n_fft, hop_length=hop_length, target_bins=target_bins, margin_ratio=margin_ratio, is_valid=is_valid, aug=aug)

    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, idx):
        if self.is_valid:
            hash_key, audio, label = super().__getitem__(idx)
            mel, _ = self.get_mel(audio)
            frame_label = self.ms_to_frame_label(label)
            if mel.shape[-1] != frame_label.shape[0]: mel = mel[:,:frame_label.shape[0]]
            return hash_key, mel, frame_label

        else:
            hash_key, audio, label = super().__getitem__(idx)
            mel, stretch_factor = self.get_mel(audio)
            
            start_idx = (mel.shape[1] - self.window_frame) // 2
            mel = mel[:,start_idx:start_idx+self.window_frame]
            if self.aug: mel = self.apply_spec_augmentation(mel)
            
            label = self.apply_time_stretch(label, stretch_factor)
            frame_label = self.ms_to_frame_label(label)
            frame_label = frame_label[start_idx:start_idx+self.window_frame]
            assert frame_label.shape[0] == mel.shape[1], f"frame_label.shape[0] != mel.shape[1]: {frame_label.shape[0]} != {mel.shape[1]}, time stretch factor: {stretch_factor}, audio length: {audio.shape[1]/self.sr} sec"
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



class PitchDataset(BaseDataset):
    def __init__(self, data_dir, label_dir, num_classes=4, sr=100, frame_rate=20, threshold=0.8, window=20, margin_ratio=1, is_valid=False, aug=False):
        super().__init__(data_dir, label_dir, num_classes, sr, window, margin_ratio, is_valid, aug)
        self.threshold = threshold
        self.frame_rate = frame_rate
        assert 100 % self.frame_rate == 0
        self.comp_ratio = self.sr // self.frame_rate
        self.window_frame = self.window * frame_rate


    def frequency_to_midi(self, frequency):
        # Convert frequency to MIDI note
        return 69 + 12 * math.log2(frequency / 440)


    def shift_contour(self, contour):
        shift = (random.random() * 12 - 6) / 12
        contour[0,:] += shift

        return contour


    def _load_norm_contour(self, csv_file):
        if 'PestoContour_untrained' not in str(self.data_dir).split('/')[-1]:
            contour_df = pd.read_csv(csv_file, header=None, names=['time', 'frequency', 'confidence'])
            frequency, confidence = contour_df['frequency'].values, contour_df['confidence'].values
        else:
            contour_df = pd.read_csv(csv_file, dtype={'frequency': np.float32, 'amplitude': np.float32})
            frequency, confidence = contour_df['frequency'].values, contour_df['amplitude'].values
            
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

        self.slice_indices = []
        self.prepare_slice_indices()


    def get_data(self):
        loaded_data = {}
        for csv_file in tqdm(list(self.data_dir.rglob('*.csv')), desc="Load Contours"):
            hash_key = unicodedata.normalize('NFC', csv_file.name.split("-")[0])
            if hash_key not in self.loaded_hash: continue

            freq_conf = self._load_norm_contour(csv_file)

            loaded_data[hash_key] = freq_conf[:,::self.comp_ratio]

        return loaded_data


    def prepare_slice_indices(self, target_hash_keys=None, random_offset=True):
        self.slice_indices = []
        window_samples = self.window * self.frame_rate

        for hash_key in self.loaded_hash:

            if target_hash_keys is not None and hash_key not in target_hash_keys: continue

            contour = self.loaded_data[hash_key]
            contour_length = contour.shape[1]
            
            if contour_length < window_samples: continue
            offset = random.randint(0, min(window_samples, contour_length - window_samples)) if random_offset else 0
            
            current_pos = offset
            while current_pos + window_samples <= contour_length:
                self.slice_indices.append((hash_key, current_pos, current_pos + window_samples))
                current_pos += window_samples

            if current_pos < contour_length and contour_length - window_samples >= 0:
                final_start = contour_length - window_samples
                self.slice_indices.append((hash_key, final_start, contour_length))

    def update_slice_indices(self, target_hash_keys):
        self.prepare_slice_indices(target_hash_keys, random_offset=True)


    def compose_validset(self, target_hash_keys):
        validset = deepcopy(self)
        validset.is_valid = True
        validset.loaded_hash = [hash_key for hash_key in target_hash_keys if hash_key in self.loaded_hash]
        validset.loaded_data = {hash_key:self.loaded_data[hash_key] for hash_key in target_hash_keys if hash_key in self.loaded_hash}
        validset.loaded_label = {hash_key:self.loaded_label[hash_key] for hash_key in target_hash_keys if hash_key in self.loaded_hash}
        return validset


    def get_split(self, target_hash_keys, split='train'):
        if split == 'train':
            self.update_slice_indices(target_hash_keys)
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
        if self.is_valid: return len(self.loaded_hash)
        return len(self.slice_indices)


    def __getitem__(self, idx):
        if self.is_valid:
            hash_key = self.loaded_hash[idx]
            contour, label = self.loaded_data[hash_key], self.loaded_label[hash_key]

            if contour.shape[1] > label.shape[0]: contour = contour[:,:label.shape[0]]
            elif contour.shape[1] < label.shape[0]: label = label[:contour.shape[1]]

            return hash_key, contour, label

        hash_key, start, end = self.slice_indices[idx]
        contour, label = self.loaded_data[hash_key], self.loaded_label[hash_key]
        if contour.shape[1] > label.shape[0]: contour = contour[:,:label.shape[0]]
        elif contour.shape[1] < label.shape[0]: label = label[:contour.shape[1]]

        margin_length = int(self.margin_ratio * (end - start))
        if start - margin_length//2 < 0: end = start + margin_length
        elif end + margin_length//2 > contour.shape[1]:
            end = contour.shape[1]
            start = end - margin_length
        else:
            start, end = (start - margin_length//2), (end + margin_length//2)

        contour = contour[:, start:end]
        label = label[start:end]

        start_idx = (contour.shape[1] - self.window_frame) // 2
        contour = contour[:, start_idx:start_idx+self.window_frame]
        label = label[start_idx:start_idx+self.window_frame]

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
            validset = deepcopy(self)
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
        self.hop_length = 16000//50 # orig MERT hop_length
        self.len_window = self.sr * self.window
        self.len_frames = int(window/(self.hop_length/sr) - 1)

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

    def update_slice_indices(self, target_hash_keys, random_offset):
        self.prepare_slice_indices(target_hash_keys, random_offset)


    def compose_validset(self, target_hash_keys):
        validset = deepcopy(self)
        validset.is_valid = True
        validset.aug = False
        validset.pitch_shift_dir = None
        validset.loaded_hash = target_hash_keys
        validset.loaded_data = {hash_key:self.loaded_data[hash_key] for hash_key in target_hash_keys}
        validset.loaded_label = {hash_key:self.loaded_label[hash_key] for hash_key in target_hash_keys}
        validset.prepare_slice_indices(target_hash_keys, False)
        return validset


    def get_split(self, target_hash_keys, split='train'):
        if split == 'train':
            self.update_slice_indices(target_hash_keys, True)
        else:
            return self.compose_validset(target_hash_keys)


    def __len__(self):
        return len(self.slice_indices)


    def __getitem__(self, idx):
        if self.is_valid:
            hash_key, start, end = self.slice_indices[idx]
            audio, label = self.loaded_data[hash_key], self.loaded_label[hash_key]
            label_len = label.shape[0]*self.sr//1000
            if end > label_len:
                end = label_len
                start = end-self.len_window
            audio = audio[:, start:end].squeeze(0)
            start_ms, end_ms = int(start/self.sr*1000), int(end/self.sr*1000)
            ms_label = label[start_ms:end_ms]
            frame_label = self.ms_to_frame(ms_label)
            return hash_key, audio, frame_label
        
        hash_key, start, end = self.slice_indices[idx]

        if self.aug and hash_key in self.pitch_shifted_audio and random.random() < 0.5: 
            audio = random.choice(self.pitch_shifted_audio[hash_key])
        else: 
            audio = self.loaded_data[hash_key]

        margin_length = int(self.margin_ratio * self.len_window)
        if start - margin_length//2 < 0: 
            end = start + margin_length
        elif end + margin_length//2 > audio.shape[1]:
            end = audio.shape[1]
            start = end - margin_length
        else: 
            start, end = (start - margin_length//2), (end + margin_length//2)

        start = (end-start)//2
        end = start+self.len_window

        audio = audio[:, start:end].squeeze(0)

        start_ms, end_ms = int(start/self.sr*1000), int(end/self.sr*1000)
        ms_label = self.loaded_label[hash_key][start_ms:end_ms]

        if self.aug: audio = self.apply_audio_augmentation(audio)

        frame_label = self.ms_to_frame(ms_label)

        return hash_key, audio, frame_label




# from transformers import AutoModel
# from transformers import Wav2Vec2FeatureExtractor
# class CMERTAudioDataset(AudioDataset):
#     def __init__(self, data_dir, label_dir, num_classes=3, sr=24000, channels='mono', window=30, margin_ratio=1, is_valid=False, aug=False, time_reduce=True):
#         super().__init__(data_dir, label_dir, num_classes, sr, channels, window, margin_ratio, is_valid, aug)
#         self.window_frame = self.window * self.sr

#         self.loaded_all = self.get_data()
#         self.loaded_data, self.loaded_label, self.loaded_meta = self.loaded_all

#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         self.processor = Wav2Vec2FeatureExtractor.from_pretrained("ntua-slp/CultureMERT-95M", trust_remote_code=True)
#         self.cmert = AutoModel.from_pretrained("ntua-slp/CultureMERT-95M", trust_remote_code=True).to(self.device)
#         if self.cmert.training:
#             self.cmert.eval()
#         print(f'CultureMERT device: {self.cmert.device}')

#         self.time_reduce = time_reduce


#     def get_data(self):
#         loaded_data, loaded_label, loaded_meta = [], [], []
#         for audio_file in tqdm(list(Path(self.data_dir).glob('*.wav')), desc="Load Audio Segment & Label"):
#             hash_key = unicodedata.normalize('NFC', audio_file.name.split("-")[0])
#             if hash_key not in self.loaded_hash: continue

#             audio = self._load_audio(audio_file)
#             hash_df = self.df[self.df['hash_key']==hash_key]
#             segment_meta = hash_df[['start', 'end','label']].to_numpy()

#             for start_ms, end_ms, label in segment_meta:
#                 start, end = int(start_ms * self.sr /1000), int(end_ms * self.sr /1000)

#                 segment = audio[:, start:end]
#                 label = self.label_map[label]

#                 loaded_data.append(segment)
#                 loaded_label.append(label)
#                 loaded_meta.append((hash_key, start, end))

#         return loaded_data, loaded_label, loaded_meta


#     def infer_cmert(self, x):
#         x = x.to(self.device)
#         inputs = self.processor(x.squeeze(0), sampling_rate=self.sr, return_tensors="pt")
#         inputs = {k: v.to(self.device) for k, v in inputs.items()}

#         with torch.no_grad():
#             feature = self.cmert(**inputs, output_hidden_states=True)

#         feature = torch.stack(feature.hidden_states).squeeze()

#         return feature


#     def get_feature(self, x):
#         if x.shape[-1] > self.window_frame:
#             chunks = []
#             chunk_num = x.shape[-1]//self.window_frame + 1
#             chunk_len = x.shape[-1]//chunk_num

#             cur_pos = 0
#             while cur_pos + self.window_frame <= x.shape[-1]:
#                 start_seg, end_seg = cur_pos, cur_pos+self.window_frame
#                 chunks.append(x[:,start_seg:end_seg])
#                 cur_pos += chunk_len
#             chunks.append(x[:,x.shape[-1]-self.window_frame:])

#             feature = torch.stack([self.infer_cmert(chunk) for chunk in chunks]).mean(dim=0)
        
#         else:
#             feature = self.infer_cmert(x)

#         return feature


#     def get_split(self, target_hash_keys, split='train'):
#         target_indices = [idx for idx, tup in enumerate(self.loaded_all[2]) if tup[0] in target_hash_keys]

#         if split == 'train':
#             self.loaded_data = [self.loaded_all[0][idx] for idx in target_indices]
#             self.loaded_label = [self.loaded_all[1][idx] for idx in target_indices]
#             self.loaded_meta = [self.loaded_all[2][idx] for idx in target_indices]

#         else:
#             validset = deepcopy(self)
#             validset.is_valid = True
#             validset.loaded_data = [self.loaded_all[0][idx] for idx in target_indices]
#             validset.loaded_label = [self.loaded_all[1][idx] for idx in target_indices]
#             validset.loaded_meta = [self.loaded_all[2][idx] for idx in target_indices]
#             return validset


#     def __len__(self):
#         return len(self.loaded_data)


#     def __getitem__(self, idx):
#         audio, label = self.loaded_data[idx], self.loaded_label[idx]
#         hash_key, _, _ = self.loaded_meta[idx]

#         if self.aug and not self.is_valid and random.random() < 0.5: 
#             audio = random.choice(self.pitch_shifted_audio[hash_key])

#         if self.aug and not self.is_valid:
#             audio = self.apply_audio_augmentation(audio)

#         feature = self.get_feature(audio)

#         if self.time_reduce:
#             feature = feature.mean(-2) # time reduced

#         return hash_key, feature, label