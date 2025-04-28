import os
import math
import json
import random
import unicodedata
from pathlib import Path
from collections import Counter
from abc import abstractmethod, ABC

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import Spectrogram, MelScale, AmplitudeToDB, TimeStretch, FrequencyMasking


class BaseDataset(Dataset, ABC):
    def __init__(self, data_dir, label_dir, sr, num_classes=4, window=20, margin_ratio=1.0, validation_mode=False, aug=False):
        super().__init__()
        self.sr = sr
        self.window = window

        self.num_classes = num_classes
        self.label_map = {"Unknown":0, "창조":1, "아니리":1, "설렁제":2, "경드름":2, "우조":2, "평조":2, "계면조": 3}

        label_csv = os.path.join(label_dir, 'label.csv')
        self.df = pd.read_csv(label_csv) if os.path.exists(label_csv) else self.get_df(label_dir)
        self.loaded_hash = list(set(self.df['hash_key'].tolist()))
        self.loaded_label = self.get_label()
        
        self.aug = aug
        self.margin_ratio = margin_ratio
        self.validation_mode = validation_mode


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


    def get_label(self):
        loaded_label = {}
        for hash_key in tqdm(self.loaded_hash, desc='Load Label'):
            target_df = self.df[self.df['hash_key']==hash_key]
            sample_duration = target_df['duration'].values[0]
            label = torch.zeros(sample_duration, self.num_classes)
            label[:,0] = 1

            for row in target_df[['start', 'end', 'label']].to_numpy():
                label[row[0]:row[1], self.label_map[row[2]]] = 1
                label[row[0]:row[1], 0] = 0

            loaded_label[hash_key] = label

        return loaded_label


    @abstractmethod
    def get_data(self): return None

    @abstractmethod
    def __len__(self): return None
    
    @abstractmethod
    def __getitem__(self, idx): return None





class AudioDataset(BaseDataset):
    def __init__(self, data_dir, label_dir, sr=16000, channels='mono', num_classes=4, window=20, margin_ratio=1, validation_mode=False, aug=False):
        super().__init__(data_dir, label_dir, sr, num_classes, window, margin_ratio, validation_mode, aug)
        self.channels = channels
        self.loaded_data = self.get_data(data_dir)

        self.pitch_shifted_audio = {}
        self.pitch_shift_dir = os.path.join(os.path.dirname(data_dir), "PitchShiftedAudio")
        if self.aug and os.path.exists(self.pitch_shift_dir): self.load_pitch_shifted_audio()

        self.slice_indices = []
        self.prepare_slice_indices()


    def get_data(self, data_dir):
        loaded_data = {}
        for audio_file in tqdm(list(Path(data_dir).glob('*.wav')), desc="Load Audio"):
            hash_key = unicodedata.normalize('NFC', audio_file.name.split("-")[0])
            if hash_key not in self.loaded_hash: continue

            audio, sr = torchaudio.load(audio_file)
            audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.sr) if self.sr!=sr else audio
            audio = audio.mean(dim=0, keepdim=True) if self.channels=='mono' else audio
            loaded_data[hash_key] = audio

        return loaded_data


    def prepare_slice_indices(self, random_offset=True):
        self.slice_indices = []
        window_samples = self.window * self.sr
        
        for hash_key in self.loaded_hash:
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


    def update_slice_indices(self):
        self.prepare_slice_indices(random_offset=True)


    def load_pitch_shifted_audio(self):
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
                
            audio, sr = torchaudio.load(os.path.join(self.pitch_shift_dir, audio_file))
            audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.sr) if self.sr!=sr else audio
            audio = audio.mean(dim=0, keepdim=True) if self.channels=='mono' else audio

            if hash_key not in self.pitch_shifted_audio: self.pitch_shifted_audio[hash_key] = []
            self.pitch_shifted_audio[hash_key].append(audio)


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


    def __len__(self):
        if self.validation_mode: return len(self.loaded_hash)
        return len(self.slice_indices)
    

    def __getitem__(self, idx):
        if self.validation_mode:
            hash_key = self.loaded_hash[idx]
            audio, label = self.loaded_data[hash_key], self.loaded_label[hash_key]
            return hash_key, audio, label
        
        hash_key, start, end = self.slice_indices[idx]
        if self.aug and hash_key in self.pitch_shifted_audio and random.random() < 0.5:
            audio = random.choice(self.pitch_shifted_audio[hash_key])
        else: audio = self.loaded_data[hash_key]


        margin_length = int(self.margin_ratio * (end - start))
        if start - margin_length//2 < 0:
            end = start + margin_length
        elif end + margin_length//2 > audio.shape[1]:
            end = audio.shape[1]
            start = end - margin_length
        else:
            start = start - margin_length//2
            end = end + margin_length//2
        audio = audio[:, start:end]

        start_ms, end_ms = int(start/self.sr*1000), int(end/self.sr*1000)
        label = self.loaded_label[hash_key][start_ms:end_ms]

        if self.aug: audio = self.apply_audio_augmentation(audio)

        return hash_key, audio, label





class MelDataset(AudioDataset):
    def __init__(self, data_dir, label_dir, sr=16000, channels='mono', num_classes=4, window=20, n_fft=2048, hop_length=512, target_bins=40, margin_ratio=1.6, validation_mode=False, aug=False):
        super().__init__(data_dir, label_dir, sr, channels, num_classes, window, margin_ratio, validation_mode, aug)
        self.hop_length = hop_length
        self.target_bins = target_bins
        self.window_frame = self.window * sr // hop_length

        self.spec_cvt = Spectrogram(n_fft=n_fft, hop_length=hop_length, power=1.0)
        self.spec2mel = MelScale(n_stft=n_fft//2+1, n_mels=target_bins, sample_rate=sr, f_min=80, f_max=2000)
        self.db_cvt = AmplitudeToDB()
        
        if self.aug:
            self.time_stretch = TimeStretch(hop_length=hop_length, n_freq=n_fft//2+1)
            self.freq_mask = FrequencyMasking(freq_mask_param=10)
    

    def get_mel(self, audio):
        spec = self.spec_cvt(audio)
        if self.aug and not self.validation_mode: 
            spec, stretch_factor = self.apply_spec_time_stretch(spec)
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


    def return_weights(self):
        label_counter = sum(val.sum(dim=0) for val in self.loaded_label.values())[2:]
        class_weights = 1.0 / label_counter
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        full_class_weights = torch.ones(4)
        full_class_weights[2:] = class_weights
        return full_class_weights


    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, idx):
        if self.validation_mode:
            hash_key, audio, label = super().__getitem__(idx)
            mel, _ = self.get_mel(audio)
            frame_label = self.ms_to_frame_label(label)
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





class PitchDataset(BaseDataset):
    def __init__(self, data_dir, label_dir, sr=100, frame_rate=20, threshold=0.8, num_classes=4, window=20, margin_ratio=1, validation_mode=False, aug=False):
        super().__init__(data_dir, label_dir, sr, num_classes, window, margin_ratio, validation_mode, aug)
        self.threshold = threshold
        self.frame_rate = frame_rate
        assert 100 % self.frame_rate == 0
        self.comp_ratio = self.sr // self.frame_rate
        self.window_frame = self.window * frame_rate

        self.loaded_data = self.get_data(data_dir)
        for hash_key in set(self.loaded_label.keys()) - set(self.loaded_data.keys()): self.loaded_hash.remove(hash_key)
        self.loaded_label = {key:self.ms_to_frame_label(val) for key, val in self.loaded_label.items()}

        self.slice_indices = []
        self.prepare_slice_indices()


    def frequency_to_midi(self, frequency):
        # Convert frequency to MIDI note
        return 69 + 12 * math.log2(frequency / 440)


    def get_data(self, data_dir):
        loaded_data = {}
        for csv_file in tqdm(list(Path(data_dir).glob('*.csv')), desc="Load Contours"):
            hash_key = unicodedata.normalize('NFC', csv_file.name.split("-")[0])
            if hash_key not in self.loaded_hash: continue

            df = pd.read_csv(csv_file, header=None, names=['time', 'frequency', 'confidence'])
            frequency, confidence = df['frequency'].values, df['confidence'].values
            midi = [self.frequency_to_midi(freq) for freq in frequency]
            tonic_counter = Counter(np.round(midi)[confidence >= self.threshold]).most_common(1)
            tonic = tonic_counter[0][0]
            norm_midi = [(mi-float(tonic))/12 for mi in midi]
            freq_conf = torch.tensor(np.stack([norm_midi, confidence], axis=0), dtype=torch.float32)
            loaded_data[hash_key] = freq_conf[:,::self.comp_ratio]
            # loaded_data[hash_key] = freq_conf

        return loaded_data


    def prepare_slice_indices(self, random_offset=True):
        self.slice_indices = []
        window_samples = self.window * self.frame_rate
        
        for hash_key in self.loaded_hash:
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


    def update_slice_indices(self):
        self.prepare_slice_indices(random_offset=True)


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
        if self.validation_mode:
            return len(self.loaded_hash)
        return len(self.slice_indices)


    def __getitem__(self, idx):
        if self.validation_mode:
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
        if start - margin_length//2 < 0:
            end = start + margin_length
        elif end + margin_length//2 > contour.shape[1]:
            end = contour.shape[1]
            start = end - margin_length
        else:
            start = start - margin_length//2
            end = end + margin_length//2
        contour = contour[:, start:end]
        label = label[start:end]

        start_idx = (contour.shape[1] - self.window_frame) // 2
        contour = contour[:, start_idx:start_idx+self.window_frame]
        label = label[start_idx:start_idx+self.window_frame]

        if self.aug and random.random() < 0.5:
            shift = random.random() * 12 - 6
            contour = contour.clone()
            contour[0,:] += shift

        assert label.shape[0] == contour.shape[1], f"label.shape[0] != contour.shape[1]: {label.shape[0]} != {contour.shape[1]}"

        return hash_key, contour, label