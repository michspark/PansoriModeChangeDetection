import os
import json
import random
import unicodedata
from pathlib import Path
from abc import abstractmethod, ABC

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import Spectrogram, MelScale, AmplitudeToDB, TimeStretch, FrequencyMasking

class SegmentDataset(Dataset, ABC):
    def __init__(self, data_dir, label_dir, sr, num_classes=3, window=20, margin_ratio=1.0, validation_mode=False, aug=False):
        super().__init__()
        self.sr = sr
        self.window = window

        self.num_classes = num_classes
        self.label_map = {"창조":0, "아니리":0, "설렁제":1, "경드름":1, "우조":1, "평조":1, "계면조": 2}

        label_csv = os.path.join(label_dir, 'label.csv')
        self.df = pd.read_csv(label_csv) if os.path.exists(label_csv) else self.get_df(label_dir)
        self.loaded_hash = list(set(self.df['hash_key'].tolist()))
        
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


    @abstractmethod
    def get_data(self): return None

    @abstractmethod
    def __len__(self): return None
    
    @abstractmethod
    def __getitem__(self, idx): return None



class AudioSegmentDataset(SegmentDataset):
    def __init__(self, data_dir, label_dir, sr=16000, channels='mono', num_classes=3, window=20, margin_ratio=1, validation_mode=False, aug=False):
        super().__init__(data_dir, label_dir, sr, num_classes, window, margin_ratio, validation_mode, aug)
        self.data_dir = data_dir
        self.channels = channels
        self.loaded_data = self.get_data()

        self.pitch_shift_dir = os.path.join(os.path.dirname(data_dir), "PitchShiftedAudio")
        if self.aug and not self.validation_mode and os.path.exists(self.pitch_shift_dir): self.pitch_shifted_audio = self.load_pitch_shifted_audio()


    def get_data(self):
        loaded_data = []
        for audio_file in tqdm(list(Path(self.data_dir).glob('*.wav')), desc="Load Audio Segment & Label"):
            hash_key = unicodedata.normalize('NFC', audio_file.name.split("-")[0])
            if hash_key not in self.loaded_hash: continue

            audio, sr = torchaudio.load(audio_file)
            audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.sr) if self.sr!=sr else audio
            audio = audio.mean(dim=0, keepdim=True) if self.channels=='mono' else audio

            hash_df = self.df[self.df['hash_key']==hash_key]
            segment_meta = hash_df[['start', 'end','label']].to_numpy()

            for start_ms, end_ms, label in segment_meta:
                start, end = int(start_ms * self.sr /1000), int(end_ms * self.sr /1000)

                segment = audio[:, start:end]
                label = torch.nn.functional.one_hot(torch.tensor(self.label_map[label]), num_classes=self.num_classes)

                loaded_data.append(((hash_key, start, end), segment, label))

        return loaded_data


    def get_valid_data(self):
        loaded_data = []
        for audio_file in tqdm(list(Path(self.data_dir).glob('*.wav')), desc="Load Audio Segment & Label"):
            hash_key = unicodedata.normalize('NFC', audio_file.name.split("-")[0])
            if hash_key not in self.loaded_hash: continue

            audio, sr = torchaudio.load(audio_file)
            audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.sr) if self.sr!=sr else audio
            audio = audio.mean(dim=0, keepdim=True) if self.channels=='mono' else audio

            hash_df = self.df[self.df['hash_key']==hash_key]
            segment_meta = hash_df[['start', 'end','label']].to_numpy()

            for start_ms, end_ms, label in segment_meta:
                label = torch.nn.functional.one_hot(torch.tensor(self.label_map[label]), num_classes=self.num_classes)
                
                start, end = int(start_ms * self.sr /1000), int(end_ms * self.sr /1000)
                seg_len, window_len = (end - start), self.window * self.sr

                if seg_len > window_len:
                    current_pos = 0
                    while current_pos + window_len <= seg_len:
                        start_seg, end_seg = start+current_pos, start+current_pos+window_len
                        # print(start_seg, end_seg)
                        segment = audio[:, start_seg:end_seg]
                        loaded_data.append(((hash_key, start_seg, end_seg), segment, label))
                        current_pos += window_len

                    if current_pos < seg_len:
                        final_start = end - window_len
                        # print(final_start, final_start+window_len)
                        segment = audio[:, final_start:final_start+window_len]
                        loaded_data.append(((hash_key, start_seg, end_seg), segment, label))

                else:
                    segment = audio[:, start:end]
                    loaded_data.append(((hash_key, start, end), segment, label))

        return loaded_data



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
                
            audio, sr = torchaudio.load(os.path.join(self.pitch_shift_dir, audio_file))
            audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.sr) if self.sr!=sr else audio
            audio = audio.mean(dim=0, keepdim=True) if self.channels=='mono' else audio

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


    def __len__(self):
        return len(self.loaded_data)


    def __getitem__(self, idx):
        if self.validation_mode:
            (hash_key, _, _), audio, label = self.loaded_data[idx]
            return hash_key, audio, label

        (hash_key, start, end), audio, label = self.loaded_data[idx]

        if self.aug and not self.validation_mode and random.random() < 0.5:  audio = random.choice(self.pitch_shifted_audio[hash_key])[:,start:end]
        if self.aug and not self.validation_mode: audio = self.apply_audio_augmentation(audio)

        audio_len, window_len = audio.shape[-1], self.window * self.sr

        if audio_len > window_len:
            start = random.randint(0, min(audio_len, audio_len - window_len))
            audio = audio[:,start:start+window_len]

        # else:
        #     num_pad = window_len - audio_len
        #     l_pad = num_pad // 2
        #     r_pad = num_pad - l_pad
        #     audio = torch.nn.functional.pad(audio, (l_pad, r_pad), mode='constant', value=0)

        return hash_key, audio, label


class MelSegmentDataset(AudioSegmentDataset):
    def __init__(self, data_dir, label_dir, sr=16000, channels='mono', num_classes=3, window=20, n_fft=2048, hop_length=512, target_bins=40, margin_ratio=1.0, validation_mode=False, aug=False):
        super().__init__(data_dir, label_dir, sr, channels, num_classes, window, margin_ratio, validation_mode, aug)
        self.hop_length = hop_length
        self.target_bins = target_bins
        self.window_frame = self.window * sr // hop_length

        self.spec_cvt = Spectrogram(n_fft=n_fft, hop_length=hop_length, power=1.0)
        self.spec2mel = MelScale(n_stft=n_fft//2+1, n_mels=target_bins, sample_rate=sr, f_min=80, f_max=2000)
        self.db_cvt = AmplitudeToDB()
        
        if self.aug and not self.validation_mode:
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

        if self.aug and not self.validation_mode: mel = self.apply_spec_augmentation(mel)
        assert mel.shape[1] == self.window_frame, f"mel.shape[1] != self.window_frame: {mel.shape[1]} != {self.window_frame}"
        return hash_key, mel, label

