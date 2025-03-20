from .AudioDataset import AudioDataset
import random
from tqdm import tqdm
import torch
from torchaudio.transforms import MelSpectrogram
import os

class MelSpecDataset(AudioDataset):
    def __init__(self, audio_dir, label_json, num_classes=5, sr=16000, channels='mono', window=20, n_fft=2048, hop_length=512, target_bins=64, shift=0.4):
        super().__init__(audio_dir, label_json, num_classes, sr, channels, window)
        self.hop_length = hop_length
        self.target_bins = target_bins
        self.window_frame = self.window * sr // hop_length
        self.mel_cvt = MelSpectrogram(sample_rate=self.sr, n_fft=n_fft, hop_length=hop_length, n_mels = 128)
        self.loaded_mel_spec = self.get_mel()
        self.max_shift = self.target_bins//4 if shift else None
        self.audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav') or f.endswith('.mp3')]

    def get_mel(self):
        mel_spec_dict = {}
        for k, v in tqdm(self.loaded_audio.items(), desc='Load MelSpectrogram'):
            mel_spec = self.mel_cvt(v).squeeze(0)
            # librosa 구현형

            max_values, _ = torch.max(mel_spec, dim=0, keepdim=True)
            max_values[max_values == 0] = 1.0 # 최댓값이 0인 경우 => 1
            mel_spec = mel_spec / max_values # 가장 강한 음높이 성분 => 1


            #mel_spec = torch.log1p(mel_spec) # Normalization using natural logarithm

            # expand mel_spec

            '''
            repeat = self.target_bins // chroma.shape[0] + 1
            expanded_chroma = chroma.repeat_interleave(repeat, dim=0)'
            '''

            mel_spec_dict[k] = mel_spec[:self.target_bins, :]
        return mel_spec_dict

    def get_label(self, filename, start):
        labels = torch.zeros((self.window_frame, self.num_classes))
        labels[:, self.label_map['Unknown']] = 1
        label_ant = self.label_dict[filename]

        start_frame = start
        end_frame = start_frame + self.window_frame

        if len(label_ant):
            for ant in label_ant:
                label_start_frame = int((ant['start']/self.hop_length) * self.sr)
                label_end_frame = int((ant['end']/self.hop_length) * self.sr)

                overlap_start = max(start_frame, label_start_frame)
                overlap_end = min(end_frame, label_end_frame)

                if overlap_start < overlap_end:
                    segment_start_idx = int(overlap_start - start_frame)
                    segment_end_idx = int(overlap_end - start_frame)
                    labels[segment_start_idx:segment_end_idx, self.label_map['Unknown']] = 0
                    label_idx = self.label_map.get(ant['label'], self.label_map['Unknown'])
                    labels[segment_start_idx:segment_end_idx, label_idx] = 1

        return labels

    def shift_mel_spec(self, mel_spec):
        if self.max_shift is None: return mel_spec
        prob = 0.8
        if random.random() < prob:
            shift_amount = random.randint(0, self.max_shift)
            mel_spec = torch.roll(mel_spec, shifts=shift_amount, dims=0)
        return mel_spec

    def get_start_time(self, filename, start_frame):
        return start_frame*(self.loaded_audio[filename].shape[1]/self.sr)/self.loaded_mel_spec[filename].shape[1]

    def __getitem__(self, idx):
        filename = self.loaded_filename[idx]
        duration = self.loaded_audio[filename].shape[1]/self.sr
        mel_spec, label = self.loaded_mel_spec[filename], self.label_dict[filename]

        if self.window:
            start_frame = random.randint(0, mel_spec.shape[1] - self.window_frame)
            mel_spec = self.shift_mel_spec(mel_spec[:, start_frame:start_frame+self.window_frame])
            label = self.get_label(filename, start_frame)
            return mel_spec, label, (filename, start_frame, duration)

        return mel_spec, label