from .AudioDataset import AudioDataset

import random
from tqdm import tqdm

import torch
from torchaudio.prototype.transforms import ChromaSpectrogram

class ChromaDataset(AudioDataset):
    def __init__(self, audio_dir, label_json, num_classes=4, sr=16000, channels='mono', window=20, n_fft=2048, hop_length=512, target_bins=25, shift=0.4):
        super().__init__(audio_dir, label_json, num_classes, sr, channels, window)
        self.hop_length = hop_length
        self.target_bins = target_bins
        self.window_frame = self.window * sr // hop_length
        self.chroma_cvt = ChromaSpectrogram(sample_rate=self.sr, n_fft=n_fft, hop_length=hop_length)
        self.loaded_chromas = self.get_chroma()
        self.shift = shift
        self.max_shift = self.target_bins//4 if self.shift > 0 else None


    def get_chroma(self):
        chroma_dict = {}
        for k, v in tqdm(self.loaded_audio.items(), desc='Load Chromagram'):
            chroma = self.chroma_cvt(v).squeeze(0)
            # librosa 구현형
            max_values, _ = torch.max(chroma, dim=0, keepdim=True)
            max_values[max_values == 0] = 1.0 # 최댓값이 0인 경우 => 1
            chroma = chroma / max_values # 가장 강한 음높이 성분 => 1
            # expand chroma
            repeat = self.target_bins // chroma.shape[0] + 1
            expanded_chroma = torch.tile(chroma, (repeat, 1))
            chroma_dict[k] = expanded_chroma[:self.target_bins, :]
        return chroma_dict

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

    def shift_chroma(self, chroma):
        if self.max_shift is None: return chroma
        if random.random() < self.shift:
            shift_amount = random.randint(0, self.max_shift) # 0일 경우 변형없는 형태로 반환
            chroma = torch.roll(chroma, shifts=shift_amount, dims=0)
        return chroma

    def get_start_time(self, filename, start_frame):
        return start_frame*(self.loaded_audio[filename].shape[1]/self.sr)/self.loaded_chromas[filename].shape[1]

    def __getitem__(self, idx):
        filename = self.loaded_filename[idx]
        duration = self.loaded_audio[filename].shape[1]/self.sr
        chroma, label = self.loaded_chromas[filename], self.label_dict[filename]

        if self.window:
            start_frame = random.randint(0, chroma.shape[1] - self.window_frame)
            chroma = self.shift_chroma(chroma[:, start_frame:start_frame+self.window_frame])
            label = self.get_label(filename, start_frame)
            return chroma, label, (filename, start_frame, duration)

        return chroma, label