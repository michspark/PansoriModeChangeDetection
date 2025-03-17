import os
import json

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset


class PrvChromaDataset(Dataset):
    def __init__(self, audio_dir, label_json, num_classes=4, sr=16000, window_size=20, hop_length=512, target_bins=25):
        super().__init__()
        self.sr = sr
        self.target_bins = target_bins
        self.audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav') or f.endswith('.mp3')]
        self.loaded_audio = self.get_audio(audio_dir)
        self.loaded_chromas = self.get_chroma()

        self.df = self.get_df(label_json)
        self.window_size = window_size
        self.window_frames = window_size*sr//hop_length
        self.segments = self.get_segments()

        self.num_classes = num_classes
        self.label_map = {"Unknown":0, "청조":0, "세면조":0, "우조":1, "계면조":2, "아니리":3}
        self.hop_size = hop_length
        self.labels = self.get_labels()

    def get_audio(self, audio_dir):
        loaded_audio = {}
        for filename in tqdm(self.audio_files, desc='Load Audio'):
            y, _ = librosa.load(os.path.join(audio_dir, filename), sr=self.sr)
            loaded_audio[filename] = y
        return loaded_audio

    def get_chroma(self):
        '''
        loaded_audio를 로드하여 file별로 chromagram 생성 후 target bins에 맞추어 expand하여 반환
        '''
        chroma_dict = {}
        for filename, audio_key_value in zip(tqdm(self.audio_files, desc="Load Chroma"), self.loaded_audio.items()):
            chroma = librosa.feature.chroma_stft(y=audio_key_value[1], sr=self.sr)
            repeat = self.target_bins // chroma.shape[0] + 1
            expanded_chroma = np.tile(chroma, (repeat, 1))
            chroma_dict[filename]=expanded_chroma[:self.target_bins, :]
        return chroma_dict

    def get_df(self, label_json):
        with open(label_json, "r", encoding="utf-8") as file: label_data = json.load(file)
        df = pd.DataFrame(columns=['filename', 'start', 'end', 'duration', 'label'])
        row = 0
        song_segments = {}
        for song in label_data:
            filename = song['file_upload'][:-3]+'wav'
            if filename not in song_segments: song_segments[filename] = []
            
            for val in song['annotations'][0]["result"]:
                start = val['value']['start']
                end = val['value']['end']
                label = val['value']['labels'][0]
                song_segments[filename].append((start, end, label))

        for filename, segments in song_segments.items():
            prev_end = None
            segments.sort(key=lambda x: x[0])
            for start, end, label in segments:
                if prev_end is not None and int(prev_end) != int(start):
                    gap_start, gap_end = prev_end+1e-6, start-1e-6
                    gap_duration = gap_end - gap_start

                    if gap_duration < 1: continue
                    df.loc[row] = [filename, gap_start, gap_end, gap_duration, "Unknown"]
                    row += 1
                
                duration = end - start
                df.loc[row] = [filename, start, end, duration, label]
                prev_end = end
                row += 1
        return df


    def get_segments(self):
        """
            audio 길이에 비례하여 20초 단위로 random sampling
            총 segment의 수는 audio length // 20sec 
            random sampled start_times를 기준으로 20sec에 해당하는 chroma의 time frame을 crop하여 segments에 추가 후 return
        """
        segments = []
        for filename in self.audio_files:
            duration = librosa.get_duration(y=self.loaded_audio[filename], sr=self.sr)
            num_segments = int(duration // self.window_size)

            start_times = np.random.uniform(0, max(1, duration - self.window_size), num_segments)
            num_frames = self.loaded_chromas[filename].shape[1]

            for start_time in start_times:
                start_frame = int(num_frames * start_time / duration)
                end_frame = int(start_frame + self.window_frames)
                segments.append((filename, start_frame, end_frame))

        return segments

    def get_labels(self):
        """
        segments: [filename, start_frame, end_frame]
        df: DataFrame with columns ['filename', 'start', 'end', 'duration', 'label']
        returns: Dictionary mapping (filename, start_frame, end_frame) to label tensor [self.window_frames, num_classes]
        """
        labels_dict = {}

        for segment in self.segments:
            filename, start_frame, end_frame = segment
            duration = librosa.get_duration(y=self.loaded_audio[filename], sr=self.sr)    

            segment_labels = np.zeros((self.window_frames, self.num_classes))
            segment_labels[:, self.label_map["Unknown"]] = 1
            
            file_df = self.df[self.df['filename'] == filename]
            
            if not file_df.empty:
                num_frames = self.loaded_chromas[filename].shape[1]
                
                for _, row in file_df.iterrows():
                    label_start_frame = (row['start'] / self.hop_size) * self.sr
                    label_end_frame = (row['end'] / self.hop_size) * self.sr
                    
                    overlap_start = int(max(start_frame, label_start_frame))
                    overlap_end = int(min(end_frame, label_end_frame))

                    if overlap_start < overlap_end:
                        segment_start_idx = int(overlap_start - start_frame)
                        segment_end_idx = int(overlap_end - start_frame)

                        segment_labels[segment_start_idx:segment_end_idx, self.label_map["Unknown"]] = 0
                        label_idx = self.label_map.get(row['label'], self.label_map["Unknown"])
                        segment_labels[segment_start_idx:segment_end_idx, label_idx] = 1

            labels_dict[segment] = torch.tensor(segment_labels, dtype=torch.float32)
        
        return labels_dict
    
    def __getitem__(self, idx):
        filename, start, end = self.segments[idx]
        chroma = self.loaded_chromas[filename][:, start:end]
        label = self.labels[self.segments[idx]] 

        return chroma, label, (filename, start, end)