import json
import random
import unicodedata
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm

import torch
from torch.utils.data import Dataset

import torchaudio

class AudioDataset(Dataset):
    def __init__(self, audio_dir, label_json, num_classes=4, sr=16000, channels='mono', window=20,):
        super().__init__()
        self.sr = sr
        self.window = window
        self.channels = channels
        self.num_classes = num_classes

        self.loaded_filename, self.loaded_audio = self.get_audio(audio_dir, ext='.wav')
        self.label_dict = self.get_label_dict(label_json)
        # self.label_map = {'계면조': 1, '우조': 2, '평조': 3, 'Unknown':0, '아니리': 0, '창조': 0, '설렁제': 0}
        self.label_map = {"Unknown":0, "창조":0, "설렁제":0, "경드름": 0, "우조": 1, "계면조": 2, "평조": 3, "아니리": 4}

    def get_audio(self, audio_dir, ext=".wav"):
        """
        지정된 디렉터리에서 특정 확장자의 오디오 파일을 로드
        Args:
            directory_path (str or Path): 오디오 파일이 있는 디렉터리 경로
            file_extension (str): 대상 확장자        
        Returns:
            dict: 파일명을 키로, 로드된 오디오 데이터를 값으로 하는 딕셔너리
        """
        audio_dir = Path(audio_dir)
        loaded_filename = []
        loaded_audio = {}
        for audio_file in tqdm(audio_dir.glob(f"*{ext}"), desc="Load Audio"):
            filename = "-".join(audio_file.name.split(".")[0].split("-")[1:])
            filename = unicodedata.normalize('NFC', filename)
            try:
                audio, sr = torchaudio.load(audio_file)
                if self.sr != sr:
                    audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.sr)
                if self.channels == 'mono' and audio.shape[0] != 1:
                    audio = audio.mean(dim=0, keepdim=True)
                loaded_audio[filename] = audio
                loaded_filename.append(filename)
            except Exception as e: print(f"Error loading {audio_file.name}: {e}")
        return loaded_filename, loaded_audio

    def get_label_dict(self, label_json):
        """
        json 파일에서 filename, start, end, label 정보 추출
        Args: json_file_path: JSON 파일 경로        
        Returns: defaultdict: 파일명을 키로, [start, end, label] 리스트를 값으로 하는 defaultdict
        """
        label_dict = defaultdict(list)
        with open(label_json, 'r', encoding='utf-8') as file: label_data = json.load(file)

        for idx, item in enumerate(tqdm(label_data, desc='Load Label')): 
            filename = "-".join(item["file_upload"].split(".")[0].split("-")[1:])
            filename = unicodedata.normalize('NFC', filename)
            if filename not in self.loaded_filename: continue
            for annotation in item["annotations"]:
                for result in annotation["result"]:
                    start, end, label = result["value"]["start"], result["value"]["end"], result["value"]["labels"][0]
                    label_dict[filename].append({"start": start, "end": end, "label": label})
        return label_dict

    def get_label(self, filename, start):
        """
        window가 주어질 때, 초 단위로 비중을 계산하여 가장 비율이 높은 레이블로 labeling
        Args:
            filename (str): Audio filename
            start (float): Start time in seconds
        Returns:
            torch.Tensor: Tensor of shape [window, num_classes] with one-hot encoded labels per second
        """
        labels = torch.zeros((self.window, self.num_classes))
        label_ant = self.label_dict[filename]

        for sec_idx in range(self.window):
            start_sec = start + sec_idx
            end_sec = start_sec + 1
            
            overlap_ratios = {'Unknown': 1.0}
            
            for ant in label_ant:
                start_ant, end_ant = ant['start'], ant['end']
                label_name = ant['label']
                
                if end_ant <= start_sec or start_ant >= end_sec: continue

                overlap_start = max(start_sec, start_ant)
                overlap_end = min(end_sec, end_ant)
                overlap_duration = overlap_end - overlap_start
                
                if overlap_duration > 0:
                    if label_name not in overlap_ratios: overlap_ratios[label_name] = 0.0
                    
                    overlap_ratios[label_name] += overlap_duration
                    overlap_ratios['Unknown'] -= overlap_duration
            
            overlap_ratios['Unknown'] = max(0, overlap_ratios['Unknown'])
            max_label = max(overlap_ratios, key=overlap_ratios.get)
            
            if max_label == 'Unknown': labels[sec_idx, self.label_map['Unknown']] = 1
            else: labels[sec_idx, self.label_map[max_label]] = 1

        return labels

    def __len__(self):
        return len(self.loaded_audio)
    
    def __getitem__(self, idx):
        filename = self.loaded_filename[idx]
        audio, label = self.loaded_audio[filename], self.label_dict[filename]

        if self.window:
            start_sec = random.randint(0, int(audio.shape[1]/self.sr) - self.window)
            end_sec = start_sec + self.window
            audio = audio[:, start_sec * self.sr:end_sec * self.sr]
            label = self.get_label(filename, start_sec)
            return audio, label, (filename, start_sec)
        else:
            return audio, label