import os
import json
import random
import unicodedata
from collections import defaultdict

from tqdm import tqdm

import torch
import torchaudio
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, audio_dir, label_json, num_classes=5, sr=16000, channels='mono', window=20, aug=True):
        super().__init__()
        self.sr = sr
        self.aug = aug
        self.window = window
        self.channels = channels
        self.num_classes = num_classes

        self.label_map = {"Unknown":0, "창조":1, "아니리":1, "설렁제":2, "경드름":2, "우조":2, "평조":2, "계면조": 3}
        # self.label_map = {"Unknown":0, "창조":0, "설렁제":0, "경드름": 0, "우조": 1, "계면조": 2, "평조": 3, "아니리": 4}
        self.loaded_hash, self.loaded_filename, self.loaded_audio = self.get_audio(audio_dir)
        self.loaded_label, self.label_dict = self.get_label(label_json)

    def get_audio(self, audio_dir):
        loaded_hash = []
        loaded_filename, loaded_audio = {}, {}

        for audio_file in tqdm(os.listdir(audio_dir), desc="Load Audio"):
            audio, sr = torchaudio.load(os.path.join(audio_dir, audio_file))
            hash_key = unicodedata.normalize('NFC', audio_file.split("-")[0])
            filename = unicodedata.normalize('NFC', "-".join(audio_file.split(".")[0].split("-")[1:]))

            audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.sr) if self.sr!=sr else audio
            audio = audio.mean(dim=0, keepdim=True) if self.channels=='mono' else audio

            loaded_hash.append(hash_key)
            loaded_filename[hash_key] = filename
            loaded_audio[hash_key] = audio

        return loaded_hash, loaded_filename, loaded_audio
    
    def get_label(self, label_json):
        loaded_label, label_dict = {}, defaultdict(list)
        with open(label_json, 'r', encoding='utf-8') as file: label_data = json.load(file)

        for item in tqdm(label_data, desc='Load Label'):
            hash_key = unicodedata.normalize('NFC', item["file_upload"].split("-")[0])
            if hash_key not in self.loaded_filename: continue

            sample_duration = self.loaded_audio[hash_key].shape[1]
            ms_duration = sample_duration // (self.sr // 1000)
            label = torch.zeros((ms_duration, self.num_classes))
            label[:,0] = 1

            for annotation in item["annotations"]:
                for result in annotation["result"]:
                    # ms 단위로 변환 (1sec = 1000ms)
                    start_ms, end_ms = int(round(result["value"]["start"]*1000)), int(round(result["value"]["end"]*1000))
                    label_ant = self.label_map[result["value"]["labels"][0]]
                    label_dict[hash_key].append({"start": start_ms, "end": end_ms, "label": label_ant})

                    if label_ant:
                        label[start_ms:end_ms, label_ant] = 1
                        label[start_ms:end_ms, 0] = 0

            loaded_label[hash_key] = label

        return loaded_label, label_dict

    def aug_tempo(self, audio, label):
        factor = round(random.uniform(0.8,1.25), 2)
        new_sr = int(self.sr/factor)

        aug_audio = torchaudio.functional.resample(audio, orig_freq=self.sr, new_freq=new_sr)
        aug_sample_duration = aug_audio.shape[1]
        aug_ms_duration = aug_sample_duration // (self.sr // 1000)
        aug_label = torch.zeros((aug_ms_duration, self.num_classes))

        if label.shape[0] != aug_label.shape[0]:
            resize_factor = label.shape[0]/aug_label.shape[0]
            for idx in range(aug_label.shape[0]):
                org_idx = min(int(idx*resize_factor), label.shape[0]-1)
                aug_label[idx] = label[org_idx]
        else: aug_label = label.clone()

        return aug_audio, aug_label

    def __len__(self):
        return len(self.loaded_hash)
    
    def __getitem__(self, idx):
        # get audio, label
        hash_key = self.loaded_hash[idx]
        filename = self.loaded_filename[hash_key]
        audio, label = self.loaded_audio[hash_key], self.loaded_label[hash_key]

        # random crop audio, label
        window_samples = self.window * self.sr
        start = random.randint(0, audio.shape[1] - window_samples)
        audio = audio[:, start:start+window_samples]

        window_ms = self.window * 1000
        start_ms = int(start/self.sr * 1000)
        end_ms = start_ms + window_ms
        label = label[start_ms:end_ms]

        # tempo aug
        if self.aug: audio, label = self.aug_tempo(audio, label)

        #  pad
        if audio.shape[1] < window_samples:
            pad_len = window_samples - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0,pad_len))
        elif audio.shape[1] > window_samples:
            audio = audio[:, :window_samples]
        
        if label.shape[0] < window_ms:
            label_len = label.shape[0]
            pad_len = window_ms - label_len
            label = torch.nn.functional.pad(label, (0,0,0,pad_len))
            label[label_len:, 0] = 1
        elif label.shape[0] > window_ms:
            label = label[:window_ms]

        return hash_key, filename, audio, label