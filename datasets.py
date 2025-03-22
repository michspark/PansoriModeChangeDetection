# TODO
# [ ] Refactor AudioDataset
# [ ] Refactor ChromdaDataset
# [ ] Refactor MelDataset

import os
import json
import random
import unicodedata
from collections import defaultdict

from tqdm import tqdm

import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import Spectrogram, MelScale, AmplitudeToDB, TimeStretch, FrequencyMasking
from torchaudio.prototype.transforms import ChromaScale

class AudioDataset(Dataset):
    def __init__(self, audio_dir, label_json, num_classes=4, sr=16000, channels='mono', window=20, validation_mode=False, margin_ratio=1.0, aug=False):
        super().__init__()
        self.sr = sr
        self.window = window
        self.channels = channels
        self.num_classes = num_classes
        self.validation_mode = validation_mode
        self.margin_ratio = margin_ratio

        self.label_map = {"Unknown":0, "창조":1, "아니리":1, "설렁제":2, "경드름":2, "우조":2, "평조":2, "계면조": 3}
        self.loaded_hash, self.loaded_filename, self.loaded_audio = self.get_audio(audio_dir)
        self.loaded_label, self.label_dict = self.get_label(label_json)
        
        # Slice indices will be a list of tuples (hash_key, start_sample, end_sample)
        self.prepare_slice_indices()
        self.aug = aug

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
    
    def prepare_slice_indices(self, random_offset=True):
        """Prepare slice indices to cover all audio files"""
        self.slice_indices = []
        window_samples = self.window * self.sr
        
        for hash_key in self.loaded_hash:
            audio = self.loaded_audio[hash_key]
            audio_length = audio.shape[1]
            
            # Skip if audio is shorter than window
            if audio_length < window_samples: continue
                
            # Calculate initial offset (random or zero)
            offset = random.randint(0, min(window_samples, audio_length - window_samples)) if random_offset else 0
            
            # Create slices with offset
            current_pos = offset
            while current_pos + window_samples <= audio_length:
                self.slice_indices.append((hash_key, current_pos, current_pos + window_samples))
                current_pos += window_samples
            
            # Add final slice from the end if needed
            if current_pos < audio_length and audio_length - window_samples >= 0:
                final_start = audio_length - window_samples
                self.slice_indices.append((hash_key, final_start, audio_length))

    def update_slice_indices(self):
        """Update slice indices with new random offsets for a new epoch"""
        self.prepare_slice_indices(random_offset=True)

    def apply_audio_augmentation(self, audio):
        """Apply various audio augmentations"""
        # Only apply augmentations with some probability
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
        if self.validation_mode:
            return len(self.loaded_hash)
        return len(self.slice_indices)
    
    def __getitem__(self, idx):
        # Validation mode: return entire audio and label
        if self.validation_mode:
            hash_key = self.loaded_hash[idx]
            filename = self.loaded_filename[hash_key]
            audio, label = self.loaded_audio[hash_key], self.loaded_label[hash_key]
            return hash_key, filename, audio, label
        
        # Training mode: return sliced audio and label
        hash_key, start, end = self.slice_indices[idx]
        audio = self.loaded_audio[hash_key]

        # adjust using margin_ratio
        margin_length = int(self.margin_ratio * (end - start))
        if start - margin_length//2 < 0:
            end = start + margin_length
        elif end + margin_length//2 > audio.shape[1]:
            end = audio.shape[1]
            start = end - margin_length
        else:
            start = start - margin_length//2
            end = end + margin_length//2
        
        filename = self.loaded_filename[hash_key]
        
        # Get the audio slice
        audio = audio[:, start:end]
        
        # Convert sample positions to ms for label slicing
        start_ms = int(start / self.sr * 1000)
        end_ms = int(end / self.sr * 1000)
        
        # Get the label slice
        label = self.loaded_label[hash_key][start_ms:end_ms]
        
        if self.aug: audio = self.apply_audio_augmentation(audio)

        return hash_key, filename, audio, label

      
class MelDataset(AudioDataset):
    def __init__(self, audio_dir, label_json, num_classes=4, sr=16000, channels='mono', window=20, n_fft=2048, hop_length=512, target_bins=40, validation_mode=False, aug=False, margin_ratio=1.6):
        super().__init__(audio_dir, label_json, num_classes, sr, channels, window, validation_mode, margin_ratio=margin_ratio)
        self.hop_length = hop_length
        self.target_bins = target_bins
        self.window_frame = self.window * sr // hop_length
        self.aug = aug
        
        self.spec_cvt = Spectrogram(n_fft=n_fft, hop_length=hop_length, power=1.0)
        self.spec2mel = MelScale(n_stft=n_fft//2+1, n_mels=target_bins, sample_rate=sr, f_min=80, f_max=2000)
        self.db_cvt = AmplitudeToDB()
        
        # Initialize augmentation transforms
        if self.aug:
            self.time_stretch = TimeStretch(hop_length=hop_length, n_freq=n_fft//2+1)
            self.freq_mask = FrequencyMasking(freq_mask_param=10)

    def apply_spec_time_stretch(self, spec):
        """Apply time stretching to spectrogram"""
        stretch_factor = 1.0
        if random.random() < 0.3:
            # Time stretching via linear interpolation along time axis
            stretch_factor = random.uniform(0.7, 1.5)
            spec = self.time_stretch(spec, stretch_factor)
            spec = spec.abs()
        
        return spec, stretch_factor

    def apply_spec_pitch_shift(self, spec):
        """Apply pitch shifting to spectrogram"""
        if random.random() < 0.3:
            # Shift pitch by rolling the frequency axis
            freq_bins = spec.shape[-2]
            shift_steps = random.randint(-4, 4)  # Shift by up to 4 steps
            
            if shift_steps == 0:
                return spec
                
            # Roll along frequency axis
            spec_shifted = torch.roll(spec, shifts=shift_steps, dims=-2)
            
            # Fill in with zeros or duplicate boundary rows
            if shift_steps > 0:  # Shifted up, fill bottom with zeros
                spec_shifted[..., :shift_steps, :] = 0
            else:  # Shifted down, fill top with zeros
                shift_steps = abs(shift_steps)
                spec_shifted[..., -shift_steps:, :] = 0
                
            return spec_shifted
        
        return spec

    def get_mel(self, audio, trim=True):
        # Generate spectrogram
        spec = self.spec_cvt(audio)
        
        # Apply spectrogram augmentations if enabled
        if self.aug and not self.validation_mode:
            spec, stretch_factor = self.apply_spec_time_stretch(spec)
            # spec = self.apply_spec_pitch_shift(spec)
        else:
            stretch_factor = 1.0
        
        # Convert to mel spectrogram
        mel = self.spec2mel(spec).squeeze(0)
        mel = self.db_cvt(mel) / 100
        
        # if trim and self.window_frame != mel.shape[1]: 
        #     mel = mel[:,:self.window_frame]
            
        return mel, stretch_factor
    
    def apply_spec_augmentation(self, mel):
        """Apply spectrogram augmentations"""
        if not self.aug or random.random() > 0.8: return mel
        if random.random() < 0.5: mel = self.apply_spec_pitch_shift(mel)

        # Frequency masking
        if random.random() < 0.5: mel = self.freq_mask(mel)
            
        return mel

    def shift_chroma(self, chroma):
        shift_amount = random.randint(-6, 6)
        chroma = torch.roll(chroma, shifts=shift_amount, dims=0)
        return chroma

    def ms_to_frame_label(self, ms_label):
        ms_per_frame = self.hop_length / self.sr * 1000 # sr=16000 => ms per frame = 32
        num_frames = int(ms_label.shape[0] / ms_per_frame) # 625
        frame_label = torch.zeros((num_frames, self.num_classes))
        
        for frame_idx in range(num_frames):
            frame_start_ms = int(frame_idx * ms_per_frame)
            frame_end_ms = int((frame_idx + 1) * ms_per_frame)
            
            # frame start가 ms label 길이를 벗어나는 경우
            if frame_start_ms >= ms_label.shape[0]: frame_label[frame_idx, 0] = 1
            else:
                frame_end_ms = min(frame_end_ms, ms_label.shape[0])
                ms_segment = ms_label[frame_start_ms:frame_end_ms] # 32, 5
                
                if ms_segment.shape[0] > 0:
                    class_sums = ms_segment.sum(dim=0) # [10, 0, 0, 22, 0] 32ms 간 비중이 높은 클래스 frame class로 할당
                    frame_label[frame_idx] = (class_sums == class_sums.max()).float() # [F F F T F] = [0 0 0 1 0]
                else: frame_label[frame_idx, 0] = 1
                    
        return frame_label
    
    def apply_time_stretch(self, label, stretch_factor):
        """
        Adjust label timing to match time-stretched audio.
        
        Args:
            label: The original label tensor with shape [time_in_ms, num_classes]
            stretch_factor: The factor by which audio has been stretched (>1 means faster, <1 means slower)
        
        Returns:
            The adjusted label tensor with the same shape
        """
        if stretch_factor == 1.0: return label  # No time stretching applied
            
        # Calculate new duration after stretching
        orig_length = label.shape[0]
        new_length = int(orig_length / stretch_factor)
        
        # Create new label tensor (filled with "Unknown" class by default)
        stretched_label = torch.zeros((new_length, label.shape[1]))
        
        # Use interpolation to map from stretched time to original time
        if new_length > 0:
            # Create source indices mapping stretched positions back to original positions
            orig_indices = torch.linspace(0, orig_length - 1, new_length).long()
            
            # Copy labels from original positions to new positions
            stretched_label[:new_length] = label[orig_indices]
            
        return stretched_label

    def __len__(self):
        # Use the parent class __len__ implementation
        return super().__len__()

    def __getitem__(self, idx):
        # Handle differently based on validation mode
        if self.validation_mode:
            # For validation, get the entire audio and label
            hash_key, filename, audio, label = super().__getitem__(idx)
            mel, _ = self.get_mel(audio, trim=False)
            frame_label = self.ms_to_frame_label(label)
            return hash_key, filename, mel, frame_label
        else:
            # For training, get the sliced audio segment
            hash_key, filename, audio, label = super().__getitem__(idx)

            # Convert to mel spectrogram with integrated spectrogram augmentations
            mel, stretch_factor = self.get_mel(audio)
            
            # slice mel to window_frame
            start_idx = (mel.shape[1] - self.window_frame) // 2
            mel = mel[:,start_idx:start_idx+self.window_frame]
            
            # Apply additional mel spectrogram augmentations
            if self.aug: mel = self.apply_spec_augmentation(mel)
            
            # Apply time stretch to label
            label = self.apply_time_stretch(label, stretch_factor)
            frame_label = self.ms_to_frame_label(label)
            frame_label = frame_label[start_idx:start_idx+self.window_frame]
            assert frame_label.shape[0] == mel.shape[1], f"frame_label.shape[0] != mel.shape[1]: {frame_label.shape[0]} != {mel.shape[1]}, time stretch factor: {stretch_factor}, audio length: {audio.shape[1]/self.sr} sec"
            return hash_key, filename, mel, frame_label



class ChromaDataset(AudioDataset):
    def __init__(self, audio_dir, label_json, num_classes=4, sr=16000, channels='mono', window=20, n_fft=2048, hop_length=512, target_bins=40, validation_mode=False, aug=False, margin_ratio=1.6):
        super().__init__(audio_dir, label_json, num_classes, sr, channels, window, validation_mode, margin_ratio=margin_ratio)
        self.hop_length = hop_length
        self.target_bins = target_bins
        self.window_frame = self.window * sr // hop_length
        self.aug = aug

        self.spec_cvt = Spectrogram(n_fft=n_fft, hop_length=hop_length, power=1.0)
        self.spec2chroma = ChromaScale(n_freqs=n_fft//2+1, n_chroma=12, sample_rate=sr)

        # Initialize augmentation transforms
        if self.aug:
            self.time_stretch = TimeStretch(hop_length=hop_length, n_freq=n_fft//2+1)
            self.freq_mask = FrequencyMasking(freq_mask_param=10)

    def get_chroma(self, audio, trim=True):
        # Generate spectrogram
        spec = self.spec_cvt(audio)
        
        # Apply spectrogram augmentations if enabled
        if self.aug and not self.validation_mode: spec, stretch_factor = self.apply_spec_time_stretch(spec)
        else: stretch_factor = 1.0
        
        # Convert to mel spectrogram
        chroma = self.spec2chroma(spec).squeeze(0)
        chroma = chroma.clamp(max=1000)

        repeat = self.target_bins // chroma.shape[0] + 1
        chroma = torch.tile(chroma, (repeat, 1))[:self.target_bins, :]

        return chroma, stretch_factor

    def apply_spec_time_stretch(self, spec):
        """Apply time stretching to spectrogram"""
        stretch_factor = 1.0
        if random.random() < 0.3:
            # Time stretching via linear interpolation along time axis
            stretch_factor = random.uniform(0.7, 1.5)
            spec = self.time_stretch(spec, stretch_factor)
            spec = spec.abs()
        
        return spec, stretch_factor

    def apply_spec_pitch_shift(self, spec):
        """Apply pitch shifting to spectrogram"""

        if random.random() < 0.3:
            # Shift pitch by rolling the frequency axis
            shift_steps = random.randint(-4, 4)  # Shift by up to 4 steps
            if shift_steps == 0: return spec

            # Roll along frequency axis
            spec_shifted = torch.roll(spec, shifts=shift_steps, dims=-2)
            
            # Fill in with zeros or duplicate boundary rows
            if shift_steps > 0:  # Shifted up, fill bottom with zeros
                spec_shifted[..., :shift_steps, :] = 0
            else:  # Shifted down, fill top with zeros
                shift_steps = abs(shift_steps)
                spec_shifted[..., -shift_steps:, :] = 0
                
            return spec_shifted
        
        return spec

    def apply_spec_augmentation(self, mel):
        """Apply spectrogram augmentations"""
        if not self.aug or random.random() > 0.8: return mel
        if random.random() < 0.5: mel = self.apply_spec_pitch_shift(mel)

        # Frequency masking
        if random.random() < 0.5: mel = self.freq_mask(mel)
            
        return mel

    def ms_to_frame_label(self, ms_label):
        ms_per_frame = self.hop_length / self.sr * 1000 # sr=16000 => ms per frame = 32
        num_frames = int(ms_label.shape[0] / ms_per_frame) # 625
        frame_label = torch.zeros((num_frames, self.num_classes))
        
        for frame_idx in range(num_frames):
            frame_start_ms = int(frame_idx * ms_per_frame)
            frame_end_ms = int((frame_idx + 1) * ms_per_frame)
            
            # frame start가 ms label 길이를 벗어나는 경우
            if frame_start_ms >= ms_label.shape[0]: frame_label[frame_idx, 0] = 1
            else:
                frame_end_ms = min(frame_end_ms, ms_label.shape[0])
                ms_segment = ms_label[frame_start_ms:frame_end_ms] # 32, 5
                
                if ms_segment.shape[0] > 0:
                    class_sums = ms_segment.sum(dim=0) # [10, 0, 0, 22, 0] 32ms 간 비중이 높은 클래스 frame class로 할당
                    frame_label[frame_idx] = (class_sums == class_sums.max()).float() # [F F F T F] = [0 0 0 1 0]
                else: frame_label[frame_idx, 0] = 1
                    
        return frame_label
    
    def apply_time_stretch(self, label, stretch_factor):
        """
        Adjust label timing to match time-stretched audio.
        
        Args:
            label: The original label tensor with shape [time_in_ms, num_classes]
            stretch_factor: The factor by which audio has been stretched (>1 means faster, <1 means slower)
        
        Returns:
            The adjusted label tensor with the same shape
        """
        if stretch_factor == 1.0: return label  # No time stretching applied
            
        # Calculate new duration after stretching
        orig_length = label.shape[0]
        new_length = int(orig_length / stretch_factor)
        
        # Create new label tensor (filled with "Unknown" class by default)
        stretched_label = torch.zeros((new_length, label.shape[1]))
        
        # Use interpolation to map from stretched time to original time
        if new_length > 0:
            # Create source indices mapping stretched positions back to original positions
            orig_indices = torch.linspace(0, orig_length - 1, new_length).long()
            
            # Copy labels from original positions to new positions
            stretched_label[:new_length] = label[orig_indices]
            
        return stretched_label

    def __len__(self):
        # Use the parent class __len__ implementation
        return super().__len__()

    def __getitem__(self, idx):
        # Handle differently based on validation mode
        if self.validation_mode:
            # For validation, get the entire audio and label
            hash_key, filename, audio, label = super().__getitem__(idx)
            chroma, _ = self.get_chroma(audio, trim=False)
            frame_label = self.ms_to_frame_label(label)
            return hash_key, filename, chroma, frame_label
        else:
            # For training, get the sliced audio segment
            hash_key, filename, audio, label = super().__getitem__(idx)

            # Convert to mel spectrogram with integrated spectrogram augmentations
            chroma, stretch_factor = self.get_chroma(audio)
            
            # slice mel to window_frame
            start_idx = (chroma.shape[1] - self.window_frame) // 2
            chroma = chroma[:,start_idx:start_idx+self.window_frame]
            
            # Apply additional mel spectrogram augmentations
            if self.aug: chroma = self.apply_spec_augmentation(chroma)
            
            # Apply time stretch to label
            label = self.apply_time_stretch(label, stretch_factor)
            frame_label = self.ms_to_frame_label(label)
            frame_label = frame_label[start_idx:start_idx+self.window_frame]
            assert frame_label.shape[0] == chroma.shape[1], f"frame_label.shape[0] != mel.shape[1]: {frame_label.shape[0]} != {chroma.shape[1]}, time stretch factor: {stretch_factor}, audio length: {audio.shape[1]/self.sr} sec"
            return hash_key, filename, chroma, frame_label