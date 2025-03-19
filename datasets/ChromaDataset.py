from .AudioDataset import *
from torchaudio.prototype.transforms import ChromaSpectrogram

class ChromaDataset(AudioDataset):
    def __init__(self, audio_dir, label_json, num_classes=5, sr=16000, channels='mono', window=20, n_fft=2048, hop_length=512, target_bins=25, aug=True):
        super().__init__(audio_dir, label_json, num_classes, sr, channels, window, aug)
        self.hop_length = hop_length
        self.target_bins = target_bins
        self.window_frame = self.window * sr // hop_length
        self.chroma_cvt = ChromaSpectrogram(sample_rate=self.sr, n_fft=n_fft, hop_length=hop_length)

    def get_chroma(self, audio):
        chroma = self.chroma_cvt(audio).squeeze(0)
        chroma = chroma.clamp(max=1000)
        if self.window_frame != chroma.shape[1]: chroma = chroma[:,:self.window_frame]
        # chroma = chroma.sqrt()
        # chroma = chroma.sqrt().clamp(max=100)

        repeat = self.target_bins // chroma.shape[0] + 1
        chroma = torch.tile(chroma, (repeat, 1))[:self.target_bins, :]

        return chroma

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

    def __len__(self):
        return len(self.loaded_hash)

    def __getitem__(self, idx):
        # get audio segment
        hash_key, filename, audio, label = super().__getitem__(idx)

        # get chroma & expand
        chroma = self.get_chroma(audio)

        # pitch shift aug
        if self.aug: chroma = self.shift_chroma(chroma)

        # ms label => frame label
        label = self.ms_to_frame_label(label)

        return hash_key, filename, chroma, label