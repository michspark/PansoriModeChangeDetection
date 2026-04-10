import torchaudio

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
import IPython.display as ipd

class DataMonitor:
    def __init__(self, audio_dir, csv_dir, sr, threshold, mode, frame_rate = 100):
        self.mode = mode

        self.audio_paths = sorted(list(Path(audio_dir).rglob('*.wav')))
        self.csv_paths = sorted(list(Path(csv_dir).rglob('*.csv')))

        self.sr = sr
        self.threshold = threshold
        self.frame_rate = frame_rate

        self.audio_files = [path.stem for path in self.audio_paths]
        self.csv_files = [path.stem for path in self.csv_paths]

    def filename_to_idx(self, filename):
        return self.audio_files.index(filename), self.csv_files.index(filename)

    def get_audio(self, audio_idx):
        if isinstance(audio_idx, str):
            audio_idx, _ = self.filename_to_idx(audio_idx)
        y, sr = torchaudio.load(self.audio_paths[audio_idx])
        if y.shape[0]!=1: y=y.mean(dim=0).unsqueeze(0)
        if sr!=self.sr: y=torchaudio.functional.resample(y, sr, self.sr)
        y = y.squeeze(0).numpy()
        return y

    def get_df(self, filename):
        _, csv_idx = self.filename_to_idx(filename)
        df = pd.read_csv(self.csv_paths[csv_idx], header=None, index_col=0)
        return df
    
    def get_pitch_plot(self, filename, filter=True):
        df = self.get_df(filename)
        df.columns = ['frequency', 'confidence']
        frequency, confidence = df['frequency'].values, df['confidence'].values
        if filter:
            frequency[confidence < self.threshold] = np.nan
            frequency = np.nan_to_num(frequency, nan=0.0)
            # from math import log
            # frequency = log(frequency)
        frequency = np.log(frequency)
        plt.plot(frequency)

    def get_crepe_sine(self, csv_idx):
        df = pd.read_csv(self.csv_paths[csv_idx], header=None, index_col=0)
        df.columns = ['frequency', 'confidence']
        frequency, confidence = df['frequency'].values, df['confidence'].values
        frequency[confidence < self.threshold] = np.nan
        frequency = np.nan_to_num(frequency, nan=0.0)

        upsample_factor = self.sr // self.frame_rate
        frequency_resampled = np.repeat(frequency, upsample_factor)
        phi = np.zeros_like(frequency_resampled)
        phi[1:] = np.cumsum(2 * np.pi * frequency_resampled[:-1] / self.sr, axis=0)
        sine = 0.9 * np.sin(phi)
        return sine

    def get_pesto_sine(self, csv_idx):
        df = pd.read_csv(self.csv_paths[csv_idx], header=None, index_col=0)
        df.columns = ['frequency', 'amplitude']
        frequency, amplitude = df['frequency'].values, df['amplitude'].values
        frequency[amplitude < self.threshold] = np.nan
        frequency = np.nan_to_num(frequency, nan=0.0)

        upsample_factor = self.sr // self.frame_rate
        frequency_resampled = np.repeat(frequency, upsample_factor)
        amplitude_resampled = np.repeat(amplitude, upsample_factor)
        phi = np.zeros_like(frequency_resampled)
        phi[1:] = np.cumsum(2 * np.pi * frequency_resampled[:-1] / self.sr, axis=0)
        sine = amplitude_resampled * np.sin(phi)
        return sine

    def synthesize_sine_wave(self, filename, display=True, with_origin=False, sine_wav_amp=1.0):
        audio_idx, csv_idx = self.filename_to_idx(filename)
        if self.mode=='CREPE': sine = self.get_crepe_sine(csv_idx)
        elif self.mode=='PESTO': sine = self.get_pesto_sine(csv_idx)

        if with_origin:
            audio = self.get_audio(audio_idx)
            sine = sine[:audio.shape[0]] * sine_wav_amp
            syn = sine + audio*0.5
        else: syn = sine

        if display: ipd.display(filename, ipd.Audio(data=syn, rate=self.sr))
        return syn