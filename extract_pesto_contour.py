import os
from math import log
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torchaudio
import torch.nn as nn

from pesto.model import Resnet1d
from pesto.data import Preprocessor
from pesto.utils import CropCQT, reduce_activations

class PESTO(nn.Module):
    def __init__(self, encoder, preprocessor, crop_kwargs, reduction = "alwa"):
        super(PESTO, self).__init__()
        self.encoder = encoder
        self.preprocessor = preprocessor

        if crop_kwargs is None: crop_kwargs = {}
        self.crop_cqt = CropCQT(**crop_kwargs)

        self.reduction = reduction

        self.register_buffer('shift', torch.zeros((), dtype=torch.float), persistent=True)

    def forward(self, audio_waveforms, sr, convert_to_freq, return_activations = True):
        batch_size = audio_waveforms.size(0) if audio_waveforms.ndim == 2 else None
        x = self.preprocessor(audio_waveforms, sr=sr).flatten(0, 1) # HCQT frame

        energy = x.mul_(log(10) / 10.).exp().squeeze_(1)
        vol = energy.sum(dim=-1)
        vol = vol.view(batch_size, -1)

        x = self.crop_cqt(x)
        activations = self.encoder(x)
        activations = activations.view(batch_size, -1, activations.size(-1))
        activations = activations.roll(-round(self.shift.cpu().item() * self.bins_per_semitone), -1)
        preds = reduce_activations(activations, reduction=self.reduction)

        if convert_to_freq: preds = 440 * 2 ** ((preds - 69) / 12)
        if return_activations: return preds, vol, activations

        return preds, vol

    @property
    def bins_per_semitone(self):
        return self.preprocessor.hcqt_kwargs["bins_per_semitone"]

    @property
    def hop_size(self):
        r"""Returns the hop size of the model (in milliseconds)"""
        return self.preprocessor.hop_size


def remap_state_dict_keys(old_state_dict):
    new_state_dict = {}
    for key, value in old_state_dict.items():
        if key.startswith("encoder.prefilt_layers."):
            parts = key.split(".")
            if len(parts) >= 4 and parts[3] not in {"0", "1"}: new_key = ".".join(parts[:3] + ["0"] + parts[3:])
            else: new_key = key
        else: new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

def load_pesto(ckpt, hop_size):
    checkpoint = torch.load(ckpt, weights_only=False)
    hparams = checkpoint['hparams']
    hcqt_params = checkpoint['hcqt_params']

    encoder = Resnet1d(**hparams["encoder"])
    preprocessor = Preprocessor(hop_size=hop_size, **hcqt_params)
    model = PESTO(encoder=encoder, preprocessor=preprocessor, crop_kwargs=hparams['pitch_shift'])

    state_dict = checkpoint["state_dict"]
    state_dict = remap_state_dict_keys(state_dict)
    model.load_state_dict(state_dict, strict=False)

    return model

def predict(x, sr, model, num_chunks = 1, convert_to_freq = True):
    preds, vols, activations = [], [], []
    try:
        for chunk in x.chunk(dim=-1, chunks=num_chunks):
            pred, vol = model(chunk, sr=sr, convert_to_freq=convert_to_freq, return_activations=False)
            preds.append(pred)
            vols.append(vol)
    except torch.cuda.OutOfMemoryError:
        raise torch.cuda.OutOfMemoryError("Got an out-of-memory error while performing pitch estimation. "
                                          "Please increase the number of chunks with option `-c`/`--chunks` "
                                          "to reduce GPU memory usage.")

    preds = torch.cat(preds, dim=0)
    vols = torch.cat(vols, dim=0)

    return preds, vols


def predict_from_files(audio_files, output_path, model, device, num_chunk = 1, convert_to_freq = True):
    pbar = tqdm(audio_files)

    with torch.inference_mode():
        for file in pbar:
            pbar.set_description(file.stem)
            # if os.path.exists(f"{output_path}/{file.stem}.csv"): continue

            try: x, sr = torchaudio.load(file)
            except Exception as e:
                print(e, f"Skipping {file}...")
                continue

            if x.shape[0]!=1: x = x.mean(dim=0).unsqueeze(0)
            x = x.to(device)

            # compute the predictions
            predictions = predict(x, sr, model=model, num_chunks=num_chunk, convert_to_freq=convert_to_freq)
            preds, vols = predictions

            if num_chunk!=1:
                preds = preds.flatten()
                vols = vols.flatten()

            # compute timesteps
            timesteps = torch.arange(preds.size(-1), device=x.device) / (1000/model.hop_size)
            timesteps = timesteps.detach().cpu().numpy()
            timesteps = [f"{val:.2f}" for val in timesteps]

            preds = preds.squeeze(0).detach().cpu().numpy()
            vols = vols.squeeze(0).detach().cpu().numpy()

            amplitude = np.sqrt(vols)
            if amplitude.max()>0: amplitude/=amplitude.max()

            df = pd.DataFrame(zip(timesteps, preds, amplitude))
            df.to_csv(f"{output_path}/{file.stem}.csv", index=None, header=None)
            # return predictions


def main():
    wav_path_list = list(Path('data/SeparatedAudio').rglob('*.wav'))
    output_path = 'data/PestoPitch'

    hop_size = 10.
    ckpt = 'pesto-full/logs/train/runs/2025-05-10_16-29-08/PESTO/9ki35fve/checkpoints/epoch=49-step=622700.ckpt'
    model = load_pesto(ckpt, hop_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    predict_from_files(wav_path_list, output_path, model, device, num_chunk=2, convert_to_freq=True)