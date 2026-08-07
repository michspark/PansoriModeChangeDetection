from models.model_zoo import Conv2DGRU
from datasets.dataset import BaseDataset
from datasets.dataset_utils import PianoRollGenerator
from utils import get_all_song_names, create_kfold_splits, plot_posteriorgram, save_test_csv, plot_confusion_matrix
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import hydra
import os
import numpy as np

def plot_posteriorgram(song_name, pred_probs):
    """
    pred_probs : (T, 5) numpy array, softmax probabilities
    Returns a matplotlib Figure.
    """
    CLASS_NAMES = ['no label', 'Ujo', 'GMjo', 'ANR', 'CJO']

    T = pred_probs.shape[0]
    n = len(CLASS_NAMES)

    fig, ax = plt.subplots(1, 1, figsize=(16, 3), constrained_layout=True)
    fig.suptitle(song_name, fontsize=10)

    im = ax.imshow(np.flipud(pred_probs.T), aspect='auto', origin='lower',
                   vmin=0, vmax=1, cmap='gray_r', interpolation='nearest',
                   extent=[0, T, -0.5, n - 0.5])
    ax.set_yticks(list(range(n)))
    ax.set_yticklabels(CLASS_NAMES[::-1])
    ax.set_title('Predicted Posteriorgram')
    ax.set_xlabel('Frame')
    fig.colorbar(im, ax=ax, label='Probability', shrink=0.8)

    return fig


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(cfg):
    midi_dir = '/home/sangheon/Desktop/Pansori_Data/JBG_rosvot'
    for filename in os.listdir(midi_dir):
        midi_path = os.path.join(midi_dir, filename)
        model_ckpt = '/home/sangheon/Desktop/PansoriMIDIDetection/outputs/2026-03-17/01-05-52/best_model_fold3.pt'
        device = cfg.device if torch.cuda.is_available() else 'cpu'
        fs = cfg.data.fs
        window_size = int(cfg.data.window_size * fs)

        model = Conv2DGRU(cfg.model).to(device)
        ckpt = torch.load(model_ckpt, map_location=device)
        state_dict = ckpt.get('state_dict', ckpt)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        gen = PianoRollGenerator(str(midi_path), fs=fs)
        piano_roll = torch.tensor(gen.generate_piano_roll(), dtype=torch.float)
        total_frames = piano_roll.shape[1]

        num_classes = cfg.model.num_classes
        segments = list(range(0, total_frames, window_size))

        all_probs = []
        with torch.no_grad():
            for start in segments:
                clip = piano_roll[:, start : start + window_size]
                if clip.shape[1] < window_size:
                    pad = np.zeros((128, window_size - clip.shape[1]))
                    clip = np.concatenate([clip, pad], axis=1)

                x = torch.tensor(clip, dtype=torch.float32).unsqueeze(0).to(device)
                logits = model(x)
                probs  = torch.softmax(logits, dim=-1)
                probs  = probs.squeeze(0).cpu().numpy()
                all_probs.append(probs)

        output = np.concatenate(all_probs, axis=0)

        song_title = os.path.splitext(os.path.basename(midi_path))[0]
        posterior = plot_posteriorgram(song_title, output)

        os.makedirs('/home/sangheon/Desktop/PansoriMIDIDetection/analysis_jbg', exist_ok=True)
        posterior.savefig(f'/home/sangheon/Desktop/PansoriMIDIDetection/analysis_jbg/{song_title}.png', dpi=150, bbox_inches='tight')

if __name__ == "__main__":
    main()
