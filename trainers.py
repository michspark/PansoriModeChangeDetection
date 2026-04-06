import datetime
import gc
import os
import re
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import seaborn as sns
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from omegaconf import OmegaConf
from scipy.ndimage import gaussian_filter1d

T = datetime.datetime.now().strftime("%m%d_%H%M")


def plot_posteriorgram(song_name, gt, pred_probs, class_names):
    """
    gt         : (T, C) numpy array, one-hot ground truth
    pred_probs : (T, C) numpy array, softmax probabilities
    class_names: list of class name strings (length C)
    Returns a matplotlib Figure.
    """
    _CMAP = plt.cm.get_cmap('tab10')
    n = len(class_names)
    _CLASS_COLORS = [_CMAP(i) for i in range(n)]

    T_len = gt.shape[0]
    gt_labels = np.argmax(gt, axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(16, 5), sharex=True, constrained_layout=True)
    fig.suptitle(song_name, fontsize=10)

    gt_cmap = mcolors.ListedColormap([_CLASS_COLORS[i] for i in range(n)])
    axes[0].imshow(gt_labels[np.newaxis, :], aspect='auto', origin='lower',
                   cmap=gt_cmap, vmin=-0.5, vmax=n - 0.5, interpolation='nearest',
                   extent=[0, T_len, -0.5, 0.5])
    axes[0].set_yticks([0])
    axes[0].set_yticklabels(['class'])
    axes[0].set_title('Ground Truth')

    legend_handles = [Patch(color=_CLASS_COLORS[i], label=class_names[i]) for i in range(n)]
    axes[0].legend(handles=legend_handles, loc='upper right', fontsize=8, framealpha=0.7)

    im = axes[1].imshow(np.flipud(pred_probs.T), aspect='auto', origin='lower',
                        vmin=0, vmax=1, cmap='gray_r', interpolation='nearest',
                        extent=[0, T_len, -0.5, n - 0.5])
    axes[1].set_yticks(list(range(n)))
    axes[1].set_yticklabels(class_names[::-1])
    axes[1].set_title('Predicted Posteriorgram')

    axes[-1].set_xlabel('Frame')
    fig.colorbar(im, ax=axes[1], label='Probability', shrink=0.8)

    return fig


class Trainer:
    def __init__(self, model, optimizer, dataset, criterion, device, save_dir, config):
        self.model = model
        self.model_state_dict = deepcopy(self.model.state_dict())
        self.optimizer = optimizer
        self.dataset = dataset
        self.device = device
        self.criterion = criterion

        self.save_dir = Path(save_dir)

        self.config = config
        self.batch_size = config.train.batch_size
        self.num_iterations = config.train.num_iterations
        self.eval_interval = config.train.get('eval_interval', 200)
        self.save_interval = config.train.get('save_interval', 1000)
        self.early_stopping_patience = config.train.get('early_stopping_patience', None)

        self.global_step = 0
        self.fold_names = None
        self.model.to(self.device)


    def plot_confusion_matrix(self, output, y, class_names):
        true_flat = y.reshape(-1).cpu().numpy()
        pred_flat = output.reshape(-1).cpu().numpy()

        assert len(true_flat) == len(pred_flat), f"Length mismatch: {len(true_flat)} vs {len(pred_flat)}"

        cm = confusion_matrix(true_flat, pred_flat, normalize='true') # , labels=list(range(self.dataset.num_classes)))

        fig = plt.figure(figsize=(14, 14))
        ax = fig.add_subplot(1, 1, 1)
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
        plt.title(f'Confusion Matrix', size=20)
        plt.ylabel('True Label', size=16)
        plt.xlabel('Predicted Label', size=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        image_np = np.array(image)
        plt.close(fig)
        buf.close()
        return image_np


    def init_selection(self):
        if self.config.train.selection == 'KFold':
            kfold = KFold(**self.config.kfold)
            selection_list = list(kfold.split(self.dataset.loaded_hash))

            selection = []
            for train_idx, test_idx in selection_list:
                train_hash_keys = [self.dataset.loaded_hash[i] for i in train_idx]
                test_hash_keys = [self.dataset.loaded_hash[i] for i in test_idx]
                selection.append([train_hash_keys, test_hash_keys])

        elif self.config.train.selection == 'Stratify':
            fold_dir = Path(self.config.train.stratified_fold_dir)
            fold_files = sorted(fold_dir.glob('*.txt'))

            chunks = []
            self.fold_names = [f.stem for f in fold_files]
            for fold_file in fold_files:
                lines = [l.strip() for l in fold_file.read_text(encoding='utf-8').splitlines() if l.strip()]
                hash_keys = [line.split('-')[0] for line in lines]
                hash_keys = [hk for hk in hash_keys if hk in self.dataset.loaded_hash]
                chunks.append(hash_keys)
                print(f"  {fold_file.name}: {len(hash_keys)} in dataset")

            k = len(chunks)
            selection = []
            for i in range(k):
                val_i = (i + 1) % k
                train_keys = [hk for j, chunk in enumerate(chunks) if j != i and j != val_i for hk in chunk]
                selection.append([train_keys, chunks[val_i], chunks[i]])

        elif self.config.train.selection == 'StratifyHalf':
            fold_dir = Path(self.config.train.stratified_half_fold_dir)
            fold_files = sorted(fold_dir.glob('*.txt'))

            from collections import defaultdict
            halves = defaultdict(dict)
            self.fold_names = []
            for fold_file in fold_files:
                m = re.match(r'^(.+)_([12])$', fold_file.stem)
                if not m:
                    continue
                base, idx = m.group(1), int(m.group(2))
                lines = [l.strip() for l in fold_file.read_text(encoding='utf-8').splitlines() if l.strip()]
                hash_keys = [line.split('-')[0] for line in lines]
                hash_keys = [hk for hk in hash_keys if hk in self.dataset.loaded_hash]
                halves[base][idx] = hash_keys
                print(f"  {fold_file.name}: {len(hash_keys)} in dataset")

            genres = sorted(halves.keys())
            self.fold_names = [f"{g}_{v}v{t}t" for g in genres for v, t in [(1, 2), (2, 1)]]
            selection = []
            for held_out in genres:
                train_keys = [hk for g in genres if g != held_out for hk in halves[g][1] + halves[g][2]]
                for val_idx, test_idx in [(1, 2), (2, 1)]:
                    val_keys  = halves[held_out][val_idx]
                    test_keys = halves[held_out][test_idx]
                    selection.append([train_keys, val_keys, test_keys])

        elif self.config.train.selection == 'Artist':
            selection = []
            df = pd.read_csv('data/Stratify/stratify.csv')

            folds = sorted(set(df['artist_stratify']))
            for fold in folds:
                print(set(df[df['artist_stratify']==fold]['artist'].tolist()))
                all_hash_keys = set(df['hash_key'].tolist())

                test_hash_keys = df[df['artist_stratify']==fold]['hash_key'].tolist()
                valid_hash_keys = df[df['artist_stratify']==(fold+1)]['hash_key'].tolist() if fold<(len(folds)-1) else df[df['artist_stratify']==0]['hash_key'].tolist()
                train_hash_keys = list(all_hash_keys-set(test_hash_keys)-set(valid_hash_keys))

                selection.append([train_hash_keys, valid_hash_keys, test_hash_keys])

        elif self.config.train.selection == 'RandomSplit':
            all_keys = list(self.dataset.loaded_hash)
            train_val_keys, test_keys = train_test_split(all_keys, test_size=0.1, random_state=self.config.train.random_seed, shuffle=True)
            train_keys, val_keys = train_test_split(train_val_keys, test_size=1/9, random_state=self.config.train.random_seed, shuffle=True)
            selection = [[train_keys, val_keys, test_keys]]

        elif self.config.train.selection == 'SharedFold':
            fold_dir = Path(self.config.train.shared_fold_dir)
            data_dir = Path(self.config.data.data_dir)
            k = self.config.train.get('k_folds', 10)

            # Build normalized_key -> hash_key mapping from audio filenames
            def _audio_key(fname):
                n = fname.replace('_vocal.wav', '').replace('_vocal.mid', '').replace('_vocal.f0.csv', '')
                n = re.sub(r'^[0-9a-f]+-\d+-', '', n)
                return n

            key_to_hash = {}
            for fname in os.listdir(data_dir):
                if fname.endswith('.wav') or fname.endswith('.mid') or fname.endswith('.csv'):
                    hk = fname.split('-')[0]
                    nk = _audio_key(fname)
                    key_to_hash[nk] = hk

            # Load fold chunks: normalized_key -> hash_key
            chunks = []
            for i in range(1, k + 1):
                fold_file = fold_dir / f'fold_{i:02d}.txt'
                norm_keys = [line.strip() for line in fold_file.read_text(encoding='utf-8').splitlines() if line.strip()]
                hash_keys = [key_to_hash[nk] for nk in norm_keys if nk in key_to_hash]
                # Keep only those present in the dataset
                hash_keys = [hk for hk in hash_keys if hk in self.dataset.loaded_hash]
                chunks.append(hash_keys)
                print(f"  Fold {i:2d}: {len(norm_keys)} keys -> {len(hash_keys)} in dataset")

            # Build train/val/test selections (rolling window: test=i, val=i+1)
            selection = []
            for i in range(k):
                val_i = (i + 1) % k
                train_keys = [hk for j in range(k) if j != i and j != val_i for hk in chunks[j]]
                selection.append([train_keys, chunks[val_i], chunks[i]])

        else: Exception("Have to select config.train.selection: [KFold, Stratify, RandomSplit, SharedFold]")

        return selection


    def init_new_fold(self, fold, fold_name=None):
        if wandb.run is not None: wandb.finish()
        fold_label = fold_name if fold_name else str(fold)
        run_name = f"{self.config.data.data_dir.split('/')[-1]}_{self.config.model.name}_{self.config.dataset.name}_Fold{fold_label}_{T}"
        wandb.init(project='Pansori_Artist', name=run_name, group=f'{self.config.dataset.name}_{self.config.train.selection}_{T}', reinit=True)
        wandb.config.update(OmegaConf.to_container(self.config))

        self.model.load_state_dict(self.model_state_dict)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.config.train.lr)
        print(f"{'='*25} Fold {fold_label} {'='*25}")


    def split_train_valid_test(self, hash_keys):
        if len(hash_keys)==2:
            train_hash_keys, test_hash_keys = hash_keys
            train_hash_keys, valid_hash_keys = train_test_split(train_hash_keys, test_size=1/9, random_state=self.config.train.random_seed, shuffle=True)
        elif len(hash_keys)==3:
            train_hash_keys, valid_hash_keys, test_hash_keys = hash_keys

        return train_hash_keys, valid_hash_keys, test_hash_keys

    def get_masked_acc(self, output, y, target_mask='Unknown'):
        mask = (y!=self.dataset.label_map[target_mask])
        total_valid = mask.sum()
        pred_masked = torch.where(mask, output, torch.tensor(-1))

        acc = (output == y).float().mean().cpu().item()
        acc_masked = ((pred_masked == y).sum()/total_valid).cpu().item()
        return acc, acc_masked

    def get_per_class_acc(self, output, y):
        per_class_acc = {}
        label_map = self.dataset.label_map
        ignore_idx = label_map['Unknown']
        inv_label_map = {v: k for k, v in label_map.items() if v != ignore_idx}
        for cls_idx, cls_name in inv_label_map.items():
            mask = (y == cls_idx)
            if mask.sum() == 0:
                per_class_acc[cls_name] = 0.0
            else:
                per_class_acc[cls_name] = ((output[mask] == cls_idx).float().sum() / mask.sum()).cpu().item()
        return per_class_acc

    def get_per_class_f1(self, output, y, ignore_label=None):
        label_map = self.dataset.label_map
        ignore_idx = label_map.get(ignore_label) if ignore_label else None
        inv_label_map = {v: k for k, v in label_map.items() if v != ignore_idx}

        true_np = y.cpu().numpy()
        pred_np = output.cpu().numpy()

        if ignore_idx is not None:
            mask = true_np != ignore_idx
            true_np = true_np[mask]
            pred_np = pred_np[mask]

        labels = sorted(inv_label_map.keys())
        f1_scores = f1_score(true_np, pred_np, labels=labels, average=None, zero_division=0)
        macro_f1 = f1_score(true_np, pred_np, labels=labels, average='macro', zero_division=0)

        per_class_f1 = {inv_label_map[lbl]: float(f1_scores[i]) for i, lbl in enumerate(labels)}
        return per_class_f1, macro_f1



class FrameTrainer(Trainer):
    def __init__(self, model, optimizer, dataset, criterion, device, save_dir, config):
        super().__init__(model, optimizer, dataset, criterion, device, save_dir, config)

    def train_batch(self, x, y):
        self.model.train()
        x, y = x.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(x)
        loss = self.criterion(outputs.permute(0,2,1), y.argmax(dim=-1))
        loss.backward()
        self.optimizer.step()

        batch_outputs = outputs.detach().argmax(dim=-1).view(-1)
        batch_acc, batch_acc_masked = self.get_masked_acc(batch_outputs, y.argmax(dim=-1).view(-1), target_mask='Unknown')

        return loss.item(), batch_acc, batch_acc_masked


    def evaluate(self, dataset):
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        total_loss = 0
        all_outputs, all_labels = [], []

        with torch.no_grad():
            for batch in dataloader:
                _, x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                if outputs.shape[1] != y.shape[1]: outputs = outputs[:, :y.shape[1]]
                loss = self.criterion(outputs.permute(0,2,1), y.argmax(dim=-1))
                total_loss += loss.item()

                probs = torch.softmax(outputs, dim=-1).cpu().numpy()
                smoothed = gaussian_filter1d(probs, self.config.filter.sigma, axis=1)
                outputs = torch.from_numpy(smoothed).to(self.device)

                all_outputs.append(outputs.detach())
                all_labels.append(y)

        all_outputs, all_labels = torch.cat(all_outputs, dim=1).argmax(dim=-1).view(-1), torch.cat(all_labels, dim=1).argmax(dim=-1).view(-1)
        valid_loss = total_loss / len(dataset)
        valid_acc, valid_acc_masked = self.get_masked_acc(all_outputs, all_labels, target_mask='Unknown')
        per_class_acc = self.get_per_class_acc(all_outputs, all_labels)
        per_class_f1, macro_f1 = self.get_per_class_f1(all_outputs, all_labels, ignore_label='Unknown')
        cm = self.plot_confusion_matrix(all_outputs, all_labels, range(dataset.num_classes))

        return valid_loss, valid_acc, valid_acc_masked, per_class_acc, per_class_f1, macro_f1, cm


    def evaluate_test(self, dataset):
        """Like evaluate(), but also collects per-song GT and softmax probs for posteriorgram generation."""
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        total_loss = 0
        all_preds, all_labels = [], []
        song_data = {}

        with torch.no_grad():
            for batch in dataloader:
                name, x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                if outputs.shape[1] != y.shape[1]:
                    outputs = outputs[:, :y.shape[1]]

                loss = self.criterion(outputs.permute(0, 2, 1), y.argmax(dim=-1))
                total_loss += loss.item()

                pred_probs = torch.softmax(outputs, dim=-1)  # (1, T, C)
                smoothed = gaussian_filter1d(pred_probs.cpu().numpy(), self.config.filter.sigma, axis=1)
                outputs = torch.from_numpy(smoothed).to(self.device)

                all_preds.append(outputs.detach().argmax(dim=-1).view(-1).cpu())
                all_labels.append(y.argmax(dim=-1).view(-1).cpu())

                sname = name[0] if isinstance(name, (list, tuple)) else name
                if sname not in song_data:
                    song_data[sname] = {'gt': [], 'pred_probs': []}
                song_data[sname]['gt'].append(y[0].cpu())
                song_data[sname]['pred_probs'].append(torch.from_numpy(smoothed[0]))

        for sname in song_data:
            song_data[sname]['gt'] = torch.cat(song_data[sname]['gt'], dim=0).numpy()
            song_data[sname]['pred_probs'] = torch.cat(song_data[sname]['pred_probs'], dim=0).numpy()

        all_preds = torch.cat(all_preds).view(-1)
        all_labels = torch.cat(all_labels).view(-1)
        test_loss = total_loss / len(dataset)
        test_acc, test_acc_masked = self.get_masked_acc(all_preds, all_labels, target_mask='Unknown')
        per_class_acc = self.get_per_class_acc(all_preds, all_labels)
        per_class_f1, macro_f1 = self.get_per_class_f1(all_preds, all_labels, ignore_label='Unknown')
        cm = self.plot_confusion_matrix(all_preds, all_labels, range(dataset.num_classes))

        return test_loss, test_acc, test_acc_masked, per_class_acc, per_class_f1, macro_f1, cm, song_data


    def save_posteriorgrams(self, song_data, out_dir, class_names):
        """Save a posteriorgram PNG for each song, then free memory."""
        out_dir.mkdir(parents=True, exist_ok=True)
        for sname, data in song_data.items():
            stem = Path(sname).stem
            fig = plot_posteriorgram(sname, data['gt'], data['pred_probs'], class_names)
            fig.savefig(out_dir / f"{stem}.png", dpi=120, bbox_inches='tight')
            fig.clf()
            plt.close(fig)
            del data['gt'], data['pred_probs']
        plt.close('all')
        gc.collect()


    def train(self):
        fold_best_acc = {}
        target_folds = self.config.train.get('target_folds', None)

        # Define selection
        selection = self.init_selection()

        # Start fold loop
        for fold, hash_keys in enumerate(selection, start=1):
            if target_folds is not None and fold not in target_folds:
                continue

            # init fold wandb, model, & optimizer
            fold_name = self.fold_names[fold - 1] if self.fold_names else None
            self.init_new_fold(fold, fold_name=fold_name)

            self.global_step, current_epoch = 0, 0
            best_acc, best_iteration = 0, 0
            patience_counter = 0

            # Get train-test split
            train_hash_keys, valid_hash_keys, test_hash_keys = self.split_train_valid_test(hash_keys)
            self.dataset.get_split(train_hash_keys, split='train')
            validset = self.dataset.get_split(valid_hash_keys, split='valid')
            testset = self.dataset.get_split(test_hash_keys, split='test')
            print('Trainset:', len(train_hash_keys), 'Validset:', len(valid_hash_keys), 'Testset:', len(test_hash_keys))
            train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

            # Start iter loop
            pbar = tqdm(total=self.num_iterations, desc=f"Fold {fold}")
            while self.global_step < self.num_iterations:
                # Forward
                for batch in train_loader:
                    _, x, y = batch
                    loss, batch_acc, batch_acc_masked = self.train_batch(x, y)
                    wandb.log({"Train/Loss": loss, "Train/ACC": batch_acc, "Train/Masked Acc": batch_acc_masked}, step=self.global_step)

                    self.global_step += 1
                    pbar.update(1)
                    pbar.set_description(f"Fold {fold} | Iter {self.global_step}/{self.num_iterations} | Loss: {loss:.4f}, Best Acc : {best_acc:.4f}")

                    # Evaluate on validation set
                    if self.global_step % self.eval_interval == 0:
                        print(f"\n[Step {self.global_step}] Running evaluation...")
                        val_loss, val_acc, val_acc_masked, per_class_acc, per_class_f1, macro_f1, cm = self.evaluate(validset)
                        print(f"[Step {self.global_step}] Eval done. Val Masked Acc: {val_acc_masked:.4f}")
                        wandb.log({"Valid/Loss": val_loss, "Valid/Acc": val_acc, "Valid/Masked Acc": val_acc_masked,
                                   "Valid/Macro F1": macro_f1,
                                   **{f"Valid/Acc_{k}": v for k, v in per_class_acc.items()},
                                   **{f"Valid/F1_{k}": v for k, v in per_class_f1.items()},
                                   "Valid Confusion Matrix": wandb.Image(Image.fromarray(cm))}, step=self.global_step)
                        pbar.set_description(f"Fold {fold} | Iter {self.global_step}/{self.num_iterations} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Masked Acc: {val_acc_masked:.4f}")

                        if val_acc_masked > best_acc:
                            best_acc = val_acc_masked
                            best_iteration = self.global_step
                            patience_counter = 0
                            torch.save(self.model.state_dict(), self.save_dir/f'fold{fold}_best_model.pt')
                            print(f"Epoch {current_epoch} with {len(self.dataset.training_instances)} segments, Best Acc {best_acc:.4f}")
                        else:
                            patience_counter += 1
                            if self.early_stopping_patience and patience_counter >= self.early_stopping_patience:
                                print(f"Early stopping at step {self.global_step} (no improvement for {patience_counter} evals)")
                                self.global_step = self.num_iterations  # trigger outer while-loop exit

                    # Save checkpoints at specified intervals
                    if self.global_step % self.save_interval == 0: torch.save(self.model.state_dict(), self.save_dir / f'fold{fold}_{self.global_step}_iter.pt')

                    # Break
                    if self.global_step >= self.num_iterations: break

                train_loader.dataset.get_split(train_hash_keys, split='train')
                current_epoch += 1

            pbar.close()
            print(f"Fold {fold} Best Accuracy: {best_acc:.4f} at iteration {best_iteration}")

            self.model.load_state_dict(torch.load(self.save_dir/f'fold{fold}_best_model.pt', weights_only=True))
            test_loss, test_acc, test_acc_masked, per_class_acc, per_class_f1, macro_f1, cm, song_data = self.evaluate_test(testset)
            wandb.log({"Test/Loss": test_loss, "Test/Acc": test_acc, "Test/Masked Acc": test_acc_masked,
                       "Test/Macro F1": macro_f1,
                       **{f"Test/Acc_{k}": v for k, v in per_class_acc.items()},
                       **{f"Test/F1_{k}": v for k, v in per_class_f1.items()},
                       "Test Confusion Matrix": wandb.Image(Image.fromarray(cm))}, step=self.global_step)
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test Masked Acc: {test_acc_masked:.4f}")
            print(f"Per-class Acc: { {k: f'{v:.4f}' for k, v in per_class_acc.items()} }")
            fold_best_acc[fold] = test_acc_masked

            CLASS_NAMES = {0: 'Unknown', 1: 'UJO', 2: 'GMJ', 3: 'ANR', 4: 'CJO'}
            class_names = [CLASS_NAMES.get(i, str(i)) for i in range(testset.num_classes)]
            fold_out_dir = self.save_dir / f'fold{fold}_posteriorgrams'
            self.save_posteriorgrams(song_data, fold_out_dir, class_names)

        if wandb.run is not None: wandb.finish()

        return fold_best_acc


class SegmentTrainer(Trainer):
    def __init__(self, model, optimizer, dataset, criterion, device, save_dir, config):
        super().__init__(model, optimizer, dataset, criterion, device, save_dir, config)

    def train_batch(self, x, y):
        self.model.train()
        x, y = x.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(x)
        loss = self.criterion(outputs, y)
        loss.backward()
        self.optimizer.step()

        batch_outputs = outputs.detach().argmax(dim=-1)
        batch_acc, batch_acc_masked = self.get_masked_acc(batch_outputs, y, target_mask='아니리')

        return loss.item(), batch_acc, batch_acc_masked


    def evaluate(self, dataloader):
        self.model.eval()

        total_loss = 0
        all_outputs, all_labels = [], []

        with torch.no_grad():
            for batch in dataloader:
                _, x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                total_loss += loss.item()
                all_outputs.append(outputs.detach())
                all_labels.append(y)

        all_outputs, all_labels = torch.cat(all_outputs, dim=0).argmax(dim=-1), torch.cat(all_labels, dim=0)
        valid_loss = total_loss / len(dataloader.dataset)
        valid_acc, valid_acc_masked = self.get_masked_acc(all_outputs, all_labels, target_mask='아니리')
        per_class_f1, macro_f1 = self.get_per_class_f1(all_outputs, all_labels, ignore_label='아니리')
        cm = self.plot_confusion_matrix(all_outputs, all_labels, range(dataloader.dataset.num_classes))

        return valid_loss, valid_acc, valid_acc_masked, per_class_f1, macro_f1, cm


    def train(self):
        fold_best_acc = {}
        selection = self.init_selection()

        for fold, hash_keys in enumerate(selection, start=1):
            fold_name = self.fold_names[fold - 1] if self.fold_names else None
            self.init_new_fold(fold, fold_name=fold_name)

            self.global_step, current_epoch = 0, 0
            best_acc, best_iteration = 0, 0

            # Get train-test split
            train_hash_keys, valid_hash_keys, test_hash_keys = self.split_train_valid_test(hash_keys)
            validset = self.dataset.get_split(valid_hash_keys, split='valid')
            testset = self.dataset.get_split(test_hash_keys, split='test')
            self.dataset.get_split(train_hash_keys, split='train')
            print('Trainset:', len(train_hash_keys), 'Validset:', len(valid_hash_keys), 'Testset:', len(test_hash_keys))

            valid_loader = DataLoader(validset, batch_size=self.batch_size, shuffle=False)
            test_loader = DataLoader(testset, batch_size=self.batch_size, shuffle=False)
            train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            train_iter = iter(train_loader)

            # Start iter loop
            pbar = tqdm(total=self.num_iterations, desc=f"Fold {fold}")
            while self.global_step < self.num_iterations:
                # Forward
                try: _, x, y = next(train_iter)
                except StopIteration:
                    # update train iter
                    train_iter = iter(train_loader)
                    _, x, y = next(train_iter)
                    current_epoch += 1

                loss, batch_acc, batch_acc_masked = self.train_batch(x, y)
                wandb.log({"Train/Loss": loss, "Train/ACC": batch_acc, "Train/Masked Acc": batch_acc_masked}, step=self.global_step)

                self.global_step += 1
                pbar.update(1)
                pbar.set_description(f"Fold {fold} | Iter {self.global_step}/{self.num_iterations} | Loss: {loss:.4f}, Best Acc : {best_acc:.4f}")


                # Evaluate at specified intervals
                if self.global_step % self.eval_interval == 0:
                    val_loss, val_acc, val_acc_masked, per_class_f1, macro_f1, cm = self.evaluate(valid_loader)
                    wandb.log({"Valid/Loss": val_loss, "Valid/Acc": val_acc, "Valid/Masked Acc": val_acc_masked,
                               "Valid/Macro F1": macro_f1,
                               **{f"Valid/F1_{k}": v for k, v in per_class_f1.items()},
                               "Valid Confusion Matrix": wandb.Image(Image.fromarray(cm))}, step=self.global_step)
                    pbar.set_description(f"Fold {fold} | Iter {self.global_step}/{self.num_iterations} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Masked Acc: {val_acc_masked:.4f}")

                    if val_acc_masked > best_acc:
                        best_acc = val_acc_masked
                        best_iteration = self.global_step
                        torch.save(self.model.state_dict(), self.save_dir/f'fold{fold}_best_model.pt')
                        print(f"Epoch {current_epoch} with {len(train_loader.dataset)} segments, Best Acc {best_acc:.4f}")

                # Save checkpoints at specified intervals
                if self.global_step % self.save_interval == 0: torch.save(self.model.state_dict(), self.save_dir / f'fold{fold}_{self.global_step}_iter.pt')

                # Break
                if self.global_step >= self.num_iterations: break

            pbar.close()
            print(f"Fold {fold} Best Accuracy: {best_acc:.4f} at iteration {best_iteration}")

            self.model.load_state_dict(torch.load(self.save_dir/f'fold{fold}_best_model.pt', weights_only=True))
            test_loss, test_acc, test_acc_masked, per_class_f1, macro_f1, cm = self.evaluate(test_loader)
            wandb.log({"Test/Loss": test_loss, "Test/Acc": test_acc, "Test/Masked Acc": test_acc_masked,
                       "Test/Macro F1": macro_f1,
                       **{f"Test/F1_{k}": v for k, v in per_class_f1.items()},
                       "Test Confusion Matrix": wandb.Image(Image.fromarray(cm))}, step=self.global_step)
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test Masked Acc: {test_acc_masked:.4f}")

            fold_best_acc[fold] = test_acc_masked

        if wandb.run is not None: wandb.finish()

        return fold_best_acc