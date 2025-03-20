from utils import FixedSampler, BaseSampler

import wandb
from omegaconf import OmegaConf

import numpy as np
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

import datetime
from io import BytesIO
from pathlib import Path

import torch
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, model, optimizer, dataset, criterion, device, save_dir, best_dir, config):
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.criterion = criterion
        self.device = device

        self.save_dir = Path(save_dir)
        self.best_dir = Path(best_dir)

        self.config = config
        self.batch_size = config.train.batch_size
        self.num_epochs = config.train.num_epochs

        self.gloabl_step = 0

    def get_segments(self, idx):
        segment_counts = {}
        for idx in idx:
            hash_key = self.dataset.loaded_hash[idx]
            segment_counts[idx] = self.dataset.loaded_audio[hash_key].shape[1]//(self.dataset.window*self.dataset.sr)
        return segment_counts

    def get_acc(self, output, label):
        pred = output.argmax(dim=-1)
        true = label.argmax(dim=-1)
        acc = (pred == true).float().mean(dim=1).cpu().numpy().mean().item()
        return acc

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        all_outputs, all_labels = [], []

        for batch_idx, (_,_,x,y) in enumerate(train_loader):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs.permute(0,2,1), y.argmax(dim=-1))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            batch_outputs = outputs.detach()
            batch_acc = self.get_acc(batch_outputs, y)

            wandb.log({"Train Step Loss":loss.item(), "Train Step Acc": batch_acc}, step=self.gloabl_step)
            self.gloabl_step += 1

            all_outputs.append(batch_outputs)
            all_labels.append(y)

        all_outputs,  all_labels = torch.cat(all_outputs, dim=0), torch.cat(all_labels, dim=0)
        epoch_loss = total_loss / (len(train_loader))
        epoch_acc = self.get_acc(all_outputs, all_labels)
        return epoch_loss, epoch_acc

    def plot_confusion_matrix(self, output, y, class_names):
        true_labels = y.argmax(dim=-1)
        pred_labels = output.argmax(dim=-1)
        
        true_flat = true_labels.reshape(-1).cpu().numpy()
        pred_flat = pred_labels.reshape(-1).cpu().numpy()
        
        assert len(true_flat) == len(pred_flat), f"Length mismatch: {len(true_flat)} vs {len(pred_flat)}"
        
        cm = confusion_matrix(true_flat, pred_flat, normalize='true')

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

    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0
        all_outputs, all_labels = [], []
        with torch.no_grad():
            for _,_,x,y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                loss = self.criterion(outputs.permute(0,2,1), y.argmax(dim=-1))

                all_outputs.append(outputs.detach())
                all_labels.append(y)

        all_outputs,  all_labels = torch.cat(all_outputs, dim=0), torch.cat(all_labels, dim=0)
        epoch_loss = total_loss / (len(test_loader))
        epoch_acc = self.get_acc(all_outputs, all_labels)
        cm = self.plot_confusion_matrix(all_outputs, all_labels, range(self.dataset.num_classes))
        return epoch_loss, epoch_acc, cm

    def train(self):
        fold_best_acc = {}
        org_model_state = self.model.state_dict()
        kfold = KFold(**self.config.kfold)

        for fold, (train_idx, test_idx) in enumerate(kfold.split(self.dataset)):
            if wandb.run is not None: wandb.finish()
            run_name = f'{self.config.models.cls}_Fold{fold+1}_{datetime.datetime.now().strftime("%m%d_%H%M")}'
            wandb.init(project='Pansori_Mode_Detection', name=run_name, reinit=True)
            wandb.config.update(OmegaConf.to_container(self.config))

            self.model.load_state_dict(org_model_state)
            self.optimizer = torch.optim.Adam(self.model.parameters(), self.config.train.lr)

            print(f"{'='*25}{fold+1} Fold{'='*25}")
            best_acc, best_epoch = 0, 0

            train_segments, test_segments = self.get_segments(train_idx), self.get_segments(test_idx)
            train_sampler, test_sampler = BaseSampler(train_segments), FixedSampler(test_segments)
            test_loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=test_sampler)

            self.global_step = 0
            train_pbar = tqdm(range(self.num_epochs), desc=f"Fold {fold+1}")
            for epoch in train_pbar:
                train_loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=train_sampler)
                train_loss, train_acc = self.train_epoch(train_loader)
                wandb.log({"Train Epoch Loss": train_loss,
                            "Train Epoch Acc": train_acc},
                            step=epoch)
                
                val_loss, val_acc, val_cm = self.evaluate(test_loader)
                wandb.log({"Valid Epoch Loss":val_loss,
                            "Valid Epoch Acc":val_acc,
                            "Valid Confusion Matrix": wandb.Image(Image.fromarray(val_cm))},
                            step=self.global_step)

                train_pbar.set_description(f"Fold {fold+1} | Epoch {epoch+1} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

                if val_acc > best_acc:
                    best_acc = val_acc
                    best_epoch = epoch
                    fold_best_acc[fold] = best_acc
                    torch.save(self.model.state_dict(), self.best_dir/f'fold{fold+1}_{best_epoch+1}epochs_best_model.pt')

                if (epoch+1)%10==0: torch.save(self.model.state_dict(), self.save_dir / f'fold{fold+1}_{epoch+1}epochs.pt')
            print(f"Fold {fold+1} Best Accuracy: {best_acc:.4f} at epoch {best_epoch+1}")

        if wandb.run is not None: wandb.finish()
        return fold_best_acc