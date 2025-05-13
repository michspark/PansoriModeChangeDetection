import wandb
from omegaconf import OmegaConf

import numpy as np
import seaborn as sns
import copy
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

import datetime
from io import BytesIO
from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, model, optimizer, dataset, criterion, device, save_dir, config):
        self.model = model
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

        self.global_step = 0
        self.model.to(self.device)

    def get_masked_acc(self, output, y):
        pred, label = output.argmax(dim=-1).view(-1), y.argmax(dim=-1).view(-1)
        
        mask = (label!=0)
        total_valid = mask.sum()
        pred_masked = torch.where(mask, pred, torch.tensor(-1))

        # acc = (pred == label).float().mean().cpu().item()
        acc_masked = ((pred_masked == label).sum()/total_valid).cpu().item()
        return acc_masked

    def train_batch(self, x, y):
        self.model.train()
        x, y = x.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(x)
        loss = self.criterion(outputs.permute(0,2,1), y.argmax(dim=-1))
        loss.backward()
        self.optimizer.step()
        
        batch_outputs = outputs.detach()
        batch_acc_masked = self.get_masked_acc(batch_outputs, y)
        
        return loss.item(), batch_outputs, batch_acc_masked

    def evaluate(self, test_dataset):
        self.model.eval()
        test_loader = DataLoader(test_dataset, batch_size=1)  # Batch size 1 for full audio evaluation
        
        total_loss = 0
        all_outputs, all_labels = [], []
        
        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 3:
                    _, x, y = batch
                else:  # Handle other dataset formats
                    x, y = batch
                
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                if outputs.shape[1] != y.shape[1]:
                    outputs = outputs[:, :y.shape[1]]
                loss = self.criterion(outputs.permute(0,2,1), y.argmax(dim=-1))
                total_loss += loss.item()
                all_outputs.append(outputs.detach())
                all_labels.append(y)

        all_outputs, all_labels = torch.cat(all_outputs, dim=1), torch.cat(all_labels, dim=1)
        epoch_loss = total_loss / len(test_loader)
        epoch_acc_masked = self.get_masked_acc(all_outputs, all_labels)
        cm = self.plot_confusion_matrix(all_outputs, all_labels, range(test_dataset.num_classes))

        wandb.log({"Valid Loss": epoch_loss, "Valid Masked Acc": epoch_acc_masked, "Valid Confusion Matrix": wandb.Image(Image.fromarray(cm))}, step=self.global_step)

        return epoch_loss, epoch_acc_masked

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

    def get_train_test_split(self, train_hash_keys, test_hash_keys):
        self.dataset.slice_indices = [si for si in self.dataset.slice_indices if si[0] in train_hash_keys]
        train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        test_dataset = self.set_test_dataset(test_hash_keys)
        test_dataset = copy.deepcopy(self.dataset)
        test_dataset.validation_mode = True
        test_hash_indices = [i for i, hash_key in enumerate(test_dataset.loaded_hash) if hash_key in test_hash_keys]
        test_dataset.loaded_hash = [test_dataset.loaded_hash[i] for i in test_hash_indices]

        return train_loader, test_dataset

    def update_train_iter(self, train_hash_keys):
        self.dataset.update_slice_indices(train_hash_keys)
        self.dataset.slice_indices = [si for si in self.dataset.slice_indices if si[0] in train_hash_keys]
        train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4)
        return iter(train_loader)

    def calc_train_metrics(self, all_outputs, all_labels, running_loss):
        if all_outputs and all_labels:
            all_out = torch.cat(all_outputs, dim=0)
            all_lab = torch.cat(all_labels, dim=0)
            train_acc_masked = self.get_masked_acc(all_out, all_lab)
            train_loss = running_loss / len(all_outputs)
            
            return train_loss, train_acc_masked

    def train(self):
        fold_best_acc = {}
        org_model_state = copy.deepcopy(self.model.state_dict())
        kfold = KFold(**self.config.kfold)

        for fold, (train_idx, test_idx) in enumerate(kfold.split(range(len(self.dataset.loaded_hash)))):
            if wandb.run is not None: wandb.finish()
            run_name = f'{self.config.model.name}_{self.config.dataset.name}_Fold{fold+1}_{datetime.datetime.now().strftime("%m%d_%H%M")}'
            wandb.init(project='Pansori', name=run_name, reinit=True)
            wandb.config.update(OmegaConf.to_container(self.config))

            self.model.load_state_dict(org_model_state)
            self.optimizer = torch.optim.Adam(self.model.parameters(), self.config.train.lr)

            print(f"{'='*25}{fold+1} Fold{'='*25}")
            best_acc, best_iteration = 0, 0

            train_hash_keys, test_hash_keys = [self.dataset.loaded_hash[i] for i in train_idx], [self.dataset.loaded_hash[i] for i in test_idx]
            train_loader, test_dataset = self.get_train_test_split(train_hash_keys, test_hash_keys)
            train_iter = iter(train_loader)

            self.global_step = 0
            current_epoch = 0

            running_loss = 0
            all_outputs, all_labels = [], []

            pbar = tqdm(total=self.num_iterations, desc=f"Fold {fold+1}")
            while self.global_step < self.num_iterations:
                try:
                    _, x, y = next(train_iter)
                except StopIteration: 
                    train_iter = self.update_train_iter(train_hash_keys)
                    current_epoch += 1
                    _, x, y = next(train_iter)
                
                loss, batch_outputs, batch_acc_masked = self.train_batch(x, y)
                running_loss += loss
                
                all_outputs.append(batch_outputs)
                all_labels.append(y.to(self.device))
                
                wandb.log({"Train Step Loss": loss, "Train Step Masked Acc": batch_acc_masked}, step=self.global_step)
                
                self.global_step += 1
                pbar.update(1)
                pbar.set_description(f"Fold {fold+1} | Iter {self.global_step}/{self.num_iterations} | Loss: {loss:.4f}, Best Acc : {batch_acc_masked:.4f}")
                
                # Evaluate at specified intervals
                if self.global_step % self.eval_interval == 0:
                    train_loss, train_acc_masked = self.calc_train_metrics(all_outputs, all_labels, running_loss)
                    wandb.log({"Train Interval Loss": train_loss, "Train Interval Masked Acc": train_acc_masked}, step=self.global_step)

                    # Reset accumulators
                    all_outputs, all_labels = [], []
                    running_loss = 0
                    
                    # Evaluate on validation set
                    val_loss, val_acc_masked = self.evaluate(test_dataset)
                    
                    pbar.set_description(f"Fold {fold+1} | Iter {self.global_step}/{self.num_iterations} | Train Loss: {train_loss:.4f}, Acc: {train_acc_masked:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc_masked:.4f}")
                    
                    if val_acc_masked > best_acc:
                        best_acc = val_acc_masked
                        best_iteration = self.global_step
                        fold_best_acc[fold] = best_acc
                        torch.save(self.model.state_dict(), self.save_dir/f'fold{fold+1}_best_model.pt')
                        print(f"Epoch {current_epoch} with {len(self.dataset.slice_indices)} segments, Best Acc {best_acc:.4f}")
                
                # Save checkpoints at specified intervals
                if self.global_step % self.save_interval == 0: torch.save(self.model.state_dict(), self.save_dir / f'fold{fold+1}_{self.global_step}_iter.pt')                

                # Break
                if self.global_step >= self.num_iterations: break
            
            pbar.close()
            print(f"Fold {fold+1} Best Accuracy: {best_acc:.4f} at iteration {best_iteration}")

        if wandb.run is not None: wandb.finish()
        return fold_best_acc