import gc
import datetime
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import seaborn as sns
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb
from tqdm import tqdm
from omegaconf import OmegaConf

class Trainer:
    def __init__(self, model, optimizer, dataset, criterion, device, save_dir, config):
        self.model = model
        self.model_state_dict = deepcopy(self.model.state_dict())
        self.optimizer = optimizer
        self.dataset = dataset
        self.loaded_data = deepcopy(dataset.loaded_data)
        self.device = device
        self.criterion = criterion # nn.CrossEntropyLoss()

        self.save_dir = Path(save_dir)

        self.config = config
        self.batch_size = config.train.batch_size
        self.num_iterations = config.train.num_iterations
        self.eval_interval = config.train.get('eval_interval', 200)
        self.save_interval = config.train.get('save_interval', 1000)

        self.global_step = 0
        self.model.to(self.device)


    def init_selection(self):
        if self.config.train.selection == 'KFold':
            kfold = KFold(**self.config.kfold)
            selection = list(kfold.split(range(len(self.dataset.loaded_hash))))

        elif self.config.train.selection == 'Stratify':
            hash_dict = {}
            df = pd.read_csv(self.config.stratify)
            for back in set(df['back']): hash_dict[back] = df[df['back']==back]['hash_key'].tolist()
            hash_dict = dict(sorted(hash_dict.items()))

            selection = []
            for back, test_hash_keys in hash_dict.items():
                print(back)
                train_hash_keys = list(set(self.dataset.loaded_hash)-set(test_hash_keys))
                train_idx = [self.dataset.loaded_hash.index(h) for h in train_hash_keys if h in self.dataset.loaded_hash]
                test_idx = [self.dataset.loaded_hash.index(h) for h in test_hash_keys if h in self.dataset.loaded_hash]
                selection.append([train_idx, test_idx])

        else: Exception("Have to select config.train.selection: [KFold, Stratify]")

        return selection


    def init_new_fold(self, fold):
        if wandb.run is not None: wandb.finish()
        run_name = f'{self.config.model.name}_{self.config.dataset.name}_Fold{fold}_{datetime.datetime.now().strftime("%m%d_%H%M")}'
        wandb.init(project='Pansori_Segment', name=run_name, reinit=True)
        wandb.config.update(OmegaConf.to_container(self.config))

        self.model.load_state_dict(self.model_state_dict)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.config.train.lr)
        print(f"{'='*25}{fold} Fold{'='*25}")


    def get_train_test_split(self, train_hash_keys, test_hash_keys):
        testset = deepcopy(self.dataset)
        testset.loaded_hash = [h for h in testset.loaded_hash if h in test_hash_keys]
        testset.validation_mode = True
        testset.loaded_data = testset.get_valid_data()

        train_loaded_data_indices = [idx for idx, tup in enumerate(self.dataset.loaded_data) if tup[0][0] in train_hash_keys]
        self.dataset.loaded_data = [self.dataset.loaded_data[idx] for idx in train_loaded_data_indices]

        return self.dataset, testset


    def get_masked_acc(self, output, y):
        pred, label = output.argmax(dim=-1), y.argmax(dim=-1)
        
        mask = (label!=0)
        total_valid = mask.sum()
        pred_masked = torch.where(mask, pred, torch.tensor(-1))

        acc = (pred == label).float().mean().cpu().item() # with aniri
        acc_masked = ((pred_masked == label).sum()/total_valid).cpu().item() # without aniri
        return acc, acc_masked


    def train_batch(self, x, y):
        self.model.train()
        x, y = x.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(x)
        loss = self.criterion(outputs, y.argmax(dim=-1))
        loss.backward()
        self.optimizer.step()
        
        batch_outputs = outputs.detach()
        batch_acc, batch_acc_masked = self.get_masked_acc(batch_outputs, y)
        
        return loss.item(), batch_outputs, batch_acc, batch_acc_masked


    def calc_train_metrics(self, all_outputs, all_labels, running_loss):
        if all_outputs and all_labels:
            all_out = torch.cat(all_outputs, dim=0)
            all_lab = torch.cat(all_labels, dim=0)
            train_acc, train_acc_masked = self.get_masked_acc(all_out, all_lab)
            train_loss = running_loss / len(all_outputs)
            
            return train_loss, train_acc, train_acc_masked


    def evaluate(self, test_loader):
        self.model.eval()
        
        total_loss = 0
        all_outputs, all_labels = [], []
        
        with torch.no_grad():
            for batch in test_loader:
                _, x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                loss = self.criterion(outputs, y.argmax(dim=-1))
                total_loss += loss.item()
                all_outputs.append(outputs.detach())
                all_labels.append(y)

        all_outputs, all_labels = torch.cat(all_outputs, dim=0), torch.cat(all_labels, dim=0)
        epoch_loss = total_loss / len(test_loader.dataset)
        epoch_acc, epoch_acc_masked = self.get_masked_acc(all_outputs, all_labels)
        cm = self.plot_confusion_matrix(all_outputs, all_labels, range(test_loader.dataset.num_classes))

        wandb.log({"Valid Loss": epoch_loss, "Valid Acc": epoch_acc, "Valid Masked Acc": epoch_acc_masked, "Valid Confusion Matrix": wandb.Image(Image.fromarray(cm))}, step=self.global_step)

        return epoch_loss, epoch_acc, epoch_acc_masked


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


    def train(self):
        fold_best_acc = {}

        # Define selection
        selection = self.init_selection()

        # Start fold loop
        for fold, (train_idx, test_idx) in enumerate(selection, start=1):
            # init fold wandb, model, & optimizer
            self.init_new_fold(fold)

            # Get train-test split
            train_hash_keys, test_hash_keys = [self.dataset.loaded_hash[i] for i in train_idx], [self.dataset.loaded_hash[i] for i in test_idx]
            trainset, testset = self.get_train_test_split(train_hash_keys, test_hash_keys)
            print(f"====== Dataset Length: Trainset {len(trainset)}, Testset {len(testset)} ======")
            train_loader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4, drop_last=False)
            test_loader = DataLoader(testset, batch_size=self.batch_size, shuffle=False, drop_last=False)

            self.global_step = 0
            current_epoch, running_loss = 0, 0
            best_acc, best_iteration = 0, 0
            all_outputs, all_labels = [], []

            # Start iter loop
            train_iter = iter(train_loader)
            pbar = tqdm(total=self.num_iterations, desc=f"Fold {fold}")
            while self.global_step < self.num_iterations:
                # Forward
                try: _, x, y = next(train_iter)
                except StopIteration: 
                    # update train iter
                    # train_loader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4)
                    train_iter = iter(train_loader)
                    _, x, y = next(train_iter)
                    current_epoch += 1

                loss, batch_outputs, batch_acc, batch_acc_masked = self.train_batch(x, y)
                running_loss += loss
                
                all_outputs.append(batch_outputs)
                all_labels.append(y.to(self.device))
                
                wandb.log({"Train Step Loss": loss, "Train Step Acc": batch_acc, "Train Step Masked Acc": batch_acc_masked}, step=self.global_step)
                
                self.global_step += 1
                pbar.update(1)
                pbar.set_description(f"Fold {fold} | Iter {self.global_step}/{self.num_iterations} | Loss: {loss:.4f}, Best Acc : {batch_acc:.4f}")

                # Evaluate at specified intervals
                if self.global_step % self.eval_interval == 0:
                    train_loss, train_acc, train_acc_masked = self.calc_train_metrics(all_outputs, all_labels, running_loss)
                    wandb.log({"Train Interval Loss": train_loss, "Train Interval Acc": train_acc, "Train Interval Masked Acc": train_acc_masked}, step=self.global_step)

                    # Reset accumulators
                    all_outputs, all_labels = [], []
                    running_loss = 0
                    
                    # Evaluate on validation set
                    val_loss, val_acc, val_acc_masked = self.evaluate(test_loader)
                    
                    pbar.set_description(f"Fold {fold} | Iter {self.global_step}/{self.num_iterations} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} , Masked Acc: {train_acc_masked:.4f}| Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Masked Acc: {val_acc_masked:.4f}")
                    
                    if val_acc > best_acc:
                        best_acc = val_acc
                        best_iteration = self.global_step
                        fold_best_acc[fold] = best_acc
                        torch.save(self.model.state_dict(), self.save_dir/f'fold{fold}_best_model.pt')
                        print(f"Epoch {current_epoch} with {len(trainset)} segments, Best Acc {best_acc:.4f}")
                
                # Save checkpoints at specified intervals
                if self.global_step % self.save_interval == 0: torch.save(self.model.state_dict(), self.save_dir / f'fold{fold}_{self.global_step}_iter.pt')
                # Break
                if self.global_step >= self.num_iterations: break
            
            pbar.close()
            self.dataset.loaded_data = self.loaded_data
            # del trainset, testset, train_loader, test_loader
            # gc.collect()
            # torch.cuda.empty_cache()
            print(f"Fold {fold} Best Accuracy: {best_acc:.4f} at iteration {best_iteration}")

        if wandb.run is not None: wandb.finish()
        return fold_best_acc