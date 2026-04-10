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
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader

import wandb
from tqdm import tqdm
from omegaconf import OmegaConf

T = datetime.datetime.now().strftime("%m%d_%H%M")

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

        self.global_step = 0
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

        elif self.config.train.selection == 'Stratify': # Madang
            hash_dict = {}
            df = pd.read_csv(self.config.stratify)
            for back in set(df['back']): hash_dict[back] = df[df['back']==back]['hash_key'].tolist()
            hash_dict = dict(sorted(hash_dict.items()))

            selection = []
            for back, test_hash_keys in hash_dict.items():
                print(back)
                train_hash_keys = list(set(self.dataset.loaded_hash)-set(test_hash_keys))
                selection.append([train_hash_keys, test_hash_keys])

        elif self.config.train.selection == 'Artist':
            selection = []
            df = pd.read_csv(self.config.stratify)

            folds = sorted(set(df['artist_stratify']))
            for fold in folds:
                print(set(df[df['artist_stratify']==fold]['artist'].tolist()))
                all_hash_keys = set(df['hash_key'].tolist())

                test_hash_keys = df[df['artist_stratify']==fold]['hash_key'].tolist()
                valid_hash_keys = df[df['artist_stratify']==(fold+1)]['hash_key'].tolist() if fold<(len(folds)-1) else df[df['artist_stratify']==0]['hash_key'].tolist()
                train_hash_keys = list(all_hash_keys-set(test_hash_keys)-set(valid_hash_keys))

                selection.append([train_hash_keys, valid_hash_keys, test_hash_keys])

        else: Exception("Have to select config.train.selection: [KFold, Stratify]")

        return selection


    def init_new_fold(self, fold):
        if wandb.run is not None: wandb.finish()
        run_name = f"{self.config.data.data_dir.split('/')[-1]}_{self.config.model.name}_{self.config.dataset.name}_Fold{fold}_{T}"
        wandb.init(project='Pansori_CMERT', name=run_name, group=f'{self.config.dataset.name}_{self.config.train.selection}_{T}', reinit=True)
        wandb.config.update(OmegaConf.to_container(self.config))

        self.model.load_state_dict(self.model_state_dict)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.config.train.lr)
        print(f"{'='*25}{fold} Fold{'='*25}")


    def split_train_valid_test(self, hash_keys):
        if len(hash_keys)==2:
            train_hash_keys, test_hash_keys = hash_keys
            train_hash_keys, valid_hash_keys = train_test_split(train_hash_keys, test_size=0.1, random_state=self.config.train.random_seed, shuffle=True)
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
                all_outputs.append(outputs.detach())
                all_labels.append(y)

        all_outputs, all_labels = torch.cat(all_outputs, dim=1).argmax(dim=-1).view(-1), torch.cat(all_labels, dim=1).argmax(dim=-1).view(-1)
        valid_loss = total_loss / len(dataset)
        valid_acc, valid_acc_masked = self.get_masked_acc(all_outputs, all_labels, target_mask='Unknown')
        cm = self.plot_confusion_matrix(all_outputs, all_labels, range(dataset.num_classes))

        return valid_loss, valid_acc, valid_acc_masked, cm


    def train(self):
        fold_best_acc = {}

        # Define selection
        selection = self.init_selection()

        # Start fold loop
        for fold, hash_keys in enumerate(selection, start=1):
            # init fold wandb, model, & optimizer
            self.init_new_fold(fold)

            self.global_step, current_epoch = 0, 0
            best_acc, best_iteration = 0, 0

            # Get train-test split            
            train_hash_keys, valid_hash_keys, test_hash_keys = self.split_train_valid_test(hash_keys)
            self.dataset.get_split(train_hash_keys, split='train')
            validset = self.dataset.get_split(valid_hash_keys, split='valid')
            testset = self.dataset.get_split(test_hash_keys, split='test')
            print('Trainset:', len(self.dataset), 'Validset:', len(validset), 'Testset:', len(testset))
            train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
            # train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
            # train_iter = iter(train_loader)

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
                        val_loss, val_acc, val_acc_masked, cm = self.evaluate(validset)
                        wandb.log({"Valid/Loss": val_loss, "Valid/Acc": val_acc, "Valid/Masked Acc": val_acc_masked, "Valid Confusion Matrix": wandb.Image(Image.fromarray(cm))}, step=self.global_step)
                        pbar.set_description(f"Fold {fold} | Iter {self.global_step}/{self.num_iterations} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Masked Acc: {val_acc_masked:.4f}")
                        
                        if val_acc_masked > best_acc:
                            best_acc = val_acc_masked
                            best_iteration = self.global_step
                            torch.save(self.model.state_dict(), self.save_dir/f'fold{fold}_best_model.pt')
                            print(f"Epoch {current_epoch} with {len(self.dataset.slice_indices)} segments, Best Acc {best_acc:.4f}")

                    # Save checkpoints at specified intervals
                    if self.global_step % self.save_interval == 0: torch.save(self.model.state_dict(), self.save_dir / f'fold{fold}_{self.global_step}_iter.pt')

                    # Break
                    if self.global_step >= self.num_iterations: break

                train_loader.dataset.get_split(train_hash_keys, split='train')
                current_epoch += 1

            pbar.close()
            print(f"Fold {fold} Best Accuracy: {best_acc:.4f} at iteration {best_iteration}")

            self.model.load_state_dict(torch.load(self.save_dir/f'fold{fold}_best_model.pt', weights_only=True))
            test_loss, test_acc, test_acc_masked, cm = self.evaluate(testset)
            wandb.log({"Test/Loss": test_loss, "Test/Acc": test_acc, "Test/Masked Acc": test_acc_masked, "Test Confusion Matrix": wandb.Image(Image.fromarray(cm))}, step=self.global_step)
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test Masked Acc: {test_acc_masked:.4f}")
            fold_best_acc[fold] = test_acc_masked

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
        # print(x.shape ,outputs.shape, y.shape)
        # if outputs.ndim!=y.ndim: outputs=outputs.unsqueeze(0)
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
                # print(x.shape ,outputs.shape, y.shape)
                # if outputs.ndim!=y.ndim: outputs=outputs.unsqueeze(0)
                loss = self.criterion(outputs, y)
                total_loss += loss.item()
                all_outputs.append(outputs.detach())
                all_labels.append(y)

        all_outputs, all_labels = torch.cat(all_outputs, dim=0).argmax(dim=-1), torch.cat(all_labels, dim=0)
        valid_loss = total_loss / len(dataloader.dataset)
        valid_acc, valid_acc_masked = self.get_masked_acc(all_outputs, all_labels, target_mask='아니리')
        cm = self.plot_confusion_matrix(all_outputs, all_labels, range(dataloader.dataset.num_classes))

        return valid_loss, valid_acc, valid_acc_masked, cm


    def train(self):
        fold_best_acc = {}
        selection = self.init_selection()

        for fold, hash_keys in enumerate(selection, start=1):
            self.init_new_fold(fold)

            self.global_step, current_epoch = 0, 0
            best_acc, best_iteration = 0, 0

            # Get train-test split
            train_hash_keys, valid_hash_keys, test_hash_keys = self.split_train_valid_test(hash_keys)
            validset = self.dataset.get_split(valid_hash_keys, split='valid')
            testset = self.dataset.get_split(test_hash_keys, split='test')
            self.dataset.get_split(train_hash_keys, split='train')
            print('Trainset:', len(self.dataset), 'Validset:', len(validset), 'Testset:', len(testset))

            valid_loader = DataLoader(validset, batch_size=self.batch_size, shuffle=False)
            test_loader = DataLoader(testset, batch_size=self.batch_size, shuffle=False)
            train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
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
                    val_loss, val_acc, val_acc_masked, cm = self.evaluate(valid_loader)
                    wandb.log({"Valid/Loss": val_loss, "Valid/Acc": val_acc, "Valid/Masked Acc": val_acc_masked, "Valid Confusion Matrix": wandb.Image(Image.fromarray(cm))}, step=self.global_step)
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
            test_loss, test_acc, test_acc_masked, cm = self.evaluate(test_loader)
            wandb.log({"Test/Loss": test_loss, "Test/Acc": test_acc, "Test/Masked Acc": test_acc_masked, "Test Confusion Matrix": wandb.Image(Image.fromarray(cm))}, step=self.global_step)
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test Masked Acc: {test_acc_masked:.4f}")

            fold_best_acc[fold] = test_acc_masked

        if wandb.run is not None: wandb.finish()

        return fold_best_acc