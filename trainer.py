from io import BytesIO
from pathlib import Path

import numpy as np
from tqdm import tqdm
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import wandb

import torch

class Trainer():
    def __init__(self, dataset, criterion, device, save_dir, best_dir, config):
        self.dataset = dataset
        self.criterion = criterion
        self.device = device

        self.save_dir = Path(save_dir)
        self.best_dir = Path(best_dir)

        self.config = config
        self.batch_size = config.train.batch_size
        self.num_epochs = config.train.num_epochs

        self.num_updated = 0

    def load_segments(self, ids):
        x, y = [], []
        for idx in ids:
            _, _, (_, _, duration) = self.dataset[idx]
            num_segments = int(duration // self.dataset.window)
            
            for i in range(num_segments):
                chroma, label, _ = self.dataset[idx]
                if isinstance(chroma, torch.Tensor): chroma = chroma.clone().detach().to(dtype=torch.float32).unsqueeze(0)
                else: chroma = torch.tensor(chroma, dtype=torch.float32).unsqueeze(0)
                if isinstance(label, torch.Tensor): label = label.clone().detach().to(dtype=torch.float32)
                else: label = torch.tensor(label, dtype=torch.float32)

                x.append(chroma)
                y.append(label)        
        x = torch.stack(x).to(self.device)
        y = torch.stack(y).to(self.device).argmax(dim=-1)
        return x, y

    def get_acc(self, output, y):
        pred_labels = torch.softmax(output, dim=-1).argmax(dim=-1)
        frame_acc = (pred_labels == y).float().mean(dim=1).cpu().numpy()
        acc = frame_acc.mean().item()
        return frame_acc, acc

    def plot_confusion_matrix(self, output, y, class_names):
        pred_labels = torch.softmax(output, dim=-1).argmax(dim=-1)
        pred_flat = pred_labels.view(-1).cpu().numpy()
        y_flat = y.view(-1).cpu().numpy()

        cm = confusion_matrix(y_flat, pred_flat, normalize='true')

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


    def train_epoch(self, train_x, train_y, model, optimizer):
        model.train()
        total_loss = 0

        ids = torch.randperm(len(train_x))
        train_x, train_y = train_x[ids], train_y[ids]
        
        all_outputs, all_labels = [], []

        for i in range(0, len(train_x), self.batch_size):
            batch_x, batch_y = train_x[i:i+self.batch_size], train_y[i:i+self.batch_size]
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = self.criterion(outputs.permute(0, 2, 1), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            all_outputs.append(outputs.detach())
            all_labels.append(batch_y)

        all_outputs, all_labels = torch.cat(all_outputs, dim=0), torch.cat(all_labels, dim=0)
        epoch_loss = total_loss / (len(train_x) // self.batch_size)
        epoch_frame_acc, epoch_acc = self.get_acc(all_outputs, all_labels)
        return epoch_loss, epoch_frame_acc, epoch_acc
    
    def evaluate(self, val_x, val_y, model):
        model.eval()
        with torch.no_grad():
            val_output = model(val_x)
            val_loss = self.criterion(val_output.permute(0, 2, 1), val_y)
            frame_acc, acc = self.get_acc(val_output, val_y)
            val_cm = self.plot_confusion_matrix(val_output, val_y, range(self.dataset.num_classes))
        return val_loss.item(), frame_acc, acc, val_cm

    def train_split(self, idx, train_idx, test_idx, model, optimizer):
        print(f"{'='*25}{idx+1} Fold{'='*25}")
        best_acc, best_epoch = 0, 0
        test_x, test_y = self.load_segments(test_idx) # segment 고정

        train_pbar = tqdm(range(self.num_epochs), desc=f"Fold {idx+1}")
        for epoch in train_pbar:
            train_x, train_y = self.load_segments(train_idx)
            train_loss, train_frame_acc, train_acc = self.train_epoch(train_x, train_y, model, optimizer)
            wandb.log({"Train Loss": train_loss,
                        "Train Frame Acc": train_frame_acc,
                        "Train Acc": train_acc},
                        step=epoch)
            val_loss, val_frame_acc, val_acc, val_cm = self.evaluate(test_x, test_y, model)
            wandb.log({"Valid Loss":val_loss,
                        "Valid Frame Acc":val_frame_acc,
                        "Valid Acc":val_acc,
                        "Valid Confusion Matrix": wandb.Image(Image.fromarray(val_cm))},
                        step=epoch)

            train_pbar.set_description(f"Fold {idx+1} | Epoch {epoch+1} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                torch.save(model.state_dict(), self.best_dir/f'fold{idx+1}_{best_epoch+1}epochs_best_model.pt')
            if (epoch+1)%10==0: torch.save(model.state_dict(), self.save_dir / f'fold{idx+1}_{epoch+1}epochs.pt')

        print(f"Fold {idx+1} Best Accuracy: {best_acc:.4f} at epoch {best_epoch+1}")
        return best_acc

    # def train(self):
    #     fold_best_acc = {}
    #     for idx, (train_idx, test_idx) in enumerate(self.selection.split(range(len(self.dataset)))):
    #         print(f"{'='*25}{idx+1} Fold{'='*25}")
    #         best_acc, best_epoch = 0, 0
    #         test_x, test_y = self.load_segments(test_idx) # segment 고정

    #         train_pbar = tqdm(range(self.num_epochs), desc=f"Fold {idx+1}")
    #         for epoch in train_pbar:
    #             train_x, train_y = self.load_segments(train_idx)
    #             train_loss, train_frame_acc, train_acc = self.train_epoch(train_x, train_y)
    #             wandb.log({"Train Loss": train_loss,
    #                         "Train Frame Acc": train_frame_acc,
    #                         "Train Acc": train_acc},
    #                         step=self.num_updated)
    #             val_loss, val_frame_acc, val_acc = self.evaluate(test_x, test_y)
    #             wandb.log({"Valid Loss":val_loss,
    #                        "Valid Frame Acc":val_frame_acc,
    #                        "Valid Acc":val_acc},
    #                        step=self.num_updated)
                
    #             self.num_updated += 1

    #             train_pbar.set_description(f"Fold {idx+1} | Epoch {epoch+1} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    #             if val_acc > best_acc:
    #                 best_acc = val_acc
    #                 best_epoch = epoch
    #                 torch.save(self.model.state_dict(), self.best_dir/f'fold{idx+1}_{self.num_updated}updated_best_model.pt')
    #             if (epoch+1)%10==0: torch.save(self.model.state_dict(), self.save_dir / f'fold{idx+1}_epoch{epoch+1}_{self.num_updated}updated.pt')

    #         print(f"Fold {idx+1} Best Accuracy: {best_acc:.4f} at epoch {best_epoch+1}")