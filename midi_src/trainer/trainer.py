import json
import shutil
import torch
import wandb
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from .metrics import masked_acc, masked_acc_per_class, masked_f1
from datasets.dataset import BaseDataset
from utils import plot_confusion_matrix, save_test_csv, plot_posteriorgram

OUTPUT_DIR = Path("/home/sangheon/Desktop/PansoriMIDIDetection/outputs")

def run_epoch(loader, model, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_tgts = [], []
    song_data = {} if not train else None

    ctx = torch.enable_grad() if train else torch.no_grad()
    desc = 'Train' if train else 'Val'

    with ctx:
        pbar = tqdm(loader, desc=desc, leave=False)
        for song_name, start_frame, piano, label in pbar:
            piano = piano.to(device)
            label = label.to(device)

            if train:
                optimizer.zero_grad()

            out = model(piano)
            tgt = label.argmax(dim=-1)

            loss = criterion(out.permute(0,2,1), tgt)

            if train:
                loss.backward()
                optimizer.step()

            preds = out.detach().argmax(dim=-1).view(-1).cpu()
            total_loss += loss.item()
            all_preds.append(preds)
            all_tgts.append(tgt.view(-1).cpu())

            if not train:
                pred_probs = torch.softmax(out, dim=-1)
                name = song_name[0] if isinstance(song_name, (list, tuple)) else song_name
                if name not in song_data:
                    song_data[name] = {'gt': [], 'pred_probs': []}
                song_data[name]['gt'].append(label[0].cpu())
                song_data[name]['pred_probs'].append(pred_probs[0].cpu())

    if not train:
        for name in song_data:
            song_data[name]['gt'] = torch.cat(song_data[name]['gt'], dim=0).numpy()
            song_data[name]['pred_probs'] = torch.cat(song_data[name]['pred_probs'], dim=0).numpy()

    all_preds = torch.cat(all_preds)
    all_tgts  = torch.cat(all_tgts)

    avg_loss    = total_loss / len(loader)
    avg_acc     = masked_acc(all_preds, all_tgts)
    acc_per_cls = masked_acc_per_class(all_preds, all_tgts)
    f1          = masked_f1(all_preds, all_tgts)

    return avg_loss, avg_acc, acc_per_cls, f1, song_data

def run_test_epoch(loader, model, criterion, device, fs=100, window_size=3000):
    """run_epoch(train=False) but also returns per-song GT, softmax probs, and per-segment metrics."""
    model.eval()
    total_loss = 0.0
    all_preds, all_tgts = [], []
    song_data = {}  # song_name -> {'gt': [Tensor(T,3)], 'pred_probs': [Tensor(T,3)]}
    segment_results = []  # per-segment: song_name, time range, loss, acc, f1s

    with torch.no_grad():
        pbar = tqdm(loader, desc='Test', leave=False)
        for song_name, start_frame, piano, label in pbar:
            piano = piano.to(device)
            label = label.to(device)
            out = model(piano)          # (1, T, C)
            tgt = label.argmax(dim=-1)  # (1, T)

            loss = criterion(out.permute(0, 2, 1), tgt)
            total_loss += loss.item()

            pred_probs = torch.softmax(out, dim=-1)  # (1, T, C)
            preds = out.detach().argmax(dim=-1).view(-1).cpu()
            all_preds.append(preds)
            all_tgts.append(tgt.view(-1).cpu())

            name = song_name[0] if isinstance(song_name, (list, tuple)) else song_name
            if name not in song_data:
                song_data[name] = {'gt': [], 'pred_probs': [], 'segments': []}
            song_data[name]['gt'].append(label[0].cpu())
            song_data[name]['pred_probs'].append(pred_probs[0].cpu())

            # Per-segment metrics
            seg_start = start_frame.item() if hasattr(start_frame, 'item') else int(start_frame)
            seg_start_sec = seg_start / fs
            seg_end_sec = seg_start_sec + window_size / fs
            song_data[name]['segments'].append({
                'gt': label[0].cpu().numpy(),
                'pred_probs': pred_probs[0].cpu().numpy(),
                'start_sec': seg_start_sec,
                'end_sec': seg_end_sec,
                'loss': loss.item(),
            })
            
            seg_f1 = masked_f1(preds.cpu(), tgt.view(-1).cpu())
            # Mean softmax probability per class over all frames in this segment
            # Class order: [no_label=0, 우조=1, 계면조=2, 아니리=3, 창조=4]
            mean_prob = pred_probs[0].cpu().numpy().mean(axis=0)  # (C,)
            segment_results.append({
                'song_name': name,
                'start_sec': f"{seg_start_sec:.1f}",
                'end_sec': f"{seg_end_sec:.1f}",
                'time_range': f"{seg_start_sec:.0f}-{seg_end_sec:.0f}s",
                'loss': round(loss.item(), 6),
                'acc': round(masked_acc(preds.cpu(), tgt.view(-1).cpu()), 6),
                'f1_ujoh': round(seg_f1['f1_ujoh'], 6),
                'f1_gyemyeon': round(seg_f1['f1_gyemyeon'], 6),
                'f1_aniri': round(seg_f1['f1_aniri'], 6),
                'f1_changjo': round(seg_f1['f1_changjo'], 6),
                'f1_macro': round(seg_f1['f1_macro'], 6),
                'prob_우조': round(float(mean_prob[1]), 6),
                'prob_계면조': round(float(mean_prob[2]), 6),
                'prob_아니리': round(float(mean_prob[3]), 6),
                'prob_창조': round(float(mean_prob[4]), 6),
            })

    for name in song_data:
        song_data[name]['gt'] = torch.cat(song_data[name]['gt'], dim=0).numpy()
        song_data[name]['pred_probs'] = torch.cat(song_data[name]['pred_probs'], dim=0).numpy()

    all_preds   = torch.cat(all_preds)
    all_tgts    = torch.cat(all_tgts)
    avg_loss    = total_loss / len(loader)
    avg_acc     = masked_acc(all_preds, all_tgts)
    acc_per_cls = masked_acc_per_class(all_preds, all_tgts)
    f1          = masked_f1(all_preds, all_tgts)

    return avg_loss, avg_acc, acc_per_cls, f1, song_data, segment_results

def train_step(batch, model, optimizer, criterion, device):
    model.train()
    _, start_frame, piano, label = batch
    piano = piano.to(device)
    label = label.to(device)
    optimizer.zero_grad()
    out = model(piano)
    tgt = label.argmax(dim=-1)
    loss = criterion(out.permute(0, 2, 1), tgt)
    loss.backward()
    optimizer.step()
    preds       = out.detach().argmax(dim=-1).view(-1).cpu()
    acc         = masked_acc(preds, tgt.view(-1).cpu())
    acc_per_cls = masked_acc_per_class(preds, tgt.view(-1).cpu())
    f1          = masked_f1(preds, tgt.view(-1).cpu())

    return loss.item(), acc, acc_per_cls, f1


class Trainer:
    def __init__(self, model, optimizer, criterion, device, cfg, fold_idx, T):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.cfg = cfg
        self.fold_idx = fold_idx
        self.T = T
        self.save_path = f"best_model_fold{fold_idx + 1}.pt"
        self.fold_out_dir = OUTPUT_DIR / f"fold{fold_idx + 1}_{T}"
        self.scheduler = self._build_scheduler()

    def _build_scheduler(self):
        sched_cfg = self.cfg.train.get('scheduler', None)
        if not sched_cfg or sched_cfg.get('name', None) is None:
            return None
        name = sched_cfg['name']
        if name == 'ReduceLROnPlateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=sched_cfg.get('factor', 0.5),
                patience=sched_cfg.get('patience', 10),
                min_lr=sched_cfg.get('min_lr', 1e-6),
            )
        if name == 'CosineAnnealingLR':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_cfg.get('T_max', self.cfg.train.get('num_epoch', self.cfg.train.get('num_iterations', 100))),
                eta_min=sched_cfg.get('min_lr', 1e-6),
            )
        raise ValueError(f"Unknown scheduler: {name}")

    def run(self, fold):
        fs = self.cfg.data.fs
        window_size = self.cfg.data.window_size
        midi_dir = self.cfg.data.dir.midi_dir
        label_path = self.cfg.data.dir.label_path

        train_dataset = BaseDataset(midi_dir, label_path, song_list=fold['train'], fs=fs, window_size=window_size, is_train=True)
        val_dataset   = BaseDataset(midi_dir, label_path, song_list=fold['val'],   fs=fs, window_size=window_size, is_train=False)
        test_dataset  = BaseDataset(midi_dir, label_path, song_list=fold['test'],  fs=fs, window_size=window_size, is_train=False)

        # Keep train song list for hard-sample mining after training
        self._train_song_list = fold['train']

        train_loader = DataLoader(train_dataset, batch_size=self.cfg.train.batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False)
        test_loader  = DataLoader(test_dataset,  batch_size=1, shuffle=False)

        run = wandb.init(
            project=self.cfg.project_name,
            name=f"fold{self.fold_idx + 1}_{self.T}",
            config=OmegaConf.to_container(self.cfg, resolve=True),
            reinit=True,
        )

        self._log_split(fold)
        self._fit(train_loader, val_loader)
        self._evaluate(test_loader)   # loads best checkpoint
        self._save_hard_train_segments()
        run.finish()

    def _save_hard_train_segments(self, top_n=30):
        """Run best-model (already loaded) inference on train set; save top-N highest-loss segments."""
        song_list = getattr(self, '_train_song_list', [])
        if not song_list:
            return
        print(f"  Hard-train-segment analysis ({len(song_list)} train songs)...")

        fs          = self.cfg.data.fs
        window_size = self.cfg.data.window_size
        midi_dir    = self.cfg.data.dir.midi_dir
        label_path  = self.cfg.data.dir.label_path

        train_eval_ds     = BaseDataset(midi_dir, label_path, song_list=song_list,
                                        fs=fs, window_size=window_size, is_train=False)
        train_eval_loader = DataLoader(train_eval_ds, batch_size=1, shuffle=False)

        _, _, _, _, train_song_data, train_seg_results = run_test_epoch(
            train_eval_loader, self.model, self.criterion, self.device,
            fs=fs, window_size=int(window_size * fs))

        hard_dir = self.fold_out_dir / 'hard_train_segments'
        hard_dir.mkdir(parents=True, exist_ok=True)

        # Top-N by loss → CSV
        top = sorted(train_seg_results, key=lambda r: r['loss'], reverse=True)[:top_n]
        save_test_csv(top, hard_dir / f'hard_train_top{top_n}.csv')

        # Posteriorgrams only for the top-N segments
        top_keys = {(r['song_name'], r['start_sec']) for r in top}
        for sname, data in train_song_data.items():
            for seg in data.get('segments', []):
                if (sname, f"{seg['start_sec']:.1f}") not in top_keys:
                    continue
                stem = Path(sname).stem
                seg_label = f"{seg['start_sec']:.0f}-{seg['end_sec']:.0f}s"
                title = f"{stem} [{seg_label}]  loss={seg['loss']:.4f}"
                fig = plot_posteriorgram(title, seg['gt'], seg['pred_probs'])
                fig.savefig(hard_dir / f"{stem}_{seg_label}.png", dpi=120, bbox_inches='tight')
                fig.clf()
                plt.close(fig)
        plt.close('all')
        print(f"  Saved top-{top_n} hard train segments → {hard_dir}")

    def _log_split(self, fold):
        self.fold_out_dir.mkdir(parents=True, exist_ok=True)
        split_info = fold.get('split_info', None)

        # Write split to file
        split_path = self.fold_out_dir / "split_info.json"
        payload = {'train': fold['train'], 'val': fold['val'], 'test': fold['test']}
        with open(split_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        # Log to wandb as a Table
        rows = ([[s, 'train'] for s in fold['train']] +
                [[s, 'val']   for s in fold['val']] +
                [[s, 'test']  for s in fold['test']])

        table = wandb.Table(columns=['song', 'split'], data=rows)
        wandb.log({'split/song_table': table,
                   'split/n_train': len(fold['train']),
                   'split/n_val':   len(fold['val']),
                   'split/n_test':  len(fold['test'])})

    def _fit(self, train_loader, val_loader):
        train_mode = self.cfg.train.get('train_mode', 'iteration')
        if train_mode == 'epoch':
            self._fit_epoch(train_loader, val_loader)
        else:
            self._fit_iteration(train_loader, val_loader)

    def _fit_iteration(self, train_loader, val_loader):
        num_iterations = self.cfg.train.num_iterations
        eval_interval  = self.cfg.train.get('iter_eval_interval', self.cfg.train.get('eval_interval', 200))
        save_interval  = self.cfg.train.get('save_interval', 1000)

        best_val_f1 = 0.0
        best_step   = 0
        global_step = 0
        train_iter  = iter(train_loader)

        pbar = tqdm(total=num_iterations, desc=f"Fold {self.fold_idx + 1}")

        while global_step < num_iterations:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            train_loss, train_acc, train_acc_per_cls, train_f1 = train_step(batch, self.model, self.optimizer, self.criterion, self.device)
            current_lr = self.optimizer.param_groups[0]['lr']
            wandb.log({
                'train/loss':             train_loss,
                'train/acc':              train_acc,
                'train/acc_ujoh':         train_acc_per_cls['acc_ujoh'],
                'train/acc_gyemyeon':     train_acc_per_cls['acc_gyemyeon'],
                'train/acc_aniri':        train_acc_per_cls['acc_aniri'],
                'train/acc_changjo':      train_acc_per_cls['acc_changjo'],
                'train/f1_macro':         train_f1['f1_macro'],
                'train/f1_ujoh':          train_f1['f1_ujoh'],
                'train/f1_gyemyeon':      train_f1['f1_gyemyeon'],
                'train/f1_aniri':         train_f1['f1_aniri'],
                'train/f1_changjo':       train_f1['f1_changjo'],
                'train/lr':               current_lr,
            }, step=global_step)

            global_step += 1
            pbar.update(1)

            if global_step % eval_interval == 0:
                val_loss, val_acc, val_acc_per_cls, val_f1, val_song_data = run_epoch(
                    val_loader, self.model, self.optimizer, self.criterion, self.device, train=False)
                val_cm = plot_confusion_matrix(val_song_data)

                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_f1['f1_macro'])
                    else:
                        self.scheduler.step()

                wandb.log({
                    'val/loss':             val_loss,
                    'val/acc':              val_acc,
                    'val/acc_ujoh':         val_acc_per_cls['acc_ujoh'],
                    'val/acc_gyemyeon':     val_acc_per_cls['acc_gyemyeon'],
                    'val/acc_aniri':        val_acc_per_cls['acc_aniri'],
                    'val/acc_changjo':      val_acc_per_cls['acc_changjo'],
                    'val/f1_macro':         val_f1['f1_macro'],
                    'val/f1_ujoh':          val_f1['f1_ujoh'],
                    'val/f1_gyemyeon':      val_f1['f1_gyemyeon'],
                    'val/f1_aniri':         val_f1['f1_aniri'],
                    'val/f1_changjo':       val_f1['f1_changjo'],
                    'val/confusion_matrix': wandb.Image(Image.fromarray(val_cm)),
                }, step=global_step)

                marker = ''
                if val_f1['f1_macro'] > best_val_f1:
                    best_val_f1 = val_f1['f1_macro']
                    best_step   = global_step
                    torch.save(self.model.state_dict(), self.save_path)
                    marker = '  ← best'

                print(f"Step {global_step}/{num_iterations} | "
                      f"Train loss {train_loss:.4f}  acc {train_acc:.3f}  f1 {train_f1['f1_macro']:.3f} | "
                      f"Val loss {val_loss:.4f}  acc {val_acc:.3f}  f1 {val_f1['f1_macro']:.3f} "
                      f"[우조 {val_f1['f1_ujoh']:.3f} / 계면조 {val_f1['f1_gyemyeon']:.3f} / 아니리 {val_f1['f1_aniri']:.3f} / 창조 {val_f1['f1_changjo']:.3f}]  lr={self.optimizer.param_groups[0]['lr']:.2e}{marker}")
                pbar.set_description(f"Fold {self.fold_idx + 1} | best f1 {best_val_f1:.3f}")

            if global_step % save_interval == 0:
                ckpt_path = (str(self.fold_out_dir / f"step{global_step}.pt")
                             if self.fold_out_dir.exists()
                             else f"fold{self.fold_idx + 1}_step{global_step}.pt")
                torch.save(self.model.state_dict(), ckpt_path)

        pbar.close()
        print(f"Fold {self.fold_idx + 1} best val f1: {best_val_f1:.4f} at step {best_step}")

    def _fit_epoch(self, train_loader, val_loader):
        num_epochs    = self.cfg.train.num_epoch
        eval_interval = self.cfg.train.get('epoch_eval_interval', self.cfg.train.get('eval_interval', 1))

        best_val_f1 = 0.0
        best_epoch  = 0

        pbar = tqdm(total=num_epochs, desc=f"Fold {self.fold_idx + 1}")

        for epoch in range(num_epochs):
            train_loss, train_acc, train_acc_per_cls, train_f1, _ = run_epoch(
                train_loader, self.model, self.optimizer, self.criterion, self.device, train=True)
            current_lr = self.optimizer.param_groups[0]['lr']
            wandb.log({
                'train/loss':             train_loss,
                'train/acc':              train_acc,
                'train/acc_ujoh':         train_acc_per_cls['acc_ujoh'],
                'train/acc_gyemyeon':     train_acc_per_cls['acc_gyemyeon'],
                'train/acc_aniri':        train_acc_per_cls['acc_aniri'],
                'train/acc_changjo':      train_acc_per_cls['acc_changjo'],
                'train/f1_macro':         train_f1['f1_macro'],
                'train/f1_ujoh':          train_f1['f1_ujoh'],
                'train/f1_gyemyeon':      train_f1['f1_gyemyeon'],
                'train/f1_aniri':         train_f1['f1_aniri'],
                'train/f1_changjo':       train_f1['f1_changjo'],
                'train/lr':               current_lr,
            }, step=epoch)

            pbar.update(1)

            if (epoch + 1) % eval_interval == 0:
                val_loss, val_acc, val_acc_per_cls, val_f1, val_song_data = run_epoch(
                    val_loader, self.model, self.optimizer, self.criterion, self.device, train=False)
                val_cm = plot_confusion_matrix(val_song_data)

                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_f1['f1_macro'])
                    else:
                        self.scheduler.step()

                wandb.log({
                    'val/loss':             val_loss,
                    'val/acc':              val_acc,
                    'val/acc_ujoh':         val_acc_per_cls['acc_ujoh'],
                    'val/acc_gyemyeon':     val_acc_per_cls['acc_gyemyeon'],
                    'val/acc_aniri':        val_acc_per_cls['acc_aniri'],
                    'val/acc_changjo':      val_acc_per_cls['acc_changjo'],
                    'val/f1_macro':         val_f1['f1_macro'],
                    'val/f1_ujoh':          val_f1['f1_ujoh'],
                    'val/f1_gyemyeon':      val_f1['f1_gyemyeon'],
                    'val/f1_aniri':         val_f1['f1_aniri'],
                    'val/f1_changjo':       val_f1['f1_changjo'],
                    'val/confusion_matrix': wandb.Image(Image.fromarray(val_cm)),
                }, step=epoch)

                marker = ''
                if val_f1['f1_macro'] > best_val_f1:
                    best_val_f1 = val_f1['f1_macro']
                    best_epoch  = epoch + 1
                    torch.save(self.model.state_dict(), self.save_path)
                    marker = '  ← best'

                print(f"Epoch {epoch + 1}/{num_epochs} | "
                      f"Train loss {train_loss:.4f}  acc {train_acc:.3f}  f1 {train_f1['f1_macro']:.3f} | "
                      f"Val loss {val_loss:.4f}  acc {val_acc:.3f}  f1 {val_f1['f1_macro']:.3f} "
                      f"[우조 {val_f1['f1_ujoh']:.3f} / 계면조 {val_f1['f1_gyemyeon']:.3f} / 아니리 {val_f1['f1_aniri']:.3f} / 창조 {val_f1['f1_changjo']:.3f}]  lr={self.optimizer.param_groups[0]['lr']:.2e}{marker}")
                pbar.set_description(f"Fold {self.fold_idx + 1} | best f1 {best_val_f1:.3f}")

        pbar.close()
        print(f"Fold {self.fold_idx + 1} best val f1: {best_val_f1:.4f} at epoch {best_epoch}")

    def _evaluate(self, test_loader):
        state_dict = torch.load(self.save_path, map_location='cpu')
        self.model.load_state_dict(state_dict)

        test_loss, test_acc, test_acc_per_cls, test_f1, song_data, segment_results = run_test_epoch(
            test_loader, self.model, self.criterion, self.device,
            fs=self.cfg.data.fs, window_size=int(self.cfg.data.window_size * self.cfg.data.fs))

        print(f"\nFold {self.fold_idx + 1} Test | acc {test_acc:.4f}  f1_macro {test_f1['f1_macro']:.4f} "
              f"[우조 {test_f1['f1_ujoh']:.4f} / 계면조 {test_f1['f1_gyemyeon']:.4f} / 아니리 {test_f1['f1_aniri']:.4f} / 창조 {test_f1['f1_changjo']:.4f}]")
        print(f"총 곡 수: {len(song_data)}")
        print(f"총 segment 수: {sum(len(d['segments']) for d in song_data.values())}")

        test_cm = plot_confusion_matrix(song_data)
        wandb.log({
            'test/loss':             test_loss,
            'test/acc':              test_acc,
            'test/acc_ujoh':         test_acc_per_cls['acc_ujoh'],
            'test/acc_gyemyeon':     test_acc_per_cls['acc_gyemyeon'],
            'test/acc_aniri':        test_acc_per_cls['acc_aniri'],
            'test/acc_changjo':      test_acc_per_cls['acc_changjo'],
            'test/f1_macro':         test_f1['f1_macro'],
            'test/f1_ujoh':          test_f1['f1_ujoh'],
            'test/f1_gyemyeon':      test_f1['f1_gyemyeon'],
            'test/f1_aniri':         test_f1['f1_aniri'],
            'test/f1_changjo':       test_f1['f1_changjo'],
            'test/confusion_matrix': wandb.Image(Image.fromarray(test_cm)),
        })

        self.fold_out_dir.mkdir(parents=True, exist_ok=True)
        save_test_csv(segment_results, self.fold_out_dir / "test_results.csv")
        self._save_plots(song_data)

        # Extract version test songs from this fold's test results
        vt_midi_names = getattr(self, '_vt_midi_names', set())
        if vt_midi_names:
            self.vt_song_data = {
                name: {
                    'gt':         data['gt'].copy(),
                    'pred_probs': data['pred_probs'].copy(),
                    'segments':   [dict(s) for s in data['segments']],
                }
                for name, data in song_data.items()
                if name in vt_midi_names
            }
            self.vt_segment_results = [r for r in segment_results if r['song_name'] in vt_midi_names]
            # Write per-fold version test CSV to shared dir (for cross-process aggregation)
            _vt_out_dir = self.cfg.train.get('vt_out_dir', None)
            if _vt_out_dir and self.vt_segment_results:
                vt_out_path = Path(_vt_out_dir)
                vt_out_path.mkdir(parents=True, exist_ok=True)
                save_test_csv(self.vt_segment_results, vt_out_path / f'fold{self.fold_idx + 1}_vt.csv')
                print(f"  Version test CSV: {len(self.vt_segment_results)} segments → {vt_out_path}/fold{self.fold_idx + 1}_vt.csv")
            # Collect paths to already-saved posteriorgram PNGs for version test songs
            vt_stems = {Path(n).stem for n in vt_midi_names}
            self.vt_png_paths = [
                p for p in self.fold_out_dir.glob('*.png')
                if p.stem.split('_')[0] in vt_stems or any(p.stem.startswith(s) for s in vt_stems)
            ]
            print(f"  Version test: {len(self.vt_song_data)} songs captured, {len(self.vt_png_paths)} PNGs collected")
        else:
            self.vt_song_data       = {}
            self.vt_segment_results = []
            self.vt_png_paths       = []

    def _save_plots(self, song_data):
        for sname, data in song_data.items():
            stem = Path(sname).stem

            for seg in data['segments']:
                seg_label = f"{seg['start_sec']:.0f}-{seg['end_sec']:.0f}s"
                fig = plot_posteriorgram(f"{stem} [{seg_label}]", seg['gt'], seg['pred_probs'])
                fig.savefig(self.fold_out_dir / f"{stem}_{seg_label}.png", dpi=120, bbox_inches='tight')
                fig.clf()
                plt.close(fig)

            fig = plot_posteriorgram(sname, data['gt'], data['pred_probs'])
            fig.savefig(self.fold_out_dir / f"{stem}_full.png", dpi=120, bbox_inches='tight')
            fig.clf()
            plt.close(fig)

        plt.close('all')

# ── Module-level helpers for version test tracking ────────────────────────────

def load_version_test_midi_names(cfg):
    """Return set of MIDI filenames that correspond to songs in version_test.txt."""
    import unicodedata, json as _json

    song_strat_dir = Path(cfg.data.dir.get('song_stratified_dir', ''))
    vt_file = song_strat_dir / 'version_test.txt'
    if not vt_file.exists():
        return set()

    vt_hashes = set()
    for line in vt_file.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            vt_hashes.add(line.split()[0])  # "hash_key singer — description"

    midi_dir   = Path(cfg.data.dir.midi_dir)
    label_path = cfg.data.dir.label_path
    with open(label_path, encoding='utf-8') as f:
        raw = _json.load(f)

    vt_midi_names = set()
    for item in raw:
        fu = unicodedata.normalize('NFC', item['file_upload'])
        if fu.split('-')[0] in vt_hashes:
            midi_name = fu.rsplit('.', 1)[0] + '_vocal.mid'
            if (midi_dir / midi_name).exists():
                vt_midi_names.add(midi_name)

    print(f"  Version test MIDI files: {len(vt_midi_names)} found")
    return vt_midi_names


def log_version_test_summary(all_vt_song_data, all_vt_segment_results, all_vt_png_paths, cfg, T):
    """After all folds: aggregate version test metrics → wandb run + CSV + posteriorgrams."""
    from .metrics import masked_acc, masked_acc_per_class, masked_f1
    import numpy as np

    if not all_vt_song_data:
        print("  No version test songs collected — skipping summary.")
        return

    # Aggregate preds and targets across all accumulated songs
    all_preds, all_tgts = [], []
    for data in all_vt_song_data.values():
        all_preds.append(torch.tensor(np.argmax(data['pred_probs'], axis=-1).flatten()))
        all_tgts.append(torch.tensor(np.argmax(data['gt'],         axis=-1).flatten()))
    all_preds = torch.cat(all_preds)
    all_tgts  = torch.cat(all_tgts)

    total_acc   = masked_acc(all_preds, all_tgts)
    acc_per_cls = masked_acc_per_class(all_preds, all_tgts)
    f1          = masked_f1(all_preds, all_tgts)

    print(f"\n===== Version Test Summary ({len(all_vt_song_data)} songs, {len(all_vt_segment_results)} segments) =====")
    print(f"  Acc: {total_acc:.4f}  Macro F1: {f1['f1_macro']:.4f}  "
          f"[우조 {f1['f1_ujoh']:.4f} / 계면조 {f1['f1_gyemyeon']:.4f} / "
          f"아니리 {f1['f1_aniri']:.4f} / 창조 {f1['f1_changjo']:.4f}]")

    # Save CSV + copy already-rendered posteriorgram PNGs from fold dirs
    out_dir = OUTPUT_DIR / f"version_test_summary_{T}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if all_vt_segment_results:
        save_test_csv(all_vt_segment_results, out_dir / "version_test_results.csv")

    copied = 0
    for src in all_vt_png_paths:
        if src.exists():
            shutil.copy2(src, out_dir / src.name)
            copied += 1
    print(f"  Copied {copied} posteriorgram PNGs → {out_dir}")

    # Dedicated wandb summary run (same project as fold runs)
    wandb.init(
        project=cfg.project_name,
        name=f"version_test_summary_{T}",
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
    )
    wandb.log({
        'version_test/acc':             total_acc,
        'version_test/acc_ujoh':        acc_per_cls['acc_ujoh'],
        'version_test/acc_gyemyeon':    acc_per_cls['acc_gyemyeon'],
        'version_test/acc_aniri':       acc_per_cls['acc_aniri'],
        'version_test/acc_changjo':     acc_per_cls['acc_changjo'],
        'version_test/f1_macro':        f1['f1_macro'],
        'version_test/f1_ujoh':         f1['f1_ujoh'],
        'version_test/f1_gyemyeon':     f1['f1_gyemyeon'],
        'version_test/f1_aniri':        f1['f1_aniri'],
        'version_test/f1_changjo':      f1['f1_changjo'],
        'version_test/n_songs':         len(all_vt_song_data),
        'version_test/n_segments':      len(all_vt_segment_results),
    })
    wandb.finish()
