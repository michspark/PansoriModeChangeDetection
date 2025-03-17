import models
import datasets
from trainer import Trainer
import random
import datetime
from pathlib import Path
import numpy as np
<<<<<<< HEAD
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
=======
from tqdm import tqdm
from sklearn.model_selection import KFold, LeaveOneOut

>>>>>>> jinin/opt
import hydra
import wandb
from omegaconf import OmegaConf
import torch
import torch.nn as nn
from torch.optim import Adam

#DEV = 'mps' if torch.mps.is_available() else 'cpu'
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu") #GPU setup for Sangheon

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Set Seed {seed}")

@hydra.main(config_path="configs", config_name="train")
def main(cfg):
    set_seed(cfg.train.random_seed)

    audio_dir = cfg.data.audio_dir
    label_json = cfg.data.label_json

    save_dir = Path(f'{cfg.dir.save_dir}/{datetime.datetime.now().strftime("%m%d_%H%M")}_{cfg.models.cls}')
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir/'config.yaml', 'w') as f: OmegaConf.save(cfg, f)

    best_dir = Path(f'{cfg.dir.best_dir}/{save_dir.name}')
    best_dir.mkdir(parents=True, exist_ok=True)
    with open(best_dir/'config.yaml', 'w') as f: OmegaConf.save(cfg, f)

    dataset_name = cfg.datasets.dset
    dataset_params = OmegaConf.to_container(cfg.datasets.cfg)
    dataset_class = getattr(datasets, dataset_name)
    dataset = dataset_class(audio_dir, label_json, **dataset_params)
    print(f"Length of dataset: {len(dataset)}")

    criterion = nn.CrossEntropyLoss()
    selection = KFold(**cfg.kfold) if cfg.train.selection=="KFold" else LeaveOneOut(**cfg.loo)

    model_name = cfg.models.cls
    model_params = OmegaConf.to_container(cfg.models.cfg)
    model_class = getattr(models, model_name)

<<<<<<< HEAD
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=cfg.train.lr)
    kfold = KFold(**cfg.kfold)

    trainer = Trainer(model=model,
                      dataset=dataset,
                      optimizer=optimizer,
                      criterion=criterion,
                      device=DEV,
                      kfold=kfold,
                      save_dir=save_dir,
=======
    trainer = Trainer(dataset=dataset, 
                      criterion=criterion,
                      device=DEV,
                      save_dir=save_dir, 
>>>>>>> jinin/opt
                      best_dir=best_dir,
                      config=cfg)

    fold_best_acc = {}
    for idx, (train_idx, test_idx) in enumerate(selection.split(range(len(trainer.dataset)))):
        if wandb.run is not None: wandb.finish()
        run_name = f'{cfg.models.cls}_Fold{idx+1}_{datetime.datetime.now().strftime("%m%d_%H%M")}'
        wandb.init(project='Pansori_Mode_Detection', name=run_name, reinit=True)
        wandb.config.update(OmegaConf.to_container(cfg))

        model = model_class(**model_params)
        model = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model.to(DEV)
        num_model_parameters = sum(p.numel() for p in model_class(**model_params).parameters())
        wandb.summary['Num model parameters'] = num_model_parameters

        optimizer = Adam(model.parameters(), lr=cfg.train.lr)

        fold_best_acc[idx] = trainer.train_split(idx, train_idx, test_idx, model, optimizer)

    print(f"KFold Average Accuracy: {sum(fold_best_acc.values())}")
    if wandb.run is not None: wandb.finish()

if __name__ == "__main__":
    main()
