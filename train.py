import models
import datasets
from trainer import Trainer

import random
import datetime
from pathlib import Path

import numpy as np

import hydra
from omegaconf import OmegaConf

import torch
import torch.nn as nn
from torch.optim import Adam

DEV = 'mps' if torch.mps.is_available() else 'cpu'

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Set Seed {seed}")

@hydra.main(config_path="configs", config_name="unified_config")
def main(cfg):
    set_seed(cfg.train.random_seed)

    audio_dir = cfg.data.audio_dir
    label_json = cfg.data.label_json

    save_dir = Path(f'{cfg.dir.save_dir}/{datetime.datetime.now().strftime("%m%d_%H%M")}_{cfg.model.name}')
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir/'config.yaml', 'w') as f: OmegaConf.save(cfg, f)

    best_dir = Path(f'{cfg.dir.best_dir}/{save_dir.name}')
    best_dir.mkdir(parents=True, exist_ok=True)
    with open(best_dir/'config.yaml', 'w') as f: OmegaConf.save(cfg, f)

    dataset_name = cfg.dataset.name
    dataset_params = OmegaConf.to_container(cfg.dataset.params)    
    dataset_class = getattr(datasets, dataset_name)
    dataset = dataset_class(audio_dir, label_json, **dataset_params)
    print(f"Length of dataset: {len(dataset)}")

    criterion = nn.CrossEntropyLoss()

    model_name = cfg.model.name
    model_params = OmegaConf.to_container(cfg.model.params)
    model_class = getattr(models, model_name)
    model = model_class(**model_params)
    # model = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model.to(DEV)
    optimizer = Adam(model.parameters(), lr=cfg.train.lr)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      dataset=dataset, 
                      criterion=criterion,
                      device=DEV,
                      save_dir=save_dir, 
                      best_dir=best_dir,
                      config=cfg)

    fold_best_acc = trainer.train()
    print(f"KFold Average Accuracy: {sum(fold_best_acc.values())/len(fold_best_acc)}")

    
if __name__ == "__main__":
    main()