import models
import losses
import trainers
import datasets
import random
import datetime
from pathlib import Path
import numpy as np
import hydra
from omegaconf import OmegaConf
import torch
import torch.nn as nn
from torch.optim import Adam

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    print(f"Set Seed {seed}")


@hydra.main(config_path="configs/frame", config_name="cmert")
def main(cfg):
    print(f"Dataset  : {cfg.dataset.name}")
    print(f"Model    : {cfg.model.name}")
    print(f"Trainer  : {cfg.train.trainer}")
    print(f"Device   : {DEV}")
    set_seed(cfg.train.random_seed)

    save_dir = Path(f"{cfg.dir.save_dir}/{datetime.datetime.now().strftime('%m%d_%H%M')}_{cfg.data.data_dir.split('/')[-1]}_{cfg.model.name}_{cfg.dataset.name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir/'config.yaml', 'w') as f: OmegaConf.save(cfg, f)

    dataset_name = cfg.dataset.name
    dataset_params = OmegaConf.to_container(cfg.dataset.params)
    dataset_class = getattr(datasets, dataset_name)
    dataset = dataset_class(**cfg.data, **dataset_params)
    print(f"Length of dataset: {len(dataset)}")

    criterion_name = cfg.loss.name
    criterion_params = OmegaConf.to_container(cfg.loss.params)
    criterion_class = getattr(losses, criterion_name)
    criterion = criterion_class(**criterion_params)

    model_name = cfg.model.name
    model_params = cfg.model.params
    model_class = getattr(models, model_name)
    model = model_class(model_params)

    optimizer = Adam(model.parameters(), lr=cfg.train.lr)

    trainer_name = cfg.train.trainer
    trainer_class = getattr(trainers, trainer_name)
    trainer = trainer_class(model=model,
                            optimizer=optimizer,
                            dataset=dataset,
                            criterion=criterion,
                            device=DEV,
                            save_dir=save_dir,
                            config=cfg)

    fold_best_acc = trainer.train()
    print(f"Split Average Accuracy: {sum(fold_best_acc.values())/len(fold_best_acc)}")

if __name__ == "__main__":
    main()

