from dataclasses import dataclass

import pyrallis
from torch import nn
from tqdm import tqdm
import torch

import wandb
from apps.deep.model_analyzer_paper1 import analyze_models
from src.deep import models, data_loaders
from src.deep.data_loaders import SeparatedRealImagDataset
from src.deep.ml_ops import Trainer
from src.deep.standalone_methods import get_platform


@dataclass
class PaperModelConfig:
    lr: float = 1e-3
    epochs: int = 3
    batch_size: int = 1
    train_val_ratio: float = 0.8
    input_data_path: str = './data/datasets/qam1024_160x20/160_samples_mu=0.001'
    output_model_path: str = './data/test_models'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    wandb_project: str = 'thesis_model_scan_test'
    model_name: str = 'paper_no_relu'


def main(config: PaperModelConfig):
    print(f"Running paper model")
    train_dataset, val_dataset = data_loaders.get_train_val_datasets(config.input_data_path, SeparatedRealImagDataset,
                                                                     train_val_ratio=config.train_val_ratio)

    wandb.init(project=config.wandb_project, entity="yarden92", name=config.model_name+f'_real',
               tags=[f'mu={train_dataset.mu}', f'{get_platform()}', config.model_name+f'_real', f'ds={len(train_dataset)}'],
               reinit=False)
    wandb.config = {
        "learning_rate": config.lr,
        "epochs": config.epochs,
        "batch_size": config.batch_size
    }
    l_metric = nn.MSELoss()  # or L1Loss
    model_real = models.PaperNNforNFTmodel()
    model_imag = models.PaperNNforNFTmodel()

    if config.device == 'cuda':
        device1, device2 = 'cuda:0', 'cuda:1'
    else:
        device1, device2 = 'cpu', 'cpu'

    optim_real = torch.optim.Adam(model_real.parameters(), lr=config.lr)
    optim_imag = torch.optim.Adam(model_imag.parameters(), lr=config.lr)

    # train
    trainer_real = Trainer(train_dataset=train_dataset, val_dataset=val_dataset, batch_size=config.batch_size,
                           model=model_real, device=device1,
                           l_metric=l_metric, optim=optim_real,
                           params=config.__dict__)

    trainer_imag = Trainer(train_dataset=train_dataset, val_dataset=val_dataset, batch_size=config.batch_size,
                           model=model_imag, device=device2,
                           l_metric=l_metric, optim=optim_imag,
                           params=config.__dict__)

    print('training real part')
    train_dataset.set_is_real(True), val_dataset.set_is_real(True)
    trainer_real.train(num_epochs=config.epochs, _tqdm=tqdm)
    trainer_real.save3(config.output_model_path, '__real')
    print('finish training real part')

    print('training imaginary part')
    wandb.init(project=config.wandb_project, entity="yarden92", name=config.model_name+f'_imag',
               tags=[f'mu={train_dataset.mu}', f'{get_platform()}', config.model_name + f'_imag',
                     f'ds={len(train_dataset)}'],
               reinit=True)
    wandb.config = {
        "learning_rate": config.lr,
        "epochs": config.epochs,
        "batch_size": config.batch_size
    }

    train_dataset.set_is_real(False), val_dataset.set_is_real(False)
    trainer_imag.train(num_epochs=config.epochs, _tqdm=tqdm)
    trainer_imag.save3(config.output_model_path, '__imag')

    # TODO: merge trainers and save the merged instead
    wandb.init(project=config.wandb_project, entity="yarden92", name=config.model_name+f'_analysis',
               tags=[f'mu={train_dataset.mu}', f'{get_platform()}', config.model_name + f'_analysis',
                     f'ds={len(train_dataset)}'],
               reinit=True)
    wandb.config = {
        "learning_rate": config.lr,
        "epochs": config.epochs,
        "batch_size": config.batch_size
    }
    analyze_models(trainer_real, trainer_imag)


if __name__ == '__main__':
    config = pyrallis.parse(PaperModelConfig)
    main(config)
