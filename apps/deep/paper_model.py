from dataclasses import dataclass

import pyrallis
from torch import nn
from tqdm import tqdm
import torch

import wandb
from apps.deep.multi_model_generator import analyze_model
from src.deep import models, data_loaders
from src.deep.data_loaders import SeparatedRealImagDataset
from src.deep.ml_ops import Trainer


@dataclass
class PaperModelConfig:
    lr: float = 1e-3
    epochs: int = 3
    batch_size: int = 1
    train_val_ratio: float = 0.8
    input_data_path: str = './data/datasets/qam1024_160x20/160_samples_mu=0.001'
    output_model_path: str = './data/test_models'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def main(config: PaperModelConfig):
    print(f"Running paper model")
    wandb.init(project="thesis_model_scan_test", entity="yarden92", name='paper_model_real')
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

    train_dataset, val_dataset = data_loaders.get_train_val_datasets(config.input_data_path, SeparatedRealImagDataset,
                                                                     train_val_ratio=config.train_val_ratio)

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
    # analyze_model(trainer_real) #TODO fix it
    del trainer_real

    print('training imaginary part')
    wandb.init(project="thesis_model_scan_test", entity="yarden92", name='paper_model_imag', reinit=True)
    wandb.config = {
        "learning_rate": config.lr,
        "epochs": config.epochs,
        "batch_size": config.batch_size
    }

    train_dataset.set_is_real(False), val_dataset.set_is_real(False)
    trainer_imag.train(num_epochs=config.epochs, _tqdm=tqdm)
    trainer_imag.save3(config.output_model_path, '__imag')


if __name__ == '__main__':
    config = pyrallis.parse(PaperModelConfig)
    main(config)
