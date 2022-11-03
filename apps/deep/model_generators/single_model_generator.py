import json
from dataclasses import dataclass

import pyrallis
import torch
import wandb
from torch import nn
from tqdm import tqdm

from src.deep import data_loaders
from src.deep.data_loaders import DatasetNormal
from src.deep.model_analyzer_src import ModelAnalyzer
from src.deep.models.paper2_model import Paper2Model
from src.deep.standalone_methods import get_platform
from src.deep.trainers import Trainer


@dataclass
class SingleModelTrainConfig:
    lr: float = 1e-3  # learning rate
    epochs: int = 3  # num of epochs
    batch_size: int = 1  # batch size
    train_val_ratio: float = 0.8  # train vs val ratio
    input_data_path: str = './data/datasets/qam1024_160x20/160_samples_mu=0.008'  # path to data
    output_model_path: str = './data/test_models'  # path to save model
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'  # device to use
    wandb_project: str = 'thesis_model_scan_test'  # wandb project name
    run_name: str = "test_model_10epochs"  # name of the run in wandb


def single_model_main(model, config: SingleModelTrainConfig, verbose: bool = False):
    # config
    print(f"Running {config.run_name}")
    if verbose: print(json.dumps(config.__dict__, indent=4))
    train_dataset, val_dataset = data_loaders.get_train_val_datasets(config.input_data_path, DatasetNormal,
                                                                     train_val_ratio=config.train_val_ratio)

    wandb.init(project=config.wandb_project, entity="yarden92", name=config.run_name,
               tags=[f'mu={train_dataset.mu}', f'{get_platform()}', f'ds={len(train_dataset)}'])
    wandb.config = {
        "learning_rate": config.lr,
        "epochs": config.epochs,
        "batch_size": config.batch_size
    }
    l_metric = nn.MSELoss()  # or L1Loss

    optim = torch.optim.Adam(model.parameters(), lr=config.lr)
    trainer = Trainer(train_dataset=train_dataset, val_dataset=val_dataset, batch_size=config.batch_size,
                      model=model, device=config.device,
                      l_metric=l_metric, optim=optim,
                      config=config.__dict__)
    # trainer.model.print_architecture(train_dataset[0])
    trainer.train(num_epochs=config.epochs, _tqdm=tqdm)
    trainer.save3(config.output_model_path)

    print(f'finished training {config.run_name}')

    ma = ModelAnalyzer(trainer)
    ma.upload_single_item_plots_to_wandb(i=0)
    ma.upload_bers_to_wandb()


if __name__ == '__main__':
    config = pyrallis.parse(config_class=SingleModelTrainConfig)
    model = Paper2Model()
    single_model_main(model, config)
