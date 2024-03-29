import json
from dataclasses import dataclass
import sys

import pyrallis
import torch
import wandb
from torch import nn
from tqdm import tqdm

from src.deep import data_loaders
from src.deep.data_loaders import DatasetNormal
from src.deep.model_analyzer_src import ModelAnalyzer
from src.deep.models import *  # essential for model name searching
from src.deep.standalone_methods import get_platform
from src.deep.trainers import Trainer


@dataclass
class SingleModelTrainConfig:
    lr: float = 1e-3  # learning rate
    min_lr: float = 1e-5  # min learning rate
    lambda_reg: float = 0.001  # regularization lambda (0 for no regularization)
    epochs: int = 3  # num of epochs
    batch_size: int = 1  # batch size
    # train_val_ratio: float = 0.8  # train vs val ratio
    train_ds_ratio: float = 0.05  # train vs dataset ratio
    val_ds_ratio: float = 0.03  # val vs dataset ratio
    test_ds_ratio: float = 0.02  # test vs dataset ratio
    input_data_path: str = './data/datasets/qam1024_160x20/160_samples_mu=0.008'  # path to data
    output_model_path: str = './data/test_models'  # path to save model
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'  # device to use
    wandb_project: str = 'thesis_model_scan_test'  # wandb project name
    model_name: str = "test_model_10epochs"  # name of the run in wandb
    ds_limit: int = None  # limit the dataset size, use 0 for unlimited (as much as exists)
    model_class: str = "Paper2Model"  # the exact class name
    is_analyze_after: bool = False  # if true, will analyze the model after training
    verbose: bool = False  # if true, will print the config


DATASETTYPE = DatasetNormal


def single_model_main(config: SingleModelTrainConfig):
    # config
    print(f"Running {config.model_name}")
    if config.verbose:
        print(json.dumps(config.__dict__, indent=4))
    try:
        ModelClass = globals()[config.model_class]
        print(f"Running {ModelClass.__name__} model")
    except Exception:
        raise f'failed to find class named {config.model_class}, make sure you wrote it correctly and imported it'

    train_dataset: DATASETTYPE
    val_dataset: DATASETTYPE
    train_dataset, val_dataset, _ = data_loaders.get_datasets_set(config.input_data_path, DATASETTYPE,
                                                                     train_ds_ratio=config.train_ds_ratio,
                                                                     val_ds_ratio=config.val_ds_ratio,
                                                                     test_ds_ratio=config.test_ds_ratio,
                                                                     )

    wandb.init(project=config.wandb_project, entity="yarden92", name=config.model_name,
               tags=[f'mu={train_dataset.cropped_mu}', f'{get_platform()}', f'ds={len(train_dataset)}'])
    wandb.config = {
        "learning_rate": config.lr,
        "epochs": config.epochs,
        "batch_size": config.batch_size
    }
    l_metric = nn.MSELoss()  # or L1Loss
    model = ModelClass()

    optim = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=config.epochs, eta_min=config.min_lr)
    trainer = Trainer(train_dataset=train_dataset, val_dataset=val_dataset, batch_size=config.batch_size,
                      model=model, device=config.device,
                      l_metric=l_metric, optim=optim,
                      scheduler=scheduler,
                      config=config.__dict__,
                      lambda_reg = config.lambda_reg,
                      )
    # trainer.model.print_architecture(train_dataset[0])
    trainer.train(num_epochs=config.epochs, _tqdm=tqdm)
    trainer.save3(config.output_model_path, config.model_name)

    print(f'finished training {config.model_name}')
    if config.is_analyze_after:
        ma = ModelAnalyzer(trainer)
        ma.upload_single_item_plots_to_wandb(i=0)
        ma.upload_bers_to_wandb()
        ma.upload_stems_to_wandb(i=0)


def show_and_choose_config(dir_path='./config/model_generator'):
    # print all the config files in the dir with numbers and let the user choose one
    # return the chosen config path
    import os
    files = os.listdir(dir_path)
    # sort by name:
    files.sort()
    for i, file in enumerate(files):
        print(f'[{i}] {file}')
    chosen = int(input('choose config file: '))
    return os.path.join(dir_path, files[chosen])


if __name__ == '__main__':
    if len(sys.argv) > 1:
        config = pyrallis.parse(SingleModelTrainConfig)
    else:
        config_path = show_and_choose_config()
        config = pyrallis.parse(SingleModelTrainConfig, config_path)
    single_model_main(config)
