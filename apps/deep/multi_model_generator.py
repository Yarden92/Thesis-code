import json
import platform
from dataclasses import dataclass
from typing import List

import numpy as np
import pyrallis
import torch

import wandb
from torch import nn

from src.deep import data_loaders
from src.deep.data_loaders import SingleMuDataSet
from src.deep.ml_ops import Trainer
from src.deep.models import NLayersModel


@dataclass
class ModelConfig:
    n_layers: int = 3  # number of layers
    sizes: List[int] = None  # list of sizes of layers (need n_layers+1)
    kernel_sizes: List[int] = None  # kernel sizes of each layer (need n_layers)
    drop_rates: List[float] = None  # drop rates of each layer (need n_layers)
    activation_name: str = 'PReLU'  # activation function
    layer_name: str = 'Conv1d'  # layer type


@dataclass
class ModelsConfig:
    models: str = '{}'  # string jsons of models '{"n_layers":1,"activation_name":"relu"};{"n_layers"...}'
    epochs: int = 10  # num of epochs
    lr: float = 1e-3  # learning rate
    batch_size: int = 1  # batch size
    input_data_path: str = './data/datasets/qam1024_150x5/150_samples_mu=0.001'  # path to data
    output_model_path: str = './data/test_models'  # path to save model
    device: str = 'auto'  # device to use
    train_val_ratio: float = 0.8  # train vs val ratio


def train_model(model: nn.Module, train_ds, val_ds, run_name: str, lr: float, epochs: int, batch_size: int, device,
                output_model_path: str, mu):
    wandb.init(project="Thesis_model_scanning", entity="yarden92", name=run_name,
               tags=[f'mu={mu}', f'{get_platform()}', f'{model.n_layers}_layers', f'ds={len(train_ds)}'],
               reinit=True)
    wandb.config = {
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size
    }

    l_metric = nn.MSELoss()  # or L1Loss
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    trainer = Trainer(train_dataset=train_ds, val_dataset=val_ds, batch_size=batch_size,
                      model=model, device=device,
                      l_metric=l_metric, optim=optim,
                      params={})

    trainer.train(num_epochs=epochs)
    trainer.save3(output_model_path)

    wandb.finish()

    print(f'finished training {run_name}')

    return trainer


def analyze_model(trainer: Trainer):
    wandb.init(project="Thesis_model_scanning", entity="yarden92",
               tags=[f'mu={trainer.train_dataset.mu}',
                     f'{get_platform()}',
                     f'{trainer.model.n_layers}_layers',
                     f'ds={len(trainer.train_dataset)}'],
               name=f'{trainer.model.n_layers}_layers__{trainer.train_dataset.mu}_mu__analysis',
               reinit=True)
    org_ber, model_ber, ber_improvement = trainer.compare_ber()
    wandb.log({'org_ber': org_ber, 'model_ber': model_ber, 'ber_improvement': ber_improvement})

    x, y, preds = trainer.test_single_item(i=0, plot=False)
    indices = np.arange(len(x))
    wandb.log({"sample test from model": wandb.plot.line_series(
        xs=indices,
        ys=[x, y, preds],
        keys=["dirty (end of channel)", "clean (before channel)", "pred (after model)"],
        title="model test",
        xname="sample index")})


def parse_models_config(model_config: str):
    models_strings = model_config.split(';')
    for model_string in models_strings:
        dict = json.loads(model_string)
        yield ModelConfig(**dict)


def get_platform():
    try:
        os = platform.system()
        if 'darwin' in os.lower():
            os = 'Mac'
    except:
        os = 'unknown'

    return os


def main(config: ModelsConfig):
    if config.device == 'auto': config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataset, val_dataset = data_loaders.get_train_val_datasets(config.input_data_path, SingleMuDataSet,
                                                                     train_val_ratio=config.train_val_ratio)
    mu = train_dataset.mu
    for model_config in parse_models_config(config.models):
        run_name = f'{model_config.n_layers}_layers__{mu}_mu'
        print(f'running model {run_name}')
        model = NLayersModel(**model_config.__dict__)
        trainer = train_model(
            model=model, train_ds=train_dataset, val_ds=val_dataset,
            run_name=run_name, lr=config.lr, epochs=config.epochs, batch_size=config.batch_size,
            device=config.device, output_model_path=config.output_model_path, mu=mu
        )

        analyze_model(trainer)

    print('finished all models')


if __name__ == '__main__':
    config = pyrallis.parse(config_class=ModelsConfig)
    main(config)
