from dataclasses import dataclass

import torch
import wandb
import pyrallis
from torch import nn
from tqdm import tqdm

from src.deep import data_loaders
from src.deep.data_loaders import SingleMuDataSet
from src.deep.ml_ops import Trainer
from src.deep.models import SingleMuModel3Layers


@dataclass
class TrainConfig:
    run_name: str = "test_model_10epochs"
    epochs: int = 10
    lr: float = 1e-3
    batch_size: int = 128
    train_val_ratio: float = 0.8
    input_data_path: str = './data/datasets/qam1024_150x5/150_samples_mu=0.001'
    output_model_path: str = './data/test_models'


def main(config: TrainConfig):
    # config
    print(f"Running {config.run_name}")
    print(config)
    wandb.init(project="Thesis", entity="yarden92", name=config.run_name)
    wandb.config = {
        "learning_rate": config.lr,
        "epochs": config.epochs,
        "batch_size": config.batch_size
    }
    l_metric = nn.MSELoss()  # or L1Loss
    model = SingleMuModel3Layers()
    train_dataset, val_dataset = data_loaders.get_train_val_datasets(config.input_data_path, SingleMuDataSet,
                                                                     train_val_ratio=config.train_val_ratio)

    optim = torch.optim.Adam(model.parameters(), lr=config.lr)
    trainer = Trainer(train_dataset=train_dataset, val_dataset=val_dataset, model=model, l_metric=l_metric, optim=optim, config=config)

    trainer.train(num_epochs=config.epochs, verbose_level=1, _tqdm=tqdm)
    trainer.save_model(config.output_model_path)

    print('finished training')


if __name__ == '__main__':
    config = pyrallis.parse(config_class=TrainConfig)
    main(config)
