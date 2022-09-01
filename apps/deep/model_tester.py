import numpy as np
import torch
import wandb
from torch import nn
from tqdm import tqdm

from src.deep import data_loaders
from src.deep.data_loaders import SingleMuDataSet
from src.deep.ml_ops import Trainer
from src.deep.models import SingleMuModel3Layers
from src.optics.channel_simulation import ChannelSimulator
from src.optics.split_step_fourier import SplitStepFourier


def main():
    # config
    wandb.init(project="Thesis", entity="yarden92")
    epochs = 50
    lr = 1e-3
    batch_size = 128
    wandb.config = {
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size
    }

    l_metric = nn.MSELoss()  # or L1Loss
    model = SingleMuModel3Layers()
    dir = '../../data/datasets/qam1024_1000x20/1000_samples_mu=0.001'
    train_dataset, val_dataset = data_loaders.get_train_val_datasets(dir, SingleMuDataSet, train_val_ratio=0.8)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = Trainer(train_dataset=train_dataset, val_dataset=val_dataset, model=model, l_metric=l_metric, optim=optim)

    trainer.test_single_item(0, verbose=True)

    trainer.train(num_epochs=epochs, verbose_level=1, _tqdm=tqdm)
    trainer.save_model('../../data/saved_models')
    trainer.plot_loss_vec()
    trainer.test_single_item(0, f'after {trainer.train_state_vec.num_epochs} epochs')
    trainer.compare_ber(tqdm=tqdm)


if __name__ == '__main__':
    main()
