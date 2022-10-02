import os

import torch
from torch import nn
from tqdm import tqdm

from apps.deep.model_analyzer import ModelAnalyzer
from src.deep import data_loaders
from src.deep.data_loaders import DatasetNormal
from src.deep.ml_ops import Trainer
from src.deep.models import SingleMuModel3Layers, Paper1ModelWrapper


def test1_model_test():
    # define model and parameters
    l_metric = nn.MSELoss()  # or L1Loss
    model = SingleMuModel3Layers()
    dir = '../data/datasets/qam1024_100x10/100_samples_mu=0.008'
    train_dataset, val_dataset = data_loaders.get_train_val_datasets(dir, DatasetNormal, train_val_ratio=0.8)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    # %%
    trainer = Trainer(train_dataset=train_dataset, val_dataset=val_dataset, model=model, l_metric=l_metric, optim=optim)
    trainer.test_single_item(0, verbose=True)

    trainer.train(num_epochs=10, mini_batch_size=10)

    trainer.test_single_item(0,f'after {trainer.train_state_vec.num_epochs} epochs')

    trainer.compare_ber(_tqdm=tqdm)

    trainer.plot_loss_vec()

    pass

def test2_model_analyzer():
    trainer = Trainer.load_trainer_from_file('./data/saved_models/model_SingleMuModel3Layers_51_epochs_mu_0.008')
    trainer.fix_datasets_paths('../../data/datasets')


    trainer.test_single_item(0, verbose=True)

    trainer.plot_loss_vec()

    trainer.compare_ber(_tqdm=tqdm)


def test3_load_trainer():
    trainer = Trainer.load_from_file('./data/saved_models/SingleMuModel3Layers_ds-128_epochs-10_mu-0.008')
    pass


def test4_paper1():
    # paper 1 model analyzer
    path_init = '../data/test_models/mu-0.008__128ds__PaperNNforNFTmodel__3epochs'

    trainer_real = Trainer.load3(path_init + '__real')
    trainer_imag = Trainer.load3(path_init + '__imag')

    train_ds = DatasetNormal(trainer_real.train_dataset.data_dir_path,
                             trainer_real.train_dataset.data_indices)
    val_ds = DatasetNormal(trainer_real.val_dataset.data_dir_path,
                           trainer_real.val_dataset.data_indices)

    model = Paper1ModelWrapper(trainer_real.model, trainer_imag.model)

    trainer = Trainer(
        train_dataset=train_ds,
        val_dataset=val_ds,
        model=model,
        device=trainer_real.device,
        batch_size=trainer_real.train_dataloader.batch_size,
        l_metric=trainer_real.l_metric,
        optim=trainer_real.optim,
        params=trainer_real.params)

    ma = ModelAnalyzer(trainer)
    ma.plot_single_item(i=0)


if __name__ == '__main__':
    test4_paper1()