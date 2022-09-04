import os

import torch
from torch import nn
from tqdm import tqdm

from src.deep import data_loaders
from src.deep.data_loaders import SingleMuDataSet
from src.deep.ml_ops import Trainer
from src.deep.models import SingleMuModel3Layers


def test1_model_test():
    # define model and parameters
    l_metric = nn.MSELoss()  # or L1Loss
    model = SingleMuModel3Layers()
    dir = '../data/datasets/qam1024_100x10/100_samples_mu=0.008'
    train_dataset, val_dataset = data_loaders.get_train_val_datasets(dir,SingleMuDataSet,train_val_ratio=0.8)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    # %%
    trainer = Trainer(train_dataset=train_dataset, val_dataset=val_dataset, model=model, l_metric=l_metric, optim=optim)
    trainer.test_single_item(0, verbose=True)

    trainer.train(num_epochs=10, mini_batch_size=10, verbose_level=1)

    trainer.test_single_item(0,f'after {trainer.train_state_vec.num_epochs} epochs')

    trainer.compare_ber(tqdm=tqdm)

    trainer.plot_loss_vec()

    pass

def test2_model_analyzer():
    trainer = Trainer.load_trainer_from_file('./data/saved_models/model_SingleMuModel3Layers_51_epochs_mu_0.008')

    # TODO: create function Trainer.fix_datasets_paths(dataset_path)
    dataset_path = '../../data/datasets'
    for ds in [trainer.train_dataset, trainer.val_dataset]:
        extension = ds.data_dir_path.split('/data/datasets')[1]
        ds.data_dir_path = os.path.abspath(f'{dataset_path}{extension}')
        print(f'updating path to:\n\t{ds.data_dir_path}')

    trainer.test_single_item(0, verbose=True)

    trainer.plot_loss_vec()

    trainer.compare_ber(tqdm=tqdm)


if __name__ == '__main__':
    test2_model_analyzer()