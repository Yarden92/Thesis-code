import os

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import wandb

from src.deep import data_loaders
from src.deep.data_loaders import FilesReadWrite, SingleMuDataSet
from src.deep.ml_ops import Trainer
from src.deep.models import SingleMuModel3Layers
from src.optics.split_step_fourier import SplitStepFourier

from src.deep.metrics import Metrics

from src.optics.channel_simulation import ChannelSimulator
from src.general_methods.visualizer import Visualizer

project_path = '..'


def test1():
    # test ber of folder
    print(os.getcwd())
    dir = f'../../apps/deep/data/3_samples_mu=0.9'
    all_x_read, all_y_read, conf_read = FilesReadWrite.read_folder(dir,
                                                                   False)  # TODO: this is old code, needs to be updated
    ber_vec, num_errors = Metrics.calc_ber_for_folder(all_x_read, all_y_read, conf_read)
    print(f'the avg ber is {np.mean(ber_vec)}')


def test2_data_generation():
    # config
    data_len = 3  # for each mu
    mu_len = 4
    num_symbols = 512
    dir = f'../data/datasets/qam1024_{data_len}x{mu_len}'
    # mu = 1e-3
    # mu_vec = [1e-3, 1e-2, 1e-1, 0.5, 0.9]
    mu_vec = np.linspace(start=0.0005, stop=0.07, num=mu_len)
    cs = ChannelSimulator(m_qam=1024,
                          num_symbols=num_symbols,
                          normalization_factor=0,  # will be overwritten during runtime
                          dt=1,
                          ssf=SplitStepFourier(
                              b2=-20e-27,
                              gamma=0.003,
                              t0=125e-12,
                              dt=1,
                              z_n=1000e3,
                              h=200
                          ),
                          verbose=False)
    # generate the date
    data_loaders.gen_data_old(data_len, num_symbols, mu_vec, cs, dir, tqdm=tqdm)

    #
    # # test generated data
    # # config
    # # config
    # data_len = 10  # for each mu
    # mu_len = 3
    # num_symbols = 512
    # dir = f'data/datasets/qam1024_{data_len}x{mu_len}'
    # # mu = 1e-3
    # # mu_vec = [1e-3, 1e-2, 1e-1, 0.5, 0.9]
    # mu_vec = np.linspace(start=0.0005, stop=0.07, num=mu_len)
    # cs = ChannelSimulator(m_qam=1024,
    #                       num_symbols=num_symbols,
    #                       normalization_factor=0,  # will be overwritten during runtime
    #                       dt=1,
    #                       ssf=SplitStepFourier(
    #                           b2=-20e-27,
    #                           gamma=0.003,
    #                           t0=125e-12,
    #                           dt=1,
    #                           z_n=1000e3,
    #                           h=200
    #                       ),
    #                       verbose=False)
    # src.deep.data_loaders.gen_data(data_len, num_symbols, mu_vec, cs, dir)


def test3_ber_vs_mu():
    # test ber vs mu from folders
    root_dir = f'{project_path}/data/datasets/qam1024_100x10'
    sub_name_filter = '*'
    ber_vec, mu_vec = Metrics.gen_ber_mu_from_folders(root_dir, sub_name_filter, 0, tqdm, 5)
    Visualizer.plot_bers(mu_vec, [ber_vec])

    # sub_name = '3_samples_mu='
    # ber_vec, mu_vec = Metrics.gen_ber_mu_from_folders('../../apps/deep/data/qam1024', sub_name)
    # indices = np.argsort(mu_vec)
    # Visualizer.plot_bers(np.array(mu_vec)[indices], [np.array(ber_vec)[indices]], [sub_name])


def test4_read_subfolder():
    dir = f'{project_path}/data/datasets/qam1024_10x3/10_samples_mu=0.035'
    data_id = 0
    zm = range(1700, 2300)

    dataloader = SingleMuDataSet(dir)

    print(f'the folder {dir} contains {len(dataloader)} samples')

    x, y = dataloader.get_numpy_xy(data_id)

    Visualizer.twin_zoom_plot('x', np.real(x), zm)
    Visualizer.twin_zoom_plot('y', np.real(y), zm)

    Metrics.calc_ber_from_dataset(dataloader, True)


def test5_data_analyzer():
    dir = f'{project_path}/data/datasets/qam1024_100x10/100_samples_mu=0.001'
    data_id = 0
    zm = range(1700, 2300)

    dataloader = SingleMuDataSet(dir)

    print(f'the folder {dir} contains {len(dataloader)} samples')

    x, y = dataloader.get_numpy_xy(data_id)

    Visualizer.twin_zoom_plot_vec('vec[0]', np.array([np.real(x), np.real(y)]).T, ['x', 'y'], zm)


def test6_wandb():
    wandb.init(project="Thesis", entity="yarden92")
    epochs = 100
    lr = 1e-3
    batch_size = 128
    wandb.config = {
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size
    }

    l_metric = nn.MSELoss()  # or L1Loss
    model = SingleMuModel3Layers()
    dir = '../data/datasets/qam1024_100x10/100_samples_mu=0.008'
    train_dataset, val_dataset = data_loaders.get_train_val_datasets(dir, SingleMuDataSet, train_val_ratio=0.8)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = Trainer(train_dataset=train_dataset, val_dataset=val_dataset, model=model, l_metric=l_metric, optim=optim)

    trainer.test_single_item(0, verbose=True)

    trainer.train(num_epochs=epochs, mini_batch_size=batch_size)

    trainer.test_single_item(0, f'after {trainer.train_state_vec.num_epochs} epochs')

    trainer.compare_ber(tqdm=tqdm)

    trainer.plot_loss_vec()


if __name__ == '__main__':
    test6_wandb()
