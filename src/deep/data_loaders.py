import json
import os
from abc import ABC
from datetime import time, datetime
from glob import glob
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.deep.standalone_methods import GeneralMethods


class OpticDataset(Dataset, ABC):
    def __init__(self, data_dir_path: str, data_indices: Union[list[int], range]) -> None:
        super().__init__()
        self.config = None
        self.mu = 0
        self.std = 1
        self.root_dir = ''

    def set_scale(self, mu, std):
        self.mu = mu
        self.std = std

    def set_root_dir(self, root_dir):
        self.root_dir = root_dir

    def __len__(self):
        return 0


def get_train_val_datasets(data_dir_path: str, dataset_type=OpticDataset, train_val_ratio=0.8):
    n_total = len(glob(f'{data_dir_path}/*{x_file_name}'))
    divider_index = int(n_total * train_val_ratio)
    train_indices = range(0, divider_index)
    val_indices = range(divider_index, n_total)
    train_ds = dataset_type(data_dir_path, train_indices)
    val_ds = dataset_type(data_dir_path, val_indices)
    mu, std = GeneralMethods.calc_statistics_for_dataset(train_ds)
    train_ds.set_scale(mu, std)
    val_ds.set_scale(mu, std)
    return train_ds, val_ds


class SingleMuDataSet(OpticDataset):
    def __init__(self, data_dir_path: str, data_indices: Union[list[int], range] = None) -> None:
        super().__init__(data_dir_path, data_indices)
        self.data_dir_path = os.path.abspath(data_dir_path)
        self.data_indices = data_indices or range(len(glob(f'{self.data_dir_path}/*{x_file_name}')))

        self.config = read_conf(self.data_dir_path)
        # self.n = len(glob(f'{self.data_dir_path}/*{x_file_name}'))
        self.n = len(self.data_indices)

    def __getitem__(self, index):
        # x = np.load(f'{self.data_dir_path}/{index}_{file_methods.x_file_name}')
        # y = np.load(f'{self.data_dir_path}/{index}_{file_methods.y_file_name}')

        file_id = self.data_indices[index]

        x, y = read_xy(self.data_dir_path, file_id)

        x, y = GeneralMethods.normalize_xy(x, y, self.mu, self.std)

        x = complex_numpy_to_torch(x)
        y = complex_numpy_to_torch(y)

        x, y = x.T, y.T

        return x, y

    def __len__(self) -> int:
        return self.n

    def get_numpy_xy(self, i):
        x, y = read_xy(self.data_dir_path, i)
        return x, y


class FilesReadWrite:

    @staticmethod
    def read_folder(dir: str, verbose: bool = False) -> (np.ndarray, np.ndarray, dict):
        # example: dir = f'data/10_samples_mu=0.001'

        conf_read = read_conf(dir)

        all_x = []
        for x_path in glob(f'{dir}/*_{x_file_name}'):
            x = np.load(x_path)
            all_x.append(x)

        all_y = []
        for y_path in glob(f'{dir}/*_{y_file_name}'):
            y = np.load(y_path)
            all_y.append(y)

        if verbose:
            print(f'loaded {len(all_x)} x files and {len(all_y)} y files.')

        return all_x, all_y, conf_read


def read_xy(dir: str, i: int):
    x = np.load(f'{dir}/{i}_{x_file_name}')
    y = np.load(f'{dir}/{i}_{y_file_name}')
    return x, y


def read_conf(dir):
    conf_path = f'{dir}/{conf_file_name}'
    with open(conf_path, 'r') as f:
        conf_read = json.load(f)
    return conf_read


def save_conf(path, conf):
    # os.path.dirname(path).mkdir(exist_ok=True)
    with open(path, 'w') as f:
        json.dump(conf, f, indent=4)


def save_xy(dir, x, y, i):
    x_path = f'{dir}/{i}_{x_file_name}'
    y_path = f'{dir}/{i}_{y_file_name}'
    with open(x_path, 'wb') as f:
        np.save(f, x)
    with open(y_path, 'wb') as f:
        np.save(f, y)


def gen_data(data_len, num_symbols, mu_vec, cs, root_dir='data', tqdm=tqdm, logger_path=None):
    vec_lens = num_symbols * cs.over_sampling + cs.N_rrc - 1
    assert vec_lens == num_symbols * 8 * 2, "the formula is not correct! check again"
    pbar = tqdm(total=len(mu_vec) * data_len)
    if logger_path:
        os.makedirs(logger_path, exist_ok=True)
    file_path = f'{logger_path}/{get_ts_filename()}'
    for mu_i, mu in enumerate(mu_vec):
        dir = f'{root_dir}/{data_len}_samples_mu={mu:.3f}'
        os.makedirs(dir, exist_ok=True)
        cs.normalization_factor = mu
        save_conf(f'{dir}/{conf_file_name}', cs.params_to_dict())
        for i in range(data_len):
            x, y = cs.gen_io_data()
            save_xy(dir, x, y, i)
            pbar.update()
            if logger_path: log_status(file_path,mu,mu_i,len(mu_vec),i,data_len,pbar)


def log_status(file_path, mu, mu_i, mu_N, sample_i, N_samples, pbar):
    timestamp = get_ts()
    elapsed = pbar.format_dict["elapsed"]
    rate = pbar.format_dict["rate"]
    remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0  # Seconds*
    remains = pbar.format_interval(remaining)
    line = f'{timestamp} | saved mu={mu} ({mu_i}/{mu_N}), sample {sample_i}/{N_samples} | ' \
           f'remains: {remains}\n'

    with open(file_path, 'a') as f:
        f.write(line)


def get_ts():
    return datetime.now().strftime("%Y-%m-%d (%H:%M:%S)")


def get_ts_filename():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


x_file_name = 'data_x.npy'
y_file_name = 'data_y.npy'
conf_file_name = 'conf.json'


def complex_numpy_to_torch(np_vec: np.ndarray):
    x = np.array([np.real(np_vec), np.imag(np_vec)])
    return torch.from_numpy(x).float()
