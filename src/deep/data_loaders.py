import concurrent
import json
import os
from abc import ABC
from concurrent.futures import ProcessPoolExecutor
from datetime import time, datetime
from glob import glob
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.deep.standalone_methods import GeneralMethods, DataType

x_file_name = 'data_x.npy'
y_file_name = 'data_y.npy'
conf_file_name = 'conf.json'


class OpticDataset(Dataset, ABC):
    def __init__(self, data_dir_path: str, data_indices: Union[list[int], range]) -> None:
        super().__init__()
        self.config = None
        self.mean = 0
        self.std = 1
        self.root_dir = ''

    def set_scale(self, mean, std):
        self.mean = mean
        self.std = std

    def set_root_dir(self, root_dir):
        self.root_dir = root_dir

    def __len__(self):
        return 0


def get_train_val_datasets(data_dir_path: str, dataset_type=OpticDataset, train_val_ratio=0.8, ds_limit: int = None):
    n_total = len(glob(f'{data_dir_path}/*{x_file_name}'))
    if ds_limit:
        n_total = min(n_total, ds_limit)
    divider_index = int(n_total*train_val_ratio)
    train_indices = range(0, divider_index)
    val_indices = range(divider_index, n_total)
    train_ds = dataset_type(data_dir_path, train_indices)
    val_ds = dataset_type(data_dir_path, val_indices)
    mean, std = GeneralMethods.calc_statistics_for_dataset(train_ds)
    train_ds.set_scale(mean, std)
    val_ds.set_scale(mean, std)
    return train_ds, val_ds


class DatasetNormal(OpticDataset):
    def __init__(self, data_dir_path: str, data_indices: Union[list[int], range] = None) -> None:
        super().__init__(data_dir_path, data_indices)
        self.data_dir_path = os.path.abspath(data_dir_path)
        self.data_indices = data_indices or range(len(glob(f'{self.data_dir_path}/*{x_file_name}')))
        self.mu = GeneralMethods.name_to_mu_val(self.data_dir_path)

        self.config = read_conf(self.data_dir_path)
        # self.n = len(glob(f'{self.data_dir_path}/*{x_file_name}'))
        self.n = len(self.data_indices)

    def __getitem__(self, index):
        # dataset returns 2x8192 vector
        # the dataloader returns 1x2x8192 where 1 is the batch size

        file_id = self.data_indices[index]

        x, y = read_xy(self.data_dir_path, file_id)

        x, y = GeneralMethods.normalize_xy(x, y, self.mean, self.std)

        x = complex_numpy_to_torch2(x)
        y = complex_numpy_to_torch2(y)

        # x, y = x.T, y.T

        return x, y

    def __len__(self) -> int:
        return self.n

    def get_numpy_xy(self, i):
        x, y = read_xy(self.data_dir_path, i)
        return x, y


class SeparatedRealImagDataset(DatasetNormal):
    def __init__(self, data_dir_path: str, data_indices: Union[list[int], range] = None, is_real=True) -> None:
        super(SeparatedRealImagDataset, self).__init__(data_dir_path, data_indices)
        self.is_real = is_real

    def set_is_real(self, is_real):
        self.is_real = is_real

    def __getitem__(self, index):
        # dataset returns 2x8192 vector
        # the dataloader returns 1x2x8192 where 1 is the batch size
        file_id = self.data_indices[index]

        x, y = read_xy(self.data_dir_path, file_id)

        x, y = GeneralMethods.normalize_xy(x, y, self.mu, self.std)

        if self.is_real:
            x = numpy_to_torch(x.real)
            y = numpy_to_torch(y.real)
        else:
            x = numpy_to_torch(x.imag)
            y = numpy_to_torch(y.imag)

        x, y = x.unsqueeze(0), y.unsqueeze(0)

        return x, y


class FilesReadWrite:

    @staticmethod
    def read_folder(dir: str, verbose: bool = False) -> (np.ndarray, np.ndarray, dict):
        # example: dir = f'data/10_samples_mu=0.001'

        conf_read = read_conf(dir)
        num_files = len(os.listdir(dir))
        N = int((num_files - 1)/2)
        assert os.path.basename(dir).startswith(f'{N}'), f'wrong number ({N}) of files in folder: {dir}'

        all_x, all_y = [], []
        for i in range(N):
            x_path = os.path.join(dir, f'{i}_{x_file_name}')
            y_path = os.path.join(dir, f'{i}_{y_file_name}')
            x = np.load(x_path)
            y = np.load(y_path)
            all_x.append(x)
            all_y.append(y)

        if verbose:
            print(f'loaded {len(all_x)} x files and {len(all_y)} y files.')

        return all_x, all_y, conf_read


def read_xy(dir: str, i: int):
    x = np.load(f'{dir}/{i}_{x_file_name}')
    y = np.load(f'{dir}/{i}_{y_file_name}')
    return x, y


def read_conf(dir, _conf_file_name=conf_file_name):
    conf_path = f'{dir}/{_conf_file_name}'
    with open(conf_path, 'r') as f:
        conf_read = json.load(f)
    return conf_read


def save_conf(path, conf):
    with open(path, 'w') as f:
        json.dump(conf, f, indent=4)


def save_xy(dir, x, y, i):
    x_path = f'{dir}/{i}_{x_file_name}'
    y_path = f'{dir}/{i}_{y_file_name}'
    with open(x_path, 'wb') as f:
        np.save(f, x)
    with open(y_path, 'wb') as f:
        np.save(f, y)


# def gen_data_old(data_len, num_symbols, mu_vec, cs, root_dir='data', tqdm=tqdm, logger_path=None):
#     vec_lens = num_symbols*cs.over_sampling + cs.N_rrc - 1
#     assert vec_lens == num_symbols*8*2, "the formula is not correct! check again"
#     pbar = tqdm(total=len(mu_vec)*data_len)
#     if logger_path:
#         os.makedirs(logger_path, exist_ok=True)
#         print(f'saving logs to {os.path.abspath(logger_path)}')
#         print(f'saving data to {os.path.abspath(root_dir)}')
#     file_path = f'{logger_path}/{get_ts_filename()}'
#     for mu_i, mu in enumerate(mu_vec):
#         dir = f'{root_dir}/{data_len}_samples_mu={mu:.3f}'
#         os.makedirs(dir, exist_ok=True)
#         cs.normalization_factor = mu
#         save_conf(f'{dir}/{conf_file_name}', cs.params_to_dict())
#         for i in range(data_len):
#             x, y = cs.gen_io_data()
#             save_xy(dir, x, y, i)
#             pbar.update()
#             if logger_path: log_status(file_path, mu, mu_i, len(mu_vec), i, data_len, pbar)


def gen_data2(data_len, num_symbols, mu_vec, cs, root_dir='data', tqdm=tqdm, logger_path=None, max_workers=1,
              type=DataType.spectrum):
    vec_lens = num_symbols*cs.over_sampling + cs.N_rrc - 1
    assert vec_lens == num_symbols*8*2, "the formula is not correct! check again"
    assert mu_vec[1] - mu_vec[0] > 0.0005, "mu_vec resolution is too low, folders will overlap"
    if logger_path:
        os.makedirs(logger_path, exist_ok=True)
        print(f'saving logs to {os.path.abspath(logger_path)}')
        print(f'saving data to {os.path.abspath(root_dir)}')
    file_path = f'{logger_path}/{get_ts_filename()}'

    print('setting up tasks...')
    # executor = ProcessPoolExecutor(max_workers=max_workers)
    # futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for mu_i, mu in enumerate(mu_vec):
            dir = f'{root_dir}/{data_len}_samples_mu={mu:.3f}'
            os.makedirs(dir, exist_ok=True)
            cs.normalization_factor = mu
            conf = cs.params_to_dict()
            conf['data_type'] = type
            save_conf(f'{dir}/{conf_file_name}', conf)
            for i in range(data_len):
                f_i = executor.submit(_gen_data_i, cs, dir, i, mu, type)
                futures[f_i] = (mu, i)

        print('initiating data generation...')
        pbar = tqdm(total=len(mu_vec)*data_len)
        for future in concurrent.futures.as_completed(futures):
            (mu, i) = futures[future]
            try:
                data = future.result()
            except Exception as exc:
                print(f'mu={mu}, i={i} generated an exception: {exc}')
            pbar.update()

    print('\nall done')


def _gen_data_i(cs, dir, i, mu, type=DataType.spectrum):
    # print(f'generating data {i}, mu {mu_i}...')
    try:
        x, y = cs.gen_io_data(type)
        save_xy(dir, x, y, i)
    except Exception as e:
        print(f'error at mu={mu}, i={i}: {e}')


def log_status(file_path, mu, mu_i, mu_N, sample_i, N_samples, pbar):
    timestamp = get_ts()
    elapsed = pbar.format_dict["elapsed"]
    rate = pbar.format_dict["rate"]
    remaining = (pbar.total - pbar.n)/rate if rate and pbar.total else 0  # Seconds*
    remains = pbar.format_interval(remaining)
    line = f'{timestamp} | saved mu={mu} ({mu_i}/{mu_N}), sample {sample_i}/{N_samples} | ' \
           f'remains: {remains}\n'

    with open(file_path, 'a') as f:
        f.write(line)


def get_ts():
    return datetime.now().strftime("%Y-%m-%d (%H:%M:%S)")


def get_ts_filename():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def complex_numpy_to_torch(np_vec: np.ndarray):
    x = np.array([np.real(np_vec), np.imag(np_vec)])
    return torch.from_numpy(x).float()


def complex_numpy_to_torch2(np_vec: np.ndarray) -> torch.Tensor:
    # complex [N] (a+bi) -> torch [2, N] (a, b)
    t = torch.Tensor([np_vec.real, np_vec.imag])
    # TODO: they say Tensor(list(numpy)) is slow and I should consider convert to numpy array first
    return t


def numpy_to_torch(np_vec: np.ndarray):
    return torch.from_numpy(np_vec).float()
