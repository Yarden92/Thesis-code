from copy import copy
from dataclasses import dataclass
import json
import os
from abc import ABC
from datetime import datetime
from glob import glob
from typing import Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from src.deep.standalone_methods import GeneralMethods
from src.general_methods.text_methods import FileNames
from src.optics.channel_simulation2 import ChannelSimulator2
from src.optics.config_manager import ChannelConfig, ConfigManager


class OpticDataset(Dataset, ABC):
    def __init__(self, data_dir_path: str, data_indices: Union[list[int], range]) -> None:
        super().__init__()
        self.config: ChannelConfig = None
        self.mean = 0
        self.std = 1
        self.cropped_mu = 0
        self.root_dir = ''
        self.data_dir_path = data_dir_path

    def set_scale(self, mean, std):
        self.mean = mean
        self.std = std

    def set_root_dir(self, root_dir):
        self.root_dir = root_dir

    def __len__(self):
        return 0


def get_datasets_set(data_dir_path: str, dataset_type=OpticDataset,
                     train_ds_ratio=0.5, val_ds_ratio=0.2, test_ds_ratio=0.3):
    assert train_ds_ratio + val_ds_ratio + test_ds_ratio <= 1, "train, val and test ratios must sum up to 1 or less"
    n_total = len(glob(f'{data_dir_path}/*{FileNames.Rx}'))
    # if ds_limit:
    #     n_total = min(n_total, ds_limit)
    stop_index_train = int(n_total*train_ds_ratio)
    stop_index_val = stop_index_train + int(n_total*val_ds_ratio)
    stop_index_test = stop_index_val + int(n_total*test_ds_ratio)

    train_indices = range(0, stop_index_train)
    val_indices = range(stop_index_train, stop_index_val)
    test_indices = range(stop_index_val, stop_index_test)

    train_ds = dataset_type(data_dir_path, train_indices) if train_ds_ratio > 0 else None
    val_ds = dataset_type(data_dir_path, val_indices) if val_ds_ratio > 0 else None
    test_ds = dataset_type(data_dir_path, test_indices) if test_ds_ratio > 0 else None

    # mean, std = GeneralMethods.calc_statistics_for_dataset(train_ds)

    # train_ds.set_scale(mean, std)  # TODO: consider if mean and std should be calculated for each dataset separately
    # val_ds.set_scale(mean, std)
    # test_ds.set_scale(mean, std)

    return train_ds, val_ds, test_ds


class DatasetNormal(OpticDataset):
    def __init__(self, data_dir_path: str, data_indices: Union[list[int], range] = None) -> None:
        super().__init__(data_dir_path, data_indices)
        self.data_dir_path = os.path.abspath(data_dir_path)
        self.data_indices = data_indices or range(len(glob(f'{self.data_dir_path}/*{FileNames.Rx}')))
        self.cropped_mu = GeneralMethods.name_to_mu_val(self.data_dir_path)

        # self.config = read_conf(self.data_dir_path)
        # self.config = FilesReadWrite.read_channel_conf2(self.data_dir_path)
        self.config = ConfigManager.read_config(self.data_dir_path)
        # self.n = len(glob(f'{self.data_dir_path}/*{x_file_name}'))
        self.n = len(self.data_indices)

    def __getitem__(self, index):
        # dataset returns 2x8192 vector
        # the dataloader returns 1x2x8192 where 1 is the batch size

        file_id = self.data_indices[index]

        Rx, Tx = read_Rx_Tx(self.data_dir_path, file_id)

        # Rx, Tx = GeneralMethods.normalize_xy(Rx, Tx, self.mean, self.std)

        Rx = complex_numpy_to_torch2(Rx)
        Tx = complex_numpy_to_torch2(Tx)



        return Rx, Tx

    def __len__(self) -> int:
        return self.n

    def get_numpy_xy(self, i):
        Rx, Tx = read_Rx_Tx(self.data_dir_path, i)
        return Rx, Tx

# TODO: move all standalone methods to a class


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

        Rx, Tx = read_Rx_Tx(self.data_dir_path, file_id)

        Rx, Tx = GeneralMethods.normalize_xy(Rx, Tx, self.mean, self.std)

        if self.is_real:
            Rx = numpy_to_torch(Rx.real)
            Tx = numpy_to_torch(Tx.real)
        else:
            Rx = numpy_to_torch(Rx.imag)
            Tx = numpy_to_torch(Tx.imag)

        Rx, Tx = Rx.unsqueeze(0), Tx.unsqueeze(0)

        return Rx, Tx


class FilesReadWrite:

    @staticmethod
    def read_folder(dir: str, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, ChannelConfig]:
        # conf_N is the number of files that supposed to be in the folder

        # conf_read = FilesReadWrite.read_channel_conf2(dir)
        conf_read = ConfigManager.read_config(dir)
        num_files = len(os.listdir(dir))
        N = int((num_files - 1)/2)

        all_x, all_y = [], []
        for i in range(N):
            x_path = os.path.join(dir, f'{i}_{FileNames.Rx}')
            y_path = os.path.join(dir, f'{i}_{FileNames.Tx}')
            x = np.load(x_path)
            y = np.load(y_path)
            all_x.append(x)
            all_y.append(y)

        if verbose:
            print(f'loaded {len(all_x)} x files and {len(all_y)} y files.')

        return all_x, all_y, conf_read

    @staticmethod
    def read_channel_conf2(dir: str, _conf_file_name=FileNames.conf_yml) -> ChannelConfig:
        raise Exception("use the function from config_manager")
        # conf_path = f'{dir}/{_conf_file_name}'
        # if is_this_a_notebook():
        #     conf = pyrallis.load(ChannelConfig, open(conf_path, "r"))
        # else:
        #     conf = pyrallis.parse(ChannelConfig, config_path=conf_path)
        # # conf = fill_config_from_loading(conf)
        # return conf


def read_Rx_Tx(dir: str, i: int):
    Rx = np.load(f'{dir}/{i}_{FileNames.Rx}')
    Tx = np.load(f'{dir}/{i}_{FileNames.Tx}')
    return Rx, Tx


def create_channels(template_conf: ChannelConfig) -> Tuple[ChannelSimulator2, ChannelSimulator2]:
    original_ssf = template_conf.with_ssf
    conf_for_input = copy(template_conf)
    conf_for_output = copy(template_conf)
    conf_for_input.with_ssf = False
    assert conf_for_output.with_ssf == original_ssf, "this code is not correct"

    cs_for_input = ChannelSimulator2(conf_for_input)
    cs_for_output = ChannelSimulator2(conf_for_output)

    return cs_for_input, cs_for_output


def save_conf(path, conf):
    with open(path, 'w') as f:
        json.dump(conf, f, indent=4)


def save_xy(dir, Rx, Tx, i):
    Rx_path = f'{dir}/{i}_{FileNames.Rx}'
    Tx_path = f'{dir}/{i}_{FileNames.Tx}'
    with open(Rx_path, 'wb') as f:
        np.save(f, Rx)
    with open(Tx_path, 'wb') as f:
        np.save(f, Tx)


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
