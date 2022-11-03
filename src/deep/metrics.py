import os
from glob import glob
import numpy as np
import torch
from tqdm import tqdm

from src.deep.data_methods import DataMethods
from src.deep.standalone_methods import GeneralMethods, DataType
from src.deep.data_loaders import OpticDataset, FilesReadWrite
from src.optics.channel_simulation import ChannelSimulator


class Metrics:
    @staticmethod
    def calc_ber_for_single_vec(x, y, cs=None, conf=None, datatype=DataType.spectrum):
        # x,y can be either complex numpy or 2D torch
        assert cs is not None or conf is not None, "either cs or conf should be given"
        # check if x is a torch
        if isinstance(x, torch.Tensor):
            x, y = GeneralMethods.torch_to_complex_numpy(x), GeneralMethods.torch_to_complex_numpy(y)
        cs = cs or ChannelSimulator.from_dict(conf)
        if datatype == DataType.spectrum:
            msg_in = cs.steps8_to_10(y)
            msg_out = cs.steps8_to_10(x)
        elif datatype == DataType.iq_samples:
            msg_in = cs.steps10(y)
            msg_out = cs.steps10(x)
        else:
            raise ValueError(f'unknown type: {type}, choose one of DataType\'s values')
        ber_i, num_errors_i = cs.cb.calc_ber(msg_in, msg_out, cs.length_of_msg, cs.sps)
        return ber_i, num_errors_i

    @staticmethod
    def calc_ber_for_folder(all_x_read, all_y_read, conf_read, verbose=True, _tqdm=None):
        num_errors = 0
        ber_vec = []
        cs = ChannelSimulator.from_dict(conf_read)
        datatype = conf_read['data_type'] if 'data_type' in conf_read else DataType.spectrum

        rng = enumerate(zip(all_x_read, all_y_read))
        if _tqdm is not None:
            rng = _tqdm(rng, total=len(all_x_read), leave=False)

        for i, (x, y) in rng:
            ber_i, num_errors_i = Metrics.calc_ber_for_single_vec(x, y, cs, datatype=datatype)
            ber_vec.append(ber_i)
            num_errors += num_errors_i
            if verbose:
                print(f'data {i} has ber of: {ber_i} with {num_errors_i}/{cs.length_of_msg} bit errors')
        return ber_vec, num_errors

    @staticmethod
    def calc_ber_from_dataset(dataset: OpticDataset, verbose=True, _tqdm=None, num_x_per_folder=None):
        num_errors = 0
        ber_vec = []
        cs = ChannelSimulator.from_dict(dataset.config)
        n = min(num_x_per_folder or len(dataset), len(dataset))
        rng = _tqdm(range(n)) if _tqdm else range(n)
        datatype = dataset.config['data_type'] if 'data_type' in dataset.config else DataType.spectrum
        for i in rng:
            x, y = dataset[i]
            # x, y = ml_ops.torch_to_complex_numpy(x), ml_ops.torch_to_complex_numpy(y)
            ber_i, num_errors_i = Metrics.calc_ber_for_single_vec(x, y, cs, datatype=datatype)
            ber_vec.append(ber_i)
            num_errors += num_errors_i
            if verbose:
                print(f'data {i} has ber of: {ber_i} with {num_errors_i}/{cs.length_of_msg} bit errors')
        return ber_vec, num_errors

    @staticmethod
    def calc_ber_from_model(dataset: OpticDataset, model, verbose=True, _tqdm=None, max_x=None, device='auto'):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        num_errors = 0
        ber_vec = []
        cs = ChannelSimulator.from_dict(dataset.config)
        n = max_x or len(dataset)
        rng = _tqdm(range(n)) if _tqdm else range(n)
        for i in rng:
            (x, y) = dataset[i]
            x = x.to(device)
            pred = model(x)
            ber_i, num_errors_i = Metrics.calc_ber_for_single_vec(pred, y, cs)
            ber_vec.append(ber_i)
            num_errors += num_errors_i
            if verbose:
                print(f'data {i} has ber of: {ber_i} with {num_errors_i}/{cs.length_of_msg} bit errors')
        return ber_vec, num_errors

    @staticmethod
    def gen_ber_mu_from_folders(root_dir, sub_name_filter, verbose_level=0, _tqdm=tqdm, num_x_per_folder=None,
                                is_matrix_ber=False):
        # num_x_per_folder - number of samples to use from each folder to determine ber (averaging across), None=all

        # walk through folder [data] and search for folders that named as [10_samples_****]
        mu_vec, ber_vec = [], []
        for dirpath in _tqdm(glob(f'{root_dir}/{sub_name_filter}')):
            if not DataMethods.is_valid_subfolder(os.path.basename(dirpath)):
                continue
            mu = GeneralMethods.name_to_mu_val(dirpath)
            all_x_read, all_y_read, conf_read = FilesReadWrite.read_folder(dirpath, verbose_level >= 1)
            all_x_read, all_y_read = trim_data(all_x_read, all_y_read, num_x_per_folder)
            sub_ber_vec, num_errors = Metrics.calc_ber_for_folder(all_x_read, all_y_read, conf_read, verbose_level >= 2)

            ber = sub_ber_vec if is_matrix_ber else np.mean(sub_ber_vec)

            mu_vec.append(mu)
            ber_vec.append(ber)

        indices = np.argsort(mu_vec)
        mu_vec = np.array(mu_vec)[indices]
        ber_vec = np.array(ber_vec)[indices]

        return ber_vec, mu_vec


def trim_data(x, y, n=None):
    if n is None:
        return x, y
    if n > len(x):
        print(f'requested {n} samples for BER but only {len(x)} were found')
        return x, y
    return x[:n], y[:n]
