import os
import platform
from enum import IntEnum

import numpy as np
from torch import Tensor


class GeneralMethods:
    @staticmethod
    def torch_to_complex_numpy(tensor: Tensor):
        np_vec = tensor.cpu().detach().numpy()
        assert np_vec.shape[0] == 2, f"input should have been 2x8192 but got {np_vec.shape} instead"
        # return np_vec[:, 0] + 1j*np_vec[:, 1]
        return np_vec[0] + 1j*np_vec[1]

    @staticmethod
    def normalize_xy(Rx, Tx, mean, std):
        return Rx, Tx  # TODO: somethings wrong here
        return (Rx - mean)/std, (Tx - mean)/std

    @staticmethod
    def calc_statistics_for_dataset(dataset):
        mean = np.mean([x.mean() for x, _ in dataset])
        std = np.mean([x.std() for x, _ in dataset])
        return mean, std

    @staticmethod
    def name_to_mu_val(dirpath: str) -> float:
        folder_name = os.path.basename(dirpath)
        mu = float(folder_name.split('=')[-1])
        return mu

    @staticmethod
    def power_ratio(Rx, Tx):
        Rx_power = np.mean(np.abs(Rx)**2)
        Tx_power = np.mean(np.abs(Tx)**2)
        if Rx_power > Tx_power:
            return Rx_power/Tx_power
        else:
            return Tx_power/Rx_power

    @staticmethod
    def sort_dirs_by_mu(dirs: list) -> list:
        dirs = [d for d in dirs if d[0] not in ['.', '_']]  # filter private dirs
        return sorted(dirs, key=lambda x: GeneralMethods.name_to_mu_val(x))


class DataType(IntEnum):
    spectrum = 0
    iq_samples = 1

    @classmethod
    def _missing_(cls, value):
        return cls.spectrum


def get_platform():
    try:
        os = platform.system()
        if 'darwin' in os.lower():
            os = 'Mac'
    except:
        os = 'unknown'

    return os
