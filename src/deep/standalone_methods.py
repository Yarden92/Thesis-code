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
    def normalize_xy(x, y, mean, std):
        return x, y # TODO: somethings wrong here
        return (x - mean)/std, (y - mean)/std

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
    def power_ratio(x, y):
        x_power = np.mean(np.abs(x)**2)
        y_power = np.mean(np.abs(y)**2)
        if x_power > y_power:
            return x_power/y_power
        else:
            return y_power/x_power



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
