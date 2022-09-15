import os

import numpy as np
from torch import Tensor

class GeneralMethods:
    @staticmethod
    def torch_to_complex_numpy(tensor: Tensor):
        try:
            tensor = tensor.cpu()
        except:
            print('tensor is already on cpu')
        np_vec = tensor.detach().numpy()
        return np_vec[:, 0] + 1j*np_vec[:, 1]

    @staticmethod
    def normalize_xy(x, y, mu, std):
        return (x - mu) / std, (y - mu) / std

    @staticmethod
    def calc_statistics_for_dataset(dataset):
        mean = np.mean([x.mean() for x, _ in dataset])
        std = np.mean([x.std() for x, _ in dataset])
        return mean, std


    @staticmethod
    def name_to_mu_val(dirpath: str) -> float:
        folder_name = os.path.basename(dirpath)
        return float(folder_name.split('=')[-1])
