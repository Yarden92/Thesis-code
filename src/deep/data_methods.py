import os
import numpy as np
from src.deep.data_loaders import DatasetNormal
from src.deep.standalone_methods import GeneralMethods


class FolderTypes:
    Data = 'data'
    Analysis = 'analysis'
    LinuxShit = 'linux_sh'
    Unknown = 'unknown'

class DataMethods:
    FolderTypes = FolderTypes
    
    @staticmethod
    def check_folder_type(sub_name):
        if sub_name.startswith('_'): return FolderTypes.Analysis
        if sub_name.startswith('.'): return FolderTypes.LinuxShit
        if 'x' in sub_name: return FolderTypes.Data
        if '=' not in sub_name: return FolderTypes.Unknown
        return FolderTypes.Data
    
    @staticmethod
    def calc_ds_power(dir_path, max_items=None, _tqdm=None):
        ds = DatasetNormal(dir_path)

        N = len(ds)
        if max_items is not None:
            N = min(N, max_items)

        indices = range(N)
        if _tqdm is not None:
            indices = _tqdm(indices)

        powers = [(np.linalg.norm(ds.get_numpy_xy(i), axis=1) ** 2) / N for i in indices]
        x_power, y_power = np.sum(powers, axis=0)

        return x_power, y_power
        

        
        
def test1_get_folder_powers():
    path = 'data/datasets/qam16_150x20/150_samples_mu=0.125'
    xp,yp = DataMethods.calc_ds_power(path, max_items=10)
    print(f'x_power={xp}, y_power={yp}')
    

def test2_get_all_folder_powers():
    path = 'data/datasets/qam16_150x20'
    for sub_name in os.listdir(path):
        sub_path = os.path.join(path, sub_name)
        folder_type = DataMethods.check_folder_type(os.path.basename(sub_path))
        if folder_type != FolderTypes.Data: continue
        
        mu = GeneralMethods.name_to_mu_val(sub_name)

        print(f'folder: {sub_name}')
        xp,yp = DataMethods.calc_ds_power(sub_path, max_items=10)
        print(f'x_power={xp}, y_power={yp}')
        print()    

if __name__ == '__main__':
    test1_get_folder_powers()