import os
from glob import glob
import numpy as np
import torch
from tqdm import tqdm

from src.deep.data_methods import DataMethods, FolderTypes
from src.deep.standalone_methods import GeneralMethods, DataType
from src.deep.data_loaders import OpticDataset, FilesReadWrite, create_channels
# from src.optics.channel_simulation import ChannelSimulator
from src.optics.channel_simulation2 import ChannelSimulator2
from src.optics.config_manager import ChannelConfig


class Metrics:
    @staticmethod
    def calc_ber_for_single_vec(Rx, Tx, 
                                in_cs: ChannelSimulator2=None, out_cs: ChannelSimulator2=None,
                                conf: ChannelConfig=None):
        # Rx, Tx can be either complex numpy or 2D torch
        assert (in_cs is not None and out_cs is not None) or conf is not None, "either cs or conf should be given"
        # check if x is a torch
        if isinstance(Rx, torch.Tensor):
            Rx, Tx = GeneralMethods.torch_to_complex_numpy(Rx), GeneralMethods.torch_to_complex_numpy(Tx)
        
        if conf is not None:
            in_cs, out_cs = create_channels(conf)
        
        msg_in = in_cs.io_to_msg(Tx)
        msg_out = out_cs.io_to_msg(Rx)
        
        num_errors_i, ber_i = in_cs.block11.calc_ber(msg_in, msg_out)
        return ber_i, num_errors_i

    @staticmethod
    def calc_ber_for_folder(all_Rx_read, all_Tx_read, 
                            in_cs: ChannelSimulator2=None, out_cs: ChannelSimulator2=None,
                            verbose=True, _tqdm=None):
        num_errors = 0
        ber_vec = []

        rng = zip(all_Rx_read, all_Tx_read)
        if _tqdm is not None:
            rng = _tqdm(rng, total=len(all_Rx_read), leave=False)

        for Rx, Tx in rng:
            ber_i, num_errors_i = Metrics.calc_ber_for_single_vec(Rx, Tx, in_cs, out_cs)
            ber_vec.append(ber_i)
            num_errors += num_errors_i

        return ber_vec, num_errors

    @staticmethod
    def calc_ber_from_dataset(dataset: OpticDataset, verbose=True, _tqdm=None, num_x_per_folder=None):
        num_errors = 0
        ber_vec = []
        cs_in, cs_out = create_channels(dataset.config)
        n = min(num_x_per_folder or len(dataset), len(dataset))
        rng = _tqdm(range(n)) if _tqdm else range(n)
        for i in rng:
            Rx, Tx = dataset[i]

            ber_i, num_errors_i = Metrics.calc_ber_for_single_vec(Rx, Tx, cs_in, cs_out)
            ber_vec.append(ber_i)
            num_errors += num_errors_i

        return ber_vec, num_errors

    @staticmethod
    def calc_ber_from_model(dataset: OpticDataset, model, verbose=True, _tqdm=None, max_x=None, device='auto'):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        num_errors = 0
        ber_vec = []
        cs_in, cs_out = create_channels(dataset.config)
        n = min(max_x or len(dataset), len(dataset))
        # TODO: can we do it all in one batch? at least the model(x) part.
        rng = _tqdm(range(n)) if _tqdm else range(n)
        for i in rng:
            (Rx, Tx) = dataset[i]
            Rx = Rx.to(device)
            Rx = Rx.unsqueeze(0)  # when fetching directly from dataset we need to wrap like dataloader does as 1x1xN
            pred_Tx = model(Rx)
            pred_Tx = pred_Tx.squeeze(0)
            ber_i, num_errors_i = Metrics.calc_ber_for_single_vec(pred_Tx, Tx, cs_in, cs_out)
            ber_vec.append(ber_i)
            num_errors += num_errors_i
        return ber_vec, num_errors

    @staticmethod
    def gen_ber_mu_from_folders(root_dir, sub_name_filter, verbose_level=0, _tqdm=tqdm, num_x_per_folder=None,
                                is_matrix_ber=False,
                                in_cs: ChannelSimulator2=None, out_cs: ChannelSimulator2=None,):
        # num_x_per_folder - number of samples to use from each folder to determine ber (averaging across), None=all

        # walk through folder [data] and search for folders that named as [10_samples_****]
        mu_vec, ber_vec = [], []
        for dirpath in _tqdm(glob(f'{root_dir}/{sub_name_filter}')):
            folder_type = DataMethods.check_folder_type(os.path.basename(dirpath))
            if folder_type != FolderTypes.Data:
                if folder_type == FolderTypes.Unknown:
                    print(f'warning: unknown folder "{dirpath}", skipping it...')
                continue
            mu = GeneralMethods.name_to_mu_val(dirpath)
            all_x_read, all_y_read, conf = FilesReadWrite.read_folder(dirpath, verbose_level >= 1)
            all_x_read, all_y_read = trim_data(all_x_read, all_y_read, num_x_per_folder)
            in_cs.update_mu(conf.mu)
            out_cs.update_mu(conf.mu)
            sub_ber_vec, num_errors = Metrics.calc_ber_for_folder(all_x_read, all_y_read, in_cs, out_cs, 
                                                                  verbose_level >= 2)

            if verbose_level >= 1:
                print(f'folder {dirpath} has {len(sub_ber_vec)} signals with total {num_errors} errors -> ber = {np.mean(sub_ber_vec)}')

            ber = sub_ber_vec if is_matrix_ber else np.mean(sub_ber_vec)

            mu_vec.append(mu)
            ber_vec.append(ber)

        indices = np.argsort(mu_vec)
        mu_vec = np.array(mu_vec)[indices]
        ber_vec = np.array(ber_vec)[indices]

        return ber_vec, mu_vec


def trim_data(Rx, Tx, n=None):
    if n is None:
        return Rx, Tx
    if n > len(Rx):
        print(f'warning: requested {n} samples for BER but only {len(Rx)} were found')
        return Rx, Tx
    return Rx[:n], Tx[:n]
