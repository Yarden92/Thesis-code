from copy import copy
import os
import re

import numpy as np
import wandb
from tqdm.auto import tqdm
from typing import Tuple
from src.deep import data_loaders
from src.deep import standalone_methods
from src.deep.data_loaders import DatasetNormal, FilesReadWrite, create_channels
from src.deep.data_methods import DataMethods, FolderTypes
from src.deep.metrics import Metrics
from src.deep.standalone_methods import DataType
from src.general_methods.signal_processing import SP
from src.general_methods.visualizer import Visualizer
from src.optics.channel_simulation2 import ChannelSimulator2
from src.optics.config_manager import ChannelConfig

class PATHS:
    analysis_dir = '_analysis'
    ber = 'ber.npy'
    mu = 'mu.npy'
    num_permuts = 'num_permutations.npy'

class DataAnalyzer():
    def __init__(self, data_folder_path: str, _tqdm=tqdm, is_box_plot=False, verbose_level=0,
                 load_ber_path=None):
        self.path = data_folder_path.rstrip('/')
        self.base_name = os.path.basename(self.path)
        self.params, conf = self.fetch_params()
        self.cs_for_input, self.cs_for_output = create_channels(conf)
        self.is_wandb_init = False
        self.is_box_plot = is_box_plot

        self._tqdm = _tqdm
        self.num_digits = None
        self.verbose_level = verbose_level
        self.ber_vec = None
        self.mu_vec = None
        self.num_permutations = 0 
        if load_ber_path is not None:
            self.try_to_load_ber(load_ber_path)

    def calc_and_save_ber(self, n: int = None) -> None:
        n = n or self.params['num_samples']  # if n is None, take all permutations
        self._calc_full_ber(n)
        self.save_ber(self.path, self.ber_vec, self.mu_vec, n)

    @staticmethod
    def save_ber(root_dir, ber_vec, mu_vec, num_permutations):
        #setup paths
        ber_path = os.path.join(root_dir, PATHS.analysis_dir, PATHS.ber)
        mu_path = os.path.join(root_dir, PATHS.analysis_dir, PATHS.mu)
        num_permuts_path = os.path.join(root_dir, PATHS.analysis_dir, PATHS.num_permuts)

        # create the dir
        os.makedirs(os.path.join(root_dir, PATHS.analysis_dir), exist_ok=True)

        #save the data
        np.save(ber_path, ber_vec)
        np.save(mu_path, mu_vec)
        np.save(num_permuts_path, num_permutations)

        print(f'ber saved to {ber_path}')


    def try_to_load_ber(self) -> bool:
        # will load ber if exists, otherwise will do nothing
        ber_path = os.path.join(self.path, PATHS.analysis_dir, PATHS.ber)
        mu_path = os.path.join(self.path, PATHS.analysis_dir, PATHS.mu)
        ber_res_path = os.path.join(self.path, PATHS.analysis_dir, PATHS.num_permuts)
        if os.path.exists(ber_path):
            self.ber_vec = np.load(ber_path)
            self.mu_vec = np.load(mu_path)
            self.num_permutations = np.load(ber_res_path)
            return True
        return False

    def fetch_params(self) -> Tuple[dict, ChannelConfig]:
        # num_samples, num_mus = [int(x) for x in self.base_name.split('_')[-1].split('x')]
        num_samples, num_mus = self._extract_values_from_folder_name()
        mu_vec = []
        conf_list = []
        self.num_digits = self._count_digits(os.listdir(self.path)[0])
        for sub_name in os.listdir(self.path):
            folder_type = DataMethods.check_folder_type(os.path.basename(sub_name))
            if folder_type != FolderTypes.Data:
                if folder_type == FolderTypes.Unknown:
                    print(f'warning: unknown folder "{sub_name}", skipping it...')
                continue
            sub_path = f"{self.path}/{sub_name}"
            # conf = data_loaders.read_conf(sub_path)
            conf = FilesReadWrite.read_channel_conf2(sub_path)
            mu_vec.append(conf.mu)
            conf_list.append(conf)
            num_files = len(os.listdir(sub_path))
            assert num_files == 1 + 2 * \
                num_samples, f'ERROR at {sub_name}: num_files={num_files} != 1+2*num_samples={num_samples}'

        assert len(conf_list) == num_mus, 'num of sub directories dont match num of mus from headline'
        assert len(mu_vec) == num_mus, 'num of sub directories dont match num of mus from headline'
        conf = conf_list[0]  # arbitrarily chosen
        params = {
            'num_samples': num_samples,
            'num_mus': num_mus,
            'mu_start': min(mu_vec),
            'mu_end': max(mu_vec),
            'num_digits': self.num_digits
        }
        return params, conf

    # -------------- plots --------------

    def plot_single_sample_b(self, mu: float = None, data_id: int = 0, is_save=True):
        # read folder and plot one of the data samples
        mu_cropped = mu or self.params['mu_start']  # default to first mu
        sub_name = self._get_sub_folder_name(mu_cropped)
        mu_actual = self._get_full_mu(sub_name)
        x, y = self._get_xy(data_id, sub_name)


        xi = self.cs_for_input.channel_config.xi
        y_dirty = x
        y_ref = y
        xlabel = r'$\xi$'
        y_name = r'b(\xi)'
        title = fr'signal compared for $\mu$={mu_actual:.2e}, i={data_id}'
        lgnd = ['x - dirty (after channel)', 'y - clean (before channel)']

        Visualizer.compare_amp_and_phase(xi, y_dirty, y_ref, xlabel, y_name, title, square=False, lgnd=lgnd)

        x_power = SP.signal_power(x)
        print(f'x_power={x_power}')

        y_power = SP.signal_power(y)
        print(f'y_power={y_power}')

        ber, num_errors = Metrics.calc_ber_for_single_vec(x, y, self.cs_for_input, self.cs_for_output)
        print(f'ber={ber}')

        if is_save:
            out_path = f'{self.path}/{PATHS.analysis_dir}/{sub_name}'
            os.makedirs(out_path, exist_ok=True)
            print(f'saving image is currently not supported')

    def plot_constelation(self,  mu: float,  data_indices: list):
        m_qam = self.cs_for_input.channel_config.M_QAM
        mu_cropped = mu or self.params['mu_start']  # default to first mu
        sub_name = self._get_sub_folder_name(mu_cropped)
        mu_actual = self._get_full_mu(sub_name)
        # update mu:
        self.cs_for_input.update_mu(mu_actual)
        self.cs_for_output.update_mu(mu_actual)

        long_x, long_y = np.array([]), np.array([])
        for i in tqdm(data_indices):
            x_i, y_i = self._get_xy(i, sub_name)
            c_out_y = self.cs_for_input.io_to_c_constellation(y_i)
            c_out_x = self.cs_for_output.io_to_c_constellation(x_i)

            long_y = np.concatenate((long_y, c_out_y))  # clean
            long_x = np.concatenate((long_x, c_out_x))  # dirty

        Visualizer.plot_constellation_map_with_k_data_vecs([long_x, long_y], m_qam,
                                                           '',
                                                           ['dirty', 'clean'])
        # Visualizer.plot_constellation_map_with_points(x9, m_qam, 'dirty signal')
        # Visualizer.plot_constellation_map_with_points(y9, m_qam, 'clean signal')
        # Visualizer.plot_constellation_map_with_points(pred9, m_qam, 'preds signal')


    def calc_ber_for_sub_folder(self, mu, n=5, _tqdm=None):
        # n is the number of x's permutations to take from each folder
        sub_name = self._get_sub_folder_name(mu)
        dir = self.path + '/' + sub_name
        dataset = DatasetNormal(dir)

        ber_vec, num_errors = Metrics.calc_ber_from_dataset(dataset, False, _tqdm, n)
        print(f'the avg ber of mu={mu} (with {n} permutations) is {np.mean(ber_vec)}')

        return ber_vec

    def plot_full_ber_graph(self, permute_limit=None, is_save_fig=True, log_mu=False):
        # n is the number of x's permutations to take from each folder
        self._calc_full_ber(permute_limit)
        out_dir = f'{self.path}/{PATHS.analysis_dir}'
        os.makedirs(out_dir, exist_ok=True)
        out_path = f'{out_dir}/ber_vs_mu.png'

        if self.is_box_plot:
            Visualizer.plot_bers_boxplot(
                us=self.mu_vec,
                bers_vecs=self.ber_vec.T,
                output_path=out_path if is_save_fig else None,
            )
        else:
            Visualizer.plot_bers(
                us=self.mu_vec,
                bers_vecs=[self.ber_vec],
                output_path=out_path if is_save_fig else None,
                log_mu=log_mu,
            )

        print(f'ber vs mu saved to {out_path}')

    # ------------ wandb ------------

    def _init_wandb(self):
        if self.is_wandb_init:
            return
        data_type = self.params['conf']['data_type'] if 'data_type' in self.params['conf'] else 0
        data_type = DataType(data_type).name
        mu_range = f'{self.params["mu_start"]}-{self.params["mu_end"]}'
        qam = self.params['conf']['m_qam']
        run_name = f'{self.base_name}__{data_type}'
        wandb.init(project="data_analyze", entity="yarden92", name=run_name,
                   tags=[
                       f'{standalone_methods.get_platform()}',
                       f'ds={self.params["num_samples"]}',
                       f'mu_range={mu_range}',
                       f'qam={qam}',
                   ])
        wandb.config = self.params
        self.is_wandb_init = True

    def wandb_log_ber_vs_mu(self, n=None):
        # n is the number of x's permutations to take from each folder
        self._calc_full_ber(n)
        self._init_wandb()
        n = n or self.params['num_samples']
        wandb.log({
            'BER vs mu': wandb.plot.line_series(
                xs=self.mu_vec,
                ys=[self.ber_vec],
                keys=['ber'],
                title=f'BER vs mu (with {n} permutations)',
                xname="mu")
        })

    def wandb_log_single_sample(self, mu: float, data_id: int):
        self._init_wandb()
        sub_name = self._get_sub_folder_name(mu)
        full_mu = self._get_full_mu(sub_name)
        x, y = self._get_xy(data_id, sub_name)
        abs_x, abs_y = np.abs(x), np.abs(y)
        indices = np.arange(len(x))
        wandb.log({'abs signal': wandb.plot.line_series(
            xs=indices,
            ys=[abs_x, abs_y],
            keys=['dirty', 'clean'],
            title=f'abs signal, mu={mu}, i={data_id}',
            xname="index")})

        # for v, title in [(x, 'x (dirty)'), (y, 'y (clean)')]:
        #     wandb.log({title: wandb.plot.line_series(
        #         xs=indices,
        #         ys=[v.real, v.imag],
        #         keys=['real', 'imag'],
        #         title=f'{title}, mu={mu}, i={data_id}',
        #         xname="index")})

        print(f'uploaded to wandb mu={mu}, i={data_id}')

    def _get_full_mu(self, sub_name):
        # read the actual mu from the config in that dir
        return DatasetNormal(self.path + '/' + sub_name).config.mu

    def _extract_values_from_folder_name(self):
        matches = re.findall(r'\d+', self.base_name)

        if len(matches) >= 2:
            num_samples = int(matches[0])
            num_mus = int(matches[1])
            return num_samples, num_mus
        else:
            raise "folder name is not in the right format of $$samples_$$mus"

    # ------------ private ------------

    def _calc_full_ber(self, n):
        sub_name_filter = '*'
        n = n or self.params['num_samples']  # if n is None, take all permutations
        if self.num_permutations < n:
            self.ber_vec, self.mu_vec = Metrics.gen_ber_mu_from_folders(
                self.path, sub_name_filter, self.verbose_level, self._tqdm, n, is_matrix_ber=self.is_box_plot,
                in_cs=self.cs_for_input, out_cs=self.cs_for_output)
            self.num_permutations = n

    def _get_sub_folder_name(self, mu):
        cropped_mu = self._get_mu_as_str(mu)
        sub_name = f"mu={cropped_mu}"
        return sub_name

    def _get_mu_as_str(self, mu):
        all_mus = []

        for dir in os.listdir(self.path):
            if 'mu=' in dir:
                mu_val = re.findall(r'mu=(\d+\.\d+)', dir)[0]
                all_mus.append((float(mu_val), mu_val))

        closest_mu = min(all_mus, key=lambda x: abs(x[0] - mu))

        return closest_mu[1]  # return the string (the cropped mu)

    def clear_ber(self):
        self.ber_vec = None
        self.mu_vec = None
        self.num_permutations = 0

    def _get_xy(self, data_id, sub_name):
        # x is the dirty signal, y is the clean signal
        dir = self.path + '/' + sub_name
        dataset = DatasetNormal(dir)
        x, y = dataset.get_numpy_xy(data_id)
        return x, y

    def _count_digits(self, subfolder_name):
        mu = subfolder_name.split('=')[1]
        return len(mu.split('.')[1])


if __name__ == '__main__':
    da = DataAnalyzer('./data/datasets/iq/qam1024_160x20')
    da.plot_single_sample_b(0.001, 0)
    da.plot_full_ber_graph()
