import os

import numpy as np
import wandb
from tqdm.auto import tqdm

from src.deep import data_loaders
from src.deep import standalone_methods
from src.deep.data_loaders import DatasetNormal
from src.deep.data_methods import DataMethods, FolderTypes
from src.deep.metrics import Metrics
from src.deep.standalone_methods import DataType
from src.general_methods.visualizer import Visualizer

analysis_dir = '_analysis'


class DataAnalyzer():
    def __init__(self, data_folder_path: str, _tqdm=tqdm, is_box_plot=False, verbose_level=0):
        self.path = data_folder_path.rstrip('/')
        self.base_name = os.path.basename(self.path)
        self.params = self.fetch_params()
        self.is_wandb_init = False
        self.is_box_plot = is_box_plot
        self.ber_vec = None
        self.mu_vec = None
        self._tqdm = _tqdm
        self.num_digits = None
        self.verbose_level = verbose_level

    def fetch_params(self):
        num_samples, num_mus = [int(x) for x in self.base_name.split('_')[-1].split('x')]
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
            conf = data_loaders.read_conf(sub_path)
            mu_vec.append(conf['normalization_factor'])
            conf_list.append(conf)
            num_files = len(os.listdir(sub_path))
            assert num_files == 1 + 2*num_samples, f'num_files={num_files} != 1+2*num_samples={num_samples}'

        assert len(conf_list) == num_mus, 'num of sub directories dont match num of mus from headline'
        assert len(mu_vec) == num_mus, 'num of sub directories dont match num of mus from headline'

        params = {
            'num_samples': num_samples,
            'num_mus': num_mus,
            'mu_start': min(mu_vec),
            'mu_end': max(mu_vec),
            'conf': conf_list[0],
            'num_digits': self.num_digits
        }
        return params

    # -------------- plots --------------

    def plot_single_sample(self, mu: float = None, data_id: int = 0, is_save=True):
        # read folder and plot one of the data samples
        mu = mu or self.params['mu_start']  # default to first mu
        sub_name = self._get_sub_folder_name(mu)
        x, y = self._get_xy(data_id, sub_name)
        out_path = f'{self.path}/{analysis_dir}/{sub_name}'
        os.makedirs(out_path, exist_ok=True)

        is_spectrum = ('data_type' not in self.params['conf']) or (self.params['conf']['data_type'] == 0)

        N = len(x)
        x_start, x_stop = int(0.2*N), int(0.3*N)
        zm = range(x_start, x_stop) if is_spectrum else range(0, 50)
        func = 'plot' if is_spectrum else 'stem'

        for v, name in [[x, 'x'], [y, 'y']]:
            Visualizer.twin_zoom_plot(
                title=name,
                full_y=np.real(v),
                zoom_indices=zm,
                function=func,
                output_name=f'{out_path}/{name}_real.png' if is_save else None
            )

        x_power = np.mean(np.abs(x) ** 2)
        print(f'x_power={x_power}')

        ber, num_errors = Metrics.calc_ber_for_single_vec(x, y, conf=self.params['conf'])
        print(f'ber={ber}')

        if is_save: print(f'images saved to {out_path}')

    def calc_ber_for_sub_folder(self, mu, n=5, _tqdm=None):
        # n is the number of x's permutations to take from each folder
        sub_name = self._get_sub_folder_name(mu)
        dir = self.path + '/' + sub_name
        dataset = DatasetNormal(dir)

        ber_vec, num_errors = Metrics.calc_ber_from_dataset(dataset, False, _tqdm, n)
        print(f'the avg ber of mu={mu} (with {n} permutations) is {np.mean(ber_vec)}')

        return ber_vec

    def plot_full_ber_graph(self, permute_limit=None, is_save=True):
        # n is the number of x's permutations to take from each folder
        self._calc_full_ber(permute_limit)
        out_dir = f'{self.path}/{analysis_dir}'
        os.makedirs(out_dir, exist_ok=True)
        out_path = f'{out_dir}/ber_vs_mu.png'

        if self.is_box_plot:
            Visualizer.plot_bers_boxplot(
                us=self.mu_vec,
                bers_vecs=self.ber_vec.T,
                output_path=out_path if is_save else None,
            )
        else:
            Visualizer.plot_bers(
                us=self.mu_vec,
                bers_vecs=[self.ber_vec],
                output_path=out_path if is_save else None,
            )

        print(f'ber vs mu saved to {out_path}')

    # ------------ wandb ------------

    def _init_wandb(self):
        if self.is_wandb_init: return
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

    # ------------ private ------------

    def _calc_full_ber(self, n):
        sub_name_filter = '*'
        if self.ber_vec is None or self.mu_vec is None or (n is not None and len(self.ber_vec) < n):
            self.ber_vec, self.mu_vec = Metrics.gen_ber_mu_from_folders(self.path, sub_name_filter, 0, self._tqdm, n,
                                                                        is_matrix_ber=self.is_box_plot)

    def _get_sub_folder_name(self, mu):
        num_samples = self.params['num_samples']
        sub_name = f"{num_samples}_samples_mu={mu:.{self.params['num_digits']}f}"
        return sub_name

    def clear_ber(self):
        self.ber_vec = None
        self.mu_vec = None

    # def _is_valid_subfolder(self, sub_name):
    #     if sub_name.startswith('_'): return False
    #     if sub_name.startswith('.'): return False
    #     if '=' not in sub_name: return False
    #     return True

    def _get_xy(self, data_id, sub_name):
        dir = self.path + '/' + sub_name
        dataset = DatasetNormal(dir)
        print(f'the folder {dir} contains {len(dataset)} samples')
        x, y = dataset.get_numpy_xy(data_id)
        return x, y

    def _count_digits(self, subfolder_name):
        mu = subfolder_name.split('=')[1]
        return len(mu.split('.')[1])


if __name__ == '__main__':
    da = DataAnalyzer('./data/datasets/iq/qam1024_160x20')
    da.plot_single_sample(0.001, 0)
    da.plot_full_ber_graph()
