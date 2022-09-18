import os

import numpy as np
import wandb
from tqdm import tqdm

from src.deep.data_loaders import SingleMuDataSet, read_conf
from src.deep.metrics import Metrics
from src.deep.standalone_methods import get_platform, DataType
from src.general_methods.visualizer import Visualizer

analyzation_dir = '_analyzation'


class DataAnalyzer():
    def __init__(self, data_folder_path: str):
        self.data_folder_path = data_folder_path.rstrip('/')
        self.base_name = os.path.basename(data_folder_path)
        self.params = self.fetch_params()
        self.is_wandb_init = False

    def _init_wandb(self):
        data_type = self.params['conf']['data_type'] if 'data_type' in self.params['conf'] else 0
        data_type = DataType(data_type).name
        run_name = f'{self.base_name}__{data_type}'
        wandb.init(project="data_analyze", entity="yarden92", name=run_name,
                   tags=[f'{get_platform()}', f'ds={self.params["num_samples"]}'])
        wandb.config = self.params
        self.is_wandb_init = True

    def wandb_log_ber_vs_mu(self, num_x_per_folder=5):
        sub_name_filter = '*'
        ber_vec, mu_vec = Metrics.gen_ber_mu_from_folders(self.data_folder_path, sub_name_filter, 0, tqdm,
                                                          num_x_per_folder)
        if not self.is_wandb_init:
            self._init_wandb()
        wandb.log({'BER vs mu': wandb.plot.line_series(
            xs=mu_vec,
            ys=[ber_vec],
            keys=['ber'],
            title='BER vs mu',
            xname="mu")})

    def save_sample_plots(self, mu: float, data_id: int):
        # read folder and plot one of the data samples
        num_samples = self.params['num_samples']
        sub_name = f"{num_samples}_samples_mu={mu}"
        dir = self.data_folder_path + '/' + sub_name

        dataset = SingleMuDataSet(dir)

        print(f'the folder {dir} contains {len(dataset)} samples')

        x, y = dataset.get_numpy_xy(data_id)
        out_path = f'{self.data_folder_path}/{analyzation_dir}/{sub_name}'
        os.makedirs(out_path, exist_ok=True)

        if self.params['conf']['data_type'] == 0:
            zm = range(1700, 2300)
            Visualizer.twin_zoom_plot('x', np.real(x), zm, output_name=out_path + '/x_real.png')
            Visualizer.twin_zoom_plot('y', np.real(y), zm, output_name=out_path + '/y_real.png')
        elif self.params['conf']['data_type'] == 1:
            zm = range(0, 50)
            Visualizer.twin_zoom_plot('x (real)', np.real(x), zm, function='stem', output_name=out_path + '/x_real.png')
            Visualizer.twin_zoom_plot('y (real)', np.real(y), zm, function='stem', output_name=out_path + '/y_real.png')
        else:
            raise ValueError(f'unknown data type of {DataType(self.params["conf"]["data_type"]).name}')
        print(f'images saved to {out_path}')

    def save_ber_vs_mu(self, num_x_per_folder=5):
        sub_name_filter = '*'
        ber_vec, mu_vec = Metrics.gen_ber_mu_from_folders(self.data_folder_path, sub_name_filter, 0, tqdm,num_x_per_folder)
        out_path = f'{self.data_folder_path}/{analyzation_dir}/ber_vs_mu.png'
        Visualizer.plot_bers(mu_vec, [ber_vec],output_path=out_path)
        print(f'ber vs mu saved to {out_path}')

    def fetch_params(self):
        num_samples, num_mus = [int(x) for x in self.base_name.split('_')[-1].split('x')]
        mu_vec = []
        conf_list = []
        for sub_name in os.listdir(self.data_folder_path):
            if sub_name.startswith('_'): continue
            sub_path = f"{self.data_folder_path}/{sub_name}"
            conf = read_conf(sub_path)
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
            'conf': conf_list[0]
        }
        return params

    def set_params_manually(self, num_samples: int, mu_start: float, mu_end: float, num_mus: int):
        self.params = {
            'num_samples': num_samples,
            # 'conf': conf,
            'mu_start': mu_start,
            'mu_end': mu_end,
            'num_mus': num_mus
        }
        dataset = SingleMuDataSet(f"{self.data_folder_path}/{num_samples}_samples_mu={mu_start}")
        self.params['conf'] = dataset.config


if __name__ == '__main__':
    da = DataAnalyzer('./data/datasets/iq/qam1024_160x20')
    da.save_sample_plots(0.001,0)
    da.save_ber_vs_mu()