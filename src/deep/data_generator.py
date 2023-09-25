from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import json
import multiprocessing
import os
import warnings

import numpy as np
import pyrallis
from tqdm.auto import tqdm
from src.deep.data_analyzer import DataAnalyzer
from src.general_methods.text_methods import FileNames
from src.optics.channel_simulation2 import ChannelSimulator2
from src.optics.config_manager import ChannelConfig, ConfigManager


@dataclass
class DataConfig:
    num_samples: int = 10  # for each mu
    num_mus: int = 3
    mu_start: float = 0.0005
    mu_end: float = 0.07
    num_workers: int = 0  # 0 means all available
    logger_path: str = './logs'
    output_path: str = '/data/yarcoh/thesis_data/data/datasets'
    to_collect_ber: bool = True
    channel_config: ChannelConfig = field(default_factory=ChannelConfig)


class DataGenerator:
    def __init__(self, config: DataConfig) -> None:
        self.cs = ChannelSimulator2(config.channel_config)
        self.mu_vec = self._gen_mu(config.mu_start, config.mu_end, config.num_mus)
        self.num_digits_mu = self._get_num_digits_mu(self.mu_vec)
        self.config = config
        self.ber_vec = np.zeros(len(self.mu_vec))
        # self.ber_vec = multiprocessing.Array('d', len(self.mu_vec))

        self.root_dir = f'{config.output_path}/{config.num_samples}samples_{config.num_mus}mu'
        if os.path.exists(self.root_dir):
            input(f'\n\nfolder: {self.root_dir} already exists.\nprevious data will be erased, press enter to continue: ')
            # delete the entire folder
            os.system(f'rm -rf {self.root_dir}')

    def _gen_mu(self, mu_start, mu_end, mu_len):
        # TODO add log scale
        return np.linspace(start=mu_start, stop=mu_end, num=mu_len)

    def _get_num_digits_mu(self, mu_vec):
        deltas = np.diff(mu_vec)
        mu_delta = np.min(deltas)
        return int(np.ceil(-np.log10(mu_delta)))
    

    def run(self) -> None:
        # vec_lens = num_symbols*cs.over_sampling + cs.N_rrc - 1
        # assert vec_lens == num_symbols*8*2, "the formula is not correct! check again"

        if self.config.logger_path:
            os.makedirs(self.config.logger_path, exist_ok=True)
            print(f'saving logs (disabled) to {os.path.abspath(self.config.logger_path)}')
            print(f'saving data to {os.path.abspath(self.root_dir)}')
        # file_path = f'{logger_path}/{get_ts_filename()}'

        # Create a pool of worker processes
        N_workers = self.config.num_workers if self.config.num_workers > 0 else multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=N_workers)
        # pool = multiprocessing.Pool(processes=1)
        # warnings.warn('using only 1 process for debugging - change it back to multiprocessing.cpu_count()')

        pbar = tqdm(total=len(self.mu_vec)*self.config.num_samples)
        for mu_index, mu in enumerate(self.mu_vec):
            dir_path = f'{self.root_dir}/mu={mu:.{self.num_digits_mu}f}'
            os.makedirs(dir_path, exist_ok=True)
            self.cs.update_mu(mu)
            conf = self.cs.channel_config
            ConfigManager.save_config(dir_path, conf)
            for i in range(self.config.num_samples):
                pool.apply_async(self._gen_data_i, (dir_path, i, mu, mu_index),
                                 callback=lambda ber, _mu_index=mu_index: self._on_exit_i(ber, _mu_index, pbar))

        # Close the pool of worker processes
        pool.close()
        pool.join()

        if self.config.to_collect_ber:
            print(f'\n\nsaving ber to {self.root_dir}')
            print(f'ber_vec={self.ber_vec}')
            DataAnalyzer.save_ber(self.root_dir, self.ber_vec, self.mu_vec, self.config.num_samples)

        print('\nall done')

    def _gen_data_i(self, dir_path, i, mu, mu_index) -> float:
        try:
            ber = None
            self.cs.update_mu(mu)  # perhaps we should copy cs to new instance
            if self.config.to_collect_ber:
                ber, num_errors = self.cs.simulate_and_analyze()
            else:
                self.cs.quick_simulate()
            x, y = self.cs.get_io_samples()
            self.save_xy(dir_path, x, y, i)
        except Exception as e:
            print(f'error at mu={mu}, i={i}: {e}')
        return ber
    
    def _on_exit_i(self, ber, mu_index, pbar):
        self.ber_vec[mu_index] += ber / self.config.num_samples
        pbar.update(1)
        # pbar.set_description(f'ber_vec={self.ber_vec}')


    def save_xy(self, dir, x, y, i):
        x_path = f'{dir}/{i}_{FileNames.x}'
        y_path = f'{dir}/{i}_{FileNames.y}'
        with open(x_path, 'wb') as f:
            np.save(f, x)
        with open(y_path, 'wb') as f:
            np.save(f, y)
