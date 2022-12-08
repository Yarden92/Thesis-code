from dataclasses import dataclass

import numpy as np
import pyrallis
from tqdm import tqdm

from src.deep import data_loaders
from src.deep.data_analyzer import DataAnalyzer
from src.deep.standalone_methods import DataType
from src.optics.channel_simulation import ChannelSimulator
from src.optics.split_step_fourier import SplitStepFourier


@dataclass
class DataConfig:
    data_len: int = 10  # for each mu
    mu_len: int = 3
    num_symbols: int = 512
    qam: int = 1024
    logger_path: str = './logs'
    output_path: str = './data/datasets/iq'
    max_workers: int = 10
    data_type: int = 0  # 0 for spectrum, 1 for iq_samples
    mu_start: float = 0.0005
    over_sampling: int = 8 # over sampling factor for pulse shaping
    mu_end: float = 0.07
    is_analyze_after: bool = False


def main(config: DataConfig):
    # config

    dir = f'{config.output_path}/qam{config.qam}_{config.data_len}x{config.mu_len}'
    mu_vec = np.linspace(start=config.mu_start, stop=config.mu_end, num=config.mu_len)
    cs = ChannelSimulator(m_qam=config.qam,
                          num_symbols=config.num_symbols,
                          normalization_factor=0,  # will be overwritten during runtime
                          dt=1,
                          ssf=SplitStepFourier(
                              b2=-20e-27,
                              gamma=0.003,
                              t0=125e-12,
                              dt=1,
                              z_n=1000e3,
                              h=200
                          ),
                          verbose=False)

    # generate the date
    data_loaders.gen_data2(config.data_len, config.num_symbols, mu_vec, cs, dir, tqdm=tqdm,
                           logger_path=config.logger_path, max_workers=config.max_workers, type=config.data_type)

    if config.is_analyze_after:
        data_analyzer = DataAnalyzer(dir)
        data_analyzer.plot_full_ber_graph(is_save=True)
        data_analyzer.wandb_log_ber_vs_mu()


if __name__ == '__main__':
    config = pyrallis.parse(DataConfig)
    main(config)
