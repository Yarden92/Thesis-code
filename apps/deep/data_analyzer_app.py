from dataclasses import dataclass

import pyrallis

from src.deep.data.data_analyzer import DataAnalyzer


@dataclass
class DataAnalyzerConfig:
    path: str = None
    is_save_ber_vs_mu: bool = True
    mu: float = None # mu to check if not None
    i: int = None # data index to check if not None
    num_x_per_folder: int = 5


def main(config):
    path = config.path
    da = DataAnalyzer(path)
    if config.mu is not None and config.i is not None:
        da.save_sample_plots(config.mu, config.i)
    if config.is_save_ber_vs_mu:
        da.save_ber_vs_mu(config.num_x_per_folder)


if __name__ == '__main__':
    config = pyrallis.parse(config_class=DataAnalyzerConfig)
    main(config)
