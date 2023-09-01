

from dataclasses import dataclass, field
import os
import time
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pyrallis
from tabulate import tabulate
from tqdm import tqdm
from src.general_methods import multi_run
from src.general_methods.visualizer import Visualizer
from src.optics.channel_simulation2 import ChannelSimulator2
from src.optics.config_manager import ChannelConfig

output_bers_path = '/data/yarcoh/thesis_data/data/outputs/ber_maps'
output_plots_path = '/data/yarcoh/thesis_data/data/outputs/ber_plots'
@dataclass
class NoiseSearcherConfig:
    mu_start: float = 0.05      # start value of mu vector (linearly spaced)
    mu_end: float = 0.4         # end value of mu vector
    num_of_mu: int = 30         # number of mu values
    n_realisation: int = 20     # number of realisations for each mu
    is_print: bool = True       # whether to print the BER map results
    is_plot: bool = True        # whether to plot the BER map results
    is_save: bool = True        # whether to save the BER map results
    load_path: str = None       # if not None, load results from the path
    channel_config: ChannelConfig = field(default_factory=ChannelConfig)


def main(config: NoiseSearcherConfig):
    ns = NoiseSearcher2(config.channel_config)

    if config.load_path:
        BERs, mu_range = ns.load_ber_map(config.load_path)
    else:
        mu_range = np.linspace(config.mu_start, config.mu_end, num=config.num_of_mu)
        BERs = ns.create_BER_vec(mu_range, n_realisation=config.n_realisation)

    if config.is_save:
        ns.save_ber_map(BERs, mu_range)

    if config.is_print:
        ns.print_ber_vec(BERs, mu_range)

    if config.is_plot:
        ns.plot_ber_map(BERs, mu_range)


class NoiseSearcher2:
    def __init__(self, channel_config: ChannelConfig) -> None:
        self.cs: ChannelSimulator2 = self._get_channel_simulator(channel_config)

    def _calculate_BER(self, mu, pbar, n_realisation=10):
        self.cs.update_mu(mu)
        ber, num_errors = multi_run.run_n_times2(self.cs, n=n_realisation, pbar=pbar)
        return ber

    def create_BER_heatmap_dummy(self, mus, Ds):
        # Create empty arrays to store mu, D, and BER values
        BERs = np.random.rand(len(mus), len(Ds))  # Example random BER values
        for i, mu in enumerate(mus):
            for j, D in enumerate(Ds):
                BER = mu * D
                BERs[i, j] = BER
        return BERs

    def create_BER_vec(self, mus, n_realisation=10):
        # Create empty arrays to store mu, and BER values
        BERs = np.zeros(len(mus))

        # Iterate through mu and D values and calculate BER
        pbar = tqdm(total=len(mus)*n_realisation)
        for i, mu in enumerate(mus):
            BER = self._calculate_BER(mu, pbar, n_realisation=n_realisation)
            BERs[i] = BER
        return BERs

    def save_ber_map(self, BERs, mus, root_dir=output_bers_path):
        timestamp = time.strftime("%Y.%m.%d-%H.%M.%S")
        sub_dir_path = os.path.abspath(os.path.join(root_dir, timestamp))
        os.makedirs(sub_dir_path, exist_ok=True)
        path_ber = os.path.join(sub_dir_path, f'ber.npy')
        path_mu = os.path.join(sub_dir_path, f'mu.npy')
        path_config = os.path.join(sub_dir_path, f'cs_config.yml')
        np.save(path_ber, BERs)
        np.save(path_mu, mus)
        pyrallis.dump(self.cs.channel_config, open(path_config,'w'))
        print(f'saved BER and mus to {sub_dir_path}')

    @classmethod
    def load_from_config(cls, channel_config: ChannelConfig):
        return cls(channel_config)

    def load_ber_map(self, dir_path) -> tuple:
        BERs = np.load(os.path.join(dir_path, 'ber.npy'))
        mus = np.load(os.path.join(dir_path, 'mu.npy'))
        return BERs, mus

    def plot_ber_map(self, BERs, mus):
        # plot BER vs mus
        Visualizer.plot_bers(mus,[BERs])

    def _save_fig(self, root_dir=output_plots_path):
        timestamp = time.strftime("%Y.%m.%d-%H.%M.%S")
        sub_dir_path = os.path.abspath(os.path.join(root_dir, timestamp))
        os.makedirs(sub_dir_path, exist_ok=True)
        path = os.path.join(sub_dir_path, f'BER_heatmap.png')
        plt.savefig(path)
        print(f'BER heatmap saved to {path}')

    def print_ber_vec(self, BERs, mus):
        # print the ber vector in a table format
        table = []
        for i, mu in enumerate(mus):
            table.append([mu, BERs[i]])

        print(tabulate(table, headers=['mu', 'BER']))
        

    def _get_channel_simulator(self, channel_config: ChannelConfig) -> ChannelSimulator2:
        return ChannelSimulator2(channel_config)


if __name__ == "__main__":
    # config = pyrallis.parse(NoiseSearcherConfig)
    config_path = './config/noise_searcher2/long_search.yml'
    config = pyrallis.parse(NoiseSearcherConfig, config_path=config_path)
    main(config)
