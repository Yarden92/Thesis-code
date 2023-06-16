

from dataclasses import dataclass
import os
import time
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pyrallis
from tabulate import tabulate
from tqdm import tqdm
from src.general_methods import multi_run
from src.optics.channel_simulation import ChannelSimulator
from src.optics.split_step_fourier import SplitStepFourier

@dataclass
class NoiseSearcherConfig:
    mu_start: float = 0.05
    mu_end: float = 0.4
    D_start: float = 1e-30
    D_end: float = 1e-10
    num_of_mu: int = 30
    num_of_D: int = 30
    n_realisation: int = 20
    is_print: bool = True
    is_plot: bool = True
    is_save: bool = True
    load_path: str = None

def main(config: NoiseSearcherConfig):
    ns = NoiseSearcher()

    if config.load_path:
        BERs, mu_range, D_range = ns.load_ber_map(config.load_path)
    else:
        mu_range = np.linspace(config.mu_start, config.mu_end, num=config.num_of_mu)
        D_range = np.logspace(np.log10(config.D_start), np.log10(config.D_end), num=config.num_of_D)
        BERs = ns.create_BER_heatmap(mu_range, D_range, n_realisation=config.n_realisation)

    if config.is_save:
        ns.save_ber_map(BERs, mu_range, D_range)
    
    if config.is_print:
        ns.print_ber_map(BERs, mu_range, D_range)
    
    if config.is_plot:
        ns.plot_ber_map(BERs, mu_range, D_range)


class NoiseSearcher:
    def __init__(self) -> None:
        self.cs = self._get_channel_simulator()

    def _calculate_BER(self, mu, D, pbar, n_realisation=10):
        self.cs.ssf.D = D
        self.cs.normalization_factor = mu
        ber, num_errors = multi_run.run_n_times(self.cs, n=n_realisation, pbar=pbar)
        return ber

    def create_BER_heatmap_dummy(self, mus, Ds):
        # Create empty arrays to store mu, D, and BER values
        BERs = np.random.rand(len(mus), len(Ds))  # Example random BER values
        for i, mu in enumerate(mus):
            for j, D in enumerate(Ds):
                BER = mu * D
                BERs[i, j] = BER
        return BERs

    def create_BER_heatmap(self, mus, Ds, n_realisation=10):
        # Create empty arrays to store mu, D, and BER values
        BERs = np.zeros((len(mus), len(Ds)))

        # Iterate through mu and D values and calculate BER
        pbar = tqdm(total=len(mus)*len(Ds)*n_realisation)
        for i, mu in enumerate(mus):
            for j, D in enumerate(Ds):
                BER = self._calculate_BER(mu, D, pbar, n_realisation=n_realisation)
                BERs[i, j] = BER
        return BERs
    
    def save_ber_map(self, BERs, mus, Ds, root_dir='./data/ber_maps'):
        timestamp = time.strftime("%Y.%m.%d-%H.%M.%S")
        sub_dir_path = os.path.abspath(os.path.join(root_dir, timestamp))
        os.makedirs(sub_dir_path, exist_ok=True)
        path_ber = os.path.join(sub_dir_path, f'ber.npy')
        path_mu = os.path.join(sub_dir_path, f'mu.npy')
        path_D = os.path.join(sub_dir_path, f'D.npy')
        np.save(path_ber, BERs)
        np.save(path_mu, mus)
        np.save(path_D, Ds)
        print(f'saved BER, mus, Ds to {sub_dir_path}')

    
    def load_ber_map(self, dir_path) -> tuple:
        BERs = np.load(os.path.join(dir_path, 'ber.npy'))
        mus = np.load(os.path.join(dir_path, 'mu.npy'))
        Ds = np.load(os.path.join(dir_path, 'D.npy'))
        return BERs, mus, Ds
    
    def plot_ber_map(self, BERs, mus, Ds):
        # fig, ax = plt.subplots()

        # Set logarithmic scale for the D axis
        plt.xscale('log')

        # Set logarithmic scale for the colorbar
        # find the nearest power of 10 to the minimum BER
        k = np.floor(np.log10(np.min(BERs)))
        k = max(k, -5)

        # give extra gap
        k -= 1

        # avoid log(0) by setting the minimum value to 10^k
        BERs[BERs == 0] = 10**k

        # Create a heatmap using pcolormesh
        cax = plt.pcolormesh(Ds, mus, BERs, shading='auto', cmap='viridis', norm=LogNorm(vmin=10**k, vmax=1))

        cbar = plt.colorbar(cax, ticks=np.logspace(k, 0, num=int(1-k)))
        cbar.set_label('BER (log scale)')

        # Set labels and titles
        plt.xlabel('D')
        plt.ylabel('mu')
        plt.title('BER heatmap')

        # Set the axis limits
        plt.xlim([min(Ds), max(Ds)])
        plt.ylim([min(mus), max(mus)])

        # Invert the y-axis so that mu increases from top to bottom
        plt.gca().invert_yaxis()
        
        # self._save_fig()
        plt.show()

    def _save_fig(self, root_dir='./data/ber_plots'):
        timestamp = time.strftime("%Y.%m.%d-%H.%M.%S")
        sub_dir_path = os.path.abspath(os.path.join(root_dir, timestamp))
        os.makedirs(sub_dir_path, exist_ok=True)
        path = os.path.join(sub_dir_path, f'BER_heatmap.png')
        plt.savefig(path)
        print(f'BER heatmap saved to {path}')

    def print_ber_map(self, BERs, mus, Ds):
        # print a 2D grid table of BERs for each mu and D
        matrix = []
        for i, mu in enumerate(mus):
            row = [f'{mu:.3f}']
            for j, D in enumerate(Ds):
                cell_value = f'{BERs[i, j]:.2e}'
                row.append(cell_value)
            matrix.append(row)

        # Prepare the table headers
        headers = ['mu \\ D'] + [f'{D:.2e}' for D in Ds]

        # Generate the table using tabulate
        table = tabulate(matrix, headers, tablefmt="pretty")

        # Print the table
        print(table)

    def _get_channel_simulator(self):
        return ChannelSimulator(m_qam=16,
                            num_symbols=64,
                            normalization_factor=0,  # will be overwritten during runtime
                            dt=1,
                            ssf=SplitStepFourier(
                                b2=-20e-27,
                                gamma=0.003,
                                t0=125e-12,
                                dt=1,
                                z_n=1000e3,
                                dz=200,
                                D=0,
                                with_noise=True,
                            ),
                            verbose=False)




if __name__ == "__main__":
    # config = pyrallis.parse(NoiseSearcherConfig)
    config = pyrallis.parse(NoiseSearcherConfig, config_path='./config/noise_searcher/load_and_plot.yml')
    main(config)
