from dataclasses import dataclass

import pyrallis

from src.deep.data_analyzer import DataAnalyzer


@dataclass
class DataAnalyzerConfig:
    path: str = None
    mu: float = -1  # mu to check, if -1: takes the first folder
    i: int = 0  # data index to check
    is_full_ber: bool = True  # if True, calculate full ber
    is_box_plot: bool = False  # if True, the ber graph will be displayed in a box plot form
    is_single_item: bool = False  # whether to plot single item (by i and mu vals)
    num_x_per_folder: int = 5
    is_save_to_file: bool = True
    is_upload_to_wandb: bool = True


def main(config):
    path = config.path
    da = DataAnalyzer(path, is_box_plot=config.is_box_plot)
    if config.mu == -1:
        config.mu = da.params['mu_start']

    if config.is_single_item:
        da.plot_single_sample(mu=config.mu, data_id=config.i, is_save=config.is_save_to_file)
        if config.is_upload_to_wandb:
            print('uploading single item to wandb is not fully supported yet...')  # TODO: complete this
            da.wandb_log_single_sample(mu=config.mu, data_id=config.i)

    if config.is_full_ber:
        da.plot_full_ber_graph(num_permut=config.num_x_per_folder, is_save=config.is_save_to_file)
        if config.is_upload_to_wandb:
            da.wandb_log_ber_vs_mu(n=config.num_x_per_folder)


if __name__ == '__main__':
    config = pyrallis.parse(config_class=DataAnalyzerConfig)
    main(config)
