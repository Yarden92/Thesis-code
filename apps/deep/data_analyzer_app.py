from dataclasses import dataclass

import pyrallis

from src.deep.data_analyzer import DataAnalyzer


@dataclass
class DataAnalyzerConfig:
    path: str = None
    mu: float = -1  # mu to check, if -1: takes the first folder
    i: int = 0  # data index to check
    is_full_ber: bool = True  # if True, calculate full ber
    is_box_plot: bool = False  # (Currently not supported on wandb) if True, the ber graph will be displayed in a box plot form
    is_single_item: bool = False  # whether to plot single item (by i and mu vals)
    num_x_per_folder: int = None  # if None, will be calculated from the data
    is_save_to_file: bool = True  # if True, will save the plots to file else will show them
    is_upload_to_wandb: bool = True  # if True, will upload the plots to wandb
    verbose_level: int = 0  # 0: no prints, 1: summary prints, 2: detailed prints


def main(config):
    path = config.path
    assert config.is_box_plot is False or config.is_upload_to_wandb is False, \
        "currently box plot is not supported on wandb, please set is_box_plot to False or is_upload_to_wandb to False"
    da = DataAnalyzer(path, is_box_plot=config.is_box_plot, verbose_level=config.verbose_level)
    if config.mu == -1:
        config.mu = da.params['mu_start']

    if config.is_single_item:
        da.plot_single_sample(mu=config.mu, data_id=config.i, is_save=config.is_save_to_file)
        if config.is_upload_to_wandb:
            print('Warning: uploading single item to wandb is not fully supported yet...')  # TODO: complete this
            da.wandb_log_single_sample(mu=config.mu, data_id=config.i)

    if config.is_full_ber:
        da.plot_full_ber_graph(permute_limit=config.num_x_per_folder, is_save=config.is_save_to_file)
        if config.is_upload_to_wandb:
            da.wandb_log_ber_vs_mu(n=config.num_x_per_folder)


if __name__ == '__main__':
    config = pyrallis.parse(config_class=DataAnalyzerConfig)
    main(config)
