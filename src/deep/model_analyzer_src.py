import os
import numpy as np
from tqdm.auto import tqdm

import wandb
from src.deep.data_loaders import DatasetNormal, create_channels, get_datasets_set
from src.deep.standalone_methods import get_platform
from src.deep.trainers import Trainer
from src.general_methods.visualizer import Visualizer
from src.deep.standalone_methods import GeneralMethods as GM
from torch.utils.data import DataLoader


class ModelAnalyzer:
    def __init__(self, trainer: Trainer, run_name: str = None):
        self.trainer = trainer
        self.wandb_project = self.trainer.config['wandb_project']
        self.run_name = run_name
        self.model_name = self.trainer.model._get_name()
        self.ds_len = len(self.trainer.train_dataset) + len(self.trainer.val_dataset)
        self.mu = self.trainer.train_dataset.cropped_mu
        self.outputs = {}
        self.cs_in, self.cs_out = create_channels(self.trainer.train_dataset.config)

    def _init_wandb(self):
        if wandb.run is not None:  # if already logged in
            return

        wandb.init(project=self.wandb_project, entity="yarden92", name=self.run_name,
                   tags=[f'mu={self.mu}', f'{get_platform()}', self.model_name, f'ds={self.ds_len}'],
                   reinit=True)
        wandb.config = {
            "learning_rate": self.trainer.config['lr'],
            "epochs": self.trainer.config['epochs'],
            "batch_size": self.trainer.config['batch_size']
        }

    def _reset_wandb(self):
        wandb.run = None

    def plot_all_bers(self, base_path, train_ds_ratio, val_ds_ratio, test_ds_ratio,
                      _tqdm=None, verbose_level=1):

        sorted_dirs = GM.sort_dirs_by_mu(os.listdir(base_path))
        # powers, org_bers, model_bers, ber_improvements = [], [], [], []
        # self.outputs['powers'] = []
        self.outputs['mu'] = []
        self.outputs['org_bers'], self.outputs['model_bers'], self.outputs['ber_improvements'] = [], [], []
        for folder_name in sorted_dirs:
            ds_path = os.path.join(base_path, folder_name)
            self.load_test_dataset(ds_path, train_ds_ratio, val_ds_ratio, test_ds_ratio, False)
            mu = self.trainer.val_dataset.cropped_mu
            # y_power = self.trainer.val_dataset.config['avg_y_power']
            org_ber, model_ber, ber_improve = self.trainer.compare_ber(verbose_level=verbose_level, _tqdm=_tqdm)

            if verbose_level > 0:
                print(
                    f'mu={mu:.3f} | org_ber={org_ber:.2e} | model_ber={model_ber:.2e} |  ber_improve={ber_improve*100:03.0f}%'
                    # f' | y_power={y_power:.2e}'
                    )
                    
                # print((
                #     f'\n----------------- BERs for mu={mu} ----------------\n'
                #     f'org_ber={org_ber}, model_ber={model_ber}, ber_improvement={ber_improve}, y_power={y_power:.2e}'
                #     '\n'
                # ))
            # powers.append(y_power)
            # org_bers.append(org_ber)
            # model_bers.append(model_ber)
            # ber_improvements.append(ber_improve)
            # self.outputs['powers'].append(y_power)
            self.outputs['mu'].append(mu)
            self.outputs['org_bers'].append(org_ber)
            self.outputs['model_bers'].append(model_ber)
            self.outputs['ber_improvements'].append(ber_improve)

        # x_axis = self.outputs['powers']
        x_axis = self.outputs['mu']
        

        Visualizer.my_plot(
            x_axis, self.outputs['org_bers'], 
            x_axis, self.outputs['model_bers'],
            name='BERs vs. y power',
            legend=['org_ber', 'model_ber'],
            # xlabel='y power', 
            xlabel='mu',
            ylabel='BER',
            function='semilogy'
        )

    def upload_all_bers_to_wandb(self, base_path, train_ds_ratio, val_ds_ratio, test_ds_ratio,
                                 _tqdm=None, verbose_level=1):

        self._init_wandb()

        sorted_dirs = GM.sort_dirs_by_mu(os.listdir(base_path))

        for folder_name in sorted_dirs:
            ds_path = os.path.join(base_path, folder_name)
            self.load_test_dataset(ds_path, train_ds_ratio, val_ds_ratio, test_ds_ratio, False)
            mu = self.trainer.val_dataset.cropped_mu
            org_ber, model_ber, ber_improvement = self.trainer.compare_ber(verbose_level=verbose_level, _tqdm=_tqdm)

            if verbose_level > 0:
                print((
                    f'\n----------------- BERs for mu={mu} ----------------\n'
                    f'org_ber={org_ber}, model_ber={model_ber}, ber_improvement={ber_improvement}'
                    '\n\n'
                ))

            wandb.log({
                'org_ber': org_ber,
                'model_ber': model_ber,
                'ber_improvement': ber_improvement,
                'mu': mu
            })

    def plot_bers(self, _tqdm=None, verbose_level=1, num_x_per_folder=None):
        _ = self.trainer.compare_ber(verbose_level=verbose_level, _tqdm=_tqdm, num_x_per_folder=num_x_per_folder)

    def upload_bers_to_wandb(self, verbose=False):
        org_ber, model_ber, ber_improvement = self.trainer.compare_ber()
        self._init_wandb()
        wandb.log({'org_ber': org_ber, 'model_ber': model_ber, 'ber_improvement': ber_improvement})
        if verbose:
            print(f'org_ber={org_ber}, model_ber={model_ber}, ber_improvement={ber_improvement}')

    def plot_single_item(self, i):
        _ = self.trainer.test_single_item(i, plot=True)

    def plot_single_item_together(self, i, zm_indices=None):
        xi_axis = self.cs_in.channel_config.xi
        x, y, preds = self.trainer.test_single_item(i, plot=False)
        if zm_indices:
            x = x[zm_indices]
            y = y[zm_indices]
            preds = preds[zm_indices]
            xi_axis = xi_axis[zm_indices]
        io_type = self.cs_in.channel_config.io_type

        if io_type == 'c':
            Visualizer.data_trio_plot(
                np.real(y),
                np.real(x),
                np.real(preds),
                zm_indices,
                title=f'c[n] - after {self.trainer.train_state_vec.num_epochs} epochs',
                xlabel=r'n',
                function='stem',
                names=[rf'$c[n]$ [Tx]', rf'$\hat c[n]$ [Rx]', rf'$\widetilde c[n]$ [pred]'],
            )
        else:
            Visualizer.my_plot(
                xi_axis, np.abs(y),
                xi_axis, np.abs(x),
                xi_axis, np.abs(preds),
                legend=[
                    rf'${io_type}(\xi)$ [Tx]',
                    rf'$\hat {io_type}(\xi)$ [Rx]',
                    rf'$\widetilde {io_type}(\xi)$ [pred]'
                ],
                name=f'{io_type} - after {self.trainer.train_state_vec.num_epochs} epochs',
                xlabel=r'$\xi$',
        )

    def plot_stems(self, i, zm_indices=None):
        x, y, preds = self.trainer.test_single_item(i, plot=False)

        c_x = self.cs_in.io_to_c_constellation(x)
        c_y = self.cs_out.io_to_c_constellation(y)
        c_preds = self.cs_out.io_to_c_constellation(preds)

        Visualizer.data_trio_plot(
            np.real(c_y),
            np.real(c_x),
            np.real(c_preds),
            zm_indices,
            title=f'c[n] - after {self.trainer.train_state_vec.num_epochs} epochs',
            xlabel=r'n',
            function='stem',
            names=[rf'$c[n]$ [Tx]', rf'$\hat c[n]$ [Rx]', rf'$\widetilde c[n]$ [pred]'],
        )

    def plot_constelation(self, indices: list):
        m_qam = self.trainer.train_dataset.config.M_QAM

        x, y, preds = np.array([]), np.array([]), np.array([])
        for i in tqdm(indices):
            x_i, y_i, preds_i = self.trainer.test_single_item(i, plot=False)
            c_out_y = self.cs_in.io_to_c_constellation(y_i)
            c_out_x = self.cs_out.io_to_c_constellation(x_i)
            c_out_preds = self.cs_out.io_to_c_constellation(preds_i)

            y = np.concatenate((y, c_out_y))  # clean
            x = np.concatenate((x, c_out_x))  # dirty
            preds = np.concatenate((preds, c_out_preds))

        Visualizer.plot_constellation_map_with_3_data_vecs(x, y, preds, m_qam,
                                                           'constellation map',
                                                           ['dirty', 'clean', 'preds'])
        # Visualizer.plot_constellation_map_with_points(x9, m_qam, 'dirty signal')
        # Visualizer.plot_constellation_map_with_points(y9, m_qam, 'clean signal')
        # Visualizer.plot_constellation_map_with_points(pred9, m_qam, 'preds signal')

    def calc_norms(self, _tqdm=None, verbose_level=1, max_items=None):
        x_norms, y_norms, preds_norms = 0, 0, 0
        N = len(self.trainer.val_dataset)
        if max_items is not None:
            N = min(N, max_items)
        rng = range(N)
        if _tqdm is not None:
            rng = _tqdm(rng)
        for i in rng:
            x, y, preds = self.trainer.test_single_item(i, plot=False)
            x_power = np.sum(np.abs(x) ** 2)
            y_power = np.sum(np.abs(y) ** 2)
            pred_power = np.sum(np.abs(preds) ** 2)
            x_norms += x_power/N
            y_norms += y_power/N
            preds_norms += pred_power/N

        return x_norms, y_norms, preds_norms

    def upload_single_item_plots_to_wandb(self, i):
        x, y, preds = self.trainer.test_single_item(i, plot=False)
        indices = np.arange(len(x))

        self._init_wandb()
        for v, title in [(x, 'x (dirty)'), (y, 'y (clean)'), (preds, 'preds')]:
            wandb.log({title: wandb.plot.line_series(
                xs=indices,
                ys=[v.real, v.imag],
                keys=['real', 'imag'],
                title=title,
                xname="sample index")})

    def upload_stems_to_wandb(self, i):
        x, y, preds = self.trainer.test_single_item(i, plot=False)
        c_x = self.cs_in.io_to_c_constellation(x)
        c_y = self.cs_out.io_to_c_constellation(y)
        c_preds = self.cs_out.io_to_c_constellation(preds)
        indices = np.arange(len(c_x))

        self._init_wandb()
        for v, title in [(c_x, 'c_x (dirty)'), (c_y, 'c_y (clean)'), (c_preds, 'c_preds')]:
            wandb.log({title: wandb.plot.line_series(
                xs=indices,
                ys=[v.real, v.imag],
                keys=['real', 'imag'],
                title=title,
                xname="sample index")})

        zm_indices = range(0, min(50, len(c_x)))

        Visualizer.data_trio_plot(
            np.real(c_y),
            np.real(c_x),
            np.real(c_preds),
            zm_indices,
            title=f'c[n] - after {self.trainer.train_state_vec.num_epochs} epochs',
            xlabel=r'n',
            function='stem',
            names=[rf'$c[n]$ [Tx]', rf'$\hat c[n]$ [Rx]', rf'$\widetilde c[n]$ [pred]'],
        )

    def load_test_dataset(self, dataset_path: str, train_ds_ratio, val_ds_ratio, test_ds_ratio,
                          is_reset_wandb=True) -> None:
        _, _, test_dataset = get_datasets_set(dataset_path, DatasetNormal, train_ds_ratio, val_ds_ratio, test_ds_ratio)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        self.trainer.val_dataset = test_dataset
        self.trainer.val_dataloader = test_dataloader
        self.trainer.train_dataset.config = test_dataset.config
        self.cs_in, self.cs_out = create_channels(self.trainer.train_dataset.config)

        if is_reset_wandb:
            self._reset_wandb()
