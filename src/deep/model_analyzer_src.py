import numpy as np

import wandb
from src.deep.standalone_methods import get_platform
from src.deep.trainers import Trainer
from src.general_methods.visualizer import Visualizer


class ModelAnalyzer:
    def __init__(self, trainer: Trainer, run_name: str = None):
        self.trainer = trainer
        self.wandb_project = self.trainer.config['wandb_project']
        self.run_name = run_name
        self.model_name = self.trainer.model._get_name()
        self.ds_len = len(self.trainer.train_dataset) + len(self.trainer.val_dataset)
        self.mu = self.trainer.train_dataset.cropped_mu

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

    def plot_bers(self, _tqdm=None, verbose_level=1):
        _ = self.trainer.compare_ber(verbose_level=verbose_level, _tqdm=_tqdm)

    def upload_bers_to_wandb(self):
        org_ber, model_ber, ber_improvement = self.trainer.compare_ber()
        self._init_wandb()
        wandb.log({'org_ber': org_ber, 'model_ber': model_ber, 'ber_improvement': ber_improvement})

    def plot_single_item(self, i):
        _ = self.trainer.test_single_item(i, plot=True)
        
    def plot_single_item_together(self, i, normalize=True):
        x,y,preds = self.trainer.test_single_item(i, plot=False)
        # delta = np.abs(x) - np.abs(y)
        indices = np.arange(len(x))
        Visualizer.my_plot(
            indices, np.abs(x),
            indices, np.abs(y),
            indices, np.abs(preds),
            # indices, np.abs(delta),
            legend=['x (dirty)', 'y (clean)', 'preds'],
            # legend=['x (dirty)', 'y (clean)', 'preds', 'delta'],
            name=f'after {self.trainer.train_state_vec.num_epochs} epochs'
        )
        
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
