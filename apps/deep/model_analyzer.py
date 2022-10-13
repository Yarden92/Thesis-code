from dataclasses import dataclass

import numpy as np
import pyrallis

import wandb
from src.deep.trainers import Trainer
from src.deep.standalone_methods import get_platform


@dataclass
class ModelAnalyzerConfig:
    path: str = None  # path to the model folder


def main(config: ModelAnalyzerConfig):
    trainer = Trainer.load3(config.path)
    ma = ModelAnalyzer(trainer)
    ma.upload_single_item_plots_to_wandb(i=0)


class ModelAnalyzer:
    def __init__(self, trainer: Trainer, run_name: str = None):
        self.trainer = trainer
        self.wandb_project = self.trainer.params['wandb_project']
        self.run_name = run_name
        self.model_name = self.trainer.model._get_name()
        self.ds_len = len(self.trainer.train_dataset) + len(self.trainer.val_dataset)
        self.mu = self.trainer.train_dataset.mu

    def _init_wandb(self):
        if wandb.run is not None:  # if already logged in
            return

        wandb.init(project=self.wandb_project, entity="yarden92", name=self.run_name,
                   tags=[f'mu={self.mu}', f'{get_platform()}', self.model_name, f'ds={self.ds_len}'],
                   reinit=True)
        wandb.config = {
            "learning_rate": self.trainer.params['lr'],
            "epochs": self.trainer.params['epochs'],
            "batch_size": self.trainer.params['batch_size']
        }

    def plot_bers(self, _tqdm=None, verbose_level=1):
        _ = self.trainer.compare_ber(verbose_level=verbose_level, _tqdm=_tqdm)

    def upload_bers_to_wandb(self):
        org_ber, model_ber, ber_improvement = self.trainer.compare_ber()
        self._init_wandb()
        wandb.log({'org_ber': org_ber, 'model_ber': model_ber, 'ber_improvement': ber_improvement})

    def plot_single_item(self, i):
        _ = self.trainer.test_single_item(i, plot=True)

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


if __name__ == '__main__':
    conf = pyrallis.parse(config_class=ModelAnalyzerConfig)
    main(conf)