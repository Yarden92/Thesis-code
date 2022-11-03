from dataclasses import dataclass

import pyrallis

from src.deep.model_analyzer_src import ModelAnalyzer
from src.deep.trainers import Trainer


@dataclass
class ModelAnalyzerConfig:
    path: str = None  # path to the model folder


def main(config: ModelAnalyzerConfig):
    trainer = Trainer.load3(config.path)
    ma = ModelAnalyzer(trainer)
    ma.upload_single_item_plots_to_wandb(i=0)


if __name__ == '__main__':
    conf = pyrallis.parse(config_class=ModelAnalyzerConfig)
    main(conf)
