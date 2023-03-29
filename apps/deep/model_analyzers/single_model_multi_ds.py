from dataclasses import dataclass

import pyrallis
from tqdm import tqdm

from src.deep.model_analyzer_src import ModelAnalyzer
from src.deep.trainers import Trainer


@dataclass
class ModelAnalyzerConfig:
    model_path: str = None  # path to the model folder
    ds_init_path: str = './data/datasets/qam16_100000x20/'  # path to the main dataset folder
    train_ds_ratio: float = 0.5  # ratio of train to val dataset
    val_ds_ratio: float = 0.2  # ratio of train to val dataset (the reset is test)
    test_ds_ratio: float = 0.3  # ratio of train to val dataset (the reset is test)
    wandb_project: str = 'models_v3_debug'  # wandb project name
    run_name: str = None  # wandb run name

def main(config: ModelAnalyzerConfig):
    trainer = Trainer.load3(config.model_path)
    ma: ModelAnalyzer = ModelAnalyzer(trainer, config.run_name)

    ma.wandb_project = config.wandb_project
    # ma.run_name = 'model_skip_'

    ma.test_all_bers(config.ds_init_path,
                     config.train_ds_ratio, config.val_ds_ratio, config.test_ds_ratio,
                     tqdm, 1)


if __name__ == '__main__':
    conf = pyrallis.parse(config_class=ModelAnalyzerConfig)
    main(conf)
