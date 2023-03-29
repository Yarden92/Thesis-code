from dataclasses import dataclass

import pyrallis

from src.deep.model_analyzer_src import ModelAnalyzer
from src.deep.trainers import Trainer


@dataclass
class ModelAnalyzerConfig:
    path: str = None  # path to the model folder
    dataset_path: str = None  # path to the dataset folder
    train_ds_ratio: float = 0.5  # ratio of train to val dataset
    val_ds_ratio: float = 0.2  # ratio of train to val dataset (the reset is test)


def main(config: ModelAnalyzerConfig):
    trainer = Trainer.load3(config.path)
    ma: ModelAnalyzer = ModelAnalyzer(trainer)
    
    if config.dataset_path is not None:
        ma.load_test_dataset(config.dataset_path, config.train_ds_ratio, config.val_ds_ratio)
        
    ma.upload_single_item_plots_to_wandb(i=0)
    # ma.plot_bers(_tqdm=tqdm, verbose_level=1)
    ma.plot_constelation(i=0)
    


if __name__ == '__main__':
    conf = pyrallis.parse(config_class=ModelAnalyzerConfig)
    main(conf)
