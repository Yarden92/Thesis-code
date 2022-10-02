from dataclasses import dataclass

import pyrallis

from apps.deep.model_analyzer import ModelAnalyzer
from src.deep.data_loaders import DatasetNormal
from src.deep.ml_ops import Trainer
from src.deep.models import Paper1ModelWrapper


@dataclass
class Paper1Config:
    real_path: str = None  # path to the model folder
    imag_path: str = None  # path to the model folder
    init_path: str = None  # path to the model folder (without __real and __imag suffixes)


def main(config: Paper1Config):
    config = verify_config(config)
    trainer_real = Trainer.load3(config.real_path)
    trainer_imag = Trainer.load3(config.imag_path)

    analyze_models(trainer_real, trainer_imag)


def analyze_models(trainer_real, trainer_imag):
    model = Paper1ModelWrapper(trainer_real.model, trainer_imag.model)
    train_ds = DatasetNormal(data_dir_path=trainer_real.train_dataset.data_dir_path,
                             data_indices=trainer_real.train_dataset.data_indices)
    val_ds = DatasetNormal(data_dir_path=trainer_real.val_dataset.data_dir_path,
                           data_indices=trainer_real.val_dataset.data_indices)
    trainer = Trainer(
        train_dataset=train_ds,
        val_dataset=val_ds,
        model=model,
        device=trainer_real.device,
        batch_size=trainer_real.train_dataloader.batch_size,
        l_metric=trainer_real.l_metric,
        optim=trainer_real.optim,
        params=trainer_real.params)

    ma = ModelAnalyzer(trainer)
    ma.upload_bers_to_wandb()
    ma.upload_single_item_plots_to_wandb(i=0)


def verify_config(config: Paper1Config) -> Paper1Config:
    assert (config.real_path is not None and config.imag_path is not None) or config.init_path is not None, \
        "Either real_path and imag_path or init_path must be specified"
    if config.real_path is None:
        config.real_path = config.init_path + "__real"
    if config.imag_path is None:
        config.imag_path = config.init_path + "__imag"
    return config


if __name__ == '__main__':
    conf = pyrallis.parse(config_class=Paper1Config)
    main(conf)
