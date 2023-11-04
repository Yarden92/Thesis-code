import json
from dataclasses import dataclass
from typing import List

import pyrallis
import torch
import wandb

from apps.deep.model_generators.single_model_generator import SingleModelTrainConfig, single_model_main
from src.deep import data_loaders
from src.deep.data_loaders import DatasetNormal
from src.deep.models.n_layers_model import NLayersModel
from src.deep.standalone_methods import get_platform


class MultiModelConfig:
    @dataclass
    class ModelConfig:
        n_layers: int = 3  # number of layers
        sizes: List[int] = None  # list of sizes of layers (need n_layers+1)
        kernel_sizes: List[int] = None  # kernel sizes of each layer (need n_layers)
        drop_rates: List[float] = None  # drop rates of each layer (need n_layers)
        activation_name: str = 'PReLU'  # activation function
        layer_name: str = 'Conv1d'  # layer type

    @dataclass
    class TrainConfig:
        models: str = '{}'  # string jsons of models '{"n_layers":1,"activation_name":"relu"};{"n_layers"...}'
        epochs: int = 10  # num of epochs
        lr: float = 1e-3  # learning rate
        batch_size: int = 1  # batch size
        input_data_path: str = './data/datasets/qam1024_150x5/150_samples_mu=0.001'  # path to data
        output_model_path: str = './data/test_models'  # path to save model
        device: str = 'auto'  # device to use
        train_val_ratio: float = 0.8  # train vs val ratio
        wandb_project: str = 'Thesis_model_scanning_test'  # wandb project name
        is_analyze_after: bool = False # if true, will analyze the model after training


    @classmethod
    def parse_models_config(cls, models_json: str):
        models_strings = models_json.split(';')
        for model_string in models_strings:
            dict = json.loads(model_string)
            yield cls.ModelConfig(**dict)

    @staticmethod
    def to_SingleModelTrainConfig(run_name: str, train_config: TrainConfig):
        return SingleModelTrainConfig(
            lr=train_config.lr,
            epochs=train_config.epochs,
            batch_size=train_config.batch_size,
            train_val_ratio=train_config.train_val_ratio,
            input_data_path=train_config.input_data_path,
            output_model_path=train_config.output_model_path,
            device=train_config.device,
            wandb_project=train_config.wandb_project,
            model_name=run_name,
            is_analyze_after=train_config.is_analyze_after
        )


def multiple_models_main(main_config: MultiModelConfig.TrainConfig):
    if main_config.device == 'auto': main_config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataset, val_dataset = data_loaders.get_datasets_set(main_config.input_data_path, DatasetNormal,
                                                                     train_val_ratio=main_config.train_val_ratio)
    mu = train_dataset.cropped_mu
    for sub_config in MultiModelConfig.parse_models_config(main_config.models):
        run_name = f'{sub_config.n_layers}_layers__{mu}_mu'
        print(f'running model {run_name}')
        model = NLayersModel(**sub_config.__dict__)
        wandb.init(project=main_config.wandb_project, entity="yarden92", name=run_name,
                   tags=[f'mu={mu}', f'{get_platform()}', f'{model.n_layers}_layers', f'ds={len(train_dataset)}'],
                   reinit=True)
        wandb.config = {
            "learning_rate": main_config.lr,
            "epochs": main_config.epochs,
            "batch_size": main_config.batch_size
        }
        # log model to wandb
        # wandb.log({"model": wandb.Histogram(np.array(model.state_dict()))})
        train_config = MultiModelConfig.to_SingleModelTrainConfig(run_name, main_config)
        single_model_main(model, train_config)

        # trainer = train_model(model=model, train_ds=train_dataset, val_ds=val_dataset, run_name=run_name,
        #                       lr=main_config.lr,
        #                       epochs=main_config.epochs, batch_size=main_config.batch_size, device=main_config.device,
        #                       output_model_path=main_config.output_model_path)
        #
        # ma = ModelAnalyzer(trainer)
        # ma.upload_bers_to_wandb()
        # ma.upload_single_item_plots_to_wandb(i=0)

    print('finished all models')





if __name__ == '__main__':
    config = pyrallis.parse(config_class=MultiModelConfig.TrainConfig)
    multiple_models_main(config)
