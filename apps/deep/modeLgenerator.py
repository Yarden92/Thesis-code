import torch
import wandb
from torch import nn
from tqdm import tqdm

from src.deep import data_loaders
from src.deep.data_loaders import SingleMuDataSet
from src.deep.ml_ops import Trainer
from src.deep.models import SingleMuModel3Layers


class Config:
    run_name = "single_mu_model_3_layers"
    epochs = 3
    lr = 1e-3
    batch_size = 128
    train_val_ratio = 0.8
    input_data_path = '../../data/datasets/qam1024_10x3/10_samples_mu=0.001'
    output_model_path = '../../data/saved_models'


def main():
    # config
    wandb.init(project="Thesis", entity="yarden92", name=Config.run_name)
    wandb.config = {
        "learning_rate": Config.lr,
        "epochs": Config.epochs,
        "batch_size": Config.batch_size
    }

    l_metric = nn.MSELoss()  # or L1Loss
    model = SingleMuModel3Layers()
    train_dataset, val_dataset = data_loaders.get_train_val_datasets(Config.input_data_path, SingleMuDataSet,
                                                                     train_val_ratio=Config.train_val_ratio)

    optim = torch.optim.Adam(model.parameters(), lr=Config.lr)
    trainer = Trainer(train_dataset=train_dataset, val_dataset=val_dataset, model=model, l_metric=l_metric, optim=optim)

    trainer.train(num_epochs=Config.epochs, verbose_level=1, _tqdm=tqdm)
    trainer.save_model(Config.output_model_path)


    print('finished training')


if __name__ == '__main__':
    main()
