import os
from typing import Optional

import numpy as np
import torch.optim
import wandb
from torch import nn, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.deep import data_loaders
from src.deep.metrics import Metrics
from src.deep.models import SingleMuModel3Layers
from src.deep.data_loaders import OpticDataset
from src.deep.standalone_methods import GeneralMethods
from src.general_methods.visualizer import Visualizer


class Trainer:
    def __init__(self,
                 train_dataset: OpticDataset,
                 val_dataset: OpticDataset,
                 model: nn.Module = None,
                 device: str = 'cpu',
                 batch_size: int = 1,
                 l_metric=None,
                 optim=None,
                 params: dict = None):
        # self.train_dataset, self.val_dataset = split_ds(dataset, train_val_split)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        assert batch_size == 1, 'batch size other than 1 is not yet supported'
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.mean, self.std = GeneralMethods.calc_statistics_for_dataset(train_dataset)
        self.model = model or SingleMuModel3Layers()
        self.device = device
        self.l_metric = l_metric or nn.MSELoss()
        self.optim = optim or torch.optim.Adam(model.parameters(), lr=1e-3)
        self.params = params or {}

        self.attach_to_device(self.device)
        # move all to device

        self.val_dataloader = self.val_dataloader  # WIP

        self.train_state_vec = TrainStateVector()

    def attach_to_device(self, device):
        self.model = self.model.to(device)
        self.l_metric = self.l_metric.to(device)
        self.train_dataloader = self.train_dataloader  # WIP
        self.val_dataloader = self.val_dataloader  # WIP

    def train(self, num_epochs: int, mini_batch_size: int = 500, _tqdm=tqdm):
        # to run quietly set _tqdm to None

        mini_batch_size = min(mini_batch_size, len(self.train_dataset), len(self.val_dataset))

        # train
        epoch_range = _tqdm(range(num_epochs)) if _tqdm else range(num_epochs)
        for epoch in epoch_range:
            train_loss_i = self.epoch_step(self.train_dataloader, self._step_train)
            val_loss_i = self.epoch_step(self.val_dataloader, self._step_val)

            wandb.log({"train_loss": train_loss_i, "val_loss": val_loss_i, "epoch": epoch})
            wandb.log({"val_loss": val_loss_i})
            self.train_state_vec.add(self.model, train_loss_i, val_loss_i)

    def epoch_step(self, dataloader, step):
        final_loss = 0
        for batch in dataloader:
            x, y = batch
            x, y = x[0].to(self.device), y[0].to(self.device)
            loss, pred = step(x, y)
            # print(f'loss={loss.item()}')
            final_loss += loss.item()/len(dataloader)

        return final_loss

    def _step_train(self, x, y):
        pred = self.model(x)
        loss: Tensor = self.l_metric(y, pred)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss, pred

    def _step_val(self, x, y):
        pred = self.model(x)
        loss: Tensor = self.l_metric(y, pred)
        return loss, pred

    def save3(self, dir_path: str = 'saved_models'):
        # create dir if it doesn't exist
        model_name = self.model.__class__.__name__
        if model_name == 'NLayersModel': model_name = f'{self.model.n_layers}_layers_model'
        ds_size = len(self.train_dataset)
        n_epochs = self.train_state_vec.num_epochs
        mu = self.train_dataset.mu
        sub_dir_path = f'{dir_path}/mu-{mu}__{ds_size}ds__{model_name}__{n_epochs}epochs'
        os.makedirs(sub_dir_path, exist_ok=True)

        # save trainer to the same dir
        torch.save(self, sub_dir_path + '/trainer.pt')

    @classmethod
    def load3(cls, dir_path: str = 'saved_models'):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.load(dir_path + '/trainer.pt', map_location=device)

    def plot_loss_vec(self):
        Visualizer.plot_loss_vec(self.train_state_vec.train_loss_vec, self.train_state_vec.val_loss_vec)

    def test_single_item(self, i: int, title=None, verbose=False):
        # test the model once before training
        x, y = self.train_dataset[i]
        if verbose: print(f'x.shape={x.shape}, y.shape={y.shape}')
        pred = self.model(x)
        x_np, y_np, pred_np = x.detach().numpy(), y.detach().numpy(), pred.detach().numpy()
        if verbose: print(f'x_np.shape={x_np.shape},y_np.shape={y_np.shape},pred_np.shape={pred_np.shape}')
        title = title or f'after {self.train_state_vec.num_epochs} epochs'
        Visualizer.data_trio_plot(x_np, y_np, pred_np, title=title)
        if verbose: print(
            f'x power={np.mean(x_np ** 2)}\ny power={np.mean(y_np ** 2)}\npred power={np.mean(pred_np ** 2):.2f}')
        # if verbose: print(f'mu(x)={np.mean(x_np)}\nstd(x)={np.std(x_np)}\nmu(y)={np.mean(y_np)}\nstd(y)={np.std(y_np)}\nmu(pred)={np.mean(pred_np)}\nstd(pred)={np.std(pred_np)}')

        return x_np, y_np, pred_np

    def compare_ber(self, verbose=False, tqdm=None, num_x_per_folder=None):
        # compare the original raw BER with the equalized model's BER

        ber_vec, num_errors = Metrics.calc_ber_from_dataset(self.val_dataset, verbose, tqdm, num_x_per_folder)
        org_ber = np.mean(ber_vec)
        print(f'the original avg ber is {org_ber}')

        # calc ber after training
        ber_vec, num_errors = Metrics.calc_ber_from_model(self.val_dataset, self.model, verbose, tqdm, num_x_per_folder)
        model_ber = np.mean(ber_vec)
        print(f'the trained avg ber is {model_ber}')

        ber_improvement = (org_ber - model_ber)/org_ber
        print(f'the ber improvement is {ber_improvement*100:.2f}%')

        return org_ber, model_ber, ber_improvement

    def fix_datasets_paths(self, dataset_path: str = '../../data/datasets', verbose=True):
        for ds in [self.train_dataset, self.val_dataset]:
            extension = ds.data_dir_path.split('/data/datasets')[1]
            ds.data_dir_path = os.path.abspath(f'{dataset_path}{extension}')
            if verbose: print(f'updating path to:\n\t{ds.data_dir_path}')


class TrainStateVector:
    # a single state of the model during training process
    # includes the weights, epoch_index, loss, and the ber
    # for easy loading the optimal model of the entire training process
    def __init__(self):
        self.models = []
        self.num_epochs = 0
        self.train_loss_vec = []
        self.val_loss_vec = []

    def add(self, model: nn.Module, train_loss: float, val_loss: float):
        self.models.append(model)  # TODO: this is not good, we need to take excplicitly the weights not the pointer.
        self.num_epochs += 1
        self.train_loss_vec.append(train_loss)
        self.val_loss_vec.append(val_loss)
