import os

import numpy as np
import torch.optim
import wandb
from torch import nn, Tensor
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
                 l_metric=None,
                 optim=None,
                 params: dict = None):
        # self.train_dataset, self.val_dataset = split_ds(dataset, train_val_split)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.mean, self.std = GeneralMethods.calc_statistics_for_dataset(train_dataset)
        self.model = model or SingleMuModel3Layers()
        self.device = device
        self.l_metric = l_metric or nn.MSELoss()
        self.optim = optim or torch.optim.Adam(model.parameters(), lr=1e-3)
        self.params = params or {}

        self.model = self.model.to(self.device)
        self.l_metric.to(self.device)
        self.train_dataset = self.train_dataset.to(self.device)
        self.val_dataset = self.val_dataset.to(self.device)

        # self.num_epoch_trained = 0
        # self.loss_vec = []
        self.train_state_vec = TrainStateVector()

    def train(self, num_epochs: int, mini_batch_size: int = 500, verbose_level=0, _tqdm=tqdm):
        # verbose_level:
        # 0 - 100% quiet, no prints at all
        # 1 - shows status bar
        # 2 - plots every first item on each epoch
        # 3 - plots every item

        mini_batch_size = min(mini_batch_size, len(self.train_dataset), len(self.val_dataset))

        # train
        epoch_range = _tqdm(range(num_epochs)) if _tqdm else range(num_epochs)
        for _ in epoch_range:
            train_loss_i = self.epoch_step(mini_batch_size, self.train_dataset, self._step_train)
            val_loss_i = self.epoch_step(mini_batch_size, self.val_dataset, self._step_val)

            wandb.log({"train_loss": train_loss_i})
            wandb.log({"val_loss": val_loss_i})
            self.train_state_vec.add(self.model, train_loss_i, val_loss_i)

    def epoch_step(self, mini_batch_size, dataset, step):
        # dataset = self.train_dataset if is_train else self.val_dataset
        # step = self._step_train if is_train else self._step_val

        # OPTION 1
        # mini_batches = self.split_to_minibatches(dataset, mini_batch_size)
        # running_loss_vec = np.zeros(len(mini_batches))
        # for minibatch in mini_batches:
        #     for i, (x, y) in enumerate(minibatch):
        #         loss, pred = step(x, y)
        #         running_loss_vec[i] += loss.item()/mini_batch_size
        # final_loss = np.mean(running_loss_vec)

        # OPTION 2
        # running_loss_vec = np.zeros(len(dataset))
        final_loss = 0
        for i, (x, y) in enumerate(dataset):
            # x1,y1 = Metrics.normalize_xy(x, y, self.mu, self.std)
            loss, pred = step(x, y)
            # running_loss_vec[i] += loss.item()/len(dataset)
            # running_loss_vec[i] += loss.item()
            final_loss += loss.item()/len(dataset)

        return final_loss

    def split_to_minibatches(self, dataset, mini_batch_size):
        # split to minibatches
        mini_batches = []
        dataset_type = type(dataset)
        for i in range(0, len(dataset), mini_batch_size):
            mini_batches.append(dataset[i:i + mini_batch_size])
        return mini_batches

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

    def save_model(self, dir_path: str = 'saved_models', verbose=True):
        # create dir if it doesn't exist
        model_name = self.model.__class__.__name__
        ds_size = len(self.train_dataset)
        n_epochs = self.train_state_vec.num_epochs
        mu = self.train_dataset.mu
        sub_dir_path = f'{dir_path}/{model_name}_ds-{ds_size}_epochs-{n_epochs}_mu-{mu}'
        os.makedirs(sub_dir_path, exist_ok=True)

        model_path = sub_dir_path + '/model.pt'
        torch.save(self.model.state_dict(), model_path)

        # save dataloader's conf to the same dir
        dataset_conf_path = sub_dir_path + '/dataset_conf.json'
        data_loaders.save_conf(dataset_conf_path, self.train_dataset.config)

        # save trainer's conf to the same dir
        trainer_conf_path = sub_dir_path + '/trainer_conf.json'
        data_loaders.save_conf(trainer_conf_path, self.params)

        # save trainer
        trainer_path = sub_dir_path + '/trainer.pt'
        torch.save(self, trainer_path)

        if verbose:
            print(f'Model saved to {sub_dir_path}')

    @classmethod
    def load_trainer_from_file(cls, folder_path: str) -> 'Trainer':
        trainer_path = folder_path + '/trainer.pt'
        trainer: Trainer = torch.load(trainer_path)
        return trainer

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

    def compare_ber(self, verbose=False, tqdm=None):
        # compare the original raw BER with the equalized model's BER

        # all_x, all_y = self.dataloader[i]

        ber_vec, num_errors = Metrics.calc_ber_from_dataset(self.val_dataset, verbose, tqdm)
        print(f'the original avg ber is {np.mean(ber_vec)}')

        # calc ber after training
        ber_vec, num_errors = Metrics.calc_ber_from_model(self.val_dataset, self.model, verbose, tqdm)
        print(f'the trained avg ber is {np.mean(ber_vec)}')

    def fix_datasets_paths(self, dataset_path: str = '../../data/datasets'):
        for ds in [self.train_dataset, self.val_dataset]:
            extension = ds.data_dir_path.split('/data/datasets')[1]
            ds.data_dir_path = os.path.abspath(f'{dataset_path}{extension}')
            print(f'updating path to:\n\t{ds.data_dir_path}')


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

