import json
import os

import numpy as np
import torch.cuda
import torch.optim
import wandb
from torch import nn, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.deep.data_loaders import OpticDataset, SeparatedRealImagDataset
from src.deep.metrics import Metrics
from src.deep.models import *
from src.deep.standalone_methods import GeneralMethods
from src.general_methods.visualizer import Visualizer

import torch

LOG_INTERVAL = 100


class Trainer:
    def __init__(self, train_dataset: OpticDataset, val_dataset: OpticDataset, model: nn.Module = None,
                 device: str = "cpu", batch_size: int = 1,
                 l_metric=None, optim=None, scheduler=None, config: dict = None):
        # self.train_dataset, self.val_dataset = split_ds(dataset, train_val_split)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        assert batch_size == 1, "batch size other than 1 is not yet supported"
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.mean, self.std = GeneralMethods.calc_statistics_for_dataset(train_dataset)
        self.model = model or SingleMuModel3Layers()
        self.device = device
        self.l_metric = l_metric or nn.MSELoss()
        self.optim = optim or torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        self.scheduler = scheduler
        self.config = config or {}

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
            self.epoch_step(self.train_dataloader, self._step_train, name="train", epoch=epoch, _tqdm=_tqdm)
            if self.scheduler:
                self.scheduler.step()
            self.epoch_step(self.val_dataloader, self._step_validate, name="val", epoch=epoch, _tqdm=_tqdm)

            self.train_state_vec.add(self.model, 0, 0)

    def epoch_step(self, dataloader, step, name: str, epoch: int, _tqdm) -> None:
        rng = enumerate(dataloader)
        if _tqdm:
            rng = _tqdm(rng, total=len(dataloader),leave=False)
        for i, batch in rng:
            x, y = batch
            loss, pred = step(x, y)
            if i % LOG_INTERVAL == 0:
                iteration_num = epoch * len(dataloader) + i
                wandb.log({f"{name}_loss": loss.item(), "iteration": iteration_num})
                if name == "train":
                    wandb.log({f'lr': self.optim.param_groups[0]['lr'], "iteration": iteration_num})

    def _step_train(self, x, y):
        loss, pred = self._step_validate(x, y)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss, pred

    def _step_validate(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        pred = self.model(x)
        loss: Tensor = self.l_metric(y, pred)
        return loss, pred

    def save3(self, dir_path: str = "saved_models", endings: str = ""):
        # create dir if it doesn't exist
        model_name = self.model.__class__.__name__
        if model_name == "NLayersModel":
            model_name = f"{self.model.n_layers}_layers_model"
        ds_size = len(self.train_dataset) + len(self.val_dataset)
        n_epochs = self.train_state_vec.num_epochs
        mu = self.train_dataset.cropped_mu
        sub_dir_path = f"{dir_path}/mu-{mu}__{ds_size}ds__{model_name}__{n_epochs}epochs{endings}"
        os.makedirs(sub_dir_path, exist_ok=True)

        # save trainer to the same dir
        torch.save(self.model.state_dict(), sub_dir_path + "/model_state_dict.pt")
        # save model class name
        with open(sub_dir_path + "/model_class_name.txt", "w") as f:
            f.write(self.model.__class__.__name__)
        model_bkp = self.model
        self.model = None
        torch.save(self, sub_dir_path + "/trainer.pt")
        self.model = model_bkp

        # save params to a readable json file (for manual inspection)
        with open(sub_dir_path + "/params.json", "w") as f:
            json.dump(self.config, f, indent=4)

    @classmethod
    def load3(cls, dir_path: str = "saved_models") -> "Trainer":
        # load a pretrained model trainer from a dir
        abs_path = os.path.abspath(dir_path)
        assert dir_path and os.path.exists(abs_path), f"cant find the following path:\n\t{abs_path}"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        trainer = torch.load(dir_path + "/trainer.pt", map_location=device)
        model_state_dict = torch.load(dir_path + "/model_state_dict.pt", map_location=device)
        # load model class name
        with open(dir_path + "/model_class_name.txt", "r") as f:
            model_class_name = f.read()
        model_class = eval(model_class_name)
        trainer.model = model_class()
        trainer.model.load_state_dict(model_state_dict)
        trainer.model.to(device)

        return trainer

    def plot_loss_vec(self):
        Visualizer.plot_loss_vec(self.train_state_vec.train_loss_vec, self.train_state_vec.val_loss_vec)

    def test_single_item(self, i: int, title=None, verbose=False, plot=True):
        # test the model once before training
        x, y = self.val_dataset[i]
        if verbose:
            print(f"x.shape={x.shape}, y.shape={y.shape}")
        x = x.to(self.device)

        pred = self.model(x)
        # x_np, y_np, pred_np = x.detach().numpy(), y.detach().numpy(), pred.detach().numpy()
        x_np, y_np, pred_np = [GeneralMethods.torch_to_complex_numpy(t) for t in [x, y, pred]]
        if verbose:
            print(f"x_np.shape={x_np.shape},y_np.shape={y_np.shape},pred_np.shape={pred_np.shape}")
        if plot:
            title = title or f"after {self.train_state_vec.num_epochs} epochs"
            Visualizer.data_trio_plot(x_np, y_np, pred_np, title=title)
        if verbose:
            print((f"x power={np.mean(x_np ** 2)}\n"
                   f"y power={np.mean(y_np ** 2)}\n"
                   f"pred power={np.mean(pred_np ** 2):.2f}"))

        return x_np, y_np, pred_np

    def compare_ber(self, verbose_level: int = 1, _tqdm=None, num_x_per_folder=None):
        # compare the original raw BER with the equalized model's BER

        ber_vec, num_errors = Metrics.calc_ber_from_dataset(
            self.val_dataset, verbose_level >= 2, _tqdm, num_x_per_folder)
        org_ber = np.mean(ber_vec)
        if verbose_level >= 1:
            print(f"the original avg ber (of validation set) is {org_ber}")

        # calc ber after training
        ber_vec, num_errors = Metrics.calc_ber_from_model(
            self.val_dataset, self.model, verbose_level >= 2, _tqdm, num_x_per_folder, self.device,)
        model_ber = np.mean(ber_vec)
        if verbose_level >= 1:
            print(f"the trained avg ber (of validation set) is {model_ber}")

        ber_improvement = (org_ber - model_ber) / org_ber
        if verbose_level >= 1:
            print(f"the ber improvement is {ber_improvement*100:.2f}%")

        return org_ber, model_ber, ber_improvement

    def fix_datasets_paths(self, new_dataset_path: str = "../../data/datasets", verbose=True):
        # fixing paths to ____
        #
        for ds in [self.train_dataset, self.val_dataset]:
            extension = ds.data_dir_path.split("/data/datasets")[1]
            ds.data_dir_path = os.path.abspath(f"{new_dataset_path}{extension}")
            if verbose:
                print(f"updating path to:\n\t{ds.data_dir_path}")


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
        self.models.append(model.state_dict())
        self.num_epochs += 1
        self.train_loss_vec.append(train_loss)
        self.val_loss_vec.append(val_loss)


class DoubleTrainer:
    def __init__(self, train_dataset: SeparatedRealImagDataset, val_dataset: SeparatedRealImagDataset,
                 model_real: nn.Module = None, model_imag: nn.Module = None,
                 device: str = "cpu", batch_size: int = 1, params: dict = None):

        l_metric_real = nn.MSELoss()  # or L1Loss
        l_metric_imag = nn.MSELoss()  # or L1Loss

        optim_real = torch.optim.Adam(model_real.parameters(), lr=params["lr"])
        optim_imag = torch.optim.Adam(model_imag.parameters(), lr=params["lr"])

        if device == "cuda":
            device1, device2 = "cuda:0", "cuda:1"
        else:
            device1, device2 = "cpu", "cpu"

        self.trainer_real = Trainer(train_dataset, val_dataset, model_real, device1,
                                    batch_size, l_metric_real, optim_real, params)
        self.trainer_imag = Trainer(train_dataset, val_dataset, model_imag, device2,
                                    batch_size, l_metric_imag, optim_imag, params)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train(self, num_epochs: int, mini_batch_size: int = 500, _tqdm=tqdm):
        print("training real part...")
        self.train_dataset.set_is_real(True), self.val_dataset.set_is_real(True)
        self.trainer_real.train(num_epochs, mini_batch_size, _tqdm)

        print("training imaginary part...")
        self.train_dataset.set_is_real(False), self.val_dataset.set_is_real(False)
        self.trainer_imag.train(num_epochs, mini_batch_size, _tqdm)

    def save3(self, dir_path: str = "saved_models", endings: str = ""):
        self.trainer_real.save3(dir_path, "_real" + endings)
        self.trainer_imag.save3(dir_path, "_imag" + endings)

    @classmethod
    def load3(cls, dir_path: str = "saved_models") -> "Trainer":
        return super().load3(dir_path)

    def test_single_item(self, i: int, title=None, verbose=False, plot=True):
        return super().test_single_item(i, title, verbose, plot)
