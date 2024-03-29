import json
import os
from ssl import TLSVersion
import warnings

import numpy as np
import torch.cuda
import torch.optim
import wandb
from torch import nn, Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.deep.data_loaders import OpticDataset, SeparatedRealImagDataset
from src.deep.metrics import Metrics
from src.deep.models import *
from src.deep.standalone_methods import GeneralMethods
from src.general_methods.visualizer import Visualizer
from torchsummary import summary

import torch
import torchviz

LOG_INTERVAL = 2


class Trainer:
    def __init__(self, train_dataset: OpticDataset, val_dataset: OpticDataset, model: nn.Module = None,
                 device: str = "cpu", batch_size: int = 1,
                 l_metric=None, optim=None, scheduler=None, config: dict = None,
                 lambda_reg: float = 0.001,
                 ):
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
        self.lambda_reg = lambda_reg

    def attach_to_device(self, device):
        self.device = device
        self.model = self.model.to(device)
        self.l_metric = self.l_metric.to(device)
        self.train_dataloader = self.train_dataloader  # WIP
        self.val_dataloader = self.val_dataloader  # WIP

    def train(self, num_epochs: int, mini_batch_size: int = 500, _tqdm=tqdm):
        # to run quietly set _tqdm to None

        mini_batch_size = min(mini_batch_size, len(self.train_dataset), len(self.val_dataset))

        # train
        epoch_range = _tqdm(range(num_epochs), 'looping on epochs') if _tqdm else range(num_epochs)
        for epoch in epoch_range:
            self.epoch_step(self.val_dataloader, self._step_validate, name="val", epoch=epoch, _tqdm=_tqdm)
            self.epoch_step(self.train_dataloader, self._step_train, name="train", epoch=epoch, _tqdm=_tqdm)
            if self.scheduler:
                self.scheduler.step()

            self.train_state_vec.add(self.model, 0, 0)

    def epoch_step(self, dataloader, step, name: str, epoch: int, _tqdm) -> None:
        rng = enumerate(dataloader)
        if _tqdm:
            rng = _tqdm(rng, total=len(dataloader), leave=False, desc=f"{name} epoch {epoch}")
        for i, batch in rng:
            Rx, Tx = batch
            loss, pred_Tx = step(Rx, Tx)
            if i % LOG_INTERVAL == 0:
                iteration_num = epoch * len(dataloader) + i
                wandb.log({f"{name}_loss": loss.item(), f"{name}_iteration": iteration_num})
                if name == "train":
                    wandb.log({f'lr': self.optim.param_groups[0]['lr'], f"{name}_iteration": iteration_num})

    def _step_train(self, Rx, Tx):
        loss, pred_Tx = self._step_validate(Rx, Tx)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss, pred_Tx

    def _step_validate(self, Rx, Tx):
        Rx, Tx = Rx.to(self.device), Tx.to(self.device)
        pred_Tx = self.model(Rx)
        loss = self._get_loss(Rx, Tx, pred_Tx)

        return loss, pred_Tx

    def _get_loss(self, Rx, Tx, pred_Tx) -> Tensor:

        loss: Tensor = self.l_metric(Tx, pred_Tx)

        # this stage was added by gpt:
        # Regularization term
        reg_loss = 0.0
        for param in self.model.parameters():
            reg_loss += torch.norm(param, p=2)  # L2 regularization

        # Add regularization term to the loss
        loss += self.lambda_reg * reg_loss

        return loss

    def save3(self, dir_path: str = "saved_models", model_name: str = "unnamed_model", endings: str = ""):
        # create dir if it doesn't exist
        # model_name = self.model.__class__.__name__
        # if model_name == "NLayersModel":
        #     model_name = f"{self.model.n_layers}_layers_model"
        unique_folder_name = self._get_unique_folder_name(dir_path, model_name)
        ds_size = len(self.train_dataset) + len(self.val_dataset)
        n_epochs = self.train_state_vec.num_epochs
        mu = self.train_dataset.cropped_mu
        # sub_dir_path = f"{dir_path}/mu-{mu}__{ds_size}ds__{model_name}__{n_epochs}epochs{endings}"
        sub_dir_path = os.path.join(dir_path, unique_folder_name)
        os.makedirs(sub_dir_path, exist_ok=True)
        print(f'saving model data to: {sub_dir_path}')

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
        trainer: Trainer = torch.load(dir_path + "/trainer.pt", map_location=device)
        model_state_dict = torch.load(dir_path + "/model_state_dict.pt", map_location=device)
        # load model class name
        with open(dir_path + "/model_class_name.txt", "r") as f:
            model_class_name = f.read()
        model_class = eval(model_class_name)
        trainer.model = model_class()
        trainer.model.load_state_dict(model_state_dict)
        trainer.attach_to_device(device)
        # trainer.model.to(device)

        # # fix h -> dz
        # if 'ssf' in trainer.train_dataset.config and 'h' in trainer.train_dataset.config['ssf']:
        #     trainer.train_dataset.config['ssf']['dz'] = trainer.train_dataset.config['ssf'].pop('h')

        return trainer

    def print_summary(self):
        rx, _, _ = self.get_single_item(0)
        shape = rx.shape
        summary(self.model, shape, device="cuda")

    def plot_architecture(self, path: str = "model_architecture", format: str = "png"):
        x, _ = self.val_dataset[0]
        pred = self.model(x.to(self.device))
        torchviz.make_dot(pred, params=dict(self.model.named_parameters())).render(path, format=format, cleanup=True)
        print(f"saved model architecture to {path}.{format}")

    def plot_loss_vec(self):
        Visualizer.plot_loss_vec(self.train_state_vec.train_loss_vec, self.train_state_vec.val_loss_vec)

    
    
    def get_single_item(self, i: int, format: str = "numpy"):
        # test the model once before training
        Rx, Tx = self.val_dataset[i]

        Rx = Rx.to(self.device)

        pred_Tx = self.model(Rx)
        # x_np, y_np, pred_np = x.detach().numpy(), y.detach().numpy(), pred.detach().numpy()
        Rx_np, Tx_np, pred_Tx_np = [GeneralMethods.torch_to_complex_numpy(t) for t in [Rx, Tx, pred_Tx]]

        if format == "torch":
            return Rx, Tx, pred_Tx
        elif format == "numpy":
            return Rx_np, Tx_np, pred_Tx_np
        
            
    def test_single_item(self, i: int, title=None, verbose=False):
        Rx_np, Tx_np, pred_Tx_np = self.get_single_item(i)
        title = title or f"after {self.train_state_vec.num_epochs} epochs"
        Visualizer.data_trio_plot(Rx_np, Tx_np, pred_Tx_np, title=title)
        if verbose:
            print(f"Rx_np.shape={Rx_np.shape},Tx_np.shape={Tx_np.shape},pred_Tx_np.shape={pred_Tx_np.shape}")
            print((f"Rx power={np.mean(Rx_np ** 2)}\n"
                   f"Tx power={np.mean(Tx_np ** 2)}\n"
                   f"pred_Tx power={np.mean(pred_Tx_np ** 2):.2f}"))

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

        ber_improvement = (org_ber - model_ber) / org_ber if org_ber != 0 else 0
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

    def _get_unique_folder_name(self, dir_path: str, model_name: str) -> str:
        # look at dir, for all folders named model_name and return new name <model_name>_v$ where $ is latest
        count = 1 + sum(1 for folder in os.listdir(dir_path) if model_name in folder)
        unique_folder_name = f"{model_name}_v{count}"

        # just to make sure that its actually unique:
        while os.path.exists(os.path.join(dir_path, unique_folder_name)):
            warnings.warn(f"model: [{model_name}] is messy, keep it clean with no higher version than {count}")
            unique_folder_name += "_1"

        return unique_folder_name


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

# TLV 7:00 -> ATHENS 9:30
