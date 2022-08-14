import os

import numpy as np
import torch.optim
from matplotlib import pyplot as plt
from torch import nn, Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from tqdm import tqdm

from src.deep import file_methods
from src.general_methods.visualizer import Visualizer

num_epochs = 3
data_dir_path = '../apps/deep/data/qam1024_50x16/50_samples_mu=0.0005'


class mModel(nn.Module):

    def __init__(self, pretrained=False, weights_path=None) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(2, 4, kernel_size=1)
        self.prelu1 = nn.PReLU(4)
        # self.pool1 = nn.MaxPool1d(2, 2, ceil_mode=True)
        self.conv2 = nn.Conv1d(4, 8, kernel_size=1)
        self.prelu2 = nn.PReLU(8)
        self.conv3 = nn.Conv1d(8, 2, kernel_size=1)
        self.prelu3 = nn.PReLU(2)
        self.training = False

        if pretrained:
            assert weights_path is not None, "if setting pretrained, one must give path to weights"
            state_dict_path = os.path.join(os.path.dirname(__file__), weights_path)
            state_dict = torch.load(state_dict_path)
            self.load_state_dict(state_dict)

    def forward(self, x: Tensor):
        # x = [np.real(x), np.imag(x)]
        x = x.unsqueeze(2)
        x = self.conv1(x)
        x = self.prelu1(x)
        # x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = x.squeeze(2)
        # x = x[0] + x[1]*1j
        return x


class mDataSet(Dataset):
    def __init__(self, data_dir_path: str) -> None:
        super().__init__()
        self.data_dir_path = data_dir_path
        x = np.load(f'{self.data_dir_path}/{file_methods.x_file_name}')
        y = np.load(f'{self.data_dir_path}/{file_methods.y_file_name}')

        x = np.array([np.real(x), np.imag(x)])
        y = np.array([np.real(y), np.imag(y)])

        x = np.rollaxis(x, 0, 3)
        y = np.rollaxis(y, 0, 3)

        self.x: Tensor = torch.from_numpy(np.float32(x))
        self.y = torch.from_numpy(np.float32(y))

        assert len(self.x) == len(self.y), \
            "not the same amount of inputs x and outputs y!"

    def __getitem__(self, index) -> T_co:
        return self.x[index], self.y[index]

    def __len__(self) -> int:
        return len(self.x)


class Trainer:
    def __init__(self,
                 dataloader: Dataset,
                 model: nn.Module = None,
                 l_metric=None,
                 optim=None):
        self.dataloader = dataloader
        self.model = model or mModel()
        self.l_metric = l_metric or nn.MSELoss()
        self.optim = optim or torch.optim.Adam(model.parameters(), lr=1e-3)

        self.num_epoch_trained = 0
        self.loss_vec = []

    def train(self, num_epochs: int, mini_back_size: int = 500, verbose_level=0, tqdm=tqdm):
        # verbose_level:
        # 0 - 100% quiet, no prints at all
        # 1 - shows status bar
        # 2 - plots every first item on each epoch
        # 3 - plots every item

        mini_back_size = min(mini_back_size, self.dataloader.__len__)

        # train
        epoch_range = tqdm(range(num_epochs)) if verbose_level > 0 else range(num_epochs)
        for _ in epoch_range:
            running_loss = 0.0
            for i, (x, y) in enumerate(self.dataloader):
                pred = self.model(x)
                loss: Tensor = self.l_metric(y, pred)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                running_loss += loss

                if i % mini_back_size == mini_back_size - 1:
                    self.loss_vec.append(loss.item())
                    running_loss = 0.0

                    if verbose_level == 2:
                        title = f'epoch #{self.num_epoch_trained}, item #{0}: loss={loss.item()}'
                        x_np, y_np, pred_np = x.detach().numpy(), y.detach().numpy(), pred.detach().numpy()
                        Visualizer.data_trio_plot(x_np, y_np, pred_np, title=title)

                if verbose_level == 3:
                    title = f'epoch #{self.num_epoch_trained}, item #{i}: loss={loss.item()}'
                    x_np, y_np, pred_np = x.detach().numpy(), y.detach().numpy(), pred.detach().numpy()
                    Visualizer.data_trio_plot(x_np, y_np, pred_np, title=title)

            self.num_epoch_trained += 1

    def plot_loss_vec(self):
        Visualizer.plot_loss_vec(self.loss_vec)

    def test_single_item(self, i: int, title=None, verbose=False):
        # test the model once before training
        x, y = self.dataloader[i]
        if verbose: print(f'x.shape={x.shape}, y.shape={y.shape}')
        pred = self.model(x)
        x_np, y_np, pred_np = x.detach().numpy(), y.detach().numpy(), pred.detach().numpy()
        if verbose: print(f'x_np.shape={x_np.shape},y_np.shape={y_np.shape},pred_np.shape={pred_np.shape}')
        Visualizer.data_trio_plot(x_np, y_np, pred_np, title=title)


def test():
    model = mModel()
    dataloader = mDataSet(data_dir_path)
    l_metric = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        for x, y in dataloader:
            pred = model(x)
            loss: Tensor = l_metric(y, pred)
            optim.zero_grad()
            loss.backward()
            optim.step()
            print(loss.item())


if __name__ == '__main__':
    test()
