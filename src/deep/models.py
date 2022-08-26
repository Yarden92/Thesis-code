import os

import torch.optim
from torch import nn, Tensor

from src.deep.data_loaders import SingleMuDataSet

num_epochs = 3


class MultiMuModel(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO TBD
        raise NotImplementedError


class SingleMuModel5Layers(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(2, 4, kernel_size=1)
        self.prelu1 = nn.PReLU(4)
        # self.pool1 = nn.MaxPool1d(2, 2, ceil_mode=True)
        self.conv2 = nn.Conv1d(4, 8, kernel_size=1)
        self.prelu2 = nn.PReLU(8)
        self.conv3 = nn.Conv1d(8, 2, kernel_size=1)
        self.prelu3 = nn.PReLU(2)
        self.training = False


class SingleMuModel3Layers(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(2, 4, kernel_size=1)
        self.prelu1 = nn.PReLU(4)
        # self.pool1 = nn.MaxPool1d(2, 2, ceil_mode=True)
        self.conv2 = nn.Conv1d(4, 8, kernel_size=1)
        self.prelu2 = nn.PReLU(8)
        self.conv3 = nn.Conv1d(8, 2, kernel_size=1)
        self.prelu3 = nn.PReLU(2)
        self.training = False

    @classmethod
    def load_pretrained_weights(cls, weights_path):
        # state_dict_path = os.path.join(os.path.dirname(__file__), weights_path)
        state_dict = torch.load(weights_path)
        return cls().load_state_dict(state_dict)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

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


def test():
    model = SingleMuModel3Layers()
    data_dir_path = '../apps/deep/data/qam1024_50x16/50_samples_mu=0.0005'
    dataset = SingleMuDataSet(data_dir_path)
    l_metric = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        for x, y in dataset:
            pred = model(x)
            loss: Tensor = l_metric(y, pred)
            optim.zero_grad()
            loss.backward()
            optim.step()
            print(loss.item())


# def test2():
#     # define parameters
#     num_epochs = 3
#     print(os.getcwd())
#     data_dir_path = '../apps/deep/data/qam1024_50x16/50_samples_mu=0.07'
#     l_metric = nn.MSELoss()  # or L1Loss
#
#     # generate model and _stuffs_
#     model = SingleMuModel()
#     dataloader = SingleMuDataSet(data_dir_path)
#     optim = torch.optim.Adam(model.parameters(), lr=1e-3)
#
#     trainer = Trainer(dataloader, model, l_metric, optim)
#     trainer.test_single_item(0, verbose=True)
#
#     trainer.train(num_epochs, verbose_level=2, tqdm=tqdm)
#
#     trainer.plot_loss_vec()


if __name__ == '__main__':
    pass
