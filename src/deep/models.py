import os
from typing import Optional

import numpy as np
import torch.optim
from torch import nn, Tensor

from src.deep.data_loaders import SingleMuDataSet

num_epochs = 3
MAX_LAYER_SIZE = 64  # maximum number of channels in each layer

activations = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'leaky_relu': nn.LeakyReLU,
    'elu': nn.ELU,
    'selu': nn.SELU,
    'celu': nn.CELU,
    'gelu': nn.GELU,
    'softplus': nn.Softplus,
    'softsign': nn.Softsign,
    'tanhshrink': nn.Tanhshrink,
    'softmin': nn.Softmin,
    'softmax': nn.Softmax,
    # 'log_softmax': nn.LogSoftmax, # not working
    'softshrink': nn.Softshrink,
    'prelu': nn.PReLU,
    'rrelu': nn.RReLU,
}

layers = {
    'linear': nn.Linear,
    'conv1d': nn.Conv1d,
    'conv2d': nn.Conv2d,
    'conv3d': nn.Conv3d,
}


class NLayersModel(nn.Module):
    def __init__(self, n_layers: int,
                 sizes: Optional[list] = None,
                 kernel_sizes: Optional[list] = None, drop_rates: Optional[list] = None,
                 activation_name: 'str' = 'PReLU', layer_name: str = 'Conv1d'):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList()

        layer_type = self.fetch_layer(layer_name)
        activation_type = self.fetch_activation(activation_name)

        self.sizes = sizes or self.default_sizes(n_layers)
        self.kernel_sizes = kernel_sizes or self.default_kernel_sizes(n_layers)
        self.drop_rates = drop_rates or self.default_drop_rates(n_layers)

        assert len(self.sizes) == n_layers + 1 and self.sizes[0] == self.sizes[-1] == 2, \
            "sizes must be None or have length n_layers+1 and start and end with 2"
        assert len(self.kernel_sizes) == n_layers, "kernel_sizes must be None or have length n_layers"
        assert len(self.drop_rates) == n_layers, "drop_rates must be None or have length n_layers"

        for in_i, out_i, ker_i, drop_i in zip(self.sizes[:-1], self.sizes[1:], self.kernel_sizes, self.drop_rates):
            self.layers.append(layer_type(in_i, out_i, kernel_size=ker_i))
            self.layers.append(activation_type(out_i))
            self.layers.append(nn.Dropout(drop_i))

    def fetch_layer(self, layer_name):
        try:
            layer_type = layers[layer_name.lower()]
        except KeyError:
            raise ValueError(f"layer_name must be one of {list(layers.keys())}")
        return layer_type

    def fetch_activation(self, activation_name):
        try:
            activation_type = activations[activation_name.lower()]
        except KeyError:
            raise ValueError(f"activation_name must be one of {list(activations.keys())}")
        return activation_type

    def forward(self, x: Tensor):
        x = x.unsqueeze(2)
        for i in range(self.n_layers):
            x = self.layers[3*i](x)
            x = self.layers[3*i + 1](x)
            x = self.layers[3*i + 2](x)
        x = x.squeeze(2)
        return x

    def print_architecture(self, x: Tensor):
        # print(self)
        # for i in range(self.n_layers):
        #     print(f"Layer {i+1}: {self.layers[3*i]}")
        #     print(f"Activation {i+1}: {self.layers[3*i + 1]}")
        #     print(f"Dropout {i+1}: {self.layers[3*i + 2]}")
        # print(f"Output layer: {self.layers[-1]}")

        # print size of x after each layer
        x = x.unsqueeze(2)
        print(f"Input size layer 0: {x.shape}")
        for i in range(self.n_layers*3):
            x = self.layers[i](x)
            print(f"Size after layer {i + 1}: {x.shape}")

    # ------------------------- default values generators -------------------------
    @staticmethod
    def default_drop_rates(n_layers):
        return [0.0]*n_layers

    @staticmethod
    def default_kernel_sizes(n):
        return [1]*n

    @staticmethod
    def default_sizes(n):
        arr = 2 ** (np.arange(n + 1) + 1)  # 2,4,8,16,...
        arr = np.clip(arr, 0, MAX_LAYER_SIZE)  # clip arr to not go above MAX_LAYER_SIZE (for performance issues)
        arr[-1] = 2  # must starts and end with 2
        return arr


class PaperNNforNFTmodel(nn.Module):
    def __init__(self, input_size = 8192, k=10):
        super().__init__()
        self.layers = nn.ModuleList()
        self._1_conv = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=k, stride=1, dilation=2)
        self._1_tanh = nn.Tanh()
        self._2_conv = nn.Conv1d(in_channels=10, out_channels=10, kernel_size=k, stride=1, dilation=2)
        self._2_tanh = nn.Tanh()
        self._3_conv = nn.Conv1d(in_channels=10, out_channels=4, kernel_size=k, stride=1, dilation=2)
        self._3_relu = nn.ReLU()
        self._4_fc = nn.Linear(in_features=4*(input_size-3*(2*(k-1))), out_features=input_size)
        self._4_relu = nn.ReLU()

    def forward(self, x: Tensor):
        x = x.unsqueeze(1).unsqueeze(1).T
        x = self._1_conv(x)
        x = self._1_tanh(x)
        x = self._2_conv(x)
        x = self._2_tanh(x)
        x = self._3_conv(x)
        x = self._3_relu(x)
        x = x.flatten()
        x = self._4_fc(x)
        x = self._4_relu(x)
        return x


class SingleMuModel3Layers(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(2, 4, kernel_size=1)
        self.prelu1 = nn.PReLU(4)
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
            x = x.to(device)
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
