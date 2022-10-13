from typing import Optional

import numpy as np
from torch import nn, Tensor

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


MAX_LAYER_SIZE = 64  # maximum number of channels in each layer
