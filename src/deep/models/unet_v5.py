import torch
from torch import nn, Tensor

class UnetV5(nn.Module):

    def __init__(self, in_channels=2, depth=3, M=4) -> None:
        # creating architecture of the following:
        # [in_channels, N]
        # -> [M, N/2]
        # -> [M*2, N/4]
        # ... until depth
        # -> [M*2**(depth-1), N/2**depth]

        # and back up:
        # [M*2**(depth-1), N/2**depth]
        # -> [M*2**(depth-2), N/2**(depth-1)]
        # -> [M*2**(depth-3), N/2**(depth-2)]
        # ... until depth
        # -> [M, N/2]
        # -> [in_channels, N]
        assert M/in_channels >= 2, f'M must be at least 2 times bigger than in_channels, got {M} and {in_channels}'

        super().__init__()

        self.down_convs = []
        self.up_convs = []
        self.pools = []
        self.tanhs = []
        self.uptanhs = []



        # down sampling
        in_channels_i = in_channels
        for i in range(depth):
            out_channels_i = M * (2 ** i) # M*[1,2,4,8,16,32,64..]
            self.down_convs.append(nn.Conv1d(in_channels_i, out_channels_i, kernel_size=3, padding="same"))
            self.up_convs.insert(0, nn.ConvTranspose1d(out_channels_i, in_channels_i, 
                                                       kernel_size=2, stride=2, padding=0)) # or stride 1 padding 0?
            in_channels_i = out_channels_i
            self.tanhs.append(nn.Tanh())
            self.uptanhs.append(nn.Tanh())
            self.pools.append(nn.MaxPool1d(2))

        # self.pools[-1] = nn.Identity() # no pooling for the last layer (replaced with dummy identity layer)



        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)
        self.pools = nn.ModuleList(self.pools)
        self.tanhs = nn.ModuleList(self.tanhs)
        self.uptanhs = nn.ModuleList(self.uptanhs)

    def forward(self, x: Tensor):
        xs = []
        xs.append(x) # [in_channels, N]
        for conv, tanh, pool in zip(self.down_convs, self.tanhs, self.pools):
            x = conv(x)
            x = tanh(x)
            x = pool(x)
            xs.append(x) # [M, N/2], [M*2, N/4], [M*4, N/8]...

        for i, (upconv, uptanh) in enumerate(zip(self.up_convs, self.uptanhs)):
            x = upconv(x)
            if i < len(self.up_convs) - 1:  # no addition for the last layer
                x = x + xs[-(i+2)]
                x = uptanh(x)

        return x

    @classmethod
    def load_pretrained_weights(cls, weights_path):
        state_dict = torch.load(weights_path)
        return cls().load_state_dict(state_dict)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)


class UnetV5Depth1(UnetV5):
    def __init__(self, in_channels=2):
        super().__init__(in_channels=in_channels, depth=1)

class UnetV5Depth2(UnetV5):
    def __init__(self, in_channels=2):
        super().__init__(in_channels=in_channels, depth=2)

class UnetV5Depth3(UnetV5):
    def __init__(self, in_channels=2):
        super().__init__(in_channels=in_channels, depth=3)

class UnetV5Depth4(UnetV5):
    def __init__(self, in_channels=2):
        super().__init__(in_channels=in_channels, depth=4)

class UnetV5Depth5(UnetV5):
    def __init__(self, in_channels=2):
        super().__init__(in_channels=in_channels, depth=5)

class UnetV5Depth6(UnetV5):
    def __init__(self, in_channels=2):
        super().__init__(in_channels=in_channels, depth=6)

class UnetV5Depth7(UnetV5):
    def __init__(self, in_channels=2):
        super().__init__(in_channels=in_channels, depth=7)

class UnetV5Depth8(UnetV5):
    def __init__(self, in_channels=2):
        super().__init__(in_channels=in_channels, depth=8)

class UnetV5Depth9(UnetV5):
    def __init__(self, in_channels=2):
        super().__init__(in_channels=in_channels, depth=9)

class UnetV5Depth10(UnetV5):
    def __init__(self, in_channels=2):
        super().__init__(in_channels=in_channels, depth=10)

class UnetV5Depth9M32(UnetV5):
    def __init__(self, in_channels=2):
        super().__init__(in_channels=in_channels, depth=9, M=32)