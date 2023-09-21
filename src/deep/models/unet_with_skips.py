import torch
import torch.optim
from torch import nn, Tensor


class UnetWithSkips(nn.Module):

    def __init__(self, in_channels=2) -> None:
        super().__init__()

        # down sampling
        self.conv1 = nn.Conv1d(2, 4, kernel_size=3, padding="same")
        self.tanh1 = nn.Tanh()
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(4, 8, kernel_size=3, padding="same")
        self.tanh2 = nn.Tanh()
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(8, 16, kernel_size=3, padding="same")
        self.tanh3 = nn.Tanh()

        # up sampling
        self.upconv1 = nn.ConvTranspose1d(16, 8, kernel_size=3, stride=1, padding=1)
        self.uptanh1 = nn.Tanh()
        self.upconv2 = nn.ConvTranspose1d(8, 4, kernel_size=2, stride=2, padding=0)
        self.uptanh2 = nn.Tanh()
        self.upconv3 = nn.ConvTranspose1d(4, 2, kernel_size=2, stride=2, padding=0)

    def forward(self, x: Tensor):
        # x.shape = (batch_size, 2, 1024)
        x1 = self.conv1(x)  # (batch_size, 4, 1024)
        x1 = self.tanh1(x1)
        x1 = self.pool1(x1)  # (batch_size, 4, 512)
        x2 = self.conv2(x1)  # (batch_size, 8, 512)
        x2 = self.tanh2(x2)
        x2 = self.pool2(x2)  # (batch_size, 8, 256)
        x3 = self.conv3(x2)  # (batch_size, 16, 256)
        x3 = self.tanh3(x3)

        y1 = self.upconv1(x3)  # (batch_size, 8, 256)
        y1 = y1 + x2
        y1 = self.uptanh1(y1)
        y2 = self.upconv2(y1)  # (batch_size, 4, 512)
        y2 = y2 + x1
        y2 = self.uptanh2(y2)
        y3 = self.upconv3(y2)
        # y3 = y3 + x

        return y3

    @classmethod
    def load_pretrained_weights(cls, weights_path):
        # state_dict_path = os.path.join(os.path.dirname(__file__), weights_path)
        state_dict = torch.load(weights_path)
        return cls().load_state_dict(state_dict)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
