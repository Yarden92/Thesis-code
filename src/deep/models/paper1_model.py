import torch
from torch import nn, Tensor


class Paper1Model(nn.Module):
    def __init__(self, input_size=8192, k=10):
        super().__init__()
        self.layers = nn.ModuleList()
        self._1_conv = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=k, stride=1, dilation=2)
        self._1_tanh = nn.Tanh()
        self._2_conv = nn.Conv1d(in_channels=10, out_channels=10, kernel_size=k, stride=1, dilation=2)
        self._2_tanh = nn.Tanh()
        self._3_conv = nn.Conv1d(in_channels=10, out_channels=4, kernel_size=k, stride=1, dilation=2)
        self._3_relu = nn.ReLU()
        self._4_fc = nn.Linear(in_features=4*(input_size - 3*(2*(k - 1))), out_features=input_size)
        # self._4_relu = nn.ReLU()

    def forward(self, x: Tensor):
        x = x.unsqueeze(1)
        x = self._1_conv(x)
        x = self._1_tanh(x)
        x = self._2_conv(x)
        x = self._2_tanh(x)
        x = self._3_conv(x)
        x = self._3_relu(x)
        x = x.flatten()
        x = self._4_fc(x)
        x = x.unsqueeze(0)
        # x = self._4_relu(x)
        return x


class Paper1ModelWrapper(nn.Module):
    def __init__(self, real_model: Paper1Model, imag_model: Paper1Model):
        super().__init__()
        self.real_model = real_model
        self.imag_model = imag_model

    def forward(self, x: Tensor):
        assert x.shape[0] == 2, f"input should have been 2x8192 but got {x.shape} instead"
        x_real = x[0].unsqueeze(0)
        x_imag = x[1].unsqueeze(0)
        pred_real = self.real_model(x_real)
        pred_imag = self.imag_model(x_imag)
        return torch.stack([pred_real, pred_imag], dim=1).squeeze(0)


class Paper1Model_v2(nn.Module):
    def __init__(self, input_size=8192, k=10):
        super().__init__()
        self._1_conv = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=k, stride=1, dilation=2)
        self._1_tanh = nn.Tanh()
        self._2_conv = nn.Conv1d(in_channels=10, out_channels=10, kernel_size=k, stride=1, dilation=2)
        self._2_tanh = nn.Tanh()
        self._3_conv = nn.Conv1d(in_channels=10, out_channels=4, kernel_size=k, stride=1, dilation=2)
        self._3_relu = nn.ReLU()
        self._4_fc = nn.Linear(in_features=4*(input_size - 3*(2*(k - 1))), out_features=input_size)
        # self._4_relu = nn.ReLU()

    def forward(self, x_in: Tensor):
        x_real = x_in[:, 0]
        x_imag = x_in[:, 1]

        for x in [x_real, x_imag]:
            x = x.unsqueeze(1).unsqueeze(1).T
            x = self._1_conv(x)
            x = self._1_tanh(x)
            x = self._2_conv(x)
            x = self._2_tanh(x)
            x = self._3_conv(x)
            x = self._3_relu(x)
            x = x.flatten()
            x = self._4_fc(x)
            # x = self._4_relu(x)

        y = torch.stack([x_real, x_imag], dim=1)
        return y
