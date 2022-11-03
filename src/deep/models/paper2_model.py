from torch import nn, Tensor


class Paper2Model(nn.Module):
    def __init__(self, input_size: int = 8192, k=10, s=1, d=2):
        super().__init__()
        self._1_conv = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=k, stride=s, dilation=d)
        self._1_relu = nn.ReLU()
        self._2_conv = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=k, stride=s, dilation=d)
        self._2_relu = nn.ReLU()
        self._3_conv = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=k, stride=s, dilation=d)
        self._3_relu = nn.ReLU()
        self._4_conv = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=k, stride=s, dilation=d)
        self._4_relu = nn.ReLU()
        self._5_conv = nn.Conv1d(in_channels=64, out_channels=4, kernel_size=k, stride=s, dilation=d)
        self._5_relu = nn.ReLU()
        # self._6_conv = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=k, stride=s, dilation=d)
        # self._6_relu = nn.ReLU()
        self._7_fc = nn.Linear(in_features=4*(input_size - 5*(2*(k - 1))), out_features=2*input_size)
        self._7_relu = nn.ReLU()
        self._8_fc = nn.Linear(in_features=2*input_size, out_features=1*input_size)

    def forward(self, x: Tensor):
        # input should be 1x2x8192, outputs the same
        x = self._1_conv(x)
        x = self._1_relu(x)
        x = self._2_conv(x)
        x = self._2_relu(x)
        x = self._3_conv(x)
        x = self._3_relu(x)
        x = self._4_conv(x)
        x = self._4_relu(x)
        x = self._5_conv(x)
        x = self._5_relu(x)
        # x = self._6_conv(x)
        # x = self._6_relu(x)
        x = x.reshape([1,1,-1])
        x = self._7_fc(x)
        x = self._7_relu(x)
        x = self._8_fc(x)
        return x
