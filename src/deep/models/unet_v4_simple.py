import torch
import torch.nn as nn

class UnetV4Simple(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # 1x1 convolution to match the number of channels
        self.match_channels = nn.Conv1d(2, 64, kernel_size=1)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(64, 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(2, 2, kernel_size=3, padding=1),
            nn.Sigmoid()  # Assuming output is in the range [0, 1]
        )

    def forward(self, x):
        # Encoder
        x1 = self.encoder(x)

        x_matched = self.match_channels(x)
        
        # Decoder with skip connection
        x2 = self.decoder(x1 + x_matched)

        return x2

