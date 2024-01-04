import torch
import torch.optim
from torch import nn, Tensor


class UnetWithSkipsV2(nn.Module):

    def __init__(self, in_channels=2) -> None:
        super().__init__()

        # Encoder
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)

        # Decoder
        self.up5 = nn.ConvTranspose1d(512, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(512, 256, kernel_size=3, padding=1)
        self.up6 = nn.ConvTranspose1d(256, 128, kernel_size=1)
        self.conv6 = nn.Conv1d(256, 128, kernel_size=1)
        self.up7 = nn.ConvTranspose1d(128, 64, kernel_size=1)
        self.conv7 = nn.Conv1d(128, 64, kernel_size=1)

        # Output layer
        self.output_layer = nn.Conv1d(64, in_channels, kernel_size=1)

    def forward(self, x: Tensor):
         # Encoder
         # x.shape = (batch_size, 2, N)
        conv1 = torch.relu(self.conv1(x))       # (batch_size, 64, N)
        conv2 = torch.relu(self.conv2(conv1))   # (batch_size, 128, N)
        conv3 = torch.relu(self.conv3(conv2))   # (batch_size, 256, N)
        conv4 = torch.relu(self.conv4(conv3))   # (batch_size, 512, N)

        # Decoder
        up5 = torch.relu(self.up5(conv4))       # (batch_size, 256, N)
        concat5 = torch.cat([conv3, up5], dim=1)
        conv5 = torch.relu(self.conv5(concat5))

        up6 = torch.relu(self.up6(conv5))
        concat6 = torch.cat([conv2, up6], dim=1)
        conv6 = torch.relu(self.conv6(concat6))

        up7 = torch.relu(self.up7(conv6))
        concat7 = torch.cat([conv1, up7], dim=1)
        conv7 = torch.relu(self.conv7(concat7))

        # Output layer
        output = torch.sigmoid(self.output_layer(conv7))

        return output

    @classmethod
    def load_pretrained_weights(cls, weights_path):
        # state_dict_path = os.path.join(os.path.dirname(__file__), weights_path)
        state_dict = torch.load(weights_path)
        return cls().load_state_dict(state_dict)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
