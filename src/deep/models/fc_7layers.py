import torch
import torch.optim
from torch import nn, Tensor

from src.deep.data_loaders import DatasetNormal


class Fc7Layers(nn.Module):

    def __init__(self, vec_size=1024) -> None:
        super().__init__()
        self.input_layer = nn.Linear(vec_size, 1024)
        self.hidden_layers = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),  
        )
        self.output_layer = nn.Linear(1024, vec_size)
        
    def forward(self, x: Tensor):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x


    @classmethod
    def load_pretrained_weights(cls, weights_path):
        # state_dict_path = os.path.join(os.path.dirname(__file__), weights_path)
        state_dict = torch.load(weights_path)
        return cls().load_state_dict(state_dict)




