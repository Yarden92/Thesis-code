import numpy as np
import torch.optim
from torch import nn, Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

num_epochs = 3
data_dir_path = '../../apps/deep/data'


class mModel(nn.Module):

    def __init__(self, vec_len: int = 512) -> None:
        super().__init__()
        self.layer1 = nn.Linear(vec_len, vec_len)

    def forward(self, x):
        x = self.layer1(x)
        return x


class mDataSet(Dataset):
    def __init__(self, data_dir_path: str) -> None:
        super().__init__()
        self.data_dir_path = data_dir_path
        x = np.load(f'{self.data_dir_path}/data_x.npy')
        y = np.load(f'{self.data_dir_path}/data_y.npy')

        self.x = torch.from_numpy(np.float32(x))
        self.y = torch.from_numpy(np.float32(y))

        assert len(self.x) == len(self.y), \
            "not the same amount of inputs x and outputs y!"

    def __getitem__(self, index) -> T_co:
        return self.x[index], self.y[index]

    def __len__(self) -> int:
        return len(self.x)


def test():
    model = mModel()
    dataloader = mDataSet(data_dir_path)
    l_metric = nn.L1Loss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        for x, y in dataloader:
            pred = model(x)
            loss: Tensor = l_metric(y, pred)
            optim.zero_grad()
            loss.backward()
            optim.step()
            print(loss.item())


if __name__ == '__main__':
    test()
