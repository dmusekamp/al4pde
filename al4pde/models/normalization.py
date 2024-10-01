import torch
from torch import Tensor
from torch.utils.data import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _get_stats(train_loader, idx):
    n = len(train_loader.sampler)
    mean_channels = 0
    for batch in train_loader:
        b = batch[idx]
        mean_channels += b.flatten(0, -2).mean(0) * len(b) / n

    std_channels = 0
    for batch in train_loader:
        b = batch[idx]
        std_channels += ((b.flatten(0, -2) - mean_channels) ** 2).mean(0) * len(b) / n

    return mean_channels, torch.sqrt(std_channels)


def _norm(x, mean, std):
    return (x - mean) / std


class TaskNormalizer(torch.nn.Module):
    """Normalizes the channels, grid and parameters. """

    def __init__(self):
        super().__init__()
        self.mean_channels = torch.nn.Parameter(torch.Tensor(), requires_grad=False)
        self.std_channels = torch.nn.Parameter(torch.Tensor(), requires_grad=False)

        self.mean_parameters = torch.nn.Parameter(torch.Tensor(), requires_grad=False)
        self.std_parameters = torch.nn.Parameter(torch.Tensor(), requires_grad=False)

        self.mean_grid_coord = torch.nn.Parameter(torch.Tensor(), requires_grad=False)
        self.std_grid_coord = torch.nn.Parameter(torch.Tensor(), requires_grad=False)

    def update(self, train_loader: DataLoader):
        self.mean_channels.data, self.std_channels.data = _get_stats(train_loader, 1)
        self.mean_grid_coord.data, self.std_grid_coord.data = _get_stats(train_loader, 2)
        self.mean_parameters.data, self.std_parameters.data = _get_stats(train_loader, 3)
        print("std channels", self.std_channels)
        print("std grid", self.std_grid_coord)
        print("std params", self.std_parameters)

    def norm_input_batch(self, xx: Tensor, yy: Tensor, grid: Tensor, pde_param: Tensor):
        return self.norm_traj(xx), self.norm_traj(yy), self.norm_grid(grid), self.norm_param(pde_param)

    def norm_traj(self, yy: Tensor):
        return _norm(yy, self.mean_channels, self.std_channels)

    def norm_param(self, pde_param: Tensor):
        return _norm(pde_param, self.mean_parameters, self.std_parameters)

    def norm_grid(self, grid: Tensor):
        return _norm(grid, self.mean_grid_coord, self.std_grid_coord)

    def denorm_traj(self, yy: Tensor):
        return yy * self.std_channels + self.mean_channels
