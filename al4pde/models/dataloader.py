import torch
from torch.utils.data import Dataset
import os
import glob
import numpy as np
import re
from al4pde.utils import subsample_grid, subsample_trajectory



def load_grid(folders, reduced_resolution):
    grid = None
    for folder in folders:
        # Define path to files
        root_path = os.path.abspath(folder)

        x_coord_fname = os.path.join(root_path, 'x_coordinate.npy')
        y_coord_fname = os.path.join(root_path, 'y_coordinate.npy')
        z_coord_fname = os.path.join(root_path, 'z_coordinate.npy')
        xy_coord_fname = os.path.join(root_path, 'xy_coordinate.npy')

        # 3D
        if os.path.exists(z_coord_fname) and os.path.exists(y_coord_fname) and  os.path.exists(x_coord_fname):
            _gridx = np.load(os.path.join(root_path, x_coord_fname))[:, np.newaxis]
            _gridy = np.load(os.path.join(root_path, y_coord_fname))[:, np.newaxis]
            _gridz = np.load(os.path.join(root_path, z_coord_fname))[:, np.newaxis]
            _gridx = torch.from_numpy(_gridx)
            _gridy = torch.from_numpy(_gridy)
            _gridz = torch.from_numpy(_gridz)
            X, Y, Z = torch.meshgrid(_gridx, _gridy, _gridz, indexing='ij')
            grid = torch.stack((X, Y, Z), dim=-1)

        # 2d
        elif os.path.exists(xy_coord_fname):
            _gridxy = np.load(os.path.join(root_path, xy_coord_fname))
            grid = torch.from_numpy(_gridxy)

        # 1d
        elif os.path.exists(x_coord_fname):
            _gridx = np.load(os.path.join(root_path, x_coord_fname))
            if len(_gridx.shape) == 1:
                _gridx = _gridx[:, np.newaxis]
            grid = torch.from_numpy(_gridx)
    if grid is None:
        raise IOError("no grid found in folders" , folders)

    grid = subsample_grid(grid.unsqueeze(0), reduced_resolution).float()[0]
    spatial_dim = len(grid.shape) - 1

    return grid, spatial_dim


def load_fdata(root_path, fname, pde_name, spatial_dim):
    fdata = np.load(os.path.join(root_path, fname))
    if len(fdata.shape) == spatial_dim + 2:
        fdata = fdata[..., None]  # assume squeezed channel dim

    fdata = fdata.transpose([0,] + list(range(2, len(fdata.shape) - 1)) + [1, -1])

    return fdata


class TrajDataset(Dataset):
    def __init__(self,
                 data,
                 pde_params,
                 grid,
                 initial_step=1,
                 num_steps=None,
                 ):

        self.initial_step = initial_step
        self.data = data
        self.pde_params = pde_params
        self.grid = grid
        if num_steps is not None:
            self._set_num_steps(num_steps)
        else:
            self._set_num_steps(data.shape[-2] - 1)
        self.t = torch.arange(data.shape[-2]).float()

    def __len__(self):
        return self.num_sub_trajs

    def __getitem__(self, idx):
        true_traj_idx = int(idx / self.num_sub_per_traj)
        start_t_idx = idx % self.num_sub_per_traj
        ic = self.data[true_traj_idx, ...,  start_t_idx:start_t_idx + self.initial_step, :]
        traj = self.data[true_traj_idx, ..., start_t_idx: start_t_idx + self.traj_len, :]
        return ic, traj, self.grid, self.pde_params[true_traj_idx], self.t[start_t_idx]

    def set_num_steps(self, num_steps):
        return TrajDataset(self.data, self.pde_params, self.grid, self.initial_step, num_steps)

    def _set_num_steps(self, num_steps):
        self.traj_len = num_steps + 1
        self.num_steps = num_steps
        self.num_sub_per_traj = (self.data.shape[-2] - num_steps)
        self.num_sub_trajs = self.num_sub_per_traj * self.data.shape[0]


class NPYDataset(TrajDataset):
    def __init__(self,
                 pde_name,
                 folders,
                 initial_step=1,
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 regexp=None,
                 max_size=None,
                 skip_initial_steps=0,
                 one_step=False,
                 ):

        # Time steps used as initial conditions
        self.initial_step = initial_step
        self.reduced_batch = reduced_batch
        self.reduced_resolution_t = reduced_resolution_t
        self.reduced_resolution = reduced_resolution
        self.skip_initial_steps = skip_initial_steps
        fdata_list, pde_param_list, grid = self.load_data_list(folders, regexp, max_size, pde_name)

        # Each npy file has multiple ICs. Hence, we batch them along 0th dim to get shape (numICs, x, t, ch)
        _data, _pde_par = self.stack_data(fdata_list, pde_param_list, max_size)
        num_step = 1 if one_step else None
        super().__init__(_data[::reduced_batch], _pde_par[::reduced_batch], grid, initial_step, num_step)
        print("number of trajectories in data:", len(_data))

    def stack_data(self, fdata_list, pde_param_list, max_size):
        _data = np.vstack(fdata_list)
        _pde_par = np.vstack(pde_param_list)
        if max_size is not None:
            _data = _data[:max_size]
            _pde_par = _pde_par[:max_size]
        if not np.all(np.isfinite(_data)):
            raise ValueError("nan or inf in data")
        _data = torch.from_numpy(_data)  # (bs, numICs, x, t, ch)
        _pde_par = torch.from_numpy(_pde_par)
        return _data, _pde_par

    def load_data_list(self, folders, regexp, max_size, pde_name):
        fdata_list = []
        pde_param_list = []
        if regexp is not None:
            regexp = re.compile(regexp)

        n_data = 0
        grid, spatial_dim = load_grid(folders, self.reduced_resolution)

        for folder in folders:
            # Define path to files
            root_path = os.path.abspath(folder)

            npy_filenames = glob.glob("*btch*_traj.npy", root_dir=root_path)
            npy_filenames = sorted(npy_filenames)

            for fname in npy_filenames:
                if regexp is None or regexp.match(fname):

                    pde_param_file = fname.split("_traj")[0] + "_param.npy"
                    pde_params = np.load(os.path.join(root_path, pde_param_file))

                    fdata = load_fdata(root_path, fname, pde_name, spatial_dim)
                    fdata = subsample_trajectory(fdata, self.reduced_resolution, self.reduced_resolution_t)
                    fdata = fdata[..., self.skip_initial_steps:, :]
                    n_data += len(fdata)

                    fdata_list.append(fdata)
                    pde_param_list.append(pde_params)

                    if max_size is not None and n_data >= max_size:
                        break

        if len(fdata_list) == 0:
            raise IOError("No data found in:", folders)

        return fdata_list, pde_param_list, grid




