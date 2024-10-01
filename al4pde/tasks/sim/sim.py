from math import ceil
import torch


class Simulator:

    def __init__(self, pde_name, num_pde_params, spatial_dim, num_channels, dt, ini_time, fin_time, channel_names=None,
                 pde_param_names=None):
        self.pde_name = pde_name
        self.spatial_dim = spatial_dim
        self.num_channels = num_channels
        self.num_pde_params = num_pde_params

        self.n_steps = ceil((fin_time - ini_time) / dt)
        self.dt = dt
        self.ini_time = ini_time
        self.t_coord = self.get_t_coord(self.ini_time, self.n_steps)
        self.fin_time = float(self.t_coord[-1])
        if channel_names is None:
            channel_names = ["channel_" + str(i) for i in range(num_channels)]
        if pde_param_names is None:
            pde_param_names = ["param_" + str(i) for i in range(num_pde_params)]
        self.pde_param_names = pde_param_names
        assert isinstance(channel_names, list)
        self.channel_names = channel_names
        self.autonomous = True

    def __call__(self, ic, pde_params, grid):
        res = self.n_step_sim(ic, pde_params, grid, self.ini_time, self.n_steps)
        if not torch.all(torch.isfinite((res[0]))):
            for i in range(len(res[0])):
                if not torch.all(torch.isfinite((res[0][i]))):
                    print("\n non-finite for", pde_params[i])
                    print(ic[i])
                    print(res[0][i])
            raise ValueError("non-finite values in simulator result")
        return res

    def get_time(self, t_idx):
        return self.ini_time + self.dt * t_idx

    def get_t_coord(self, ini_time, n_steps):
        return torch.arange(n_steps + 1) * self.dt + ini_time

    def n_step_sim(self, ic, pde_params, grid, init_time, n_steps):
        raise NotImplementedError

