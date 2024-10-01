import os
import torch
from torch import Tensor
import jax.numpy as jnp
from tensordict import TensorDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Task:
    """Combines all the interfaces and information about the pde and its parameters. """

    def __init__(self, ic_gen, param_gen, sim, data_path, run_save_path, initial_step, reduced_batch,
                 reduced_resolution, reduced_resolution_t, skip_initial_steps=0,
                 data_gen=None, use_test=False):

        self.ic_gen = ic_gen
        self.param_gen = param_gen
        self.sim = sim
        self.data_gen_params = data_gen
        self.reduced_batch = reduced_batch
        self.initial_step = initial_step
        self.reduced_resolution = reduced_resolution
        self.reduced_resolution_t = reduced_resolution_t
        self.skip_initial_steps = skip_initial_steps
        self.channel_names = sim.channel_names
        self.pde_name = sim.pde_name
        self.num_pde_param = param_gen.num_pde_params
        self.spatial_dim = sim.spatial_dim
        self.num_channels = sim.num_channels
        self.pde_param_names = sim.pde_param_names

        if use_test:
            self.eval_set_path = os.path.join(data_path, self.pde_name, "test")
            print("using test, loading", self.eval_set_path)
        else:
            self.eval_set_path = os.path.join(data_path, self.pde_name, "val")
            print("using val, loading", self.eval_set_path)

        self.data_path = data_path
        self.traj_save_path = os.path.join(run_save_path, "data")
        self.run_save_path = run_save_path
        os.makedirs(self.traj_save_path, exist_ok=True)
        self.img_save_path = os.path.join(run_save_path, "img")
        os.makedirs(self.img_save_path, exist_ok=True)
        self.train_data_folders = [self.traj_save_path]

    def get_grid(self, n=1):
        return self.ic_gen.get_grid(n)

    def get_ic_params(self, n: int) -> TensorDict:
        return self.ic_gen.initialize_ic_params(n)

    def get_ic(self, ic_params: TensorDict, pde_params: Tensor) -> Tensor:
        return self.ic_gen.generate_initial_conditions(ic_params, pde_params)

    def get_pde_params_normed(self, n: int) -> Tensor:
        return self.param_gen.get_normed_pde_params(n)

    def get_pde_params(self, pde_params_normed: Tensor) -> Tensor:
        return self.param_gen.get_pde_params(pde_params_normed)

    def evolve_ic(self, ic: Tensor, pde_params: Tensor, grid: Tensor = None):
        if grid is None:
            grid = self.get_grid(1)[0]
        return self.sim(ic, pde_params, grid)

    def n_step_sim(self, ic: Tensor, pde_params: Tensor, grid: Tensor, init_time: Tensor, n_steps: int):
        return self.sim.n_step_sim(ic, pde_params, grid, init_time, n_steps)

    def save_trajectories(self, u_trajectories, pde_params, u_grid_coords, u_tcoords,  al_iter, opt_batch_num,
                          save_path=None, ic_params=None, pde_params_normed=None):
        if save_path is None:
            save_path = self.traj_save_path

        jnp_save_fname = self.pde_name + "_alstp_" + str(al_iter) + "_btch" + str(opt_batch_num)

        names = [jnp_save_fname + "_traj", jnp_save_fname + "_param", None, "t_coordinate",
                 jnp_save_fname + "_pde_params_normed"]
        if self.spatial_dim == 1:
            names[2] = "x_coordinate"
        elif self.spatial_dim == 2:
            names[2] = "xy_coordinate"
        elif self.spatial_dim == 3:
            names[2] = "xyz_coordinate"

        data = [u_trajectories, pde_params, u_grid_coords, u_tcoords, pde_params_normed]
        optional = [False, False, True, True, True]
        if ic_params is not None:
            for key in ic_params.keys():
                names.append(jnp_save_fname + "_ic_params_" + key)
                data.append(ic_params[key])
                optional.append(True)

        for i in range(len(data)):
            if not optional[i] or data[i] is not None:
                arr = data[i]
                if isinstance(data[i], torch.Tensor):
                    arr = arr.detach().cpu().numpy()
                jnp.save(os.path.join(save_path, names[i]), arr)

    def set_seed(self, seed):
        rng = torch.Generator(device=device).manual_seed(seed)
        self.ic_gen.set_rng(rng)
        self.param_gen.set_rng(rng)

    def to_sim_format(self, traj):
        """bx0x1x2tc to btx0x1x2c"""
        if self.spatial_dim == 1:
            return traj.permute(0, 2, 1, 3)
        elif self.spatial_dim == 2:
            return traj.permute(0, 3, 1, 2, 4)
        elif self.spatial_dim == 3:
            return traj.permute(0, 4, 1, 2, 3, 5)
        else:
            raise ValueError(self.spatial_dim)

    def to_ml_format(self, traj):
        """btx0x1x2c to bx0x1x2tc"""
        if self.spatial_dim == 1:
            return traj.permute(0, 2, 1, 3)
        elif self.spatial_dim == 2:
            return traj.permute(0, 2, 3, 1, 4)
        elif self.spatial_dim == 3:
            return traj.permute(0, 2, 3, 4, 1, 5)
        else:
            raise ValueError(self.spatial_dim)
