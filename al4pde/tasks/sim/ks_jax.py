import torch
import jax
import jax.numpy as jnp
from jax_cfd.base import funcutils
from jax_cfd.base import grids
from jax_cfd.spectral import equations as spectral_equations
from jax_cfd.spectral import time_stepping
from jax_cfd.spectral import utils as spectral_utils
import numpy as np
import dataclasses
import numbers
import operator
from al4pde.tasks.sim.ks_scipy import  ParametricKSSim


@dataclasses.dataclass(init=False, frozen=True)
class Grid(grids.Grid):
    """Overwrite Grid so that it can be used within jax.jit"""

    def __init__(
            self,
            shape,
            step=None,
            domain=None,
    ):
        """Construct a grid object."""
        shape = tuple(operator.index(s) for s in shape)
        object.__setattr__(self, 'shape', shape)

        if step is not None and domain is not None:
            raise TypeError('cannot provide both step and domain')
        elif domain is not None:
            if isinstance(domain, (int, float)):
                domain = ((0, domain),) * len(shape)
            else:
                if len(domain) != self.ndim:
                    raise ValueError('length of domain does not match ndim: '
                                     f'{len(domain)} != {self.ndim}')
                for bounds in domain:
                    if len(bounds) != 2:
                        raise ValueError(
                            f'domain is not sequence of pairs of numbers: {domain}')
            domain = tuple((lower, upper) for lower, upper in domain)
        else:
            if step is None:
                step = 1
            if isinstance(step, numbers.Number):
                step = (step,) * self.ndim
            elif len(step) != self.ndim:
                raise ValueError('length of step does not match ndim: '
                                 f'{len(step)} != {self.ndim}')
            domain = tuple(
                (0.0, float(step_ * size)) for step_, size in zip(step, shape))

        object.__setattr__(self, 'domain', domain)

        step = tuple(
            (upper - lower) / size for (lower, upper), size in zip(domain, shape))
        object.__setattr__(self, 'step', step)


def build_jax_sim(nx, dt, nt):
    @jax.jit
    def _evolve(carry, ):
        print("compiling jax")
        u0, param = carry
        viscosity = param[0]
        domain_length = param[1]
        grid = Grid((nx,), domain=((0, domain_length),))
        @dataclasses.dataclass
        class KuramotoSivashinsky(spectral_equations.KuramotoSivashinsky):
            grid: grids.Grid
            smooth: bool = True

            def __post_init__(self):
                self.kx, = self.grid.rfft_axes()
                self.two_pi_i_k = 2j * jnp.pi * self.kx
                self.linear_term = -self.two_pi_i_k ** 2 - viscosity * self.two_pi_i_k ** 4
                self.rfft = spectral_utils.truncated_rfft if self.smooth else jnp.fft.rfft
                self.irfft = spectral_utils.padded_irfft if self.smooth else jnp.fft.irfft

        step_fn = time_stepping.crank_nicolson_rk4(KuramotoSivashinsky(grid, smooth=True), dt)
        rollout_fn = funcutils.trajectory(step_fn, nt + 1, start_with_input=True)

        u0_hat = jnp.fft.rfft(u0)
        _, trajectory_hat = jax.device_get(rollout_fn(u0_hat))
        return jnp.fft.irfft(trajectory_hat).real

    return jax.pmap(jax.vmap(_evolve, axis_name='j'), axis_name='i')


class ParametricKSJaxSim(ParametricKSSim):

    def __init__(self, ini_time,  dt, fin_time, pde_name="KS", L=64):
        super().__init__(ini_time, dt, fin_time, pde_name, L)
        self.sim_fun = None
        self.last_n_steps = -1

    def n_step_sim(self, ic, pde_params, grid, init_time, n_steps):

        if self.sim_fun is None or self.last_n_steps != n_steps:
            self.sim_fun = build_jax_sim(grid.shape[-2], self.dt, n_steps)
            self.last_n_steps = n_steps

        ic = ic.detach().cpu().numpy()
        pde_params = pde_params.detach().cpu().numpy()
        t_coord = np.array([init_time + self.dt * i for i in range(n_steps + 1)])

        u = jax.device_put(ic)
        local_devices = jax.local_device_count()
        u = u.reshape([local_devices, len(ic) // local_devices, -1])
        pde_param = pde_params.reshape([local_devices, len(ic) // local_devices, 2])
        uu = self.sim_fun((u, pde_param)).block_until_ready().reshape((-1, n_steps + 1, grid.shape[-2]))
        uu_traj = torch.from_numpy(np.array(jax.device_put(uu, jax.devices("cpu")[0]))).unsqueeze(-1)
        return uu_traj, grid.detach().cpu().numpy().squeeze().astype(np.float32), t_coord.astype(np.float32)
