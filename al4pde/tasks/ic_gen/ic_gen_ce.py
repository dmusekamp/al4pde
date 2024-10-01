import numpy as np
import torch
from tensordict import TensorDict
from al4pde.tasks.sim.mp_pde_solvers.generate.generate_data import initial_conditions
from al4pde.tasks.ic_gen.ic_gen import ICGenerator
from al4pde.tasks.sim.mp_pde_solvers.equations.PDEs import CE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ICGenCE(ICGenerator):

    def __init__(self, nx, requires_grad=True, single_fixed=False, length=16):
        super().__init__(requires_grad, single_fixed)
        self.nx = nx
        self.length = length
        self.grid = torch.from_numpy(np.linspace(0, self.length, self.nx, dtype=np.float32)).to(device)

    def get_grid(self, n):
        grid = self.grid[:, None]
        return grid.expand([n, ] + list(grid.shape))

    def _initialize_ic_params(self, n: int) -> TensorDict:
        return TensorDict({}, batch_size=n)

    def generate_initial_conditions(self, ic_params: TensorDict, pde_params: torch.Tensor) -> torch.Tensor:
        num_forcing_waves = int((pde_params.shape[-1] - 3) / 4)

        forcing_a = pde_params[:, 3: 3 + num_forcing_waves]
        forcing_omega = pde_params[:, 3 + num_forcing_waves: 3 + num_forcing_waves * 2]
        phi = pde_params[:, 3 + num_forcing_waves * 2: 3 + num_forcing_waves * 3]
        forcing_l = pde_params[:, 3 + num_forcing_waves * 3: 3 + num_forcing_waves * 4]
        pde = CE(device=forcing_a.device, L=self.length)
        ic = initial_conditions(forcing_a.unsqueeze(1), forcing_omega.unsqueeze(1), phi.unsqueeze(1),
                                forcing_l.unsqueeze(1), pde)(self.get_grid(len(ic_params)), 0)[:, None]

        return ic.transpose(1, 2).unsqueeze(-1)
