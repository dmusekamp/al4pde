# MIT Licence. https://github.com/brandstetter-johannes/LPSDA.
import torch
import numpy as np
from tensordict import TensorDict
from torch import Tensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from al4pde.tasks.sim.sim import Simulator
from al4pde.tasks.sim.mp_pde_solvers.generate.generate_data import initial_conditions
from al4pde.tasks.sim.mp_pde_solvers.equations.PDEs import CE
from al4pde.tasks.sim.mp_pde_solvers.temporal.solvers import Solver, RKSolver,Dopri45


def generate_data_combined_equation(u0, pde_params, grid, t, use_forcing=True):
    u0 = u0.double().to(device)
    pde_params = pde_params.double().to(device)
    grid = grid.double().to(device)
    t = t.double().to(device)
    alpha = pde_params[:, 0]
    beta = pde_params[:, 1]
    gamma = pde_params[:, 2]
    num_forcing_waves = int((pde_params.shape[-1] - 3) / 4)

    pde = CE(t[0], t[-1], (len(t), len(grid)), device=alpha.device, L=grid.max())
    if use_forcing:
        forcing_a = pde_params[:, 3: 3 + num_forcing_waves]
        forcing_omega = pde_params[:, 3 + num_forcing_waves: 3 + num_forcing_waves * 2]
        forcing_phi = pde_params[:, 3 + num_forcing_waves * 2: 3 + num_forcing_waves * 3] * 2 * np.pi
        forcing_l = pde_params[:, 3 + num_forcing_waves * 3: 3 + num_forcing_waves * 4]

        batch_size = len(alpha)
        # Time dependent force term
        def force(t):
            return initial_conditions(forcing_a.unsqueeze(1), forcing_omega.unsqueeze(1), forcing_phi.unsqueeze(1),
                                      forcing_l.unsqueeze(1), pde)(grid, t)[:, None]
    else:
        def force(t):
            return 0.0
    # Initialize PDE parameters and get initial condition
    pde.alpha = alpha[:, None, None]
    pde.beta = beta[:,  None, None]
    pde.gamma = gamma[:,  None, None]
    pde.force = force
    # The spatial method is the WENO reconstruction for uux and FD for the rest
    spatial_method = pde.WENO_reconstruction

    # Solving full trajectories and runtime measurement
    solver = Solver(RKSolver(Dopri45(), device=device), spatial_method)
    x0 = u0[..., 0, 0][:, None].to(device)
    with torch.no_grad():
        sol = solver.solve(x0=x0.to(device), times=t[None, :].to(device))
    # Save solutions
    return sol.float().cpu()


class CombinedEquation(Simulator):

    def __init__(self, pde_name, dt, ini_time, fin_time, num_forcing_waves=5):
        self.num_forcing_waves = num_forcing_waves
        num_pde_params = 3 + self.num_forcing_waves * 4
        self.autonomous = False
        super().__init__(pde_name, num_pde_params, 1, 1, dt, ini_time, fin_time)

    def n_step_sim(self, ic: Tensor, pde_params: Tensor, grid: Tensor, init_time: Tensor, n_steps: int):
        t_coord = self.get_t_coord(init_time, n_steps)
        sol = generate_data_combined_equation(ic, pde_params, grid, self.get_t_coord(init_time, n_steps))
        return sol.transpose(-2, -1), grid[:, 0], t_coord


class CombinedEquationNoForcing(CombinedEquation):

    def __init__(self, pde_name, dt, ini_time, fin_time):
        super().__init__(pde_name, dt, ini_time, fin_time, 0)
        self.autonomous = True

    def n_step_sim(self, ic: Tensor, pde_params: Tensor, grid: Tensor, init_time: Tensor, n_steps: int):
        t_coord = self.get_t_coord(init_time, n_steps)
        sol = generate_data_combined_equation(ic, pde_params, grid, self.get_t_coord(init_time, n_steps))
        return sol.transpose(-2, -1), grid[:, 0], t_coord


