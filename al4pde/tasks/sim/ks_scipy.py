# MIT Licence. Adapted from https://github.com/brandstetter-johannes/LPSDA.
import numpy as np
import torch
from scipy.fftpack import diff as psdiff
from scipy.integrate import solve_ivp
from al4pde.tasks.sim.sim import Simulator


class ParametricKS:
    """
    The Kuramoto-Sivashinsky equation:
    ut + (0.5*u**2 + ux + uxxx)x = 0
    """
    def __init__(self,
                 tmin: float=None,
                 tmax: float=None,
                 grid_size: list=None,
                 L: float=None,
                 viscosity: float=None,
                 device: torch.cuda.device = "cpu"):
        super().__init__()
        # Start and end time of the trajectory
        self.tmin = 0 if tmin is None else tmin
        self.tmax = 40. if tmax is None else tmax
        # Number of different waves
        self.N = 10
        # Length of the spatial domain
        self.L = 64. if L is None else L
        self.grid_size = (100, 2 ** 8) if grid_size is None else grid_size
        # The effective time steps used for learning and inference
        self.nt = self.grid_size[0]
        self.nx = self.grid_size[1]
        # dt and dx are slightly different due to periodicity in the spatial domain
        self.dt = self.tmax / (self.grid_size[0]-1)
        self.dx = self.L / (self.grid_size[1])
        self.device = device
        if self.device != "cpu":
            raise NotImplementedError

        # Parameters for Lie Point symmetry data augmentation
        self.time_shift = 0
        self.max_x_shift = 0.0
        self.max_velocity = 0.0

        self.viscosity = viscosity


    def __repr__(self):
        return f'KS'

    def pseudospectral_reconstruction(self, t: float, u: np.ndarray, L: float) -> np.ndarray:
        """
        Pseudospectral reconstruction of the spatial derivatives of the KS equation, discretized in x.
        Args:
            t (float): time point
            u (np.ndarray): 1D input field
            L (float): length of the spatial domain
        Returns:
            np.ndarray: reconstructed pseudospectral time derivative
        """
        # Compute the x derivatives using the pseudo-spectral method.
        ux = psdiff(u, period=L)
        uxx = psdiff(u, period=L, order=2)
        uxxxx = self.viscosity * psdiff(u, period=L, order=4)
        # Compute du/dt.
        dudt = - u*ux - uxx - uxxxx
        return dudt

    def fvm_reconstruction(self, t: float, u: np.ndarray, L: float) -> np.ndarray:
        """
        FVM reconstruction of the spatial derivatives of the KS equation, discretized in x.
        Args:
            t (float): time point
            u (np.ndarray): 1D input field
            L (float): length of the spatial domain
        Returns:
            np.ndarray: reconstructed FVM time derivative
        """
        dx = L / len(u)
        # Integrate: exact at half nodes
        iu = np.cumsum(u) * dx
        # Derivatives
        u = psdiff(iu, order=0 + 1, period=L)
        ux = psdiff(iu, order=1 +1 , period=L)
        uxxx = self.viscosity * psdiff(iu, order=3 + 1, period=L)
        # Compute du/dt.
        Jrhs = 0.5 * (u ** 2) + ux + uxxx
        Jlhs = np.roll(Jrhs, 1)
        dudt = -(Jrhs - Jlhs) / dx
        return dudt


def generate_trajectory(pde, u0, t, ):

    # parallel data generation is not yet implemented

    pde_string = str(pde)


    tcoord = {}
    xcoord = {}
    dx = {}
    dt = {}
    h5f_u = {}

    # Tolerance of the solver
    tol = 1e-9
    nx = pde.grid_size[1]
    L = pde.L
    T = pde.tmax
    # For the Heat (Burgers') equation, a fixed grid is used
    # For KdV and KS, the grid is flexible ->  this is due to scale symmetries which we want to exploit
    #if pde_string == 'Heat':
    #    T = pde.tmax
    #    L = pde.L

    #else:
        #t1 = pde.tmax - pde.tmax / 10
        #t2 = pde.tmax + pde.tmax / 10
        #T = (t1 - t2) * np.random.rand() + t2
        #l1 = pde.L - pde.L / 10
        #l2 = pde.L + pde.L / 10
        #L = (l1 - l2) * np.random.rand() + l2
        #L = pde.L
        #T = pde.tmax
    #t = np.linspace(pde.tmin, T, nt)
    #x = np.linspace(0, (1 - 1.0 / nx) * L, nx)

    # We use pseudospectral reconstruction as spatial solver
    spatial_method = pde.pseudospectral_reconstruction

    # Solving for the full trajectories
    # For integration in time, we use an implicit Runge-Kutta method of Radau IIA family, order 5
    solved_trajectory = solve_ivp(fun=spatial_method,
                                  t_span=[t[0], t[-1]],
                                  y0=u0,
                                  method='Radau',
                                  t_eval=t,
                                  args=(L, ),
                                  atol=tol,
                                  rtol=tol)

    # Saving the trajectories, if successfully solved
    if solved_trajectory.success:
        idx = 0
        sol = solved_trajectory.y.T

    else:
        raise Exception

    return sol


class ParametricKSSim(Simulator):

    def __init__(self, ini_time,  dt, fin_time, pde_name="KS", L=64):
        super().__init__(pde_name, 1, 1, 1, dt, ini_time, fin_time)
        self.pde_param_names = "viscosity"
        self.L = L

    def __call__(self, ic, pde_params, grid):
        return self.n_step_sim(ic, pde_params, grid, self.ini_time, self.n_steps)

    def to_ml_format(self, sim_format):
        return sim_format.permute(0, 2, 1).unsqueeze(-1)

    def n_step_sim(self, ic, pde_params, grid, init_time, n_steps):
        ic = ic.detach().cpu().numpy()
        pde_params = pde_params.detach().cpu().numpy()
        fin_time = self.dt * n_steps + init_time
        t_coord = np.array([init_time + self.dt * i for i in range(n_steps + 1)])
        print(n_steps, t_coord, t_coord.shape)
        u = []
        for i in range(len(ic)):
            L = self.L
            if pde_params.shape[1] == 2:
                L = pde_params[i, 1]
            pde = ParametricKS(tmin=init_time, tmax=fin_time, grid_size=[n_steps, grid.shape[-1]], L=L,
                               viscosity=pde_params[i, 0])

            u0 = ic[i, :, 0, 0]
            u.append(generate_trajectory(pde, u0, t_coord).astype(np.float32))
        uu_traj = torch.from_numpy(np.stack(u))
        return uu_traj, grid.detach().cpu().numpy().squeeze().astype(np.float32), t_coord.astype(np.float32)
