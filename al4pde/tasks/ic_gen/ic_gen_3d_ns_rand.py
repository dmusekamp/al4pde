import os
import torch
from torch import randint, rand, multinomial
from tensordict import TensorDict
from al4pde.tasks.ic_gen.ic_gen import ICGenerator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def uniform_(n, low, high, generator, device):
    return torch.empty(n, device=device).uniform_(low, high, generator=generator)


class ICGenNSRand3D(ICGenerator):
    """Generates parameters that are responsible for generating the initial conditions of the given pde.
    E.g. IC Params: (amplitude, phase)
    Needs torch 2.1 or higher, because of the chunk_size argument in vmap
    """

    def __init__(self, k_tot, xL, xR, yL, yR, zL, zR, nx, ny, nz, gamma, mach_min, mach_max, d0Min, d0Max, T0Min,
                 T0Max, delDMin, delDMax, delPMin, delPMax, init_field_type, requires_grad=True, single_fixed=False,
                 constrain_max=False):
        super().__init__(requires_grad, single_fixed)
        self.init_field_type = init_field_type
        self.k_tot = k_tot
        self.xL = xL
        self.xR = xR
        self.yL = yL
        self.yR = yR
        self.zL = zL
        self.zR = zR
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.gamma = gamma
        self.mach_min = mach_min
        self.mach_max = mach_max
        self.d0Min = torch.tensor(d0Min, dtype=torch.float)
        self.d0Max = torch.tensor(d0Max, dtype=torch.float)
        self.T0Min = torch.tensor(T0Min, dtype=torch.float)
        self.T0Max = torch.tensor(T0Max, dtype=torch.float)
        self.delDMin = torch.tensor(delDMin, dtype=torch.float)
        self.delDMax = torch.tensor(delDMax, dtype=torch.float)
        self.delPMin = torch.tensor(delPMin, dtype=torch.float)
        self.delPMax = torch.tensor(delPMax, dtype=torch.float)
        dx = (self.xR - self.xL) / self.nx
        dy = (self.yR - self.yL) / self.ny
        dz = (self.zR - self.zL) / self.nz

        # cell edge and cell center coordinate
        xe = torch.linspace(self.xL, self.xR, self.nx + 1, device=device)
        ye = torch.linspace(self.yL, self.yR, self.ny + 1, device=device)
        ze = torch.linspace(self.zL, self.zR, self.nz + 1, device=device)

        xc = xe[:-1] + 0.5 * dx
        yc = ye[:-1] + 0.5 * dy
        zc = ze[:-1] + 0.5 * dz
        self.xc = xc.to(device)
        self.yc = yc.to(device)
        self.zc = zc.to(device)
        self.constrain_max = constrain_max

    def get_grid(self, n):
        """
        n : number of initial conditions
        """
        gridx, gridy, gridz = torch.meshgrid(self.xc, self.yc, self.zc, indexing='ij')
        grid = torch.stack([gridx, gridy, gridz], dim=-1) # [nx, ny, nz, 3]
        return grid.expand([n, ] + list(grid.shape))      # [bs, nx, ny, nz, 3]

    def _initialize_ic_params(self, n: int) -> TensorDict:
        """
        n: number of initial conditions
        currently making only the phase and mach number to be optimized for data acquisition
        """
        total_phases = len(range(-self.k_tot, self.k_tot)) ** 3
        phs = 2.0 * torch.pi * rand(size=[n, total_phases, 5], generator=self.rng, device=device,
                                    requires_grad=self.requires_grad)
        mach = (self.mach_max - self.mach_min) * rand((n, 1), generator=self.rng, device=device,
                                                      requires_grad=self.requires_grad) + self.mach_min

        # (r2 - r1) * torch.rand(a, b) + r1    # shape: (a,b)
        # https://stackoverflow.com/a/44375813
        d0 = (self.d0Max - self.d0Min) * rand((n, 1), generator=self.rng, device=device) + self.d0Min
        T0 = (self.T0Max - self.T0Min) * rand((n, 1), generator=self.rng, device=device) + self.T0Min
        delD = (self.delDMax - self.delDMin) * rand((n, 1), generator=self.rng, device=device) + self.delDMin
        delP = (self.delPMax - self.delPMin) * rand((n, 1), generator=self.rng, device=device) + self.delPMin

        xL = uniform_(n, 0.1, 0.45, generator=self.rng, device=self.xc.device)[:, None]
        xR = uniform_(n, 0.55, 0.9, generator=self.rng, device=self.xc.device)[:, None]
        yL = uniform_(n, 0.1, 0.45, generator=self.rng, device=self.yc.device)[:, None]
        yR = uniform_(n, 0.55, 0.9, generator=self.rng, device=self.yc.device)[:, None]
        zL = uniform_(n, 0.1, 0.45, generator=self.rng, device=self.zc.device)[:, None]
        zR = uniform_(n, 0.55, 0.9, generator=self.rng, device=self.zc.device)[:, None]

        probs = torch.tensor([0.5, 0.5], device=device)
        cond = torch.multinomial(probs, num_samples=n, generator=self.rng, replacement=True)

        return TensorDict({"phs": phs, "mach": mach, "d0": d0, "T0": T0, "delD": delD, "delP": delP,
                           "xL": xL, "xR": xR, "yL": yL, "yR": yR, "zL": zL, "zR": zR, "cond": cond}, batch_size=n)

    def generate_initial_conditions(self, ic_params: TensorDict, pde_params: torch.Tensor) -> torch.Tensor:
        # Appendix D.5: https://arxiv.org/abs/2210.07182
        # v(x,y,0)

        """
            Notes: @kmario23
            Rand -- Random initial field; Turb -- turbulence initial field
            M -- Mach Number;           TUNABLE M
            Eta - Shear viscosity;      TUNABLE η
            Zeta - Bulk viscosity;      TUNABLE ζ
            --------------------------------------------------------------
            ρ - mass density of fluid
            v - velocity of fluid (2D: vx, vy)
            p - gas pressure

            M = |v|/cs, where cs = sqrt(Gamma_p/ρ) is the velocity of sound   # TUNABLE M
        """
        phs = ic_params.get("phs")
        mach = ic_params.get("mach")
        d0 = ic_params.get("d0")
        T0 = ic_params.get("T0")
        delD = ic_params.get("delD")
        delP = ic_params.get("delP")
        xL = ic_params.get("xL")
        xR = ic_params.get("xR")
        yL = ic_params.get("yL")
        yR = ic_params.get("yR")
        zL = ic_params.get("zL")
        zR = ic_params.get("zR")
        cond = ic_params.get("cond")

        u = torch.zeros([len(ic_params), 5, self.nx, self.ny, self.nz], device=device)
        nb = u.shape[0]

        def _create_3DRand_init(u, mach, phs, d0, T0, delD, delP):
            # print("Entering _create 3D Rand Init!")
            # nx, ny, nz = self.xc.shape[0], self.yc.shape[0], self.zc.shape[0]
            nx, ny, nz = self.nx, self.ny, self.nz
            p0 = d0 * T0
            cs = torch.sqrt(T0 * self.gamma)
            u0 = mach * cs

            dx = self.xc[1] - self.xc[0]
            dy = self.yc[1] - self.yc[0]
            dz = self.zc[1] - self.zc[0]

            qLx = dx * nx
            qLy = dy * ny
            qLz = dz * nz

            kx0 = 2.0 * torch.pi / qLx    # initial (wavenumber)-- need n_i ~ [1, n_max]
            ky0 = 2.0 * torch.pi / qLy    # same as above, but for the y-axis
            kz0 = 2.0 * torch.pi / qLz    # same as above, but for the z-axis

            # random/zero velocity field
            d = torch.zeros([nx, ny, nz]).to(phs.device)
            p = torch.zeros([nx, ny, nz]).to(phs.device)
            vx = torch.zeros([nx, ny, nz]).to(phs.device)
            vy = torch.zeros([nx, ny, nz]).to(phs.device)
            vz = torch.zeros([nx, ny, nz]).to(phs.device)

            # generate unique positive indices (0,1,...) for the index combinations (k,j,i), excluding k*j*i=0
            idx_combo = [(k, j, i) for k in range(-self.k_tot, self.k_tot+1) for j in range(-self.k_tot, self.k_tot+1) for i in range(-self.k_tot, self.k_tot+1) if k*j*i != 0]
            idx_combo_idxmap = {item: idx for idx, item in enumerate(idx_combo)}

            for k in range(-self.k_tot, self.k_tot + 1):
                kz = kz0 * k  # from 1 to k_tot
                for j in range(-self.k_tot, self.k_tot + 1):
                    ky = ky0 * j  # from 1 to k_tot
                    for i in range(-self.k_tot, self.k_tot + 1):
                        kx = kx0 * i  # from 1 to k_tot
                        if k * j * i == 0:  # avoiding uniform velocity
                            continue

                        # random phase;
                        phs_idx = idx_combo_idxmap[(k, j, i)]
                        phs_ = phs[phs_idx]    # (vx, vy, vz, p, d)

                        uk = (1.0 / torch.sqrt(kx ** 2 + ky ** 2 + kz ** 2)).to(phs.device)
                        kdx = (kx * self.xc[:, None, None] + ky * self.yc[None, :, None] + kz * self.zc[None, None, :]).to(phs.device)
                        vx = vx + uk * torch.sin(kdx + phs_[0])
                        vy = vy + uk * torch.sin(kdx + phs_[1])
                        vz = vz + uk * torch.sin(kdx + phs_[2])
                        p = p + uk * torch.sin(kdx + phs_[3])
                        d = d + uk * torch.sin(kdx + phs_[4])

            del (kdx, uk, phs_)

            # renormalize total velocity
            vtot = torch.sqrt(vx ** 2 + vy ** 2 + vz ** 2).mean()

            # in-place ops blocked in vmap; hence the alternative
            norm_div = u0 / vtot
            vx = vx * norm_div
            vy = vy * norm_div
            vz = vz * norm_div

            if self.constrain_max:
                d_div = torch.abs(d).max()
                p_div = torch.abs(p).max()
            else:
                d_div = torch.abs(d).mean()
                p_div = torch.abs(p).mean()

            d = d0 * (1.0 + delD * d / d_div)
            p = p0 * (1.0 + delP * p / p_div)

            u[0, ...] = d
            u[1, ...] = vx
            u[2, ...] = vy
            u[3, ...] = vz
            u[4, ...] = p

            return u
        
        u = torch.vmap(_create_3DRand_init, in_dims=(0, 0, 0, 0, 0, 0, 0), out_dims=0, randomness="same",
                       chunk_size=nb)(u, mach, phs, d0, T0, delD, delP)  # apply vmap over the 1st dimension (idx 0 -- batch)

        # perform windowing
        num_init_conds = u.shape[0]
        mask = torch.ones([num_init_conds, self.nx, self.ny, self.nz], device=device)
        _xc = torch.repeat_interleave(self.xc[None, :], repeats=num_init_conds, dim=0)
        _yc = torch.repeat_interleave(self.yc[None, :], repeats=num_init_conds, dim=0)
        _zc = torch.repeat_interleave(self.zc[None, :], repeats=num_init_conds, dim=0)

        trns = 0.01 * torch.ones_like(cond, device=self.xc.device)[:, None]

        # @kmario23: check windowing again
        def _select_W(_xc, _yc, _zc, xL, xR, yL, yR, zL, zR, trns):
            xwin = 0.5 * (torch.tanh((_xc - xL) / trns) - torch.tanh((_xc - xR) / trns))
            ywin = 0.5 * (torch.tanh((_yc - yL) / trns) - torch.tanh((_yc - yR) / trns))
            zwin = 0.5 * (torch.tanh((_zc - zL) / trns) - torch.tanh((_zc - zR) / trns))
            mask = xwin[:, None, None] * ywin[None, :, None] * zwin[None, None, :]

            return mask

        if cond.sum() > 0:
            to_be_wind_IDXS = (torch.where(cond == 1)[0]).to(mask.device)
            mask_fv = torch.vmap(_select_W, randomness="same", chunk_size=nb)(_xc, _yc, _zc, xL, xR, yL, yR, zL, zR, trns)
            mask_fv = mask_fv.to(mask.device)
            mask[to_be_wind_IDXS, ...] = mask_fv[to_be_wind_IDXS, ...]

        # apply windowing on all fields based on the mask
        u[:, :, ...] = u[:, :, ...] * mask[:, None, :, :, :]
        u[:, 0, ...] = u[:, 0, ...] + d0[:, :, None, None] * (1.0 - mask[:, :, :, :])
        u[:, 4, ...] = u[:, 4, ...] + d0[:, :, None, None] * T0[:, :, None, None] * (1.0 - mask[:, :, :, :])

        # u: [bs, 5, nx, ny, nz]
        # for active learning requirement, reformat to [bs, nx, ny, nz, nc=5]
        u = u.permute(0, 2, 3, 4, 1)  # [bs, nx, ny, nz, nc=5])
        # add a singleton dim for t (nt=1)
        u = u.unsqueeze(-2)  # [bs, nx, ny, nz, nt=1, nc=5])

        if not torch.all(torch.isfinite(u)):
            for i in range(len(u)):
                if not torch.all(torch.isfinite((u[i]))):
                    print(u[i])
            raise ValueError("non-finite values in simulator result")
        return u      # [bs, nx, ny, nz, nt, 5])

