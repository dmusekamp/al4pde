"""
       PDEBench

  File:     ic_gen_2d_ns_rand.py
  Authors:  Makoto Takamoto (makoto.takamoto@neclab.eu)
            Marimuthu Kalimuthu (marimuthu.kalimuthu@ki.uni-stuttgart.de)
            Daniel Musekamp (daniel.musekamp@ki.uni-stuttgart.de)

NEC Laboratories Europe GmbH, Copyright (c) 2024, All rights reserved.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.

       PROPRIETARY INFORMATION ---

SOFTWARE LICENSE AGREEMENT

ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
DOWNLOAD THE SOFTWARE.

This is a license agreement ("Agreement") between your academic institution
or non-profit organization or self (called "Licensee" or "You" in this
Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
Agreement).  All rights not specifically granted to you in this Agreement
are reserved for Licensor.

RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
ownership of any copy of the Software (as defined below) licensed under this
Agreement and hereby grants to Licensee a personal, non-exclusive,
non-transferable license to use the Software for noncommercial research
purposes, without the right to sublicense, pursuant to the terms and
conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
Agreement, the term "Software" means (i) the actual copy of all or any
portion of code for program routines made accessible to Licensee by Licensor
pursuant to this Agreement, inclusive of backups, updates, and/or merged
copies permitted hereunder or subsequently supplied by Licensor,  including
all or any file structures, programming instructions, user interfaces and
screen formats and sequences as well as any and all documentation and
instructions related to it, and (ii) all or any derivatives and/or
modifications created or made by You to any of the items specified in (i).

CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
proprietary to Licensor, and as such, Licensee agrees to receive all such
materials and to use the Software only in accordance with the terms of this
Agreement.  Licensee agrees to use reasonable effort to protect the Software
from unauthorized use, reproduction, distribution, or publication. All
publication materials mentioning features or use of this software must
explicitly include an acknowledgement the software was developed by NEC
Laboratories Europe GmbH.

COPYRIGHT: The Software is owned by Licensor.

PERMITTED USES:  The Software may be used for your own noncommercial
internal research purposes. You understand and agree that Licensor is not
obligated to implement any suggestions and/or feedback you might provide
regarding the Software, but to the extent Licensor does so, you are not
entitled to any compensation related thereto.

DERIVATIVES: You may create derivatives of or make modifications to the
Software, however, You agree that all and any such derivatives and
modifications will be owned by Licensor and become a part of the Software
licensed to You under this Agreement.  You may only use such derivatives and
modifications for your own noncommercial internal research purposes, and you
may not otherwise use, distribute or copy such derivatives and modifications
in violation of this Agreement.

BACKUPS:  If Licensee is an organization, it may make that number of copies
of the Software necessary for internal noncommercial use at a single site
within its organization provided that all information appearing in or on the
original labels, including the copyright and trademark notices are copied
onto the labels of the copies.

USES NOT PERMITTED:  You may not distribute, copy or use the Software except
as explicitly permitted herein. Licensee has not been granted any trademark
license as part of this Agreement.  Neither the name of NEC Laboratories
Europe GmbH nor the names of its contributors may be used to endorse or
promote products derived from this Software without specific prior written
permission.

You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
whole or in part, or provide third parties access to prior or present
versions (or any parts thereof) of the Software.

ASSIGNMENT: You may not assign this Agreement or your rights hereunder
without the prior written consent of Licensor. Any attempted assignment
without such consent shall be null and void.

TERM: The term of the license granted by this Agreement is from Licensee's
acceptance of this Agreement by downloading the Software or by using the
Software until terminated as provided below.

The Agreement automatically terminates without notice if you fail to comply
with any provision of this Agreement.  Licensee may terminate this Agreement
by ceasing using the Software.  Upon any termination of this Agreement,
Licensee will delete any and all copies of the Software. You agree that all
provisions which operate to protect the proprietary rights of Licensor shall
remain in force should breach occur and that the obligation of
confidentiality described in this Agreement is binding in perpetuity and, as
such, survives the term of the Agreement.

FEE: Provided Licensee abides completely by the terms and conditions of this
Agreement, there is no fee due to Licensor for Licensee's use of the
Software in accordance with this Agreement.

DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
RELATED MATERIALS.

SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
provided as part of this Agreement.

EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
permitted under applicable law, Licensor shall not be liable for direct,
indirect, special, incidental, or consequential damages or lost profits
related to Licensee's use of and/or inability to use the Software, even if
Licensor is advised of the possibility of such damage.

EXPORT REGULATION: Licensee agrees to comply with any and all applicable
export control laws, regulations, and/or other laws related to embargoes and
sanction programs administered by law.

SEVERABILITY: If any provision(s) of this Agreement shall be held to be
invalid, illegal, or unenforceable by a court or other tribunal of competent
jurisdiction, the validity, legality and enforceability of the remaining
provisions shall not in any way be affected or impaired thereby.

NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
or remedy under this Agreement shall be construed as a waiver of any future
or other exercise of such right or remedy by Licensor.

GOVERNING LAW: This Agreement shall be construed and enforced in accordance
with the laws of Germany without reference to conflict of laws principles.
You consent to the personal jurisdiction of the courts of this country and
waive their rights to venue outside of Germany.

ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
entire agreement between Licensee and Licensor as to the matter set forth
herein and supersedes any previous agreements, understandings, and
arrangements between the parties relating hereto.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
"""
import os
import torch
from torch import randint, rand, multinomial
from tensordict import TensorDict
from al4pde.tasks.ic_gen.ic_gen import ICGenerator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def uniform_(n, low, high, generator, device):
    return torch.empty(n, device=device).uniform_(low, high, generator=generator)


class ICGenNSRand(ICGenerator):
    """Generates parameters that are responsible for generating the initial conditions of the given pde.
    E.g. IC Params: (amplitude, phase)
    Needs torch 2.1 or higher, because of the chunk_size argument in vmap
    """

    def __init__(self,  k_tot, xL, xR, yL, yR, zL, zR, nx, ny, nz, gamma, mach_min, mach_max, d0Min, d0Max, T0Min,
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
        gridx, gridy = torch.meshgrid(self.xc, self.yc, indexing='ij')
        grid = torch.stack([gridx, gridy], dim=-1)
        return grid.expand([n, ] + list(grid.shape))     # [bs, nx, ny, 2]

    def _initialize_ic_params(self, n: int) -> TensorDict:
        """
        n: number of initial conditions
        currently making only the phase and mach number to be optimized for data acquisition
        """
        total_phases = len(range(-self.k_tot, self.k_tot)) ** 2
        phs = 2.0 * torch.pi * rand(size=[n, total_phases, 4], generator=self.rng, device=device,
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

        probs = torch.tensor([0.5, 0.5], device=device)
        cond = torch.multinomial(probs, num_samples=n, generator=self.rng, replacement=True)

        return TensorDict({"phs": phs, "mach": mach, "d0": d0, "T0": T0, "delD": delD, "delP": delP,
                           "xL": xL, "xR": xR, "yL": yL, "yR": yR, "cond": cond}, batch_size=n)

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
        cond = ic_params.get("cond")

        u = torch.zeros([len(ic_params), 5, self.nx, self.ny, self.nz], device=device)
        nb = u.shape[0]

        def _create_2DRand_init(u, mach, phs, d0, T0, delD, delP):
            # print("Entering _create 2D Rand Init!")
            # nx, ny, nz = self.xc.shape[0], self.yc.shape[0], self.zc.shape[0]
            nx, ny, nz = self.nx, self.ny, self.nz
            p0 = d0 * T0
            cs = torch.sqrt(T0 * self.gamma)
            u0 = mach * cs

            dx = self.xc[1] - self.xc[0]
            dy = self.yc[1] - self.yc[0]

            qLx = dx * nx
            qLy = dy * ny

            # random/zero velocity field
            d = torch.zeros([nx, ny, nz]).to(phs.device)
            p = torch.zeros([nx, ny, nz]).to(phs.device)
            vx = torch.zeros([nx, ny, nz]).to(phs.device)
            vy = torch.zeros([nx, ny, nz]).to(phs.device)

            kx0 = 2.0 * torch.pi / qLx    # initial (wavenumber)-- need n_i ~ [1, n_max]
            ky0 = 2.0 * torch.pi / qLy    # same as above, but for the y-axis

            # generate unique positive indices (0,1,...) for the index combinations (j,i), excluding j*i=0
            idx_combo = [(j, i) for j in range(-self.k_tot, self.k_tot+1) for i in range(-self.k_tot, self.k_tot+1) if i * j != 0]
            idx_combo_idxmap = {item: idx for idx, item in enumerate(idx_combo)}

            for j in range(-self.k_tot, self.k_tot + 1):
                ky = ky0 * j  # from 1 to k_tot
                for i in range(-self.k_tot, self.k_tot + 1):
                    kx = kx0 * i  # from 1 to k_tot
                    if i * j == 0:  # avoiding uniform velocity
                        continue

                    # random phase;
                    phs_idx = idx_combo_idxmap[(j, i)]
                    phs_ = phs[phs_idx]    # (vx, vy, p, d)

                    uk = (1.0 / torch.sqrt(torch.sqrt(kx ** 2 + ky ** 2))).to(phs.device)
                    kdx = (kx * self.xc[:, None, None] + ky * self.yc[None, :, None]).to(phs.device)
                    vx = vx + uk * torch.sin(kdx + phs_[0])
                    vy = vy + uk * torch.sin(kdx + phs_[1])
                    p = p + uk * torch.sin(kdx + phs_[2])
                    d = d + uk * torch.sin(kdx + phs_[3])

            del (kdx, uk, phs)

            # renormalize total velocity
            vtot = torch.sqrt(vx ** 2 + vy ** 2).mean()

            # in-place ops blocked in vmap; hence the alternative
            norm_div = u0 / vtot
            vx = vx * norm_div
            vy = vy * norm_div
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
            u[4, ...] = p

            return u
        
        u = torch.vmap(_create_2DRand_init, in_dims=(0, 0, 0, 0, 0, 0, 0), out_dims=0, randomness="same",
                       chunk_size=nb)(u, mach, phs, d0, T0, delD, delP)

        # perform windowing
        num_init_conds = u.shape[0]
        mask = torch.ones([num_init_conds, self.nx, self.ny], device=device)
        _xc = torch.repeat_interleave(self.xc[None, :], repeats=num_init_conds, dim=0)
        _yc = torch.repeat_interleave(self.yc[None, :], repeats=num_init_conds, dim=0)

        trns = 0.01 * torch.ones_like(cond, device=self.xc.device)[:, None]

        def _select_W(_xc, _yc, xL, xR, yL, yR, trns):
            xwin = 0.5 * (torch.tanh((_xc - xL) / trns) - torch.tanh((_xc - xR) / trns))
            ywin = 0.5 * (torch.tanh((_yc - yL) / trns) - torch.tanh((_yc - yR) / trns))
            mask = xwin[:, None] * ywin[None, :]

            return mask

        if cond.sum() > 0:
            to_be_wind_IDXS = (torch.where(cond == 1)[0]).to(mask.device)
            mask_fv = torch.vmap(_select_W, randomness="same", chunk_size=nb)(_xc, _yc, xL, xR, yL, yR, trns)
            mask_fv = mask_fv.to(mask.device)
            mask[to_be_wind_IDXS, ...] = mask_fv[to_be_wind_IDXS, ...]

        # apply windowing on all fields based on the mask
        u[:, :, ...] = u[:, :, ...] * mask[:, None, :, :, None]
        u[:, 0, ...] = u[:, 0, ...] + d0[:, :, None, None] * (1.0 - mask[:, :, :, None])
        u[:, 4, ...] = u[:, 4, ...] + d0[:, :, None, None] * T0[:, :, None, None] * (1.0 - mask[:, :, :, None])
        # u: [bs, 5, nx, ny, nz],   nz = 1 for 2D
        # for active learning requirement, remove z dimension and reformat to [bs, nx, ny, 5]
        u = u.squeeze().permute(0, 2, 3, 1)  # [bs, nx, ny, 5])
        # remove the extraneous Vz field since the ML model uses only 4 fields (DD, Vx, Vy, PP) for 2D
        u = u[:, :, :, (0, 1, 2, 4)].unsqueeze(-2)
        if not torch.all(torch.isfinite(u)):
            for i in range(len(u)):
                if not torch.all(torch.isfinite((u[i]))):
                    print(u[i])
            raise ValueError("non-finite values in simulator result")
        return u      # [bs, nx, ny, nt, 4])
