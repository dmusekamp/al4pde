"""
       PDEBench

  File:     ic_gen_burgers.py
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
import torch
from torch import randint, rand, multinomial
from tensordict import TensorDict
from al4pde.tasks.ic_gen.ic_gen import ICGenerator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def uniform_(n, low, high, generator, device, requires_grad=False):
    return torch.empty(n, device=device).uniform_(low, high, generator=generator)


class ICGenBurgers(ICGenerator):
    """Generates parameters that are responsible for generating the initial conditions of the given pde.
    E.g. IC Params: (amplitude, phase)
    """

    def __init__(self, k_tot, num_choice_k, xL, xR, nx, requires_grad=True, single_fixed=False):
        super().__init__(requires_grad, single_fixed)
        self.k_tot = k_tot
        self.num_choice_k = num_choice_k
        self.xL = xL
        self.xR = xR
        self.nx = nx
        dx = (self.xR - self.xL) / self.nx
        # cell edge and cell center coordinate
        xe = torch.linspace(self.xL, self.xR, self.nx + 1)
        xc = xe[:-1] + 0.5 * dx
        self.xc = xc.to(device)

    def get_grid(self, n):
        grid = self.xc[:, None]
        return grid.expand([n, ] + list(grid.shape))

    def _initialize_ic_params(self, n: int) -> TensorDict:
        # init multi def
        selected = randint(low=0, high=self.k_tot, size=(n, self.num_choice_k), generator=self.rng,
                           device=device)

        selected = torch.nn.functional.one_hot(selected, num_classes=self.k_tot).sum(dim=1)

        kk = torch.pi * 2. * torch.arange(1, self.k_tot + 1, device=device) * selected / (self.xc[-1] - self.xc[0])

        amp = rand(size=[n, self.k_tot, 1], generator=self.rng, device=device,
                   requires_grad=self.requires_grad)

        phs = rand(size=[n, self.k_tot, 1], generator=self.rng,  device=device,
                   requires_grad=self.requires_grad)

        probs = torch.tensor([0.9, 0.1],  device=device)

        cond0 = multinomial(probs, num_samples=n, generator=self.rng, replacement=True).to(device)
        sgn = randint(0, 2, size=(n, 1), generator=self.rng, device=device) * 2 - 1
        xL = uniform_(n, 0.1, 0.45, generator=self.rng, device=device)[:, None]
        xR = uniform_(n, 0.55, 0.9, generator=self.rng, device=device)[:, None]

        cond1 = multinomial(probs, num_samples=n, generator=self.rng, replacement=True).to(device)
        return TensorDict({"amp": amp, "phs": phs, "kk": kk, "cond0": cond0, "cond1": cond1,
                           "sgn": sgn, "xL": xL, "xR": xR}, batch_size=n)

    def generate_initial_conditions(self, ic_params: TensorDict, pde_params: torch.Tensor) -> torch.Tensor:
        # Appendix II: https://arxiv.org/abs/1808.04930
        # u(x,0)

        amp = ic_params.get("amp").clamp(0, 1)
        phs = ic_params.get("phs").clamp(0, 1)
        phs = 2. * torch.pi * phs

        _u = amp * torch.sin(ic_params.get("kk")[:, :, None] * self.xc[None, None, :] + phs)
        _u = torch.sum(_u, dim=1)

        # perform absolute value function
        cond = ic_params.get("cond0")
        if cond.sum() > 0:
            abs_u_waves = torch.abs(_u[cond.bool()])
            _u[cond.bool()] = abs_u_waves

        _u = _u * ic_params.get("sgn").to(device)

        # perform window function
        cond = ic_params.get("cond1")
        cond = cond.to(device)

        _xc = torch.repeat_interleave(self.xc[None, :], repeats=len(amp), dim=0)
        mask = torch.ones_like(_xc)

        trns = 0.01 * torch.ones_like(cond)[:, None]

        # move to device
        xL = ic_params.get("xL").to(device)
        xR = ic_params.get("xR").to(device)
        trns = trns.to(device)

        # 'select_W' that returns a processed `mask` which will be applied to `_u`
        if cond.sum() > 0:
            mask = 0.5 * (torch.tanh((_xc - xL) / trns) - torch.tanh((_xc - xR) / trns))
            windowed_u_waves = _u[cond.bool()] * mask[cond.bool()]
            _u[cond.bool()] = windowed_u_waves

        return _u[..., None, None]   # [bs, nx, nt, nc]
