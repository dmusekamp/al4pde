"""
       PDEBench

  File:     burgers.py
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
import jax
import jax.numpy as jnp
from jax import device_put, lax
import numpy as np
from al4pde.tasks.solver_utils import Courant, Courant_diff, bc, limiting
from al4pde.tasks.sim.sim import Simulator
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BurgersSim(Simulator):
    """Evolve the initial conditions for the pde to get the full trajectory."""

    def __init__(self, ini_time, fin_time, dt, CFL, show_steps, if_norm, if_second_order, pde_name="Burgers"):
        super().__init__(pde_name, 1, 1, 1, dt, ini_time, fin_time)
        self.pde_param_names = "Nu"
        self.CFL = CFL
        self.show_steps = show_steps
        self.if_norm = if_norm
        self.if_second_order = if_second_order
        # self.traj_data_path = traj_data_path
        self.sim_fun = None

    def __call__(self, ic, pde_params, grid):
        ic = jnp.array(ic.detach().cpu().numpy().squeeze())
        pde_params = jnp.array(pde_params.detach().cpu().numpy().squeeze())
        grid = jnp.array(grid.detach().cpu().numpy().squeeze())

        uu_traj, uu_xc, uu_tc = self.evolve_ic_jax(ic, pde_params, grid)
        uu_traj = torch.from_numpy(np.array(uu_traj))
        uu_xc = torch.from_numpy(np.array(uu_xc))
        uu_tc = torch.from_numpy(np.array(uu_tc))
        return uu_traj, uu_xc, uu_tc

    def build_jax_sim(self, grid):
        pi_inv = 1. / jnp.pi
        dx = grid[1] - grid[0]
        dx_inv = 1. / dx
        nx = len(grid)
        it_tot = self.n_steps + 1
        ini_time = float(self.ini_time)
        fin_time = float(self.fin_time)

        def _pass(carry):
            return carry

        @jax.jit
        def evolve(carry):
            print("compiling jax sim")
            u, pde_param = carry
            t = ini_time
            tsave = t
            steps = 0
            i_save = 0
            dt = 0.
            uu = jnp.zeros([it_tot, u.shape[0]])
            uu = uu.at[0].set(u)

            cond_fun = lambda x: x[0] < fin_time

            def _body_fun(carry):
                def _show(_carry):
                    u, tsave, i_save, uu = _carry
                    uu = uu.at[i_save].set(u)
                    tsave += self.dt
                    i_save += 1
                    return (u, tsave, i_save, uu)

                t, tsave, steps, i_save, dt, u, uu, pde_param = carry

                carry = (u, tsave, i_save, uu)
                u, tsave, i_save, uu = lax.cond(t >= tsave, _show, _pass, carry)

                carry = (u, t, dt, steps, tsave, pde_param)
                u, t, dt, steps, tsave, params = lax.fori_loop(0, self.show_steps, simulation_fn, carry)

                return (t, tsave, steps, i_save, dt, u, uu, pde_param)

            carry = t, tsave, steps, i_save, dt, u, uu, pde_param
            t, tsave, steps, i_save, dt, u, uu, pde_param = lax.while_loop(cond_fun, _body_fun, carry)
            uu = uu.at[-1].set(u)

            return uu

        @jax.jit
        def simulation_fn(i, carry):
            u, t, dt, steps, tsave, pde_param_epsilon = carry
            dt_adv = Courant(u, dx) * self.CFL
            dt_dif = Courant_diff(dx, pde_param_epsilon * pi_inv) * self.CFL
            dt = jnp.min(jnp.array([dt_adv, dt_dif, fin_time - t, tsave - t]))

            def _update(carry):
                u, dt, pde_param_epsilon = carry
                # preditor step for calculating t+dt/2-th time step
                u_tmp = update(u, u, dt * 0.5, pde_param_epsilon)
                # update using flux at t+dt/2-th time step
                u = update(u, u_tmp, dt, pde_param_epsilon)
                return u, dt, pde_param_epsilon

            carry = u, dt, pde_param_epsilon
            u, dt, pde_param_epsilon = lax.cond(dt > 1.e-8, _update, _pass, carry)
            t += dt
            steps += 1
            return u, t, dt, steps, tsave, pde_param_epsilon

        @jax.jit
        def update(u, u_tmp, dt, pde_param_epsilon):
            f = flux(u_tmp, pde_param_epsilon)
            u -= dt * dx_inv * (f[1:nx + 1] - f[0:nx])
            return u

        def flux(u, pde_param_epsilon):
            _u = bc(u, dx, Ncell=nx)  # index 2 for _U is equivalent with index 0 for u
            uL, uR = limiting(_u, nx, if_second_order=1.)
            fL = 0.5 * uL ** 2
            fR = 0.5 * uR ** 2
            # upwind advection scheme
            f_upwd = 0.5 * (fR[1:nx + 2] + fL[2:nx + 3]
                            - 0.5 * jnp.abs(uL[2:nx + 3] + uR[1:nx + 2]) * (uL[2:nx + 3] - uR[1:nx + 2]))
            # source term
            f_upwd += - pde_param_epsilon * pi_inv * (_u[2:nx + 3] - _u[1:nx + 2]) * dx_inv
            return f_upwd

        return jax.pmap(jax.vmap(evolve, axis_name='j'), axis_name='i')

    def evolve_ic_jax(self, ic_u, pde_param, grid):
        # basic parameters
        xc = grid
        it_tot = self.n_steps + 1
        tc = jnp.array(self.t_coord)
        num_init_conds = ic_u.shape[0]
        print(f"number of ICs to be processed: {num_init_conds}")

        if self.sim_fun is None:
            self.sim_fun = self.build_jax_sim(grid)

        # entry point
        # this is the point of entry to call the solver to evolve the initial condition
        # which comes from the active learning variance maximation model
        # u_ic_active_learn = jnp.load('u_ic_active_learn.npy')
        # u = init_multi(xc, numbers=cfg.multi.numbers, k_tot=4, init_key=cfg.multi.init_key)
        u = ic_u
        u = device_put(u)  # putting variables in GPU (not necessary??)
        pde_param = device_put(pde_param)
        # print(f"num samples requested: {numbers}")

        local_devices = jax.local_device_count()
        u = u.reshape([local_devices, num_init_conds // local_devices, -1])   # @kmario23: reshapes the traj tensor accord. to the number of devices
        pde_param = pde_param.reshape([local_devices, num_init_conds // local_devices])
        uu = self.sim_fun((u, pde_param)).block_until_ready()
        uu = uu.reshape([num_init_conds, it_tot, -1, 1])
        return uu, xc, tc

