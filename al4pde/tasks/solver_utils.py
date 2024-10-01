"""
       PDEBench

  File:     solver_utils.py
  Authors:  Makoto Takamoto (makoto.takamoto@neclab.eu)

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

import jax.numpy as jnp

def VLlimiter(a, b, c, alpha=2.):
    return jnp.sign(c) \
        * (0.5 + 0.5 * jnp.sign(a * b)) \
        * jnp.minimum(alpha * jnp.minimum(jnp.abs(a), jnp.abs(b)), jnp.abs(c))


def limiting(u, Ncell, if_second_order):
    # under construction
    duL = u[1:Ncell + 3] - u[0:Ncell + 2]
    duR = u[2:Ncell + 4] - u[1:Ncell + 3]
    duM = (u[2:Ncell + 4] - u[0:Ncell + 2]) * 0.5
    gradu = VLlimiter(duL, duR, duM) * if_second_order
    uL, uR = jnp.zeros_like(u), jnp.zeros_like(u)
    uL = uL.at[1:Ncell + 3].set(u[1:Ncell + 3] - 0.5 * gradu)  # left of cell
    uR = uR.at[1:Ncell + 3].set(u[1:Ncell + 3] + 0.5 * gradu)  # right of cell
    return uL, uR


def limiting_HD(u, if_second_order):
    nd, nx, ny, nz = u.shape
    uL, uR = u, u
    nx -= 4

    duL = u[:, 1:nx + 3, :, :] - u[:, 0:nx + 2, :, :]
    duR = u[:, 2:nx + 4, :, :] - u[:, 1:nx + 3, :, :]
    duM = (u[:, 2:nx + 4, :, :] - u[:, 0:nx + 2, :, :]) * 0.5
    gradu = VLlimiter(duL, duR, duM) * if_second_order

    uL = uL.at[:, 1:nx + 3, :, :].set(u[:, 1:nx + 3, :, :] - 0.5 * gradu)  # left of cell
    uR = uR.at[:, 1:nx + 3, :, :].set(u[:, 1:nx + 3, :, :] + 0.5 * gradu)  # right of cell

    uL = jnp.where(uL[0] > 0., uL, u)
    uL = jnp.where(uL[4] > 0., uL, u)
    uR = jnp.where(uR[0] > 0., uR, u)
    uR = jnp.where(uR[4] > 0., uR, u)

    return uL, uR


def Courant(u, dx):
    stability_adv = dx / (jnp.max(jnp.abs(u)) + 1.e-8)
    return stability_adv


def Courant_diff(dx, epsilon=1.e-3):
    stability_dif = 0.5 * dx ** 2 / (epsilon + 1.e-8)
    return stability_dif


def Courant_HD(u, dx, dy, dz, gamma):
    cs = jnp.sqrt(gamma * u[4] / u[0])  # sound velocity
    stability_adv_x = dx / (jnp.max(cs + jnp.abs(u[1])) + 1.e-8)
    stability_adv_y = dy / (jnp.max(cs + jnp.abs(u[2])) + 1.e-8)
    stability_adv_z = dz / (jnp.max(cs + jnp.abs(u[3])) + 1.e-8)
    stability_adv = jnp.min(jnp.array([stability_adv_x, stability_adv_y, stability_adv_z]))
    return stability_adv


def Courant_vis_HD(dx, dy, dz, eta, zeta):
    visc = 4. / 3. * eta + zeta  # maximum
    stability_dif_x = 0.5 * dx ** 2 / (visc + 1.e-8)
    stability_dif_y = 0.5 * dy ** 2 / (visc + 1.e-8)
    stability_dif_z = 0.5 * dz ** 2 / (visc + 1.e-8)
    stability_dif = jnp.min(jnp.array([stability_dif_x, stability_dif_y, stability_dif_z]))
    return stability_dif


def bc(u, dx, Ncell, mode='periodic'):
    _u = jnp.zeros(Ncell + 4)  # because of 2nd-order precision in space
    _u = _u.at[2:Ncell + 2].set(u)
    if mode == 'periodic':  # periodic boundary condition
        _u = _u.at[0:2].set(u[-2:])  # left hand side
        _u = _u.at[Ncell + 2:Ncell + 4].set(u[0:2])  # right hand side
    elif mode == 'reflection':
        _u = _u.at[0].set(- u[3])  # left hand side
        _u = _u.at[1].set(- u[2])  # left hand side
        _u = _u.at[-2].set(- u[-3])  # right hand side
        _u = _u.at[-1].set(- u[-4])  # right hand side
    elif mode == 'copy':
        _u = _u.at[0].set(u[3])  # left hand side
        _u = _u.at[1].set(u[2])  # left hand side
        _u = _u.at[-2].set(u[-3])  # right hand side
        _u = _u.at[-1].set(u[-4])  # right hand side

    return _u




def bc_HD(u, mode):
    _, Nx, Ny, Nz = u.shape
    Nx -= 2
    Ny -= 2
    Nz -= 2
    if mode == 'periodic':  # periodic boundary condition
        # left hand side
        u = u.at[:, 0:2, 2:-2, 2:-2].set(u[:, Nx - 2:Nx, 2:-2, 2:-2])  # x
        u = u.at[:, 2:-2, 0:2, 2:-2].set(u[:, 2:-2, Ny - 2:Ny, 2:-2])  # y
        u = u.at[:, 2:-2, 2:-2, 0:2].set(u[:, 2:-2, 2:-2, Nz - 2:Nz])  # z
        # right hand side
        u = u.at[:, Nx:Nx + 2, 2:-2, 2:-2].set(u[:, 2:4, 2:-2, 2:-2])
        u = u.at[:, 2:-2, Ny:Ny + 2, 2:-2].set(u[:, 2:-2, 2:4, 2:-2])
        u = u.at[:, 2:-2, 2:-2, Nz:Nz + 2].set(u[:, 2:-2, 2:-2, 2:4])
    elif mode == 'trans':  # periodic boundary condition
        # left hand side
        u = u.at[:, 0, 2:-2, 2:-2].set(u[:, 3, 2:-2, 2:-2])  # x
        u = u.at[:, 2:-2, 0, 2:-2].set(u[:, 2:-2, 3, 2:-2])  # y
        u = u.at[:, 2:-2, 2:-2, 0].set(u[:, 2:-2, 2:-2, 3])  # z
        u = u.at[:, 1, 2:-2, 2:-2].set(u[:, 2, 2:-2, 2:-2])  # x
        u = u.at[:, 2:-2, 1, 2:-2].set(u[:, 2:-2, 2, 2:-2])  # y
        u = u.at[:, 2:-2, 2:-2, 1].set(u[:, 2:-2, 2:-2, 2])  # z
        # right hand side
        u = u.at[:, -2, 2:-2, 2:-2].set(u[:, -3, 2:-2, 2:-2])
        u = u.at[:, 2:-2, -2, 2:-2].set(u[:, 2:-2, -3, 2:-2])
        u = u.at[:, 2:-2, 2:-2, -2].set(u[:, 2:-2, 2:-2, -3])
        u = u.at[:, -1, 2:-2, 2:-2].set(u[:, -4, 2:-2, 2:-2])
        u = u.at[:, 2:-2, -1, 2:-2].set(u[:, 2:-2, -4, 2:-2])
        u = u.at[:, 2:-2, 2:-2, -1].set(u[:, 2:-2, 2:-2, -4])
    elif mode == 'KHI':  # x: periodic, y, z : trans
        # left hand side
        u = u.at[:, 0:2, 2:-2, 2:-2].set(u[:, Nx - 2:Nx, 2:-2, 2:-2])  # x
        u = u.at[:, 2:-2, 0, 2:-2].set(u[:, 2:-2, 3, 2:-2])  # y
        u = u.at[:, 2:-2, 2:-2, 0].set(u[:, 2:-2, 2:-2, 3])  # z
        u = u.at[:, 2:-2, 1, 2:-2].set(u[:, 2:-2, 2, 2:-2])  # y
        u = u.at[:, 2:-2, 2:-2, 1].set(u[:, 2:-2, 2:-2, 2])  # z
        # right hand side
        u = u.at[:, Nx:Nx + 2, 2:-2, 2:-2].set(u[:, 2:4, 2:-2, 2:-2])
        u = u.at[:, 2:-2, -2, 2:-2].set(u[:, 2:-2, -3, 2:-2])
        u = u.at[:, 2:-2, 2:-2, -2].set(u[:, 2:-2, 2:-2, -3])
        u = u.at[:, 2:-2, -1, 2:-2].set(u[:, 2:-2, -4, 2:-2])
        u = u.at[:, 2:-2, 2:-2, -1].set(u[:, 2:-2, 2:-2, -4])
    return u