"""
       CAPE

  File:     fno_wrapper.py
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
import hydra.utils
import torch
import functools
from al4pde.modules.fno import FNO1d, FNO2d, FNO3d
from al4pde.models.wrapper import ModelWrapper
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FNOWrapper(ModelWrapper):
    """Wrapper for FNO."""

    def forward(self, xx, grid, pde_param=None, t_idx=None, return_features=False):

        if self.norm_mode is not None:
            xx = self.task_norm.norm_traj(xx)
            grid = self.task_norm.norm_grid(grid)
            if pde_param is not None:
                pde_param = self.task_norm.norm_param(pde_param)
        dimensions = len(xx.shape)
        if pde_param is not None:
            if t_idx is not None and not self.task.sim.autonomous:
                pde_param = torch.concat([pde_param, t_idx[:, None]], -1)
            if dimensions == 4:
                xx = torch.cat((xx, pde_param[:, None, None, :].repeat(1, xx.shape[1], xx.shape[-2], 1)), dim=-1)
            elif dimensions == 5:
                xx = torch.cat((xx, pde_param[:, None, None, None, :].repeat(1, xx.shape[1], xx.shape[2],
                                                                             xx.shape[-2], 1)),
                               dim=-1)
            elif dimensions == 6:
                xx = torch.cat((xx, pde_param[:, None, None, None, None, :].repeat(1, xx.shape[1], xx.shape[2],
                                                                                   xx.shape[3], xx.shape[-2], 1)),
                               dim=-1)
            xx = torch.flatten(xx, -2)
            if return_features:
                out, feat = self.model(xx, grid, True)
                out = out[..., : -pde_param.shape[-1]]
            else:
                out = self.model(xx, grid)[..., :-pde_param.shape[-1]]
        else:
            if return_features:  # if statement because not everywhere implemented and normal call should still work
                out, feat = self.model(xx, grid, True)
            else:
                out = self.model(xx, grid)
        if self.norm_mode is not None:
            out = self.task_norm.denorm_traj(out)
        if return_features:
            return out, feat
        return out


def build_model(task, cfg):
    # make the model "condition" on the pde parameter (e.g. burgers nu)
    if cfg.if_conditional:
        _num_channels = task.num_channels + task.num_pde_param
    else:
        _num_channels = task.num_channels
    if not task.sim.autonomous:
        _num_channels += 1
    if task.spatial_dim == 1:
        model = FNO1d(num_channels=_num_channels,  # originally num_channels
                      width=cfg.width,
                      modes=cfg.modes,
                      initial_step=task.initial_step, predict_delta=cfg.predict_delta).to(device)
        print("MODEL: Building FNO1D")
    elif task.spatial_dim == 2:
        model = FNO2d(num_channels=_num_channels,  # originally num_channels
                      width=cfg.width,
                      modes1=cfg.modes,
                      modes2=cfg.modes,
                      initial_step=task.initial_step).to(device)
        print("MODEL: Building FNO2D")
    elif task.spatial_dim == 3:
        model = FNO3d(num_channels=_num_channels,  # originally num_channels
                      width=cfg.width,
                      modes1=cfg.modes,
                      modes2=cfg.modes,
                      modes3=cfg.modes,
                      initial_step=task.initial_step).to(device)
        print("MODEL: Building FNO3D")
    else:
        raise ValueError
    return model


def build_fno_wrapper(task, cfg, name=""):
    model_factory = functools.partial(build_model, task=task, cfg=cfg.model)
    optimizer_factory = hydra.utils.instantiate(cfg.optimizer, _partial_=True)
    scheduler_factory = hydra.utils.instantiate(cfg.scheduler, _partial_=True)
    loss = hydra.utils.instantiate(cfg.loss)
    return FNOWrapper(task, cfg.training_type, cfg.t_train, cfg.batch_size, model_factory, optimizer_factory,
                        scheduler_factory, cfg.num_workers, cfg.val_period, cfg.vis_period,
                        cfg.block_grad, loss, name=name,
                        step_batch=cfg.step_batch, num_train_steps=cfg.num_train_steps, max_grad_norm=cfg.max_grad_norm,
                        norm_mode=cfg.norm_mode, noise=cfg.noise, noise_mode=cfg.noise_mode)
