import torch
from al4pde.models.model import Model


class ProbModel(Model):

    def uncertainty(self, xx, grid, pde_param=None, t_idx=None, return_state=False):
        raise NotImplementedError

    def unc_roll_out(self, xx, grid, final_step,  pde_param=None, t_idx=None, return_features=False):
        unc_cell = []
        if self.training_type in ['autoregressive', 'teacher_forcing']:
            pred = xx
            unc_cell = [torch.zeros_like(xx)]
            for t in range(self.initial_step, final_step):
                m, unc = self.uncertainty(xx, grid, t_idx, pde_param, True)
                pred = torch.cat((pred, m), -2)
                xx = torch.cat((xx[..., 1:, :], m), dim=-2)
                unc_cell.append(unc)
                if t_idx is not None:
                    t_idx += 1
            return pred, torch.concat(unc_cell, dim=-2)

        else:
            raise ValueError(self.training_type)

    def roll_out(self, xx, grid, final_step, pde_param=None, t_idx=None, return_features=False):
        out = self.unc_roll_out(xx, grid, final_step, pde_param,  t_idx, return_features=return_features)
        return (out[0], out[2]) if return_features else out[0]

    def sample_trajectory(self, xx, grid, final_step, pde_param=None):
        raise NotImplementedError
