import os
import time
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn as nn

from al4pde.evaluation.visualization import plot_worst_traj
from al4pde.prob_models.prob_model import ProbModel
from al4pde.models.build_wrapper import build_wrapper
from al4pde.evaluation.stats import LossUncCorr, UncAvg


class Ensemble(ProbModel):

    def __init__(self, task, unc_roll_out_mode, base_models, synchronized_training=True, adapt_unc_to_loss=True,
                 first_model_prediction=False):
        b0 = base_models[0]
        super().__init__(task,  b0.training_type, b0.t_train, b0.batch_size, b0.val_period, b0.vis_period, b0.loss)
        self.base_models = nn.ModuleList(base_models)
        self.unc_roll_out_mode = unc_roll_out_mode
        self.synchronized_training = synchronized_training
        self.adapt_unc_to_loss = adapt_unc_to_loss
        self.first_model_prediction = first_model_prediction

        if not synchronized_training:
            self.base_models[-1].ens_val_callback = self.validate

        self.stats.append(UncAvg("unc"))
        self.stats.append(LossUncCorr("corr_unc_loss", self.loss))

    def train_single_epoch(self, current_epoch, total_epoch, num_epoch):
        for m in self.base_models:
            m.train_single_epoch(current_epoch, total_epoch, num_epoch)
        # validate

    def train_n_epoch(self, al_iter: int, num_epoch: int, step_offset: int, vis: bool = True,
                      prefix: str = "",is_last=False) -> float:
        if self.synchronized_training:
            return super().train_n_epoch(al_iter, num_epoch, step_offset)
        else:
            total_time = 0
            for m_idx, m in enumerate(self.base_models):
                m.init_training(al_iter)
                if m_idx > 0 and is_last:
                    break
                if m_idx == 0:
                    self.task_norm = m.task_norm
                for i in range(num_epoch):
                    t = time.time()
                    m.train_single_epoch(i, step_offset + i, num_epoch)
                    total_time += time.time() - t
                    if m_idx == 0:
                        if i % self.val_period == 0:
                            m.validate(step_offset + i, prefix=prefix)
                        if vis and ((i > 0 and i % self.vis_period == 0) or i == num_epoch - 1):
                            m.visualize(step_offset + i)
            #self.validate(num_epoch + step_offset)
            #self.visualize(num_epoch + step_offset)
            return total_time

    def init_training(self, al_iter, load_train_data=True):
        for m in self.base_models:
            m.init_training(al_iter, load_train_data)
        self.task_norm = self.base_models[0].task_norm

    @property
    def val_loader(self):
        return self.base_models[0].val_loader

    @property
    def train_loader(self):
        return self.base_models[0].train_loader

    @property
    def train_loader_full_traj(self):
        return self.base_models[0].train_loader_full_traj

    def eval_pred(self, xx, yy, grid, param=None, t_idx=None):
        xx = xx.to(device)
        grid = grid.to(device)
        yy = yy.to(device)
        param = param.to(device)
        t_idx = t_idx.to(device)
        pred, unc = self.unc_roll_out(xx, grid, yy.shape[-2], param, t_idx)
        onestep_pred = self.one_step_pred(yy, grid, param, t_idx)
        return {"pred": pred, "onestep_pred": onestep_pred, "yy": yy, "param": param, "unc": unc}

    def unc_roll_out(self, xx, grid, final_step, pde_param=None, t_idx=None, return_features=False):
        if self.unc_roll_out_mode == "independent":
            return self.unc_independent_rollout(xx, grid, final_step, pde_param, t_idx, return_features)
        elif self.unc_roll_out_mode == "mean":
            if return_features:
                raise NotImplementedError
            return super().unc_roll_out(xx, grid, final_step, pde_param, t_idx)
        else:
            return ValueError(self.unc_roll_out_mode)

    def uncertainty(self, xx, grid, pde_param=None, t_idx=None, return_state=False):
        m_outputs = torch.stack([m(xx, grid, pde_param) for m in self.base_models])
        unc = m_outputs.var(dim=0)
        if return_state:
            return m_outputs.mean(dim=0), unc
        else:
            return unc

    def forward(self, xx, grid, pde_param=None, t_idx=None, return_features=False):
        if self.first_model_prediction:
            return self.base_models[0](xx, grid, pde_param, t_idx)
        m_outputs = torch.stack([m(xx, grid, pde_param, t_idx) for m in self.base_models])
        return m_outputs.mean(dim=0)

    def unc_independent_rollout(self, xx, grid, final_step, pde_param=None, t_idx=None, return_features=False):
        out = self._roll_out_all(xx, grid, final_step, pde_param, t_idx, return_features)
        traj = out if not return_features else out[0]
        mean_traj = torch.mean(traj, dim=0)
        if self.adapt_unc_to_loss:
            estimated_losses = []
            for t in traj:
                estimated_losses.append(self.loss(t, mean_traj, reduction=None, initial_step=0))
            unc_traj = torch.stack(estimated_losses, 0).mean(0)
        else:
            unc_traj = torch.var(traj, dim=0)

        if self.first_model_prediction:
            mean_traj = traj[0]
        if return_features:
            return mean_traj, unc_traj, out[1]

        return mean_traj, unc_traj

    def unc_mean_propagation_rollout(self, xx, grid, final_step, pde_param=None, t_idx=None):
        raise NotImplementedError() # does not deal with first model and different losses yet
        #return super().unc_roll_out(xx, grid, final_step, pde_param)

    def visualize(self, total_epoch):
        super().visualize(total_epoch)
        # over t
        img_save_path = os.path.join(self.task.run_save_path, "img")

        # plot trajectories
        if self.task.spatial_dim == 1:
            plot_worst_traj(self, self.val_loader, img_save_path, "traj_" + str(total_epoch))

    def _roll_out_all(self, xx, grid, final_step, pde_param=None, t_idx=None, return_features=False):
        b_m_out = [b.roll_out(xx, grid, final_step, pde_param, t_idx, return_features) for b in self.base_models]
        if return_features:
            traj = torch.stack([o[0] for o in b_m_out], dim=0)
            feat = torch.concat([o[1] for o in b_m_out], dim=-1)
            return traj, feat
        return torch.stack(b_m_out, dim=0)

    def sample_trajectory(self, xx, grid, final_step, pde_param=None, t_idx=None):
        traj = self._roll_out_all(xx, grid, final_step, pde_param, t_idx)
        idx = torch.randint(0, len(self.base_models), size=(xx.shape[0],), device=device)
        sample = torch.concat([traj[idx[i], i:i+1] for i in range(xx.shape[0])], 0)
        return sample


def build_ensemble(task, cfg):
    base_models = [build_wrapper(task, cfg.model_wrapper, name="ens_member_" + str(i))
                   for i in range(cfg.num_models_ensemble)]
    return Ensemble(task, cfg.unc_roll_out_mode, base_models, cfg.synchronized_training, first_model_prediction=
                    cfg.first_model_prediction)
