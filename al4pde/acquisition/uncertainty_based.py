import torch
import wandb
from torch.utils.data import TensorDataset, DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from al4pde.acquisition.pool_based import PoolBased
from al4pde.prob_models.prob_model import ProbModel


def top_k(unc: torch.Tensor, k: int) -> torch.Tensor:
    return torch.argsort(unc, descending=True)[:k]


def power_sampling(unc: torch.Tensor, k: int, beta=1) -> torch.Tensor:
    weights = torch.pow(unc, beta)
    prob = weights / weights.sum()
    return torch.multinomial(prob, k, replacement=False)


def random(n: int, k: int) -> torch.Tensor:
    weights = torch.ones(n)
    prob = weights / weights.sum()
    return torch.multinomial(prob, k, replacement=False)


class UncertaintyBased(PoolBased):

    def __init__(self, task, data_schedule, batch_size, pool_size, unc_eval_mode,
                 unc_num_rollout_steps_rel, selection_mode, power_beta=1, pred_batch_size=128):
        super().__init__(task, data_schedule, batch_size, pool_size, unc_eval_mode, unc_num_rollout_steps_rel,
                         pred_batch_size=pred_batch_size)
        self.selection_mode = selection_mode
        assert selection_mode in ["random", "top_k", "power"]
        self.power_beta = power_beta

    def select_next(self, prob_model: ProbModel, ic_pool: torch.Tensor, pde_param_pool: torch.Tensor,
                    ic_train: torch.Tensor, pde_param_train: torch.Tensor, grid: torch.Tensor, al_iter: int,
                    train_loader=None) -> torch.Tensor:
        dataset = TensorDataset(ic_pool, pde_param_pool)
        unc = []
        if self.selection_mode != "random":
            loader = DataLoader(dataset, batch_size=self.pred_batch_size)
            grid = grid.to(device)
            for i, batch in enumerate(loader):
                ic = batch[0].to(device)
                pde_param = batch[1].to(device)
                grid_batch = grid.expand([len(ic), ] + list(grid.shape))
                unc_batch = self.model_uncertainty(prob_model, ic, grid_batch, pde_param).detach().cpu()
                unc.append(unc_batch.reshape((len(unc_batch), -1)).mean(1))
            unc = torch.concat(unc, dim=0)
        n_samples = self.num_batches(al_iter) * self.batch_size
        if self.selection_mode == "top_k":
            sel_idx = top_k(unc, n_samples)
        elif self.selection_mode == "power":
            sel_idx = power_sampling(unc, n_samples, self.power_beta)
        elif self.selection_mode == "random":
            sel_idx = random(len(ic_pool), n_samples)
        else:
            raise ValueError(self.selection_mode)
        if self.selection_mode != "random":
            wandb.log({"al/acq_avg_unc": unc[sel_idx].mean(), "al/al_iter": al_iter})
        return sel_idx

    @property
    def name(self):
        return self.selection_mode
