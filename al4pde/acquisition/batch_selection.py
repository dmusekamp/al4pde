import time
import torch
import wandb
import numpy as np
from tensordict import TensorDict
from al4pde.tasks.task import Task
from al4pde.acquisition.data_schedule import DataSchedule
from al4pde.evaluation.analysis import batch_errors
from al4pde.prob_models.prob_model import ProbModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BatchSelection:
    """Base class for all selection methods.
    """

    def __init__(self, task: Task, data_schedule: DataSchedule, batch_size: int, unc_eval_mode: str,
                 unc_num_rollout_steps_rel: float):
        self.task = task
        self.batch_size = batch_size
        self.data_schedule = data_schedule
        self.unc_num_rollout_steps_rel = unc_num_rollout_steps_rel
        assert 0 < self.unc_num_rollout_steps_rel <= 1
        self.unc_eval_mode = unc_eval_mode
        self.unc_num_rollout_steps = None
        self.set_num_time_steps(len(task.sim.t_coord[::task.reduced_resolution_t]))

    def get_next_params(self, prob_model: ProbModel) -> tuple[TensorDict, torch.Tensor]:
        """Return next batch of ic and pde params."""
        raise NotImplementedError

    def set_num_time_steps(self, num_idx: int) -> None:
        self.unc_num_rollout_steps = int(num_idx * self.unc_num_rollout_steps_rel)

    def num_batches(self, al_iter: int) -> int:
        """Returns how many batches to select in this AL iteration."""
        return self.data_schedule(al_iter)

    def generate(self, prob_model: ProbModel, al_iter: int, train_loader: torch.utils.data.DataLoader) -> None:
        """Selects a set of new inputs, simulates and saves them. Starting point of the selection phase. """

        t = time.time()
        ics = []
        pde_params = []
        ic_params = []
        pde_params_normed = []
        for idx in range(self.num_batches(al_iter)):
            ic_params_b, pde_params_normed_b = self.get_next_params(prob_model)
            ic_params.append(ic_params_b.detach().cpu())
            pde_params_normed.append(pde_params_normed_b.detach().cpu())
            pde_params_b = self.task.get_pde_params(pde_params_normed_b)
            ics.append(self.task.get_ic(ic_params_b, pde_params_b).detach().cpu())
            pde_params.append(pde_params_b.detach().cpu())
        wandb.log({"al/al_iter": al_iter, "al/sel_time": time.time() - t})
        ic_params = torch.cat(ic_params, dim=0)
        self.simulate_all(prob_model, torch.concat(ics, 0), torch.concat(pde_params, 0), al_iter,
                          ic_params=ic_params, pde_params_normed=torch.concat(pde_params_normed, 0))

    def model_uncertainty(self, prob_model: ProbModel, x: torch.Tensor, grid: torch.Tensor,
                          pde_params: torch.Tensor = None) -> torch.Tensor:
        """Get the accumulated uncertainty for the whole trajectory."""
        with torch.no_grad():
            t_idx = torch.zeros([len(x)], device=x.device)
            if self.unc_eval_mode == "max":
                x, grid = prob_model.reduce_input_res(x, grid)
                _, unc = prob_model.unc_roll_out(x, grid, self.unc_num_rollout_steps, pde_param=pde_params, t_idx=t_idx)
                return unc.reshape([unc.shape[0], -1]).max(dim=1)[0]

            elif self.unc_eval_mode == "mean":
                x, grid = prob_model.reduce_input_res(x, grid)
                _, unc = prob_model.unc_roll_out(x, grid, self.unc_num_rollout_steps, pde_param=pde_params, t_idx=t_idx)
                return unc.reshape([unc.shape[0], -1]).mean(dim=1)

            else:
                raise ValueError(self.unc_eval_mode)

    def simulate_all(self, prob_model: ProbModel, ics: torch.Tensor, pde_params: torch.Tensor, al_iter: int,
                     ic_params: TensorDict, pde_params_normed: torch.Tensor) -> None:
        """Passes all selected inputs to the simulator."""
        prob_model.cpu()
        with torch.no_grad():
            torch.cuda.empty_cache()
        t = time.time()
        for i in range(int(np.ceil(float(len(ics)) / self.batch_size))):
            ic_batch = ics[i * self.batch_size: (i + 1) * self.batch_size]
            ic_params_batch = ic_params[i * self.batch_size: (i + 1) * self.batch_size]
            pde_param_batch = pde_params[i * self.batch_size: (i + 1) * self.batch_size]
            pde_params_normed_batch = pde_params_normed[i * self.batch_size: (i + 1) * self.batch_size]

            u_trajectories, u_xcoords, u_tcoords = self.task.evolve_ic(ic_batch, pde_param_batch)

            print(f"obtained trajectories shape: {u_trajectories.shape}")
            # save generated trajectories
            self.task.save_trajectories(u_trajectories, pde_param_batch,
                                        u_xcoords, u_tcoords, al_iter, i, ic_params=ic_params_batch,
                                        pde_params_normed=pde_params_normed_batch)
        wandb.log({"al/al_iter": al_iter, "al/sim_time": time.time() - t})
        print("simulation  time", time.time() - t)

        prob_model.to(device)

    @property
    def name(self):
        raise NotImplementedError
