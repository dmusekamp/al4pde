import time
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn as nn
import wandb

from al4pde.evaluation.visualization import plot_traj
from al4pde.evaluation.visualization import plot_traj_2d
from al4pde.utils import subsample
from al4pde.evaluation.stats import build_standard_stats
from al4pde.evaluation.vis import build_standard_plots

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Model(nn.Module):
    """Base class for all models"""

    def __init__(self, task, training_type, t_train, batch_size, val_period, vis_period, loss):
        super().__init__()
        self.training_type = training_type
        self.t_train = t_train
        self.task = task
        self.batch_size = batch_size
        self._val_loader = None
        self.val_period = val_period
        self.vis_period = vis_period
        self.initial_step = task.initial_step
        self.reduced_resolution = task.reduced_resolution
        self.reduced_resolution_t = task.reduced_resolution_t
        self.skip_initial_steps = task.skip_initial_steps
        self.reduced_batch = task.reduced_batch
        self._train_loader = None
        self.loss = loss
        self.task_norm = None
        self.stats = build_standard_stats(self.loss, task)
        self.plots = build_standard_plots(self.loss, task.img_save_path, task)

    def init_training(self, al_iter: int, load_train_data=True):
        """Initialize model, optimizer, scheduler to (re-)start training. """
        raise NotImplementedError

    @property
    def val_loader(self) -> DataLoader:
        """Returns the data loader of the validation data set."""
        return self._val_loader

    @property
    def train_loader(self) -> DataLoader:
        """Returns the data loader of the train data set like it is used for the actual training (e.g. only pairs)."""
        return self._train_loader

    @property
    def train_loader_full_traj(self) -> DataLoader:
        """Returns the data loader of the training data set with full, unaltered trajectories."""
        return self._train_loader

    def forward(self, xx: Tensor, grid: Tensor, param: Tensor = None, t_idx: Tensor = None,
                return_features: bool = False) -> Tensor:
        """Computes the forward pass

        @param xx: initial state [b, nx1, nx2, ..., nt=1, nc]
        @param grid: grid coordinates  [b, nx1, nx2, ..., num_dim]
        @param param: [b, n_param]
        @param t_idx: [b, 1]. Time index of xx (only important for non-autonomous PDEs)
        @param return_features: Should the model return latent features?
        @return: The next state.  [b, nx1, nx2, ..., nt=1, nc]
        """

        raise NotImplementedError

    def roll_out(self, xx: Tensor, grid: Tensor, final_step: int, param: Tensor = None,
                 t_idx: Tensor = None, return_features=False):
        """Computes the rollout of a model.

        @param xx: initial state [b, nx1, nx2, ..., nt=1, nc]
        @param grid: grid coordinates  [b, nx1, nx2, ..., num_dim]
        @param param: [b, n_param]
        @param t_idx: [b, 1]. Time index of xx (only important for non-autonomous PDEs)
        @param return_features: Should the model return latent features?
        @return: The rollout including xx as the first entry,  [b, nx1, nx2, ...,  final_step + 1, nc]
        """
        pred = xx
        features = []

        for t in range(self.initial_step, final_step):
            if return_features:
                im, features_t = self(xx, grid, param, t_idx + t - self.initial_step, return_features)
                features.append(features_t)
            else:
                im = self(xx, grid, param, t_idx + t - self.initial_step, return_features)
            _batch = xx.size(0)
            pred = torch.cat((pred, im), -2)
            xx = torch.cat((xx[..., 1:, :], im), dim=-2)
            if t_idx is not None:
                t_idx = t_idx + 1
        if return_features:
            return pred, torch.concat(features, dim=-2)
        return pred

    def reduce_input_res(self, xx: Tensor, grid: Tensor) -> Tensor:
        """Reduce from sim to model resolution."""
        return subsample(xx, grid, self.reduced_resolution, self.reduced_resolution_t)

    def eval_pred(self, xx: Tensor, yy: Tensor, grid: Tensor, param: Tensor = None, t_idx: Tensor = None):
        """Helper for necessary model calls during evaluation.

        @param xx: initial state [b, nx1, nx2, ..., nt=1, nc]
        @param yy: target [b, nx1, nx2, ..., nt, nc]
        @param grid: grid coordinates  [b, nx1, nx2, ..., num_dim]
        @param param: [b, n_param]
        @param t_idx: [b, 1]. Time index of xx (only important for non-autonomous PDEs)
        """
        xx = xx.to(device)
        grid = grid.to(device)
        yy = yy.to(device)
        param = param.to(device)
        t_idx = t_idx.to(device)
        pred = self.roll_out(xx, grid, yy.shape[-2], param, t_idx)
        onestep_pred = self.one_step_pred(yy, grid, param, t_idx)
        return {"pred": pred, "onestep_pred": onestep_pred, "yy": yy, "param": param, "grid": grid, "t_idx": t_idx}

    def validate(self, step: int, val_loader: DataLoader = None, train_loader: DataLoader = None,
                 prefix: str = "") -> None:
        """Starts the evaluation of the model during training.
        @param step: logging iteration
        @param val_loader: validation data loader
        @param train_loader: training data loader
        @param prefix: will be put before the name of all logged metrics.
        """
        if val_loader is None:
            val_loader = self.val_loader
        self.evaluate(step, val_loader, prefix + "val/")
        if train_loader is None:
            train_loader = self.train_loader_full_traj
            self.evaluate(step, train_loader, prefix + "train/")

    def one_step_pred(self, xx: Tensor, grid: Tensor, param: Tensor, t_idx: Tensor = None) -> Tensor:
        """Predicts the next state for all time steps in xx

        @param xx: trajectory [b, nx1, nx2, ..., nt, nc]
        @param grid: grid coordinates  [b, nx1, nx2, ..., num_dim]
        @param param: [b, n_param]
        @param t_idx: [b, 1]. Time index of xx (only important for non-autonomous PDEs)
        @return: The rollout including xx as the first entry,  [b, nx1, nx2, final_step + 1, nt=1, nc]
        """
        pred_one_steps = [xx[..., :1, :]]
        for rel_t_idx in range(xx.shape[-2] - 1):
            x_t = xx[..., rel_t_idx: rel_t_idx + 1, :]
            pred_one_steps.append(self(x_t, grid, param, t_idx + rel_t_idx))
        return torch.concat(pred_one_steps, dim=-2)

    def evaluate(self, step: int, loader: DataLoader, prefix: str, time_step_name: str = "total_epoch") -> None:
        """Evaluates the model on the dataset.
        @param step: Logging iteration
        @param loader:  Data loader
        @param prefix: Will be put before the name of all logged metrics.
        @param time_step_name: Name of the iteration variable (e.g. epoch, al_iter)
        """
        with torch.no_grad():
            for s in self.stats:
                s.init_eval(self.task_norm, self.initial_step)
            for batch_idx, (xx, yy, grid, param, t_idx) in enumerate(loader):
                eval_pred = self.eval_pred(xx, yy, grid, param, t_idx)
                for s in self.stats:
                    s.add_batch(eval_pred)
            res = {time_step_name: step}
            for s in self.stats:
                res.update(s.get_result(step))
            print(res)
            res = {prefix + key: res[key] for key in res}
            wandb.log(res)

    def visualize(self, total_epoch: int) -> None:
        """Generates visualizations on the validation data set. """
        with torch.no_grad():
            with torch.no_grad():
                for p in self.plots:
                    p.init_eval(self.task_norm, self.initial_step)
                for batch_idx, (xx, yy, grid, param, t_idx) in enumerate(self.val_loader):
                    eval_pred = self.eval_pred(xx, yy, grid, param, t_idx)
                    for p in self.plots:
                        p.add_batch(eval_pred)
                for p in self.plots:
                    p.get_result(total_epoch)
            # plot trajectories
            if self.task.spatial_dim == 1:
                plot_traj(self, self.val_loader, self.task.img_save_path, "traj_" + str(total_epoch), 4)
            elif self.task.spatial_dim == 2:
                plot_traj_2d(self.task, self, self.val_loader, self.task.img_save_path, "traj2d_" + str(total_epoch), 4)

    def train_single_epoch(self, current_epoch, total_epoch, num_epoch, ):
        """ Train model for one epoch.
        @param current_epoch: Current training epoch in the current AL iteration.
        @param total_epoch: Total epoch added up over all AL iterations.
        @param num_epoch: Number of epochs to train for.
        """
        raise NotImplementedError

    def train_n_epoch(self, al_iter: int, num_epoch: int, step_offset: int, vis: bool = True,
                      prefix: str = "", is_last=False) -> float:
        """ Train model for num_epoch epochs.
        @param al_iter: Current AL iteration.
        @param num_epoch: Number of epochs to train for.
        @param step_offset: Total epoch added up over all AL iterations.
        @param vis: Whether to visualize.
        @param prefix: Will be put before the name of all logged metrics.
        @return: Training duration in seconds.
        """
        total_time = 0
        self.init_training(al_iter)
        for i in range(num_epoch):
            t = time.time()
            self.train_single_epoch(i, step_offset + i, num_epoch)
            total_time += time.time() - t
            if i % self.val_period == 0:
                self.validate(step_offset + i, prefix=prefix)
            if vis:
                if (i > 0 and i % self.vis_period == 0) or i == num_epoch - 1:
                    self.visualize(step_offset + i)
        return total_time
