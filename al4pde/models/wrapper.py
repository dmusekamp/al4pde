import torch
import numpy as np
from timeit import default_timer
import wandb

from al4pde.models.pdebench_loss import get_loss
from al4pde.evaluation.analysis import batch_errors
from al4pde.models.model import Model
from al4pde.models.dataloader import NPYDataset
from al4pde.models.normalization import TaskNormalizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModelWrapper(Model):


    def __init__(self, task, training_type, t_train, batch_size, model_factory, optimizer_factory,
                 scheduler_factory, num_workers, val_period, vis_period, block_grad, loss, name="", step_batch=-1,
                 num_train_steps=None, max_grad_norm=None, norm_mode="first", noise=0.001, noise_mode="constant"):
        super().__init__(task, training_type, t_train, batch_size, val_period, vis_period, loss)
        self.model_factory = model_factory
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.t_train = t_train
        self.training_type = training_type
        self.optimizer = None
        self.scheduler = None
        self.loss_val_min = None
        self.model = None
        self.num_workers = num_workers
        self.name = name
        self.block_grad = block_grad
        self.train_data = None
        self._train_loader_full = None
        self.step_batch = step_batch
        self.first_iter_nb = None
        self.num_train_steps = num_train_steps
        self.max_grad_norm = max_grad_norm
        self.norm_mode = norm_mode
        self.noise = noise
        self.noise_mode = noise_mode
        self.train_one_step_avg = 1
        assert noise_mode in ["constant", "channel_std_rel"]
        assert norm_mode in ["first", "every", None]
        self.norm_max_counter = 0
        self.max_norm_stat = 0
        self.max_norm = 0

    def init_training(self, al_iter, load_train_data=True):
        print("init training")

        self.model = self.model_factory().to(device)
        self.model.train()
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Total parameters = {total_params}')
        self.optimizer = self.optimizer_factory(self.model.parameters())
        self.scheduler = self.scheduler_factory(self.optimizer)
        self.loss_val_min = np.infty
        self.max_norm_stat = 0
        self.train_one_step_avg = 1
        self.norm_max_counter = 0
        self.max_norm = 0
        if load_train_data:
            train_data = NPYDataset(
                pde_name=self.task.pde_name,
                folders=self.task.train_data_folders,
                reduced_resolution=self.reduced_resolution,
                reduced_resolution_t=self.reduced_resolution_t,
                reduced_batch=self.reduced_batch,
                initial_step=self.initial_step,
                skip_initial_steps=self.skip_initial_steps,
            ).set_num_steps(self.num_train_steps)
            self.train_data = train_data
            self._train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size,
                                                             num_workers=self.num_workers, shuffle=True)
            self._train_loader_full = torch.utils.data.DataLoader(train_data.set_num_steps(None),
                                                                  batch_size=self.batch_size,
                                                                  num_workers=self.num_workers,
                                                                  shuffle=True)
        if self._val_loader is None:
            val_data = NPYDataset(
                pde_name=self.task.pde_name,
                folders=[self.task.eval_set_path, ],
                reduced_resolution=self.reduced_resolution,
                reduced_resolution_t=self.reduced_resolution_t,
                reduced_batch=self.reduced_batch,
                initial_step=self.initial_step,
                skip_initial_steps=self.skip_initial_steps
            )

            self._val_loader = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size,
                                                           num_workers=self.num_workers, shuffle=False)

        if al_iter == 0:
            self.task_norm = TaskNormalizer()
            self.task_norm.update(self.train_loader)
            self.task_norm.to(device)
            self.first_iter_nb = len(self.train_loader)
            self.loss.init_training(self.task_norm, self.initial_step)
        elif self.norm_mode == "every":
            self.task_norm = TaskNormalizer()
            self.task_norm.cpu().update(self.train_loader)
            self.task_norm.to(device)

    @property
    def train_loader_full_traj(self):
        return self._train_loader_full


    def train_single_epoch(self, current_epoch, total_epoch, num_epoch):

        t1 = default_timer()

        self.model.train()
        train_mse_sum = 0
        train_loss_sum = 0
        n = 0
        train_max_l2 = 0
        max_grad = -1
        grad_sum = 0
        grad_sum_n = 0
        norm_sum = 0
        norm_n = 0

        if self.noise_mode == "channel_std_rel":
            std = self.noise * self.task_norm.std_channels
        elif self.noise_mode == "constant":
            std = self.noise
        else:
            raise ValueError(self.noise_mode)

        for batch_idx, (xx, yy, grid, pde_param, t_idx) in enumerate(self.train_loader):
            xx = xx.to(device)
            yy = yy.to(device)
            grid = grid.to(device)
            pde_param = pde_param.to(device)
            t_idx = t_idx.to(device)
            _batch = yy.size(0)

            pred, loss = get_loss(xx, yy, grid, pde_param, t_idx, self, current_epoch, num_epoch,
             self.training_type, std, self.block_grad)

            self.optimizer.zero_grad()

            loss.backward()
            total_norm = 0

            for param in self.model.parameters():
                if param.requires_grad and param.grad is not None:
                    grad = param.grad
                    grad_sum += grad.abs().mean().detach().cpu()
                    grad_sum_n += 1
                    max_grad = max(max_grad, grad.abs().max().detach().cpu().item())
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            norm_sum += total_norm
            norm_n += 1
            tb_idx = (batch_idx + current_epoch * len(self.train_loader))

            self.max_norm_stat = max(total_norm, self.max_norm_stat)
            if self.max_grad_norm is not None:
                if self.max_grad_norm < total_norm:
                    self.norm_max_counter += 1
                if tb_idx == 0:
                    self.max_grad_norm = 5 * total_norm
                elif current_epoch < 5:
                    self.max_grad_norm = 5 * max(total_norm, self.max_grad_norm / 5)  # add some slack in beginning
                else:
                    norm_update = min(total_norm, self.max_grad_norm)
                    self.max_grad_norm = 5 * (self.max_grad_norm / 5 * 0.95 + norm_update * 0.05)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()
            all_se = batch_errors(pred, yy, self.initial_step)
            train_mse_sum += all_se.mean().detach() * _batch
            train_max_l2 = max(torch.sqrt(all_se).max().detach().item(), train_max_l2)
            train_loss_sum += loss.detach().item() * _batch
            n += _batch

            if self.step_batch != -1 and tb_idx % self.step_batch == 0 and tb_idx < self.first_iter_nb * num_epoch:
                self.scheduler.step()

        train_rmse = torch.sqrt(train_mse_sum / n).item()
        train_avg_loss = train_loss_sum / n
        t2 = default_timer()
        avg_grad = grad_sum / grad_sum_n
        if self.step_batch == -1:
            self.scheduler.step()

        print("epoch:", str(current_epoch), "time:", np.round(t2 - t1, 2),
              "train_rmse:", np.round(train_rmse, 4), "train_loss", np.round(train_avg_loss, 4),
              "avg_grad", np.round(avg_grad.item(), 4), "max_grad", np.round(max_grad, 3),
              "grad_norm", np.round(norm_sum/ norm_n, 4), "max_grad_norm", self.max_grad_norm)

        if current_epoch % self.val_period == 0:
            wandb.log({
                self.name + "/train_rmse": train_rmse,
                self.name + "/train_avg_loss": train_avg_loss,
                self.name + "/train_max_l2": train_max_l2,
                self.name + "/lr": self.scheduler.get_lr()[0],
                self.name + "/max_grad": max_grad,
                self.name + "/grad_norm": norm_sum/ norm_n,
                self.name + "/norm_cap_counter": self.norm_max_counter,
                self.name + "/norm_max": self.max_norm_stat,
                self.name + "/avg_grad": avg_grad,
                "total_epoch": total_epoch,
            },
            )
            self.norm_max_counter = 0
            self.max_norm_stat = 0

    def get_extra_state(self):
        return {"optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict()}

    def set_extra_state(self, state):
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])

