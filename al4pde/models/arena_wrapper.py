import hydra
from al4pde.utils import bxtc_to_btcx, btcx_to_bxtc
from al4pde.models.wrapper import ModelWrapper


class ArenaWrapper(ModelWrapper):
    """Provides an interface to the base models defined in PDEArena"""

    def __init__(self, task, training_type, t_train, batch_size, model_factory, optimizer_factory, scheduler_factory,
                 num_workers, val_period, vis_period, block_grad, difference_weight, predict_delta, loss, name="",
                 step_batch=-1, num_train_steps=None, max_grad_norm=None, norm_mode="first", noise=0.001,
                 noise_mode="constant",):
        super().__init__(task, training_type, t_train, batch_size, model_factory, optimizer_factory, scheduler_factory,
                         num_workers, val_period, vis_period, block_grad,
                         loss, name,
                         step_batch=step_batch, num_train_steps=num_train_steps,
                         max_grad_norm=max_grad_norm, norm_mode=norm_mode, noise=noise, noise_mode=noise_mode,
                         )
        assert 0 <= difference_weight <= 1
        self.difference_weight = difference_weight
        self.predict_delta = predict_delta

    def forward(self, xx, grid, pde_param=None, t_idx=None, return_features=False):
        xx = self.task_norm.norm_traj(xx)
        if pde_param is not None:
            pde_param = self.task_norm.norm_param(pde_param)
        # transform PDEBench data format to PDEArena data format
        xx_arena = bxtc_to_btcx(xx)  # PDEArena models consume data of shape [nb, nt, nc, nx, ny, nz]
        features = None
        if self.task.sim.autonomous:  # don't use time as an input if not necessary
            t_idx = None
        if return_features:
            next_x, features = self.model(xx_arena, z=pde_param, return_features=return_features, time=t_idx)
            features = btcx_to_bxtc(features)
        else:
            next_x = self.model(xx_arena, z=pde_param, time=t_idx)
        next_x = btcx_to_bxtc(next_x)
        if self.predict_delta:
            next_x = next_x * self.difference_weight + xx[..., -1:, :]
        out = self.task_norm.denorm_traj(next_x)
        if return_features:
            return out, features
        else:
            return out


def build_arena_model_factory(task, cfg):
    factory = hydra.utils.instantiate(cfg.model,
                                      _partial_=True,
                                      n_input_scalar_components=task.num_channels,
                                      n_output_scalar_components=task.num_channels,
                                      n_input_vector_components=0,
                                      n_output_vector_components=0,
                                      time_history=task.initial_step,
                                      time_future=1,
                                      param_conditioning="scalar_" + str(int(task.num_pde_param)),
                                      )
    return factory


def build_arena_wrapper(task, cfg, name=""):
    model_factory = build_arena_model_factory(task, cfg)
    optimizer_factory = hydra.utils.instantiate(cfg.optimizer, _partial_=True)
    scheduler_factory = hydra.utils.instantiate(cfg.scheduler, _partial_=True)
    loss = hydra.utils.instantiate(cfg.loss)
    return ArenaWrapper(task, cfg.training_type, cfg.t_train, cfg.batch_size, model_factory, optimizer_factory,
                        scheduler_factory, cfg.num_workers, cfg.val_period, cfg.vis_period,
                        cfg.block_grad, cfg.difference_weight, cfg.predict_delta, loss=loss, name=name,
                        step_batch=cfg.step_batch, num_train_steps=cfg.num_train_steps, max_grad_norm=cfg.max_grad_norm,
                        norm_mode=cfg.norm_mode, noise=cfg.noise, noise_mode=cfg.noise_mode)
