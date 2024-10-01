import time
import torch
import hydra
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.utils.data import TensorDataset, DataLoader
from bmdal_reg.bmdal.feature_data import TensorFeatureData
from bmdal_reg.bmdal.feature_maps import IdentityFeatureMap
from bmdal_reg.bmdal.features import Features
from al4pde.utils import subsample_grid, subsample_trajectory
from al4pde.prob_models.prob_model import ProbModel
from al4pde.acquisition.pool_based import PoolBased


def max_dist_selection(batch_size, train_mat, pool_mat, sel_method_factory):
    train_mat = train_mat.to(device)
    pool_mat = pool_mat.to(device)

    n_features = train_mat.shape[1]

    train_features = Features(IdentityFeatureMap(n_features=n_features), TensorFeatureData(train_mat))
    pool_features = Features(IdentityFeatureMap(n_features=n_features), TensorFeatureData(pool_mat))

    sel_method = sel_method_factory(pool_features=pool_features, train_features=train_features)
    return sel_method.select(batch_size).detach().cpu()  # returns index into the pool set that have been selected


class DistancePoolBased(PoolBased):

    def __init__(self, task, data_schedule, batch_size, pool_size, unc_eval_mode,
                 unc_num_rollout_steps_rel, num_rollout_steps_rel,
                 selection, predict_train, use_latent_space, pred_batch_size=128, traj_transform="identity"):

        self.num_rollout_steps_rel = num_rollout_steps_rel
        self.num_rollout_steps = None
        self.sel_method_factory = selection
        self.use_latent_space = use_latent_space
        if use_latent_space:
            predict_train = True
        self.predict_train = predict_train
        super().__init__(task, data_schedule, batch_size, pool_size, unc_eval_mode,
                         unc_num_rollout_steps_rel, pred_batch_size=pred_batch_size)
        self.sketch_ftm = None
        self.pred_batch_size = pred_batch_size
        self.traj_transform = traj_transform
        assert traj_transform in ["identity", "fourier", "spatial_mean", "spatial_max", "add_periodic",
                                  "fourier_real_imag", "fourier_amplitude", "traj_mean"]

    def set_num_time_steps(self, num_idx):
        super().set_num_time_steps(num_idx)
        self.num_rollout_steps = int(num_idx * self.num_rollout_steps_rel)

    def _apply_sketch(self, traj):
        features = traj.flatten(1, -1)
        n_features = features.shape[1]
        features = Features(IdentityFeatureMap(n_features=n_features), TensorFeatureData(features))
        if self.sketch_ftm is None:
            sketch_ftm = features.sketch_tfm(512)
            self.sketch_ftm = sketch_ftm
        return self.sketch_ftm(features).get_feature_matrix()

    def _add_periodic_translation(self, traj):
        all_translations = []
        spatial_dims = [i + 1 for i in range(self.task.spatial_dim)]
        all_idx_gris = torch.meshgrid([torch.arange(0, traj.shape[i], 1) for i in spatial_dims], indexing="ij")
        all_idx = torch.stack(all_idx_gris, -1).reshape([-1, self.task.spatial_dim])
        for i in range(len(all_idx)):
            all_translations.append(self._apply_sketch(torch.roll(traj, tuple(all_idx[i]), spatial_dims)))
        return torch.concat(all_translations, 0)

    def _fourier_features(self, traj):

        t_fft = torch.fft.rfftn(traj, dim=[1 + i for i in range(self.task.spatial_dim)])
        n_freq_rel = 0.5
        for i in range(self.task.spatial_dim):
            idx = torch.arange(t_fft.shape[1 + i])

            if i < self.task.spatial_dim - 1:
                offset = int(len(idx) * n_freq_rel * 0.5)
                idx = torch.concat([idx[:offset], idx[-offset:]])
            else:
                offset = int(len(idx) * n_freq_rel)
                idx = idx[:offset]

            t_fft = torch.index_select(t_fft, 1 + i, idx)

        if self.traj_transform == "fourier":
            return torch.concat([torch.sqrt(t_fft.real ** 2 + t_fft.imag ** 2),
                                 torch.atan2(t_fft.imag, t_fft.real)], -1)

        elif self.traj_transform == "fourier_real_imag":
            return torch.concat([t_fft.real, t_fft.imag], -1)

        elif self.traj_transform == "fourier_amplitude":
            return t_fft.abs()
        else:
            raise ValueError(self.traj_transform)

    def get_features(self, traj, is_train):
        if self.traj_transform.startswith("fourier"):
            return self._apply_sketch(self._fourier_features(traj))
        elif self.traj_transform == "spatial_mean":
            return self._apply_sketch(traj.flatten(1, -3).mean(1))
        elif self.traj_transform == "spatial_max":
            return self._apply_sketch(traj.flatten(1, -3).max(1)[0])
        elif self.traj_transform == "traj_mean":
            return self._apply_sketch(traj.flatten(1, -2).mean(1))
        elif self.traj_transform == "identity":
            return self._apply_sketch(traj)
        elif self.traj_transform == "add_periodic":
            if is_train:
                return self._add_periodic_translation(traj)
            else:
                return self._apply_sketch(traj)
        else:
            raise ValueError(self.traj_transform)

    def create_features_pool(self, data_loader, prob_model):
        features = []

        for i, batch in enumerate(data_loader):
            x = batch[0].to(device)
            pde_param = batch[-1].to(device)
            grid_b = self.task.get_grid(len(x))
            grid_b = subsample_grid(grid_b, self.task.reduced_resolution)
            x = subsample_trajectory(x, self.task.reduced_resolution, self.task.reduced_resolution_t)
            t_idx = torch.zeros([len(x)], device=device)
            out = prob_model.roll_out(x, grid_b, self.num_rollout_steps, pde_param, t_idx=t_idx,
                                      return_features=self.use_latent_space)
            if self.use_latent_space:
                out = out[1]
            elif prob_model.loss.normalize_channels:
                out = prob_model.loss.task_norm.norm_traj(out)
            features.append(self.get_features(out.detach().cpu(), is_train=False))
        return torch.concat(features, dim=0)

    def create_features_train(self, data_loader, prob_model):
        features = []
        for i, (xx, yy, grid, pde_param, t_idx) in enumerate(data_loader):
            if self.predict_train:
                xx = xx.to(device)
                grid = grid.to(device)
                pde_param = pde_param.to(device)
                t_idx = t_idx.to(device)
                out = prob_model.roll_out(xx, grid, self.num_rollout_steps, pde_param, t_idx=t_idx,
                                          return_features=self.use_latent_space)
            else:
                out = yy
            if self.use_latent_space:
                out = out[1]
            elif prob_model.loss.normalize_channels:
                out = prob_model.loss.task_norm.norm_traj(out)
            features.append(self.get_features(out.detach().cpu(), is_train=True))
        return torch.concat(features, dim=0)

    def select_next(self, prob_model: ProbModel, ic_pool: torch.Tensor, pde_param_pool: torch.Tensor,
                    ic_train: torch.Tensor, pde_param_train: torch.Tensor, grid: torch.Tensor, al_iter: int,
                    train_loader: torch.utils.data.DataLoader = None) -> torch.Tensor:

        dataset = TensorDataset(ic_pool, pde_param_pool)

        pool_loader = DataLoader(dataset, batch_size=self.pred_batch_size)
        with torch.no_grad():
            t = time.time()

            train_features = self.create_features_train(train_loader, prob_model)
            print("train_preparation_time", time.time() - t)
            t = time.time()
            pool_features = self.create_features_pool(pool_loader, prob_model)
            print("pool_preparation_time", time.time() - t)
            n_samples = self.num_batches(al_iter) * self.batch_size
            t = time.time()
            sel_idx = max_dist_selection(n_samples, train_features, pool_features, self.sel_method_factory)
            print("pure_selection_time", time.time() - t)
        return sel_idx

    @property
    def name(self):
        return "cov_dist"


def build_distance_pool_based(task, cfg):
    sel_factory = hydra.utils.instantiate(cfg.selection, _partial_=True)
    datas_schedule = hydra.utils.instantiate(cfg.data_schedule)
    return hydra.utils.instantiate(cfg, task=task, data_schedule=datas_schedule,
                                   _recursive_=False, selection=sel_factory)
