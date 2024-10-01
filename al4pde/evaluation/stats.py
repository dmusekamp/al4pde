import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr

from al4pde.evaluation.analysis import corr_over_t
from al4pde.evaluation.pdebench_metrics import metric_func
from al4pde.models.loss import FrameRelMSE, MSE, PredIdentity, MAELoss, Loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Stat:
    """Class to compute statistics on an evaluation batch and gather them. """

    def __init__(self, name: str):
        self.name = name
        self.initial_step = None
        self.task_norm = None

    def init_eval(self, task_norm, initial_step):
        self.task_norm = task_norm
        self.initial_step = initial_step

    def add_batch(self, batch_dict):
        """Compute stats over the batch.

        @param batch_dict: Dictionary of predictions as returned by model.eval_pred
        """
        raise NotImplementedError

    def get_result(self, step: int):
        """Collect the results of the statistic."""
        return {self.name: self._get_result()}

    def _get_result(self):
        raise NotImplementedError


class LossStat(Stat):

    def __init__(self, name, loss: Loss, sqrt=False, onestep=False, reduction=None):
        super().__init__(name)
        self.loss = loss
        self.onestep = onestep
        self.reduction = reduction
        self.batch_results = []
        self.sqrt = sqrt

    def init_eval(self, task_norm, initial_step):
        self.loss.init_training(task_norm, initial_step)
        self.batch_results = []

    def add_batch(self, batch_dict):
        pred = batch_dict["onestep_pred"] if self.onestep else batch_dict["pred"]
        loss = self.loss(pred, batch_dict["yy"], reduction=self.reduction).cpu()
        return self._add_batch(loss)

    def _add_batch(self, reduced_loss):
        self.batch_results.append(reduced_loss)

    def get_result(self, step):
        res = self._get_result()
        if self.sqrt:
            return np.sqrt(res)
        return {self.name: res}


class MeanLossStat(LossStat):

    def __init__(self, name, loss, sqrt=False, onestep=False):
        super().__init__(name, loss, sqrt, onestep, reduction="mean_traj")

    def _get_result(self):
        return torch.concat(self.batch_results, 0).mean()


class MaxLossStat(LossStat):
    def _add_batch(self, reduced_loss):
        self.batch_results.append(reduced_loss.max())

    def _get_result(self):
        return torch.stack(self.batch_results).max()


class Quantiles(LossStat):

    def __init__(self, name, loss, quantiles, onestep=False, reduction=None):
        super().__init__(name, loss, False, onestep, reduction)
        self.quantiles = quantiles

    def get_result(self, step=None):
        all_err = torch.concat(self.batch_results, 0).numpy().flatten()
        qv = np.quantile(all_err, self.quantiles)
        return {self.name + "_" + str(int(100 * self.quantiles[i])): qv[i] for i in range(len(qv))}


class UncAvg(MeanLossStat):

    def __init__(self, name):
        super().__init__(name, PredIdentity(), False)

    def add_batch(self, batch_dict):
        loss = self.loss(batch_dict["unc"], batch_dict["yy"], reduction=self.reduction).cpu()
        return self._add_batch(loss)


class LossUncCorr(LossStat):
    def __init__(self, name, loss, reduction="mean_traj"):
        super().__init__(name, loss, False, False, reduction=reduction)
        self.unc_avg = []
        self.loss = loss
        self.unc_ident = PredIdentity()

    def init_eval(self, task_norm, initial_step):
        self.unc_avg = []
        self.unc_ident.init_training(task_norm, initial_step)
        super().init_eval(task_norm, initial_step)

    def add_batch(self, batch_dict):
        super().add_batch(batch_dict)
        self.unc_avg.append(self.unc_ident(batch_dict["unc"], batch_dict["yy"], reduction=self.reduction).cpu())

    def get_result(self, step):
        all_mse = torch.concat(self.batch_results, 0).flatten()
        all_unc = torch.concat(self.unc_avg, 0).flatten()
        mask = torch.isfinite(all_unc) & torch.isfinite(all_mse)
        all_unc = all_unc[mask]
        all_mse = all_mse[mask]
        return {self.name + "_pearson": pearsonr(all_unc, all_mse).statistic,
                self.name + "_spearman": spearmanr(all_unc, all_mse).statistic}


class CorrelationOverT(Stat):
    def __init__(self):
        super().__init__("time_till_corr_lower")
        self.corr = []
        self.n = 0

    def init_eval(self, task_norm, initial_step):
        self.corr = []
        self.n = 0

    def add_batch(self, batch_dict):
        pred = batch_dict["pred"]
        self.corr.append([corr_over_t(pred, batch_dict["yy"], i).sum(0).cpu() for i in range(pred.shape[-1])])
        self.n += len(pred)

    def get_result(self, step):
        n_channels = len(self.corr[0])
        res = {}
        for channel_idx in range(n_channels):
            corr_t = torch.stack([self.corr[i][channel_idx] for i in range(len(self.corr))], 0).sum(0) / self.n
            for threshold in [0.8, 0.95]:
                t = (corr_t > threshold).float().sum()
                res[self.name + "_" + str(threshold) + "_channel_" + str(channel_idx)] = t
        return res


class PDEBenchStats(Stat):
    def __init__(self):
        super().__init__("PDEBenchStats")
        self.batch_results = []
        self.initial_step = None
        self.n = []
        self.names = ["RMSE", "nRMSE", "CSV", "Max", "BD", "F", ]

    def init_eval(self, task_norm, initial_step):
        self.initial_step = initial_step
        self.batch_results = []
        self.n = []

    def add_batch(self, batch_dict):
        pred = batch_dict["pred"]
        yy = batch_dict["yy"]
        Lx, Ly, Lz = 1., 1., 1.
        batch = metric_func(pred, yy, if_mean=True, Lx=Lx, Ly=Ly, Lz=Lz, initial_step=self.initial_step)
        self.batch_results.append([b.cpu() for b in batch])
        self.n.append(len(pred))

    def get_result(self, step):
        res = {}
        n = sum([self.n[i] for i in range(len(self.n))])
        for stat_idx in range(len(self.names)):
            all_res = sum([self.batch_results[i][stat_idx] * self.n[i] for i in range(len(self.batch_results))]) / n
            res[self.names[stat_idx]] = all_res
        return res


def build_standard_stats(primary_loss, task):
    stats = [MeanLossStat("loss_avg", primary_loss, False, False),
             MeanLossStat("onestep_loss_avg", primary_loss, False, True),
             MaxLossStat("loss_cell_max", primary_loss, reduction="mean_cell"),
             MaxLossStat("loss_frame_max", primary_loss, reduction="mean_frame"),
             MaxLossStat("loss_traj_max", primary_loss, reduction="mean_frame"),
             MaxLossStat("onestep_loss_cell_max", primary_loss, False, True, reduction="mean_cell"),
             MaxLossStat("onestep_loss_frame_max", primary_loss, False, True, reduction="mean_frame"),
             Quantiles("loss_quantiles_cell", primary_loss, (0.5, 0.95, 0.99), False, reduction="mean_cell"),
             Quantiles("loss_quantiles_frame", primary_loss, (0.5, 0.95, 0.99), False, reduction="mean_frame"),
             Quantiles("loss_quantiles_traj", primary_loss, (0.5, 0.95, 0.99,), False, reduction="mean_traj"),
             CorrelationOverT(),
             PDEBenchStats()]

    standard_losses = [MSE(normalize_channels=True),
                       MSE(normalize_channels=False),
                       FrameRelMSE(normalize_channels=False),
                       MAELoss(normalize_channels=False)
                       ]
    names = ["c_norm_mse", "mse", "frame_rel_mse", "mae"]
    for l_idx in range(len(names)):
        stats.append(MeanLossStat(names[l_idx], standard_losses[l_idx], False, False))
        stats.append(MeanLossStat("onestep_" + names[l_idx], standard_losses[l_idx], False, True))

    return stats
