import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from al4pde.evaluation.stats import LossStat
from al4pde.evaluation.analysis import mean_confidence_interval


class LossOverT(LossStat):

    def __init__(self, path, name, loss, sqrt=False, onestep=False, ):
        super().__init__(name, loss, sqrt, onestep, reduction="mean_frame")
        self.path = path

    def init_eval(self, task_norm, initial_step):
        self.loss.init_training(task_norm, 0)
        self.batch_results = []

    def get_result(self, step):
        fig = plt.figure()
        file_name = self.name.replace(" ", "_") + "_over_time_" + str(step)
        batch_results = torch.concat(self.batch_results, 0).numpy().T
        if self.sqrt:
            batch_results = np.sqrt(batch_results)
        mean = []
        unc = []
        for i in range(len(batch_results)):
            m, u = mean_confidence_interval(batch_results[i, :])
            mean.append(m)
            unc.append(u)

        mean = np.array(mean)
        unc = np.array(unc)

        np.save(os.path.join(self.path, file_name), np.stack([mean, unc]))
        plt.plot(np.arange(len(mean)), mean)
        ax = plt.gca()
        ax.fill_between(np.arange(len(mean)), mean - unc, mean + unc, alpha=0.2, linewidth=0)
        ax.set_xlabel("t")
        ax.set_ylabel(self.loss.name)
        ylim = ax.get_ylim()[0]
        ax.set_ylim(bottom=max(ylim, 0))
        plt.tight_layout()
        plt.savefig(os.path.join(self.path, file_name))
        plt.close(fig)
        return {}


class LossOverParam(LossStat):

    def __init__(self, path, task, name, loss, bins=30, onestep=False, ):
        super().__init__(name, loss, False, onestep, reduction="mean_traj")
        self.params = []
        self.path = path
        self.task = task
        self.n_bins = bins

    def init_eval(self, task_norm, initial_step):
        self.params = []
        super().init_eval(task_norm, initial_step)

    def add_batch(self, batch_dict):
        self.params.append(batch_dict["param"].cpu())
        super().add_batch(batch_dict)

    def get_result(self, step):
        traj_loss = torch.concat(self.batch_results, 0).numpy()
        param = torch.concat(self.params, 0).numpy()
        for pidx in range(param.shape[-1]):
            file_name = self.name.replace(" ", "_") + "_" + str(step) + "_over_param_" + str(pidx)
            param_i = param[:, pidx]
            if self.task.param_gen.log_scale[pidx]:
                param_i = np.log(param_i + 1e-20)
            min_x = np.min(param_i)
            max_x = np.max(param_i)
            delta = (max_x - min_x) / self.n_bins
            mse_means = []
            x_center = []
            for xl in np.arange(min_x, max_x, max(delta, 1e-99)):
                mse_mean = np.mean(traj_loss[(param_i >= xl) & (param_i < xl + delta)])
                mse_means.append(mse_mean)
                x_center.append(xl + delta / 2)

            if self.task.param_gen.log_scale[pidx]:
                x_center = np.exp(x_center)
            fig = plt.figure()
            plt.plot(x_center, mse_means)
            np.save(os.path.join(self.path, file_name), np.stack([x_center, mse_means], -1))

            ax = plt.gca()
            if self.task.param_gen.log_scale[pidx]:
                ax.set_xscale('log')
            plt.legend()
            ax.set_ylabel(self.name)
            ax.set_xlabel("PDE parameter " + str(pidx))
            plt.tight_layout()
            plt.savefig( os.path.join(self.path, file_name))
            plt.close(fig)
        return {}


class LossOverFreq(LossStat):

    def __init__(self, path, channel,  name, loss, t_min=None, t_max=None):
        super().__init__(name, loss, False, False, reduction=None)
        self.path = path
        self.channel = channel
        self.t_min = t_min
        self.t_max = t_max

    def add_batch(self, batch_dict):
        loss_c = self.loss(batch_dict["pred"], batch_dict["yy"], reduction=None)[..., self.channel]
        pred = batch_dict["pred"][..., self.channel]
        onestep_pred = batch_dict["onestep_pred"][..., self.channel]
        yy_b = batch_dict["yy"][..., self.channel]
        if self.t_min is not None:
            loss_c = loss_c[..., self.t_min:self.t_max]
            pred = pred[..., self.t_min:self.t_max]
            onestep_pred = onestep_pred[..., self.t_min:self.t_max]
            yy_b = yy_b[..., self.t_min:self.t_max]

        ampl_sums_pred = torch.fft.rfft(pred, dim=1).abs().mean(-1).detach().cpu()
        ampl_sums_gt = torch.fft.rfft(yy_b, dim=1).abs().mean(-1).detach().cpu()
        error_ampl_sums_gt = torch.fft.rfft(loss_c, dim=1).abs().mean(-1).detach().cpu()
        ampl_sums_pred_onestep = torch.fft.rfft(onestep_pred, dim=1).abs().mean(-1).detach().cpu()
        self.batch_results.append({
            "Prediction": ampl_sums_pred,
            "Ground Truth": ampl_sums_gt,
            "Loss": error_ampl_sums_gt,
            "One Step Prediction": ampl_sums_pred_onestep,
        })

    def get_result(self, step):

        title = " "
        if self.t_min is not None:
            title = str(self.t_min) + "_" + str(self.t_max) + "_"
        fig = plt.figure()
        file_name = "pred_freq_" + title + str(step)
        stacked_all_res = []
        for key in self.batch_results[0]:
            all_res = torch.concat([b[key] for b in self.batch_results], 0).mean(0)
            plt.plot(all_res, label=key)
            stacked_all_res.append(all_res.detach().cpu().numpy())
        np.save(os.path.join(self.path, file_name), np.stack(stacked_all_res, -1))
        ax = plt.gca()
        ax.set_xlabel("k")
        ax.set_ylabel("amplitude channel " + str(self.channel))
        ax.set_yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.path,  file_name + ".png"))
        plt.close(fig)


def build_standard_plots(primary_loss, path, task):
    plots = [
        LossOverT(path, "Loss", primary_loss),
        LossOverT(path, "Onestep Loss", primary_loss, onestep=True),
        LossOverParam(path, task, "Loss", primary_loss),
        LossOverParam(path, task, "Onestep Loss", primary_loss, onestep=True),
    ]
    if task.spatial_dim == 1:
        plots += [
        LossOverFreq(path, 0, "Frequency", primary_loss),
        LossOverFreq(path, 0, "FrequencyIC", primary_loss, t_min=0, t_max=1),
        LossOverFreq(path, 0, "Frequency10", primary_loss, t_min=10, t_max=11),
        LossOverFreq(path, 0, "Frequency20", primary_loss, t_min=20, t_max=21),]
    return plots
