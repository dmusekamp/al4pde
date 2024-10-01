import scipy.stats
import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr
from pdearena.modules.loss import pearson_correlation
from al4pde.utils import bxtc_to_btcx
from al4pde.models.dataloader import NPYDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def stat_over_param(prob_model, data_loader, stat, n_bins=30, pidx=0, normalize_w_sq=False):
    stats = []
    with torch.no_grad():
        for batch_idx, (xx, yy, grid, param, t_idx) in enumerate(data_loader):
            eval_batch_dict = {"xx": xx, "yy": yy, "grid": grid, "param": param}
            eval_batch_dict.update(prob_model.eval_pred(xx, yy, grid, param, t_idx))
            stats.append(prob_model.batch_stats(eval_batch_dict))
            stats[-1]["param"] = param[:, pidx]
            stats[-1]["norm"] = yy.flatten(1, -2).norm(dim=-1).mean(1)
    mse = [s[stat] for s in stats]
    all_mse = torch.concat(mse, 0)
    all_mse = all_mse.reshape([all_mse.shape[0], -1]).mean(-1).numpy()

    norm = [s["norm"] for s in stats]
    norm = torch.concat(norm, 0)

    param = [s["param"] for s in stats]
    all_param = torch.concat(param, dim=0)
    all_param = all_param.reshape([-1, 1]).numpy()
    p_log = np.log(all_param[:, 0] + 1e-20)
    min_x = np.min(p_log)
    max_x = np.max(p_log)
    delta = (max_x - min_x) / n_bins
    mse_means = []
    x_center = []
    for xl in np.arange(min_x, max_x, max(delta, 1e-99)):
        mse_mean = np.mean(all_mse[(p_log >= xl) & (p_log < xl + delta)])
        squared_norm_mean = norm[(p_log >= xl) & (p_log < xl + delta)].mean() ** 2
        if normalize_w_sq:
            mse_mean = mse_mean / squared_norm_mean
        mse_means.append(mse_mean)
        x_center.append(xl + delta / 2)
    return x_center, mse_means


def stat_over_t(prob_model, data_loader, stat):
    stats = []
    with torch.no_grad():
        for batch_idx, (xx, yy, grid, param, t_idx) in enumerate(data_loader):
            eval_batch_dict = {"xx": xx, "yy": yy, "grid": grid, "param": param}
            eval_batch_dict.update(prob_model.eval_pred(xx, yy, grid, param, t_idx))
            stats.append(prob_model.batch_stats(eval_batch_dict))
            stats[-1]["param"] = param
    avg_stat = []
    for s in stats:
        bs = s["n"]
        s = s[stat].reshape([-1, s[stat].shape[-1]])
        avg_stat.append(s.mean(dim=0) * bs)
    n_total = sum([s["n"] for s in stats])
    avg_stat = torch.stack(avg_stat).sum(0) / n_total
    return avg_stat.cpu().numpy()


def compare_rollouts(prob_model, data_loader):

    def traj_mse(p, gt):
        err = (p.cpu().detach() - gt.cpu().detach()) ** 2
        err = err.reshape([err.shape[0], -1]).mean(dim=-1)
        return err

    # what using ensemble or individual models is best?
    with torch.no_grad():
        independent_rollout_mean_mse = []
        mean_propa_mse = []
        base_m_mse = [[] for i in range(len(prob_model.base_models))]
        mean_prop_unc_all = []
        all_unc = []
        for idx, (xx_b, yy_b, grid_b, param_b, t_idx) in enumerate(data_loader):
            independent_mean, unc = prob_model.unc_independent_rollout(xx_b.to(device), grid_b.to(device),
                                                                       yy_b.shape[-2], param_b.to(device),
                                                                       t_idx.to(device))
            independent_rollout_mean_mse.append(traj_mse(independent_mean, yy_b))
            mean_prop_mean,  mean_prop_unc = prob_model.unc_mean_propagation_rollout(xx_b.to(device), grid_b.to(device),
                                                                                     yy_b.shape[-2], param_b.to(device),
                                                                                     t_idx.to(device))
            mean_prop_unc = mean_prop_unc.cpu().detach()
            mean_prop_unc = mean_prop_unc.reshape([mean_prop_unc.shape[0], -1]).mean(dim=-1)
            mean_prop_unc_all.append(mean_prop_unc)

            mean_propa_mse.append(traj_mse(mean_prop_mean, yy_b))
            for i in range(len(prob_model.base_models)):

                base_m_mse[i].append(traj_mse(prob_model.base_models[i].roll_out(xx_b.to(device), grid_b.to(device),
                                                                                 yy_b.shape[-2], param_b.to(device),
                                                                                 t_idx.to(device)),
                                              yy_b))
            unc = unc.cpu().detach()
            unc = unc.reshape([unc.shape[0], -1]).mean(dim=-1)
            all_unc.append(unc)
        mean_prop_unc = torch.concat(mean_prop_unc_all, dim=0)
        unc = torch.concat(all_unc, dim=0)
        independent_rollout_mean_mse = torch.concat(independent_rollout_mean_mse, dim=0)
        mean_propa_mse = torch.concat(mean_propa_mse, dim=0)
        print("mean_prop_unc_corr")
        print("pearson", pearsonr(mean_prop_unc, mean_propa_mse).statistic)
        print("spearman", spearmanr(mean_prop_unc, mean_propa_mse).statistic)

        base_m_mse = [torch.concat(b, dim=0) for b in base_m_mse]

        names = ["independent", "mean_propagation"] + [str(i) for i in range(len(base_m_mse))]
        mses = [independent_rollout_mean_mse, mean_propa_mse] + [b for b in base_m_mse]
        for i in range(len(names)):
            print(names[i])
            print("mse", torch.mean(mses[i]))
            print("pearson",  pearsonr(unc, mses[i]).statistic)
            print("spearman",  spearmanr(unc, mses[i]).statistic)


def eval_on_new(task, prob_model, al_iter):
    last_data = NPYDataset(
        pde_name=task.pde_name,
        folders=task.train_data_folders,
        reduced_resolution=prob_model.reduced_resolution,
        reduced_resolution_t=prob_model.reduced_resolution_t,
        reduced_batch=prob_model.reduced_batch,
        initial_step=prob_model.initial_step,
        skip_initial_steps=prob_model.skip_initial_steps,
        regexp=".*alstp_" + str(al_iter) + ".*")
    data_loader = torch.utils.data.DataLoader(last_data, batch_size=prob_model.batch_size, shuffle=False)
    prob_model.evaluate(al_iter, data_loader, prefix="al_new_data/", time_step_name="al_iter")


def batch_errors(pred, yy, initial_step):
    pred = pred[..., initial_step:, :]
    yy = yy[..., initial_step:, :]
    return torch.square(pred - yy).sum(dim=-1)  # || ||² over channels


def resim_mse(pred, yy_b, param_b, grid_b, t_idx, task, prob_model):
    # resimulated error - feed each state back into the simulator to evaluate the prediction on the right input
    resim_mse_batch = []
    for rel_t_idx in range(yy_b.shape[-2] - 1):
        start_time = task.sim.get_time((t_idx + rel_t_idx) * prob_model.reduced_resolution_t)
        fin_time = task.sim.get_time((t_idx + rel_t_idx + 1) * prob_model.reduced_resolution_t)
        u, _, _ = task.sim.n_step_sim(pred[..., rel_t_idx: rel_t_idx + 1, :], param_b.cpu(), grid_b[0].cpu(), start_time,
                                      fin_time, prob_model.reduced_resolution_t)
        u = task.sim.to_ml_format(u)
        if rel_t_idx == 0:
            diff = yy_b[..., rel_t_idx + 1, :].cpu() - u[..., -1, :]
            if diff.abs().sum() > 1.0e-4:
                raise ValueError("big difference between one step simulation and original one")
        resim_mse_batch.append(batch_errors(pred[..., rel_t_idx:rel_t_idx + 1, :].cpu(), u[..., -1:, :], 0))
    return torch.concat(resim_mse_batch, dim=-1)


def one_step_mse(yy_b, param_b, grid_b, t_idx, prob_model):
    mses_one_step_batch = []
    for rel_t_idx in range(yy_b.shape[-2] - 1):
        y_t = yy_b[..., rel_t_idx + 1: rel_t_idx + 2, :]
        x_t = yy_b[..., rel_t_idx: rel_t_idx + 1, :]
        pred = prob_model(x_t, grid_b, param_b, t_idx + rel_t_idx)
        mses_one_step_batch.append(batch_errors(y_t.cpu(), pred.cpu(), 0))
    return torch.concat(mses_one_step_batch, dim=-1)


def n_step_mse(yy_b, param_b, grid_b, t_idx, prob_model, n):
    mses_one_step_batch = []
    for rel_t_idx in range(yy_b.shape[-2] - n):
        y_t = yy_b[..., rel_t_idx + 1: rel_t_idx + n + 1, :]
        x_t = yy_b[..., rel_t_idx: rel_t_idx + 1, :]
        pred = prob_model.roll_out(x_t, grid_b, n + 1, param_b, t_idx + rel_t_idx)[..., 1:, :]
        mse = batch_errors(y_t.cpu(), pred.cpu(), 0)
        mses_one_step_batch.append(mse.mean(dim=-1, keepdim=True))
    return torch.concat(mses_one_step_batch, dim=-1)


def n_step_mse_reduction(yy_b, param_b, grid_b, t_idx, prob_model, n, ):
    mse_next_one_known = []
    mse_full = []
    for rel_t_idx in range(yy_b.shape[-2] - n):
        y_t = yy_b[..., rel_t_idx + 1: rel_t_idx + n + 1, :]
        x_t = yy_b[..., rel_t_idx: rel_t_idx + 1, :]
        pred = prob_model.roll_out(x_t, grid_b, n + 1, param_b, rel_t_idx + t_idx)[..., 1:, :]
        mse = batch_errors(y_t.cpu(), pred.cpu(), 0)
        full_mse_sum = mse.sum(dim=-1, keepdim=True)
        mse_full.append(full_mse_sum)
        mse_next_one_known.append(mse[..., :-1].sum(dim=-1, keepdim=True))

    next_one_known = torch.concat(mse_next_one_known, dim=-1)[..., 1:]
    full = torch.concat(mse_full, dim=-1)[..., :-1]
    return (full - next_one_known) / n


def autoregressive_mse(xx_b, yy_b, param_b, grid_b, t_idx, prob_model):
    traj = prob_model.roll_out(xx_b, grid_b, yy_b.shape[-2], param_b, t_idx)
    return batch_errors(traj, yy_b, prob_model.initial_step)


def corr_over_t(pred, target, channel):
    pred = bxtc_to_btcx(pred)[:, :, channel]
    target = bxtc_to_btcx(target)[:, :, channel]
    corr = pearson_correlation(pred, target, False)
    return corr


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h
