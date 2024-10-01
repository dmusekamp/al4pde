import os
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
import numpy as np
import seaborn as sns
import torch.utils.data
from al4pde.evaluation.analysis import stat_over_t, stat_over_param, compare_rollouts, eval_on_new, batch_errors, \
    resim_mse, one_step_mse, autoregressive_mse, n_step_mse, n_step_mse_reduction
from al4pde.models.dataloader import NPYDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_single_traj(traj, grid, labels, title, path, filename, common_color_bar=True):
    traj = [t.cpu().detach().numpy() for t in traj]
    nx = int(np.ceil(np.sqrt(len(labels))))
    ny = int(np.ceil(len(labels) / nx))
    fig, axs = plt.subplots(nx, ny, sharex='all', sharey='all')
    if len(axs.shape) == 1:
        axs = axs.reshape([len(axs), 1])
    grid_np = grid.cpu().detach().numpy()
    if common_color_bar:
        v_min = np.min([np.min(t[..., 0]) for t in traj])
        v_max = np.max([np.max(t[..., 0]) for t in traj])
    else:
        v_min = None
        v_max = None
    for i in range(len(axs)):
        for j in range(len(axs[i])):
            idx = i * len(axs[i]) + j
            if idx < len(traj):
                im = axs[i][j].imshow(
                    traj[idx][:, :, 0],
                    aspect="auto",
                    origin="lower",
                    extent=[0, traj[idx].shape[-2], np.min(grid_np), np.max(grid_np)],
                    vmin=v_min, vmax=v_max
                )
                axs[i][j].title.set_text(labels[idx])
                axs[-1][j].set_xlabel("t")
                axs[i][0].set_ylabel("x")
                plt.colorbar(im, ax=axs[i][j])

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(path, filename))
    plt.close(fig)


def visualize_channel_2d(traj, gt_frames, pred_frames, viz_num_t, random_tsteps, title, path, filename,
                         channel_idx, channel_name):
    density_frames_gt = [gt_frame[..., channel_idx].unsqueeze(-1).cpu().detach().numpy() for gt_frame in gt_frames]
    density_frames_pred = [pred_frame[..., channel_idx].unsqueeze(-1).cpu().detach().numpy() for pred_frame in
                           pred_frames]
    fig, axes = plt.subplots(len(traj), viz_num_t + 1, figsize=(18, 8))  # (2 rows - gt/pred, 5 cols - tsteps)
    minval = min(pred_frames[..., channel_idx].min(), gt_frames[..., channel_idx].min())
    maxval = max(pred_frames[..., channel_idx].max(), gt_frames[..., channel_idx].max())

    for traj_type_idx, _ in enumerate(traj):  # (gt, pred)
        for tstep, ax in enumerate(axes[traj_type_idx]):
            if traj_type_idx == 0:  # GT
                ax.imshow(density_frames_gt[tstep], vmin=minval, vmax=maxval)
                ax.title.set_text(f"GT@t{random_tsteps[tstep]}")
            elif traj_type_idx == 1:  # pred
                ax.imshow(density_frames_pred[tstep], vmin=minval, vmax=maxval)
                ax.title.set_text(f"Pred@t{random_tsteps[tstep]}")
            ax.set_aspect('equal')
            ax.autoscale(False)

    fig.suptitle(channel_name + " field with PDE params: " + title, fontsize=16)
    plt.subplots_adjust(wspace=0, hspace=0, top=1, bottom=0, right=1, left=0)
    plt.tight_layout()
    plt.savefig(os.path.join(path, filename + channel_name), bbox_inches='tight')
    plt.close(fig)


def plot_single_traj_2d(task, traj, grid, labels, title, path, filename, common_color_bar=True):
    max_t = traj[0].shape[-2]
    viz_num_t = min(5, max_t - 1)  # how many timesteps to visualize (excluding IC)
    random_tsteps = tuple([0] + np.sort(np.random.choice(np.arange(1, max_t), viz_num_t, replace=False)).tolist())

    traj = [tensor.permute(2, 0, 1, 3) for tensor in traj]  # (H, W, T, C) -> (T, H, W, C)
    gt_pred_pair = torch.stack((traj[0][random_tsteps, ...], traj[1][random_tsteps, ...]), dim=0)  # (2, T, H, W, C)
    gt_frames = torch.stack([gt_pred_pair[0][img_tstep] for img_tstep in range(gt_pred_pair.shape[1])], 0)
    pred_frames = torch.stack([gt_pred_pair[1][img_tstep] for img_tstep in range(gt_pred_pair.shape[1])], 0)

    for c_idx in range(task.num_channels):
        visualize_channel_2d(traj, gt_frames, pred_frames, viz_num_t, random_tsteps, title, path, filename, c_idx,
                             task.channel_names[c_idx])


def plot_traj(prob_model, loader, path, file_name_prefix, n):
    xx_b, yy_b, grid_b, param_b, t_idx_b = loader.__iter__().__next__()
    for traj_idx in range(min(n, len(xx_b))):
        xx = xx_b[traj_idx:traj_idx + 1].to(device)
        yy = yy_b[traj_idx:traj_idx + 1].to(device)
        grid = grid_b[traj_idx:traj_idx + 1].to(device)
        param = param_b[traj_idx:traj_idx + 1].float().to(device)
        t_idx = t_idx_b[traj_idx:traj_idx + 1].to(device)
        traj = [yy[0], prob_model.roll_out(xx, grid, yy.shape[-2], param, t_idx)[0]]
        labels = ["ground_truth", "model"]
        title = list(param[0])
        filename = file_name_prefix + "_" + str(traj_idx) + ".png"
        plot_single_traj(traj, grid, labels, title, path, filename)


def plot_traj_2d(task, prob_model, loader, path, file_name_prefix, n):
    """
    Visualize trajectory, containing different fields (D, Vx, Vy, P), over time steps.
    For every trajectory, plot the ground truth and the model predictions (D, Vx, Vy, P).
    Randomly sample time steps in the range of the trajectory and visualize the ground truth and the model predictions.
    """
    xx_b, yy_b, grid_b, param_b, t_idx = loader.__iter__().__next__()
    print(f"n - how many to visualize: {n}")
    for traj_idx in range(min(n, len(xx_b))):
        xx = xx_b[traj_idx:traj_idx + 1].to(device)
        yy = yy_b[traj_idx:traj_idx + 1].to(device)
        grid = grid_b[traj_idx:traj_idx + 1].to(device)
        param = param_b[traj_idx:traj_idx + 1].float().to(device)
        t_idx = t_idx[traj_idx:traj_idx + 1].float().to(device)
        traj = [yy[0], prob_model.roll_out(xx, grid, yy.shape[-2], param, t_idx)[0]]
        labels = ["ground_truth", "prediction"]
        pde_params = list(param[0].cpu().detach().numpy())
        title = ["|eta|=" + str(p) if idx == 0 else "|zeta|=" + str(p) for idx, p in enumerate(pde_params)]
        title = " & ".join(title)
        filename = file_name_prefix + "_traj" + str(traj_idx) + "_"  # + ".png"
        plot_single_traj_2d(task, traj, grid, labels, title, path, filename)


def plot_worst_traj(prob_model, loader, path, file_name_prefix):
    with torch.no_grad():
        max_mse = 0
        max_traj = None
        max_ground_truth = None
        max_param = None
        max_unc = None
        max_unc_independent = None
        max_independent_traj = None

        for idx, (xx_b, yy_b, grid_b, param_b, t_idx) in enumerate(loader):
            traj, unc = prob_model.unc_roll_out(xx_b.to(device), grid_b.to(device), yy_b.shape[-2], param_b.to(device),
                                                t_idx.to(device))
            independent_traj, independent_unc = prob_model.unc_independent_rollout(xx_b.to(device), grid_b.to(device),
                                                                                   yy_b.shape[-2],
                                                                                   param_b.to(device),
                                                                                   t_idx.to(device))
            traj = traj.cpu().detach()
            unc = unc.cpu().detach()
            independent_traj = independent_traj.cpu().detach()
            independent_unc = independent_unc.cpu().detach()

            mse = (traj - yy_b) ** 2
            mse = mse.reshape([mse.shape[0], -1]).max(dim=-1)[0]
            max_b, max_idx_b = mse.max(dim=0)
            if max_b > max_mse:
                max_mse = max_b
                max_traj = traj[max_idx_b]
                max_ground_truth = yy_b[max_idx_b]
                max_param = param_b[max_idx_b]
                max_unc = unc[max_idx_b]
                max_unc_independent = independent_unc[max_idx_b]
                max_independent_traj = independent_traj[max_idx_b]
        print(max_mse)
        traj = [max_ground_truth, max_traj, torch.abs(max_ground_truth - max_traj), max_unc,
                torch.abs(max_ground_truth - max_independent_traj), max_unc_independent]
        labels = ["ground_truth", "model", "error", "unc", "independent_error", "independent_rollout_unc"]
        title = str(list(max_param.cpu().detach().numpy()))
        filename = file_name_prefix + "_max_mse_traj.png"
        plot_single_traj(traj, grid_b, labels, title, path, filename, common_color_bar=False)


def plot_sampled_pde_params(task, al_iter, bins=30, label=None):
    data_path = os.path.join(task.run_save_path, "data")
    npy_filenames = glob.glob("*_alstp_" + str(al_iter) + "_*_param.npy", root_dir=data_path)
    all_params = np.array([np.load(os.path.join(data_path, f)) for f in npy_filenames])
    all_params = all_params.reshape([-1, ] + list(all_params.shape[2:]))
    all_params = all_params.reshape((len(all_params), -1))

    for i in range(task.num_pde_param):
        fig = plt.figure()
        sns.histplot(all_params[:, i].flatten(), bins=bins, label=label, log_scale=task.param_gen.log_scale[i])
        ax = plt.gca()
        ax.set_xlabel("PDE Parameter")
        ax.set_ylabel("N")
        plt.tight_layout()
        plt.savefig(os.path.join(task.img_save_path, "pde_param_" + str(i) + "_dist_" + str(al_iter) + ".png"))
        plt.close(fig)


def plot_initial_conditions(task, al_iter, n):
    for al_iter in range(al_iter + 1):

        last_data = NPYDataset(
            pde_name=task.pde_name,
            folders=task.train_data_folders,
            initial_step=1, regexp=".*alstp_" + str(al_iter) + ".*")
        data_loader = torch.utils.data.DataLoader(last_data, n)

        with torch.no_grad():
            for batch_idx, (xx, yy, grid, param, t_idx) in enumerate(data_loader):
                fig = plt.figure()
                for i in range(len(xx)):
                    plt.plot(grid[0, :, 0], xx[i, :, 0, 0], label=str(np.round(float(param[i, 0]), 4)))
                l1 = plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
                ax = plt.gca()
                ax.set_xlabel("x")
                ax.set_ylabel("IC")
                ax.title.set_text("ICs in AL iteration " + str(al_iter))
                plt.tight_layout()
                plt.savefig(os.path.join(task.run_save_path, "img", "ic_samples_" + str(al_iter) + ".png"))
                plt.close(fig)
                break


def _plot_quantiles(x, y, quantiles, label, stat="mean"):
    idx = torch.argsort(x, descending=True)
    y_sorted = y[idx]
    p_mse = []
    for p in quantiles:
        if stat == "mean":
            quant_y = y_sorted[:int(len(x) * (1 - p))]
            q_v = quant_y.mean()
        elif stat == "sum":
            quant_y = y_sorted[:int(len(x) * p)]
            q_v = quant_y.sum()
        else:
            raise ValueError(stat)
        p_mse.append(q_v)
    plt.plot(quantiles, p_mse, label=label)
    return idx


def quantile_mse_plot(unc, mse, task, step):
    quantiles = np.arange(0, 1, 0.01)

    fig = plt.figure()

    _plot_quantiles(unc, mse, quantiles, "unc")
    _plot_quantiles(mse, mse, quantiles, "true mse")

    ax = plt.gca()
    ax.set_xlabel("selection quantile")
    ax.set_ylabel("mse normed avg")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(task.run_save_path, "img", "selection_quantiles_mse_" + str(step) + ".png"))

    plt.close(fig)


def mse_quantiles_iteratively(mse, task, al_iter):
    q = torch.arange(0, 1, 0.01)
    v = torch.quantile(mse, q)

    data_dir = os.path.join(task.run_save_path, "img", "log")
    os.makedirs(data_dir, exist_ok=True)
    data = np.stack([q, v], axis=1)
    np.save(os.path.join(data_dir, "mse_quantiles_" + str(al_iter) + ".npy"), data)

    fig = plt.figure()
    for i in range(al_iter + 1):
        data = np.load(os.path.join(data_dir, "mse_quantiles_" + str(i) + ".npy"))
        plt.plot(data[:, 0], data[:, 1], label=str(i))

    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(task.run_save_path, "img", "mse_quantiles" ".png"))
    plt.close(fig)


def unc_mse_scatter(mse, unc, task, step):
    fig = plt.figure()
    plt.scatter(mse, unc)
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("mse_normed")
    ax.set_ylabel("unc")
    plt.tight_layout()
    plt.savefig(os.path.join(task.run_save_path, "img", "unc_mse_traj" + str(step) + ".png"))
    plt.close(fig)


def unc_mse_traj_plots(task, prob_model, step, loader):
    with torch.no_grad():
        all_mse = []
        all_unc = []
        for idx, (xx_b, yy_b, grid_b, param_b, t_idx) in enumerate(loader):
            traj, unc = prob_model.unc_roll_out(xx_b.to(device), grid_b.to(device), yy_b.shape[-2], param_b.to(device),
                                                t_idx.to(device))

            traj = traj.cpu().detach()
            unc = unc.cpu().detach()

            mse = (traj - yy_b) ** 2
            mse = mse.reshape([mse.shape[0], -1]).mean(dim=-1)
            unc = unc.reshape([unc.shape[0], -1]).mean(dim=-1)

            all_mse.append(mse)
            all_unc.append(unc)

        mse = torch.concat(all_mse, dim=0)
        mse = mse / torch.mean(mse)
        unc = torch.concat(all_unc, dim=0)
    quantile_mse_plot(unc, mse, task, step)
    show_misranked_examples(unc, mse, loader, task, prob_model)
    unc_mse_scatter(mse, unc, task, step)
    mse_quantiles_iteratively(mse, task, step)
    compare_rollouts(prob_model, loader)


def end_of_al_iter_plots(task, prob_model, al_iter, last=False):
    with torch.no_grad():
        for idx, (xx_b, yy_b, grid_b, param_b, t_idx) in enumerate(prob_model.val_loader):
            np.save(os.path.join(task.img_save_path, "ex_params_" + str(al_iter) + ".npy"), param_b.numpy())
            np.save(os.path.join(task.img_save_path, "ex_grid_" + str(al_iter) + ".npy"), grid_b.numpy())
            np.save(os.path.join(task.img_save_path, "ex_gt_" + str(al_iter) + ".npy"), yy_b)
            pred, unc = prob_model.unc_roll_out(xx_b.to(device), grid_b.to(device), yy_b.shape[-2],
                                                pde_param=param_b.to(device), t_idx=t_idx)
            np.save(os.path.join(task.img_save_path, "ex_mean_" + str(al_iter) + ".npy"), pred.cpu().detach().numpy())
            np.save(os.path.join(task.img_save_path, "ex_unc_" + str(al_iter) + ".npy"), unc.cpu().detach().numpy())
            break
    prob_model.evaluate(al_iter, prob_model.val_loader, prefix="al/", time_step_name="al_iter")

    os.makedirs(os.path.join(task.run_save_path, "img"), exist_ok=True)
    if not last:
        eval_on_new(task, prob_model, al_iter)
    compare_error_averages(task, prob_model, prob_model.val_loader, al_iter)
    if not last:
        if task.spatial_dim == 1:
            plot_initial_conditions(task, al_iter, 10)

        plot_sampled_pde_params(task, al_iter)

    try:
        unc_mse_traj_plots(task, prob_model, al_iter, prob_model.val_loader)
    except:  # may fail if no uncertainty is available
        return


def show_misranked_examples(unc, mse, loader, task, prob_model, quantile=0.95, n=5, suffix=""):
    # take all samples from a certain ranking quantile and see which one has the worst ranking error

    traj_idx = np.arange(len(mse))

    quantile_unc_val = torch.quantile(unc, quantile)
    quantile_mask = unc >= quantile_unc_val

    unc_idx = torch.argsort(unc, descending=True)
    unc_ranks = torch.arange(len(mse))
    unc_ranks[unc_idx] = torch.arange(len(mse))

    mse_idx = torch.argsort(mse, descending=True)
    mse_ranks = torch.arange(len(mse))
    mse_ranks[mse_idx] = torch.arange(len(mse))

    mse_ranks_in_quant = mse_ranks[quantile_mask]
    worst_mismatched = torch.argsort(mse_ranks_in_quant, descending=True)[:n]
    mse_ranks_in_quant = mse_ranks_in_quant[worst_mismatched]
    traj_idx_in_quant = traj_idx[quantile_mask][worst_mismatched]
    unc_in_quant = unc[quantile_mask][worst_mismatched]
    mse_in_quant = mse[quantile_mask][worst_mismatched]
    unc_ranks_in_quant = unc_ranks[quantile_mask][worst_mismatched]

    plt_count = 0
    total_idx = 0
    with torch.no_grad():
        for idx, (xx_b, yy_b, grid_b, param_b, t_idx) in enumerate(loader):
            for i in range(len(xx_b)):
                for j in range(len(traj_idx_in_quant)):
                    if traj_idx_in_quant[j] == total_idx:
                        yy_b = yy_b.to(device)
                        traj, unc = prob_model.unc_roll_out(xx_b.to(device), grid_b.to(device), yy_b.shape[-2],
                                                            param_b.to(device), t_idx.to(device))
                        independent_traj, independent_unc = prob_model.unc_independent_rollout(xx_b.to(device),
                                                                                               grid_b.to(device),
                                                                                               yy_b.shape[-2],
                                                                                               param_b.to(device),
                                                                                               t_idx.to(device))

                        traj = [yy_b[i], traj[i], torch.abs(yy_b - traj)[i], unc[i],
                                torch.abs(yy_b - independent_traj)[i], independent_unc[i]]
                        labels = ["ground_truth", "model", "error", "unc", "independent_error",
                                  "independent_rollout_unc"]
                        title = str(list(param_b[i].cpu().detach().numpy()))
                        title += "_unc_q_" + str(np.round(float(1 - unc_ranks_in_quant[j] / len(mse)), 2))
                        title += "_mse_q_" + str(np.round(float(1 - mse_ranks_in_quant[j] / len(mse)), 2))
                        filename = "worst_mismatched" + suffix + "_" + str(plt_count) + ".png"
                        plt_count += 1
                        plot_single_traj(traj, grid_b, labels, title, os.path.join(task.run_save_path, "img"),
                                         filename, common_color_bar=False)
                total_idx += 1


def compare_error_averages(task, prob_model, data_loader, step, mode="autoregressive", n_max=10000):
    """
    Compare cell-wise, frame-wise and traj-wise error over rank.
    """

    with torch.no_grad():

        mse_cells = []
        mse_frames = []
        mse_trajs = []
        for idx, batch in enumerate(data_loader):
            (xx_b, yy_b, grid_b, param_b, t_idx) = (e.to(device) for e in batch)
            if mode == "autoregressive":
                mse_batch = autoregressive_mse(xx_b, yy_b, param_b, grid_b, t_idx, prob_model)
            elif mode == "one_step":
                mse_batch = one_step_mse(yy_b, param_b, grid_b, t_idx, prob_model)
            elif mode == "five_step":
                mse_batch = n_step_mse(yy_b, param_b, grid_b, t_idx, prob_model, 5)
            elif mode == "10_step_red":
                mse_batch = n_step_mse_reduction(yy_b, param_b, grid_b, t_idx, prob_model, 10)
            else:
                raise ValueError(mode)
            mse_cells.append(mse_batch.flatten().cpu())
            mse_frame = mse_batch.flatten(start_dim=1, end_dim=-2).mean(dim=1)  # avg spatial dimensions
            mse_frames.append(mse_frame.flatten().cpu())
            mse_trajs.append(mse_batch.flatten(start_dim=1).mean(1).flatten().cpu())

        mses = [mse_cells, mse_frames, mse_trajs]
        labels = ["point", "frame", "traj"]
        avg_mse = torch.concat(mse_cells).mean()
        mse_cell_normed = (torch.concat(mse_cells).flatten() / avg_mse).flatten()
        # simple histogram
        fig = plt.figure()
        bins = np.histogram_bin_edges(mse_cell_normed, bins=50)
        x_max = 0
        bin_width = bins[1] - bins[0]
        x_min = np.inf
        for i in range(len(mses)):
            x = (bins[1:] + bins[:-1]) / 2
            y = np.histogram((torch.concat(mses[i]).flatten() / avg_mse).flatten(), bins=bins, density=True)[0]
            y *= bin_width
            y = np.cumsum(y[::-1])[::-1]
            xmax_i = np.max(x[y > 0.001])
            x_max = xmax_i if xmax_i > x_max else x_max

            x_min_i = np.max(x[y > 0.2])
            x_min = x_min_i if x_min_i < x_min else x_min

            plt.plot(x, y, label=labels[i])
        plt.title(task.pde_name)
        plt.legend()
        ax = plt.gca()
        ax.set_xlabel("mse normed")
        ax.set_ylabel("density cumsum")
        ax.set_xlim([x_min, x_max])
        ax.set_xscale("log")
        ax.set_ylim([0, 0.2])
        plt.tight_layout()
        plt.savefig(os.path.join(task.run_save_path, "img", "mse_hist_" + mode + "_" + str(step) + ".png"))
        plt.close(fig)

        fig = plt.figure()
        for i in range(len(mses)):
            mse = torch.sort(torch.concat(mses[i]).flatten() / avg_mse)[0]
            skip = int(np.ceil(len(mse) / n_max))
            mse = mse[::skip]
            relative_rank = np.arange(len(mse)) / len(mse)
            plt.plot(relative_rank, mse, label=labels[i])
        plt.title(task.pde_name)
        plt.legend()
        ax = plt.gca()
        ax.set_xlabel("relative rank")
        ax.set_ylabel("normed " + mode + " MSE")
        ax.set_ylim(bottom=1.0e-5)
        ax.set_yscale("log")
        plt.tight_layout()
        plt.savefig(os.path.join(task.run_save_path, "img", "mse_rank_curves_" + mode + "_" + str(step) + ".png"))
        plt.close(fig)

        q = torch.linspace(0, 0.6, 1000)
        fig = plt.figure()
        for i in range(len(mses)):
            mse = torch.concat(mses[i]) / torch.concat(mses[i]).sum()
            _plot_quantiles(mse, mse, q, labels[i], "sum")
        plt.legend()
        ax = plt.gca()
        ax.set_xlabel("relative top-k fraction (fraction of data selected)")
        ax.set_ylabel("fraction of total se sum " + mode)
        plt.title(task.pde_name)
        plt.tight_layout()
        plt.savefig(os.path.join(task.run_save_path, "img", "mse_sel_level_comparison_" + mode + "_" +
                                 str(step) + ".png"))
        plt.close(fig)
