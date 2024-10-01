import torch


class Loss(torch.nn.Module):

    def __init__(self, name, normalize_channels=True):
        super().__init__()
        self.name = name
        self.task_norm = None
        self.initial_step = None
        self.normalize_channels = normalize_channels

    def init_training(self, task_norm, initial_step):
        self.task_norm = task_norm
        self.initial_step = initial_step

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, reduction: str | None = "mean",
                 initial_step: int = None) -> torch.Tensor:
        """ Computes the loss between pred and target, ignoring entries before the intial_step time step.

        @param pred:
        @param target:
        @param reduction: how to and which dimensions to average (mean_traj, mean_frame, mean_cell, mean)
        @param initial_step: ignore first initial_step timesteps for calculating the loss
        @return:
        """

        if initial_step is None:
            initial_step = self.initial_step
        if self.normalize_channels:
            pred = self.task_norm.norm_traj(pred)
            target = self.task_norm.norm_traj(target)

        element_wise_loss = self._calc_loss(pred[..., initial_step:, :], target[..., initial_step:, :])

        if reduction == "mean":
            return element_wise_loss.mean()
        elif reduction == "mean_cell":
            return element_wise_loss.mean(-1)
        elif reduction == "mean_frame":
            cell_wise_loss = element_wise_loss.mean(-1)
            return cell_wise_loss.flatten(1, -2).mean(1)
        elif reduction == "mean_traj":
            return element_wise_loss.flatten(1, -1).mean(1)
        elif reduction is None:
            return element_wise_loss
        else:
            raise ValueError(reduction)

    def _calc_loss(self, pred, target):
        raise NotImplementedError


class MSE(Loss):

    def __init__(self, normalize_channels=True):
        name = "MSE"
        if normalize_channels:
            name = "c_norm_" + name
        super().__init__(name, normalize_channels)

    def _calc_loss(self, pred, target):
        return (pred - target) ** 2


class MAELoss(Loss):

    def __init__(self, normalize_channels=True):
        name = "MAE"
        if normalize_channels:
            name = "c_norm_" + name
        super().__init__(name, normalize_channels)

    def _calc_loss(self, pred, target):
        return torch.abs(pred - target)


class FrameRelMSE(Loss):
    def __init__(self, normalize_channels=False, eps=1e-6):
        name = "FrameRelMSE"
        if normalize_channels:
            name = "c_norm_" + name
        super().__init__(name, normalize_channels)
        self.eps = eps

    def _calc_loss(self, pred, target):
        se = (pred - target) ** 2
        n_dim = len(target.shape) - 3
        norm = target.flatten(1, -3).square().mean(1)
        for i in range(n_dim):
            norm = norm.unsqueeze(1)
        return se / (norm + self.eps)

    def __call__(self, pred, target, reduction="mean", initial_step=None):
        if reduction in ("mean", "mean_traj", "mean_frame"):
            if initial_step is None:
                initial_step = self.initial_step
            if self.normalize_channels:
                pred = self.task_norm.norm_traj(pred)
                target = self.task_norm.norm_traj(target)
            pred = pred[..., initial_step:, :]
            target = target[..., initial_step:, :]
            se = (pred - target) ** 2
            norm = target.flatten(1, -3).square().mean(1)
            se = se.flatten(1, -3).square().mean(1)
            loss = se / (norm + self.eps)
            if reduction == "mean":
                return loss.mean()
            elif reduction == "mean_traj":
                return loss.flatten(1, -1).mean(1)
            elif reduction == "mean_frame":
                return loss.mean(-1)
        else:
            return super().__call__(pred, target, reduction, initial_step)


class PredIdentity(Loss):
    def __init__(self):
        name = "Mean"
        super().__init__(name, False)

    def _calc_loss(self, pred, target):
        return pred
