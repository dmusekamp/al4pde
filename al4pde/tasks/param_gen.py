import math
import torch
import numpy as np
from omegaconf import ListConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PDEParamGenerator(object):
    """Generates pde parameters for the pde. """

    def __init__(self, min_val, max_val, log_scale, const_val=None):
        if not (isinstance(min_val, list) or isinstance(min_val, ListConfig)):
            min_val = [min_val]
            max_val = [max_val]
            log_scale = [log_scale]
        self.min_val = torch.from_numpy(np.array(list(min_val))).float()  # list or tuple
        self.max_val = torch.from_numpy(np.array(list(max_val))).float()
        self.log_scale = list(log_scale)
        self.num_pde_params = len(min_val)
        self.rng = None
        self.const_val = const_val
        if const_val is not None:
            if not isinstance(const_val, list):
                const_val = [const_val]
            self.const_val = torch.from_numpy(np.array(const_val).astype("float32")).to(device)

    def get_normed_pde_params(self, n):
        if self.const_val is None:
            pde_param_values = torch.rand(size=[n, self.num_pde_params], generator=self.rng,
                                          device=device, requires_grad=True)
            return pde_param_values
        else:
            return torch.ones(size=[n, self.num_pde_params], device=device)

    def get_pde_params(self, pde_params_normed):
        """
        f(x) = a exp(b*x)
        f(0) = a != l0

        f(1) = l0 exp(b) != l1
        exp(b) = l1 / l0
        b = ln (l1 / l0)

        """
        if self.const_val:
            return pde_params_normed * self.const_val
        else:
            res = []
            for i in range(self.num_pde_params):
                param_i = pde_params_normed[..., i]
                param_i = param_i.clamp(0, 1)
                if self.log_scale[i]:
                    param_i = self.min_val[i] * torch.exp(param_i * math.log(self.max_val[i] / self.min_val[i]))
                else:
                    param_i = param_i * (self.max_val[i] - self.min_val[i]) + self.min_val[i]
                res.append(param_i)

            return torch.stack(res, dim=-1)

    def set_rng(self, rng):
        self.rng = rng



class PDEParamGeneratorInt:
    def __init__(self, min_val, max_val, n):
        self.min_val = min_val  # list or tuple or container
        self.max_val = max_val
        self.rng = None
        self.log_scale = [False] * n
        self.num_pde_params = n

    def get_normed_pde_params(self, n):
        return torch.randint(low=self.min_val, high=self.max_val + 1, size=(n, self.num_pde_params), generator=self.rng,
                             device=device, dtype=torch.int32)

    def get_pde_params(self, pde_params_normed):
        return pde_params_normed

    def set_rng(self, rng):
        self.rng = rng
