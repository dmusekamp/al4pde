import torch
from tensordict import TensorDict
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ICGenerator:

    def __init__(self, requires_grad=True, single_fixed=False):
        self.requires_grad = requires_grad
        self.single_fixed = single_fixed
        self.fixed_params = None
        self.rng = None
        if self.single_fixed and self.requires_grad:
            raise ValueError("Single fixed should only be used with requires_grad=False")

    def get_grid(self, n):
        raise NotImplementedError

    def initialize_ic_params(self, n: int) -> TensorDict:
        if self.single_fixed:
            return torch.cat([self.fixed_params for _ in range(n)], dim=0)
        else:
            return self._initialize_ic_params(n)

    def _initialize_ic_params(self, n: int) -> TensorDict:
        raise NotImplementedError

    def generate_initial_conditions(self, ic_params: TensorDict, pde_params: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def set_rng(self, rng):
        self.rng = rng
        if self.single_fixed and self.fixed_params is None:
            self.rng = torch.Generator(device=device).manual_seed(398764592)
            self.fixed_params = self._initialize_ic_params(1)

