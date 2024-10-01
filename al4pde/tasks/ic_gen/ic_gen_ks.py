import numpy as np
import torch
from tensordict import TensorDict
from al4pde.tasks.ic_gen.ic_gen import ICGenerator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ICGenKS(ICGenerator):

    def __init__(self, N, nx, lmin, lmax, amp_max=1, requires_grad=True, single_fixed=False, length=1, in_fourier_domain=False):
        super().__init__(requires_grad, single_fixed)
        self.N = N
        self.nx = nx
        self.lmin = lmin
        self.lmax = lmax
        self.length = length
        self.grid = torch.from_numpy(np.linspace(0, self.length, self.nx, dtype=np.float32)).to(device)
        self.amp_max = amp_max
        self.in_fourier_domain = in_fourier_domain

    def get_grid(self, n):
        grid = self.grid[:, None]
        return grid.expand([n, ] + list(grid.shape))

    def _initialize_ic_params(self, n: int) -> TensorDict:
        amplitude_normed = torch.rand(size=[n, self.N, 1], generator=self.rng, device=device,
                                      requires_grad=self.requires_grad)
        phi_normed = torch.rand(size=(n, self.N), generator=self.rng, device=device,
                                requires_grad=self.requires_grad)
        freq_idx = torch.randint(self.lmin, self.lmax, (n, self.N), generator=self.rng, device=device)
        return TensorDict({"amp": amplitude_normed, "phi": phi_normed, "freq": freq_idx}, batch_size=n)

    def generate_initial_conditions(self, ic_params: TensorDict, pde_params: torch.Tensor) -> torch.Tensor:
        if self.in_fourier_domain:
            u = self.create_ic_fourier_domain(ic_params)
        else:
            u = self.create_ic_real_domain(ic_params)
        return u[..., None, None]  # [bs, nx, nt, nc]

    def create_ic_real_domain(self, ic_params):
        amp = 2 * self.amp_max * (ic_params.get("amp") - 0.5)
        phi = 2.0 * torch.pi * ic_params.get("phi")
        x = 2 * np.pi * ic_params.get("freq").unsqueeze(1) * self.get_grid(len(amp)) / self.length + phi.unsqueeze(1)
        return torch.sum(amp[:, 0:, 0].unsqueeze(1) * torch.sin(x), -1)

    def create_ic_fourier_domain(self, ic_params):
        amp = 2 * self.amp_max * (ic_params.get("amp") - 0.5)[..., 0]
        phase = 2.0 * torch.pi * ic_params.get("phi")
        idx = ic_params.get("freq")
        # Compute the real and imaginary parts
        real = amp * torch.cos(phase - np.pi / 2) * self.nx / 2
        imag = amp * torch.sin(phase - np.pi / 2) * self.nx / 2

        # Combine into a complex tensor
        complex_tensor = torch.view_as_complex(torch.stack((real, imag), dim=-1))
        res = torch.zeros((len(amp), self.nx // 2 + 1), device=amp.device, dtype=torch.cfloat)
        for i in range(self.lmin, self.lmax):
            mi = (complex_tensor * (idx == i)).sum(-1)
            res[:, i] = mi
        return torch.fft.irfft(res, n=self.nx)
