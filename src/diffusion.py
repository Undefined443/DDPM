import torch
import torch.nn.functional as F


class Diffusion:
    def __init__(
        self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cuda"
    ):

        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(
            beta_start, beta_end, num_timesteps, dtype=torch.float32, device=device
        )

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = self.alphas_cumprod**0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        self.sqrt_inv_alphas = torch.sqrt(1.0 / self.alphas)
        self.noise_coef = self.betas / self.sqrt_one_minus_alphas_cumprod
        self.variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def diffusion(self, x_0, noise, t):
        # p(x_t | x_0)
        coef1 = self.sqrt_alphas_cumprod[t]
        coef2 = self.sqrt_one_minus_alphas_cumprod[t]
        return coef1 * x_0 + coef2 * noise

    def denoise(self, pred_noise, t, x_t):
        # p(x_{t-1} | x_t, x_0)
        coef1 = self.sqrt_inv_alphas[t]
        coef2 = self.noise_coef[t]
        coef3 = self.variance[t] ** 0.5
        noise = torch.randn_like(pred_noise)
        return coef1 * (x_t - coef2 * pred_noise) + coef3 * noise