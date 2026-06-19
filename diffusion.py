"""DDPM-style beta schedule and forward/reverse diffusion utilities."""
import torch


class Diffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=2e-2, device="cpu"):
        self.timesteps = timesteps
        self.device = device

        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=device), alphas_cumprod[:-1]]
        )

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        # posterior variance for reverse sampling
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

    @staticmethod
    def _gather(values, t, shape):
        """Pick per-sample scalars at timesteps t and broadcast to image shape."""
        out = values.gather(0, t)
        return out.reshape(t.shape[0], *([1] * (len(shape) - 1)))

    def q_sample(self, x0, t, noise):
        """Forward noising: q(x_t | x_0)."""
        sa = self._gather(self.sqrt_alphas_cumprod, t, x0.shape)
        soma = self._gather(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sa * x0 + soma * noise

    def predict_x0_from_eps(self, x_t, t, eps):
        sa = self._gather(self.sqrt_alphas_cumprod, t, x_t.shape)
        soma = self._gather(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - soma * eps) / sa

    @torch.no_grad()
    def p_sample(self, model_out, x_t, t, predict="eps"):
        """One reverse step x_t -> x_{t-1}."""
        if predict == "eps":
            eps = model_out
            x0 = self.predict_x0_from_eps(x_t, t, eps)
        else:  # model predicts x0
            x0 = model_out
            sa = self._gather(self.sqrt_alphas_cumprod, t, x_t.shape)
            soma = self._gather(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
            eps = (x_t - sa * x0) / soma

        beta = self._gather(self.betas, t, x_t.shape)
        alpha = self._gather(self.alphas, t, x_t.shape)
        soma = self._gather(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        mean = (x_t - beta / soma * eps) / torch.sqrt(alpha)
        if t.min() == 0:
            return mean
        var = self._gather(self.posterior_variance, t, x_t.shape)
        return mean + torch.sqrt(var) * torch.randn_like(x_t)

    def to_eps(self, model_out, x_t, t, predict="eps"):
        """Return predicted noise regardless of the model's parameterisation."""
        if predict == "eps":
            return model_out
        sa = self._gather(self.sqrt_alphas_cumprod, t, x_t.shape)
        soma = self._gather(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sa * model_out) / soma

    @torch.no_grad()
    def ddim_step(self, model_out, x_t, t, t_prev, predict="eps"):
        """One deterministic DDIM step (eta=0): x_t -> x_{t_prev}."""
        eps = self.to_eps(model_out, x_t, t, predict)
        x0 = self.predict_x0_from_eps(x_t, t, eps).clamp(-1, 1)
        acp_prev = self._gather(self.alphas_cumprod, t_prev, x_t.shape)
        return torch.sqrt(acp_prev) * x0 + torch.sqrt(1 - acp_prev) * eps
