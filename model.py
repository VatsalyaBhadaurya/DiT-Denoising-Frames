"""Diffusion Transformer (DiT) for image denoising."""
import math
import torch
import torch.nn as nn


def timestep_embedding(t, dim, max_period=10000):
    """Sinusoidal timestep embedding (as in DDPM / Transformer)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class PatchEmbed(nn.Module):
    """Image -> sequence of patch tokens via a strided conv."""

    def __init__(self, image_size, patch_size, channels, dim):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.grid = image_size // patch_size
        self.num_patches = self.grid ** 2
        self.proj = nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                 # (B, dim, grid, grid)
        return x.flatten(2).transpose(1, 2)  # (B, num_patches, dim)


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero timestep conditioning."""

    def __init__(self, dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )
        # produces shift/scale/gate for attn and mlp
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.ada(c).chunk(6, dim=1)
        h = modulate(self.norm1(x), shift_a, scale_a)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_a.unsqueeze(1) * attn
        h = modulate(self.norm2(x), shift_m, scale_m)
        x = x + gate_m.unsqueeze(1) * self.mlp(h)
        return x


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        dim = cfg.hidden_dim
        # conditional denoising: concat the noisy observation on the channel axis
        in_ch = cfg.channels * 2 if cfg.conditional else cfg.channels
        self.patch_embed = PatchEmbed(cfg.image_size, cfg.patch_size, in_ch, dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches, dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.t_mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.blocks = nn.ModuleList(
            [DiTBlock(dim, cfg.num_heads, cfg.mlp_ratio) for _ in range(cfg.depth)]
        )
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.head = nn.Linear(dim, cfg.patch_size ** 2 * cfg.channels)

        nn.init.zeros_(self.ada_out[1].weight)
        nn.init.zeros_(self.ada_out[1].bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def unpatchify(self, x):
        c, p = self.cfg.channels, self.cfg.patch_size
        g = self.patch_embed.grid
        x = x.reshape(x.shape[0], g, g, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, g * p, g * p)

    def forward(self, x_t, t, cond=None):
        """x_t: (B,C,H,W) noisy sample; t: (B,) timesteps; cond: (B,C,H,W) noisy observation."""
        if cond is not None:
            x_t = torch.cat([x_t, cond], dim=1)
        x = self.patch_embed(x_t) + self.pos_embed
        c = self.t_mlp(timestep_embedding(t, self.cfg.hidden_dim))
        for blk in self.blocks:
            x = blk(x, c)
        shift, scale = self.ada_out(c).chunk(2, dim=1)
        x = modulate(self.norm_out(x), shift, scale)
        x = self.head(x)
        return self.unpatchify(x)
