"""Configuration for DiT denoising."""
from dataclasses import dataclass


@dataclass
class Config:
    # data
    image_size: int = 128
    patch_size: int = 8
    channels: int = 3          # 1 = grayscale, 3 = RGB
    data_root: str = "data"    # expects data/noisy and data/clean (paired) OR data/clean only
    synthetic_noise_std: float = 0.25  # used when no paired noisy images exist
    dataset_repeat: int = 128  # virtually replicate the image set (handy for single-image demos)

    # model
    hidden_dim: int = 256
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0

    # diffusion
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    predict: str = "eps"       # "eps" or "x0"
    conditional: bool = True   # condition the model on the noisy observation
    sample_steps: int = 50     # DDIM steps for generate-from-noise sampling
    # supervised denoising inference (SDEdit-style: start from the noisy image)
    denoise_t_start: int = 200 # how far back to noise before refining (higher = stronger cleanup)
    denoise_steps: int = 20    # DDIM refinement steps

    # training
    batch_size: int = 16
    epochs: int = 100
    lr: float = 2e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    num_workers: int = 0      # 0 is safest on Windows
    amp: bool = True
    val_every: int = 5
    ckpt_path: str = "checkpoints/sih.pt"

    device: str = "cuda"  # falls back to cpu automatically in code
