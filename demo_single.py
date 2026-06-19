"""Single-image denoising demo on data/clean/sih.png (the SIH logo).

Trains a small DiT to denoise synthetically-noised versions of one image,
then writes a noisy/denoised/clean comparison strip.

    python demo_single.py            # train + sample
"""
import torch
from torchvision.utils import save_image

import train
from config import Config


def make_cfg():
    return Config(
        image_size=128,        # higher res -> less pixelation
        patch_size=8,
        channels=3,
        data_root="data",
        synthetic_noise_std=0.25,
        dataset_repeat=128,     # one image -> 128 virtual samples / epoch
        hidden_dim=256,
        depth=6,
        num_heads=8,
        timesteps=1000,
        predict="eps",
        conditional=True,
        batch_size=8,
        epochs=150,
        lr=2e-4,
        val_every=10,
        num_workers=0,          # 0 is safest on Windows
        ckpt_path="checkpoints/sih.pt",
    )


def main():
    cfg = make_cfg()
    train.train(cfg)

    # build a noisy view of the target and denoise it
    device = train.get_device(cfg)
    model, diff = train.build(cfg, device)
    train.load_ckpt(cfg.ckpt_path, model, map_location=device)

    ds = __import__("dataset").DenoiseDataset(cfg, train=False)
    sample = ds[0]
    clean = sample["clean"].unsqueeze(0).to(device)
    noisy = sample["noisy"].unsqueeze(0).to(device)
    out = train.denoise(model, diff, noisy, cfg, device)  # SDEdit-style, stable

    mse = ((out - clean) ** 2).mean().item()
    print(f"PSNR(denoised vs clean): {10 * torch.log10(torch.tensor(4.0 / mse)):.2f} dB")
    strip = torch.cat([noisy, out, clean], dim=0)  # noisy | denoised | clean
    save_image((strip + 1) / 2, "sih_compare.png", nrow=3)
    print("wrote sih_compare.png  (noisy | denoised | clean)")


if __name__ == "__main__":
    main()
