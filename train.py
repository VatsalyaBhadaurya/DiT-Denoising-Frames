"""Train / validate / sample a DiT denoiser.

Usage:
    python train.py                 # train
    python train.py --denoise IMG   # denoise a single image with the saved ckpt
"""
import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image

from config import Config
from dataset import DenoiseDataset
from diffusion import Diffusion
from model import DiT


def get_device(cfg):
    return torch.device(cfg.device if torch.cuda.is_available() else "cpu")


def build(cfg, device):
    model = DiT(cfg).to(device)
    diff = Diffusion(cfg.timesteps, cfg.beta_start, cfg.beta_end, device)
    return model, diff


def loss_fn(model, diff, batch, cfg, device):
    clean = batch["clean"].to(device)
    cond = batch["noisy"].to(device) if cfg.conditional else None
    t = torch.randint(0, cfg.timesteps, (clean.shape[0],), device=device)
    noise = torch.randn_like(clean)
    x_t = diff.q_sample(clean, t, noise)
    pred = model(x_t, t, cond)
    target = noise if cfg.predict == "eps" else clean
    return F.mse_loss(pred, target)


# ----------------------------------------------------------------------------- #
def save_ckpt(path, model, opt, epoch, cfg):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "opt": opt.state_dict(),
         "epoch": epoch, "cfg": vars(cfg)},
        path,
    )


def load_ckpt(path, model, opt=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if opt is not None and "opt" in ckpt:
        opt.load_state_dict(ckpt["opt"])
    return ckpt.get("epoch", 0)


# ----------------------------------------------------------------------------- #
@torch.no_grad()
def validate(model, diff, loader, cfg, device):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        total += loss_fn(model, diff, batch, cfg, device).item() * batch["clean"].size(0)
        n += batch["clean"].size(0)
    model.train()
    return total / max(n, 1)


@torch.no_grad()
def sample(model, diff, cond, cfg, device, steps=None, ddim=True):
    """Reverse diffusion from pure noise, conditioned on `cond` (noisy image).

    DDIM (deterministic, default) is stable and works well with few steps;
    set ddim=False for stochastic DDPM ancestral sampling over all timesteps.
    """
    model.eval()
    b = cond.shape[0]
    x = torch.randn(b, cfg.channels, cfg.image_size, cfg.image_size, device=device)
    c = cond if cfg.conditional else None

    if ddim:
        steps = steps or cfg.sample_steps
        seq = torch.linspace(cfg.timesteps - 1, 0, steps, dtype=torch.long).tolist()
        for cur, nxt in zip(seq, seq[1:] + [0]):
            t = torch.full((b,), cur, device=device, dtype=torch.long)
            t_prev = torch.full((b,), nxt, device=device, dtype=torch.long)
            out = model(x, t, c)
            x = diff.ddim_step(out, x, t, t_prev, cfg.predict)
    else:
        for step in reversed(range(cfg.timesteps)):
            t = torch.full((b,), step, device=device, dtype=torch.long)
            x = diff.p_sample(model(x, t, c), x, t, cfg.predict)

    model.train()
    return x.clamp(-1, 1)


@torch.no_grad()
def denoise(model, diff, noisy, cfg, device, t_start=None, steps=None):
    """Supervised denoising (SDEdit-style): start the reverse process FROM the
    noisy observation at timestep `t_start` and refine with DDIM. Far more
    stable than generating from pure noise for noisy->clean tasks."""
    t_start = t_start or cfg.denoise_t_start
    steps = steps or cfg.denoise_steps
    model.eval()
    cond = noisy if cfg.conditional else None
    x = noisy.clone()
    seq = torch.linspace(t_start, 0, steps, dtype=torch.long).tolist()
    for cur, nxt in zip(seq, seq[1:] + [0]):
        t = torch.full((noisy.shape[0],), cur, device=device, dtype=torch.long)
        t_prev = torch.full((noisy.shape[0],), nxt, device=device, dtype=torch.long)
        x = diff.ddim_step(model(x, t, cond), x, t, t_prev, cfg.predict)
    model.train()
    return x.clamp(-1, 1)


# ----------------------------------------------------------------------------- #
def train(cfg):
    device = get_device(cfg)
    print(f"device: {device}")

    ds = DenoiseDataset(cfg, train=True)
    n_val = min(max(1, int(0.1 * len(ds))), len(ds) - 1) if len(ds) > 1 else 0
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])
    drop_last = len(train_ds) >= cfg.batch_size
    train_loader = DataLoader(train_ds, cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, drop_last=drop_last)
    val_loader = DataLoader(val_ds, cfg.batch_size, num_workers=cfg.num_workers)
    print(f"dataset: {len(ds)} images (paired={ds.paired}), "
          f"train={len(train_ds)} val={len(val_ds)}")

    model, diff = build(cfg, device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = float("inf")
    for epoch in range(cfg.epochs):
        running = 0.0
        for i, batch in enumerate(train_loader):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = loss_fn(model, diff, batch, cfg, device)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            running += loss.item()
        sched.step()
        print(f"epoch {epoch+1:03d}/{cfg.epochs}  loss={running/len(train_loader):.4f}  "
              f"lr={sched.get_last_lr()[0]:.2e}")

        if len(val_ds) > 0 and (epoch + 1) % cfg.val_every == 0:
            vloss = validate(model, diff, val_loader, cfg, device)
            print(f"           val_loss={vloss:.4f}")
            if vloss < best:
                best = vloss
                save_ckpt(cfg.ckpt_path, model, opt, epoch, cfg)
                print(f"           saved -> {cfg.ckpt_path}")

    save_ckpt(cfg.ckpt_path, model, opt, cfg.epochs, cfg)
    print(f"done. final ckpt -> {cfg.ckpt_path}")


# ----------------------------------------------------------------------------- #
def denoise_image(cfg, img_path, out_path="denoised.png"):
    from PIL import Image
    from torchvision import transforms

    device = get_device(cfg)
    model, diff = build(cfg, device)
    load_ckpt(cfg.ckpt_path, model, map_location=device)

    mode = "L" if cfg.channels == 1 else "RGB"
    tf = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * cfg.channels, [0.5] * cfg.channels),
    ])
    noisy = tf(Image.open(img_path).convert(mode)).unsqueeze(0).to(device)
    out = denoise(model, diff, noisy, cfg, device)
    save_image((out + 1) / 2, out_path)
    print(f"denoised -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--denoise", type=str, default=None, help="path to a noisy image")
    ap.add_argument("--out", type=str, default="denoised.png")
    args = ap.parse_args()

    cfg = Config()
    if args.denoise:
        denoise_image(cfg, args.denoise, args.out)
    else:
        train(cfg)
