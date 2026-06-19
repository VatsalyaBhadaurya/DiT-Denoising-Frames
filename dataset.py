"""Paired (noisy/clean) image dataset with synthetic-noise fallback.

Layouts supported:
  data_root/
    clean/   *.png|jpg            -> synthetic Gaussian noise added on the fly
    noisy/   *.png|jpg  (optional)-> paired noisy inputs matched by filename
"""
import os
from glob import glob

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")


def _list_images(folder):
    files = []
    for e in _EXTS:
        files += glob(os.path.join(folder, e))
    return sorted(files)


class DenoiseDataset(Dataset):
    def __init__(self, cfg, train=True):
        self.cfg = cfg
        clean_dir = os.path.join(cfg.data_root, "clean")
        noisy_dir = os.path.join(cfg.data_root, "noisy")
        self.clean_files = _list_images(clean_dir)
        if not self.clean_files:
            raise FileNotFoundError(f"No clean images found in {clean_dir}")

        self.paired = os.path.isdir(noisy_dir) and bool(_list_images(noisy_dir))
        self.noisy_map = (
            {os.path.basename(p): p for p in _list_images(noisy_dir)}
            if self.paired else {}
        )

        mode = "L" if cfg.channels == 1 else "RGB"
        self.mode = mode
        tf = [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),                # -> [0,1]
            transforms.Normalize([0.5] * cfg.channels, [0.5] * cfg.channels),  # -> [-1,1]
        ]
        if train:
            tf.insert(1, transforms.RandomHorizontalFlip())
        self.transform = transforms.Compose(tf)

    def __len__(self):
        return len(self.clean_files) * self.cfg.dataset_repeat

    def _load(self, path):
        return self.transform(Image.open(path).convert(self.mode))

    def __getitem__(self, i):
        clean_path = self.clean_files[i % len(self.clean_files)]
        clean = self._load(clean_path)
        if self.paired:
            name = os.path.basename(clean_path)
            noisy = self._load(self.noisy_map[name]) if name in self.noisy_map else clean
        else:
            noise = torch.randn_like(clean) * self.cfg.synthetic_noise_std
            noisy = (clean + noise).clamp(-1, 1)
        return {"clean": clean, "noisy": noisy}
