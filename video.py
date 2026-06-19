"""Frame-by-frame video denoising with the trained DiT denoiser (OpenCV I/O).

Each frame is resized to the model resolution, denoised with the SDEdit-style
`train.denoise`, then resized back to the original resolution. Frames are
processed in batches for speed. Frames are cleaned independently, so slight
temporal flicker is possible (see README for the temporally-consistent option).
"""
import cv2
import torch

import train


def _frames_to_tensor(frames, cfg, device):
    """List of HxWx3 BGR uint8 -> (B,C,size,size) in [-1,1]."""
    out = []
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        if cfg.channels == 1:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[..., None]
        rgb = cv2.resize(rgb, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0  # [0,1]
        out.append(t * 2 - 1)                                        # [-1,1]
    return torch.stack(out).to(device)


def _tensor_to_frames(x, out_size):
    """(B,C,size,size) in [-1,1] -> list of HxWx3 BGR uint8 at out_size=(w,h)."""
    x = ((x + 1) / 2).clamp(0, 1)
    frames = []
    for t in x:
        arr = (t.permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
        if arr.shape[2] == 1:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        arr = cv2.resize(arr, out_size, interpolation=cv2.INTER_CUBIC)
        frames.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    return frames


def denoise_video(cfg, in_path, out_path="denoised.mp4", batch_size=8):
    device = train.get_device(cfg)
    model, diff = train.build(cfg, device)
    train.load_ckpt(cfg.ckpt_path, model, map_location=device)

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {in_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    print(f"input: {in_path}  {w}x{h} @ {fps:.1f}fps  ~{total} frames")

    done, buf = 0, []

    def flush():
        nonlocal done
        if not buf:
            return
        x = _frames_to_tensor(buf, cfg, device)
        out = train.denoise(model, diff, x, cfg, device)
        for fr in _tensor_to_frames(out, (w, h)):
            writer.write(fr)
        done += len(buf)
        buf.clear()
        print(f"  denoised {done}/{total} frames", end="\r")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        buf.append(frame)
        if len(buf) >= batch_size:
            flush()
    flush()

    cap.release()
    writer.release()
    print(f"\ndenoised video -> {out_path}")
