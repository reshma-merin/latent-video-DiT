import numpy as np
import torch
from PIL import Image


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_stats(path, device):
    s = torch.load(path, map_location=device)
    return s["mean"].to(device), s["std"].to(device)  # each [1, 1, C, 1, 1]


def load_vae(name, device):
    from diffusers import AutoencoderKL
    return AutoencoderKL.from_pretrained(name).to(device).eval()


@torch.no_grad()
def decode_latents(vae, z, mean, std, chunk=256):
    """z: normalized latents [B, T, C, h, w] -> uint8 numpy [B, T, H, W] (grayscale)."""
    B, T = z.shape[:2]
    z = (z * std + mean).reshape(B * T, *z.shape[2:])
    dtype = next(vae.parameters()).dtype
    outs = []
    for i in range(0, z.shape[0], chunk):
        x = vae.decode(z[i:i + chunk].to(dtype)).sample
        x = ((x.float().mean(1) + 1) / 2).clamp(0, 1)
        outs.append((x * 255).round().byte().cpu())
    x = torch.cat(outs)
    return x.reshape(B, T, *x.shape[1:]).numpy()


def save_gif_grid(clips, path, ncols=4, fps=8):
    """clips: uint8 [N, T, H, W]."""
    N, T, H, W = clips.shape
    nrows = (N + ncols - 1) // ncols
    frames = []
    for t in range(T):
        canvas = np.zeros((nrows * H, ncols * W), np.uint8)
        for i in range(N):
            r, c = divmod(i, ncols)
            canvas[r * H:(r + 1) * H, c * W:(c + 1) * W] = clips[i, t]
        frames.append(Image.fromarray(canvas))
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0)