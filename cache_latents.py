"""Encode every frame with a pretrained image VAE once, so training runs on latents only.

    python cache_latents.py --data data/train.npy

Writes float16 latents [N, T, 4, H/8, W/8] and per-channel mean/std used to normalize them.
Also prints a reconstruction PSNR on the first batch so you can see how much the VAE costs you.
"""
import argparse

import numpy as np
import torch

from utils import get_device, load_vae


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/train.npy")
    p.add_argument("--out", default="data/train_latents.npy")
    p.add_argument("--stats", default="data/latent_stats.pt")
    p.add_argument("--vae", default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--bs", type=int, default=256, help="frames per VAE batch")
    args = p.parse_args()

    device = get_device()
    vae = load_vae(args.vae, device)
    x = np.load(args.data, mmap_mode="r")
    N, T, H, W = x.shape
    C, h, w = 4, H // 8, W // 8
    out = np.lib.format.open_memmap(args.out, mode="w+", dtype=np.float16, shape=(N, T, C, h, w))

    s1 = torch.zeros(C, dtype=torch.float64)
    s2 = torch.zeros(C, dtype=torch.float64)
    cnt = 0
    cpb = max(1, args.bs // T)
    first = True
    with torch.no_grad():
        for i in range(0, N, cpb):
            b = torch.from_numpy(np.ascontiguousarray(x[i:i + cpb])).to(device)
            f = b.reshape(-1, 1, H, W).float() / 127.5 - 1
            f = f.repeat(1, 3, 1, 1)
            z = vae.encode(f).latent_dist.mean  # [B*T, C, h, w]
            if first:
                rec = ((vae.decode(z).sample.mean(1) + 1) / 2).clamp(0, 1)
                ref = (f[:, 0] + 1) / 2
                mse = ((rec - ref) ** 2).mean().item()
                print(f"VAE reconstruction PSNR on first batch: {10 * np.log10(1 / max(mse, 1e-10)):.2f} dB")
                first = False
            s1 += z.double().sum((0, 2, 3)).cpu()
            s2 += (z.double() ** 2).sum((0, 2, 3)).cpu()
            cnt += z.shape[0] * h * w
            out[i:i + b.shape[0]] = z.reshape(b.shape[0], T, C, h, w).cpu().numpy().astype(np.float16)
            if (i // cpb) % 20 == 0:
                print(f"{i + b.shape[0]}/{N} clips")
    out.flush()
    mean = s1 / cnt
    std = (s2 / cnt - mean ** 2).sqrt()
    torch.save({"mean": mean.float().view(1, 1, C, 1, 1), "std": std.float().view(1, 1, C, 1, 1)}, args.stats)
    print("latents", out.shape, "mean", mean.tolist(), "std", std.tolist())


if __name__ == "__main__":
    main()