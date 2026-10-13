"""Generate clips from a checkpoint (EMA weights) and decode them to pixels.

    python sample.py --ckpt runs/base/last.pt --n 1000 --steps 50 --out samples/gen.npy
"""
import argparse
import os

import numpy as np
import torch

from diffusion import sample_latents
from model import VideoDiT
from utils import decode_latents, get_device, load_stats, load_vae, save_gif_grid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--bs", type=int, default=50)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--sampler", default="euler", choices=["euler", "heun"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="samples/gen.npy")
    p.add_argument("--stats", default=None)
    p.add_argument("--vae", default=None)
    p.add_argument("--gif_n", type=int, default=16)
    args = p.parse_args()

    device = get_device()
    ck = torch.load(args.ckpt, map_location=device)
    cfg = ck["cfg"]
    model = VideoDiT(**cfg).to(device).eval()
    model.load_state_dict(ck["ema"])
    stats = args.stats or ck["args"]["stats"]
    vae_name = args.vae or ck["args"]["vae"]
    mean, std = load_stats(stats, device)
    vae = load_vae(vae_name, device)
    torch.manual_seed(args.seed)

    T, C, h = cfg["frames"], cfg["in_ch"], cfg["size"]
    out = []
    for i in range(0, args.n, args.bs):
        b = min(args.bs, args.n - i)
        z = sample_latents(model, (b, T, C, h, h), device, steps=args.steps, sampler=args.sampler)
        out.append(decode_latents(vae, z, mean, std))
        print(f"{i + b}/{args.n}")
    clips = np.concatenate(out)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, clips)
    save_gif_grid(clips[:args.gif_n], os.path.splitext(args.out)[0] + ".gif")
    print("saved", clips.shape, "->", args.out)


if __name__ == "__main__":
    main()