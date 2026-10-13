"""Train the latent video DiT on cached Moving MNIST latents.

    python train.py --out runs/base

Resumes automatically from runs/base/last.pt if it exists (use --no_resume to disable).
"""
import argparse
import contextlib
import copy
import os
import time

import numpy as np
import torch

from diffusion import flow_loss, sample_latents
from model import VideoDiT
from utils import decode_latents, get_device, load_stats, load_vae, save_gif_grid


@torch.no_grad()
def ema_update(ema, model, decay):
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.lerp_(pm, 1 - decay)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latents", default="data/train_latents.npy")
    p.add_argument("--stats", default="data/latent_stats.pt")
    p.add_argument("--vae", default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--out", default="runs/base")
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--patch", type=int, default=1)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--ema", type=float, default=0.9999)
    p.add_argument("--t_dist", default="logit_normal", choices=["logit_normal", "uniform"])
    p.add_argument("--amp", default="auto", choices=["auto", "bf16", "none"])
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--sample_every", type=int, default=5000)
    p.add_argument("--no_resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    os.makedirs(args.out, exist_ok=True)

    lat = np.load(args.latents)  # [N, T, C, h, w] float16, kept in RAM
    N, T, C, h, w = lat.shape
    mean, std = load_stats(args.stats, device)

    cfg = dict(in_ch=C, size=h, frames=T, patch=args.patch, dim=args.dim, depth=args.depth, heads=args.heads)
    model = VideoDiT(**cfg).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    print(f"device={device} clips={N} latent={tuple(lat.shape[1:])} params={sum(q.numel() for q in model.parameters()) / 1e6:.1f}M")

    step = 0
    last = os.path.join(args.out, "last.pt")
    if os.path.exists(last) and not args.no_resume:
        ck = torch.load(last, map_location=device)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print("resumed from step", step)

    use_bf16 = device.type == "cuda" and (args.amp == "bf16" or (args.amp == "auto" and torch.cuda.is_bf16_supported()))
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_bf16 else contextlib.nullcontext
    print("bf16 autocast:", use_bf16)

    vae = None

    def save():
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "step": step, "cfg": cfg, "args": vars(args)}, last)
        torch.save({"ema": ema.state_dict(), "cfg": cfg, "args": vars(args), "step": step},
                   os.path.join(args.out, f"ema_{step:07d}.pt"))

    def preview():
        nonlocal vae
        try:
            if vae is None:
                vae = load_vae(args.vae, device)
            z = sample_latents(ema, (8, T, C, h, w), device, steps=50)
            clips = decode_latents(vae, z, mean, std)
            save_gif_grid(clips, os.path.join(args.out, f"samples_{step:07d}.gif"), ncols=4)
        except Exception as e:  # a failed preview should not kill a long run
            print("preview failed:", repr(e))

    t0, run_loss = time.time(), None
    model.train()
    while step < args.steps:
        idx = np.random.randint(0, N, args.bs)
        x = (torch.from_numpy(lat[idx]).to(device).float() - mean) / std
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, (step + 1) / args.warmup)
        with amp():
            loss = flow_loss(model, x, args.t_dist)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema_update(ema, model, min(args.ema, (1 + step) / (10 + step)))
        step += 1

        run_loss = loss.item() if run_loss is None else 0.98 * run_loss + 0.02 * loss.item()
        if step % 100 == 0:
            print(f"step {step} loss {run_loss:.4f} {100 / (time.time() - t0):.2f} it/s")
            t0 = time.time()
        if step % args.sample_every == 0:
            preview()
        if step % args.save_every == 0 or step == args.steps:
            save()


if __name__ == "__main__":
    main()