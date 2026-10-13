"""Generate Moving MNIST clips (uint8, [N, T, H, W]).

Train clips use digits from the MNIST train split, eval clips use digits from the
MNIST test split, so the eval set never contains a digit the model trained on.

    python data.py --out data --train 20000 --eval 2000
"""
import argparse
import os

import numpy as np


def generate(digits, n, frames, size, rng, num_digits=2, digit=28):
    """digits: uint8 [M, 28, 28]. Returns uint8 [n, frames, size, size]."""
    out = np.zeros((n, frames, size, size), np.uint8)
    lim = size - digit
    for i in range(n):
        imgs = digits[rng.integers(0, len(digits), num_digits)]
        pos = rng.uniform(0, lim, (num_digits, 2))
        ang = rng.uniform(0, 2 * np.pi, num_digits)
        speed = rng.uniform(2.0, 4.0, num_digits)  # pixels per frame
        vel = np.stack([np.cos(ang), np.sin(ang)], 1) * speed[:, None]
        for t in range(frames):
            frame = np.zeros((size, size), np.uint8)
            for k in range(num_digits):
                x = min(max(int(round(pos[k, 0])), 0), lim)
                y = min(max(int(round(pos[k, 1])), 0), lim)
                frame[y:y + digit, x:x + digit] = np.maximum(
                    frame[y:y + digit, x:x + digit], imgs[k])
            out[i, t] = frame
            pos += vel
            for k in range(num_digits):
                for d in range(2):  # bounce off the walls
                    if pos[k, d] < 0:
                        pos[k, d] = -pos[k, d]
                        vel[k, d] = -vel[k, d]
                    elif pos[k, d] > lim:
                        pos[k, d] = 2 * lim - pos[k, d]
                        vel[k, d] = -vel[k, d]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data")
    p.add_argument("--mnist_root", default="data/mnist_raw")
    p.add_argument("--train", type=int, default=20000)
    p.add_argument("--eval", type=int, default=2000)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from torchvision.datasets import MNIST

    os.makedirs(args.out, exist_ok=True)
    tr = MNIST(args.mnist_root, train=True, download=True).data.numpy()
    te = MNIST(args.mnist_root, train=False, download=True).data.numpy()

    train = generate(tr, args.train, args.frames, args.size, np.random.default_rng(args.seed))
    ev = generate(te, args.eval, args.frames, args.size, np.random.default_rng(args.seed + 1))
    np.save(os.path.join(args.out, "train.npy"), train)
    np.save(os.path.join(args.out, "eval.npy"), ev)
    print("train", train.shape, "eval", ev.shape)


if __name__ == "__main__":
    main()