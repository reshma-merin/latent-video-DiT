"""Factorized space-time Diffusion Transformer for video latents.

Each block does: spatial self-attention (tokens within one frame) ->
temporal self-attention (same spatial position across frames) -> MLP,
all modulated by the timestep embedding with adaLN-Zero (as in DiT).

Input/output: [B, T, C, h, w] latents; the network predicts the flow velocity.
Run `python model.py` for a quick shape / parameter-count self-test.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.cos(), args.sin()], dim=-1)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):  # [B, N, D]
        B, N, D = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, D))


class STBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        ln = lambda: nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.n1, self.n2, self.n3 = ln(), ln(), ln()
        self.sa = Attention(dim, heads)
        self.ta = Attention(dim, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(approximate="tanh"),
                                 nn.Linear(int(dim * mlp_ratio), dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))

    def forward(self, x, c):  # x: [B, T, N, D], c: [B, D]
        B, T, N, D = x.shape
        m = self.ada(c)[:, None, None, :].chunk(9, dim=-1)

        h = modulate(self.n1(x), m[0], m[1])
        h = self.sa(h.reshape(B * T, N, D)).reshape(B, T, N, D)
        x = x + m[2] * h

        h = modulate(self.n2(x), m[3], m[4])
        h = h.permute(0, 2, 1, 3).reshape(B * N, T, D)
        h = self.ta(h).reshape(B, N, T, D).permute(0, 2, 1, 3)
        x = x + m[5] * h

        h = modulate(self.n3(x), m[6], m[7])
        return x + m[8] * self.mlp(h)


class VideoDiT(nn.Module):
    def __init__(self, in_ch=4, size=8, frames=16, patch=1, dim=384, depth=12, heads=6):
        super().__init__()
        assert size % patch == 0
        self.patch, self.gs, self.in_ch = patch, size // patch, in_ch
        self.N = self.gs ** 2
        self.embed = nn.Conv2d(in_ch, dim, patch, patch)
        self.pos_s = nn.Parameter(torch.zeros(1, 1, self.N, dim))
        self.pos_t = nn.Parameter(torch.zeros(1, frames, 1, dim))
        self.t_mlp = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([STBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada_f = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.out = nn.Linear(dim, patch * patch * in_ch)
        self._init()

    def _init(self):
        def basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.apply(basic)
        nn.init.trunc_normal_(self.pos_s, std=0.02)
        nn.init.trunc_normal_(self.pos_t, std=0.02)
        w = self.embed.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.embed.bias)
        for b in self.blocks:  # adaLN-Zero: every block starts as identity
            nn.init.zeros_(b.ada[-1].weight)
            nn.init.zeros_(b.ada[-1].bias)
        nn.init.zeros_(self.ada_f[-1].weight)
        nn.init.zeros_(self.ada_f[-1].bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t):  # x: [B, T, C, H, W], t: [B] in [0, 1]
        B, T, C, H, W = x.shape
        h = self.embed(x.reshape(B * T, C, H, W))  # [B*T, D, g, g]
        h = h.flatten(2).transpose(1, 2).reshape(B, T, self.N, -1)
        h = h + self.pos_s + self.pos_t[:, :T]
        c = self.t_mlp(timestep_embedding(t * 1000, 256))
        for blk in self.blocks:
            h = blk(h, c)
        shift, scale = self.ada_f(c)[:, None, None, :].chunk(2, dim=-1)
        h = self.out(modulate(self.norm(h), shift, scale))  # [B, T, N, p*p*C]
        p, g = self.patch, self.gs
        h = h.view(B, T, g, g, p, p, C).permute(0, 1, 6, 2, 4, 3, 5)
        return h.reshape(B, T, C, g * p, g * p)


if __name__ == "__main__":
    m = VideoDiT()
    x = torch.randn(2, 16, 4, 8, 8)
    y = m(x, torch.rand(2))
    print("params (M):", sum(p.numel() for p in m.parameters()) / 1e6)
    print("in", tuple(x.shape), "out", tuple(y.shape))
    assert y.shape == x.shape
    # Zero-init output layer => exactly zero at init; take one step to check gradients flow.
    loss = ((m(x, torch.rand(2)) - x) ** 2).mean()
    loss.backward()
    print("init loss", loss.item(), "| grad on out.weight:", m.out.weight.grad.abs().sum().item() > 0)
    # Patch=2 variant
    m2 = VideoDiT(patch=2, depth=2)
    assert m2(x, torch.rand(2)).shape == x.shape
    print("self-test OK")