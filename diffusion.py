"""Rectified-flow objective and samplers.

Path: x_t = (1 - t) * x0 + t * eps, t in [0, 1]  (t=0 data, t=1 noise).
The network predicts the velocity v = eps - x0. Sampling integrates from t=1 to t=0.
"""
import torch
import torch.nn.functional as F


def flow_loss(model, x0, t_dist="logit_normal"):
    B = x0.shape[0]
    if t_dist == "logit_normal":
        t = torch.sigmoid(torch.randn(B, device=x0.device))
    else:
        t = torch.rand(B, device=x0.device)
    eps = torch.randn_like(x0)
    tt = t.view(B, 1, 1, 1, 1)
    xt = (1 - tt) * x0 + tt * eps
    pred = model(xt, t)
    return F.mse_loss(pred.float(), (eps - x0).float())


@torch.no_grad()
def sample_latents(model, shape, device, steps=50, sampler="euler"):
    """Returns normalized latents of the given shape [B, T, C, h, w]."""
    x = torch.randn(shape, device=device)
    B = shape[0]
    ts = torch.linspace(1, 0, steps + 1).tolist()
    for i in range(steps):
        t, tn = ts[i], ts[i + 1]
        dt = tn - t  # negative
        v = model(x, torch.full((B,), t, device=device)).float()
        if sampler == "heun" and tn > 0:
            xe = x + dt * v
            v2 = model(xe, torch.full((B,), tn, device=device)).float()
            v = 0.5 * (v + v2)
        x = x + dt * v
    return x