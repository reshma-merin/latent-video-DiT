"""Pure numpy/OpenCV metrics (no torch): Frechet distance and temporal-consistency scores.

All clip arrays are uint8 [N, T, H, W].
"""
import cv2
import numpy as np
from scipy import linalg


def feature_stats(feats):
    feats = np.asarray(feats, dtype=np.float64)
    return feats.mean(0), np.cov(feats, rowvar=False)


def frechet_distance(mu1, s1, mu2, s2, eps=1e-6):
    diff = mu1 - mu2
    covmean = linalg.sqrtm(s1.dot(s2))
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * eps
        covmean = linalg.sqrtm((s1 + off).dot(s2 + off))
    covmean = np.real(covmean)
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def flow_warp_error(clips):
    """Per-clip mean |warp(frame[t+1]) - frame[t]| with Farneback flow, in [0, 1].

    Lower = smoother, physically consistent motion. Always compare against the same
    number computed on real clips; Farneback is imperfect, so the absolute value alone
    means little.
    """
    N, T, H, W = clips.shape
    gx, gy = np.meshgrid(np.arange(W), np.arange(H))
    res = np.zeros(N)
    for i in range(N):
        errs = []
        for t in range(T - 1):
            a, b = clips[i, t], clips[i, t + 1]
            flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mx = (gx + flow[..., 0]).astype(np.float32)
            my = (gy + flow[..., 1]).astype(np.float32)
            warped = cv2.remap(b.astype(np.float32), mx, my, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
            errs.append(np.abs(warped - a.astype(np.float32)).mean() / 255.0)
        res[i] = np.mean(errs)
    return res


def flicker(clips):
    """Per-clip mean |change in global mean intensity| between adjacent frames, in [0, 1]."""
    m = clips.reshape(clips.shape[0], clips.shape[1], -1).mean(-1) / 255.0
    return np.abs(np.diff(m, axis=1)).mean(1)


def frame_diff(clips):
    """Per-clip mean absolute pixel change between adjacent frames, in [0, 1]."""
    c = clips.astype(np.float32) / 255.0
    return np.abs(np.diff(c, axis=1)).mean((1, 2, 3))