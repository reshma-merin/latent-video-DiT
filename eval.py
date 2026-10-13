"""Evaluate generated clips: FVD + temporal-consistency metrics, with real-data references.

    python eval.py --gen samples/gen.npy --real data/eval.npy --n 1000

FVD uses a VideoMAE (Kinetics-finetuned) feature extractor by default. That is NOT the
standard I3D extractor, so numbers are not comparable to published FVD values; they are
only comparable to other runs of this same script. Pass --extractor i3d --i3d_path <file>
to use the TorchScript I3D from the StyleGAN-V repo instead.

The "noise floor" is the FVD between two disjoint halves of real eval clips: it is the
best score a perfect generator could expect at this sample size.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from metrics import feature_stats, flicker, flow_warp_error, frame_diff, frechet_distance
from utils import get_device


class Extractor:
    def __init__(self, kind, device, i3d_path=None):
        self.kind, self.device = kind, device
        if kind == "videomae":
            from transformers import VideoMAEForVideoClassification
            self.m = VideoMAEForVideoClassification.from_pretrained(
                "MCG-NJU/videomae-base-finetuned-kinetics").to(device).eval()
            self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
        else:
            assert i3d_path, "--i3d_path is required for --extractor i3d"
            self.m = torch.jit.load(i3d_path).eval().to(device)

    @torch.no_grad()
    def __call__(self, clips):  # uint8 numpy [B, T, H, W]
        x = torch.from_numpy(clips).to(self.device)
        B, T, H, W = x.shape
        if self.kind == "videomae":
            assert T == 16, "VideoMAE expects 16 frames"
            f = x.float().div(255).reshape(B * T, 1, H, W).repeat(1, 3, 1, 1)
            f = F.interpolate(f, size=(224, 224), mode="bilinear", align_corners=False)
            f = ((f.reshape(B, T, 3, 224, 224)) - self.mean) / self.std
            seq = self.m.videomae(pixel_values=f)[0]
            feat = self.m.fc_norm(seq.mean(1)) if self.m.fc_norm is not None else seq[:, 0]
        else:
            v = x[:, None].repeat(1, 3, 1, 1, 1)  # [B, 3, T, H, W] uint8
            feat = self.m(v, rescale=False, resize=True, return_features=True)
        return feat.float().cpu().numpy()


def extract_all(ex, clips, bs):
    return np.concatenate([ex(clips[i:i + bs]) for i in range(0, len(clips), bs)])


def lpips_adjacent(clips, device, bs=16):
    import lpips
    fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    res = []
    with torch.no_grad():
        for i in range(0, len(clips), bs):
            x = torch.from_numpy(clips[i:i + bs]).to(device).float() / 127.5 - 1
            b, T, H, W = x.shape

            def prep(y):
                y = y.reshape(-1, 1, H, W).repeat(1, 3, 1, 1)
                return F.interpolate(y, size=(128, 128), mode="bilinear", align_corners=False)

            d = fn(prep(x[:, :-1]), prep(x[:, 1:])).reshape(b, T - 1).mean(1)
            res.append(d.cpu().numpy())
    return np.concatenate(res)


def summarize(clips, device, use_lpips):
    out = {"flow_warp_error": flow_warp_error(clips), "flicker": flicker(clips),
           "mean_abs_frame_diff": frame_diff(clips)}
    if use_lpips:
        out["lpips_adjacent"] = lpips_adjacent(clips, device)
    return {k: {"mean": float(v.mean()), "std": float(v.std())} for k, v in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen", required=True)
    p.add_argument("--real", default="data/eval.npy")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--extractor", default="videomae", choices=["videomae", "i3d"])
    p.add_argument("--i3d_path", default=None)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--out", default="results/metrics.json")
    p.add_argument("--no_lpips", action="store_true")
    args = p.parse_args()

    device = get_device()
    gen = np.load(args.gen)[:args.n]
    real = np.load(args.real)
    n = min(len(gen), len(real) // 2) if len(real) >= 2 * len(gen) else len(gen)
    real_a = real[:n]
    real_b = real[n:2 * n] if len(real) >= 2 * n else None
    print(f"gen={len(gen)} real_a={len(real_a)} real_b={None if real_b is None else len(real_b)}")

    ex = Extractor(args.extractor, device, args.i3d_path)
    fg, fa = extract_all(ex, gen, args.bs), extract_all(ex, real_a, args.bs)
    fvd = frechet_distance(*feature_stats(fg), *feature_stats(fa))
    result = {"n_gen": len(gen), "extractor": args.extractor, "fvd": fvd}
    if real_b is not None:
        fb = extract_all(ex, real_b, args.bs)
        result["fvd_noise_floor_real_vs_real"] = frechet_distance(*feature_stats(fb), *feature_stats(fa))
    else:
        print("note: need >= 2x gen-count real clips to compute the noise floor")

    result["generated"] = summarize(gen, device, not args.no_lpips)
    result["real_reference"] = summarize(real_a, device, not args.no_lpips)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nFVD ({args.extractor}): {fvd:.2f}"
          + (f"   noise floor: {result['fvd_noise_floor_real_vs_real']:.2f}" if real_b is not None else ""))
    print(f"{'metric':<22}{'generated':>14}{'real':>14}")
    for k in result["generated"]:
        print(f"{k:<22}{result['generated'][k]['mean']:>14.4f}{result['real_reference'][k]['mean']:>14.4f}")
    print("saved ->", args.out)


if __name__ == "__main__":
    main()