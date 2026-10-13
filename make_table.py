import json

runs = [("9k steps", "results/metrics_9k.json"),
        ("20k steps", "results/metrics_20k.json"),
        ("VAE ceiling*", "results/vae_ceiling.json")]
data = [(name, json.load(open(path))) for name, path in runs]
ref = data[1][1]  # real-clip reference (the same real clips in every run)

keys = [("flow_warp_error", "Flow-warp error"), ("flicker", "Flicker"),
        ("mean_abs_frame_diff", "Mean frame diff"), ("lpips_adjacent", "LPIPS (adjacent)")]

lines = ["| Metric | " + " | ".join(n for n, _ in data) + " | Real clips |",
         "|---|" + "---|" * (len(data) + 1)]
lines.append("| FVD (VideoMAE) | " + " | ".join(f"{d['fvd']:.2f}" for _, d in data)
             + f" | {ref['fvd_noise_floor_real_vs_real']:.2f} (noise floor) |")
for k, label in keys:
    lines.append(f"| {label} | " + " | ".join(f"{d['generated'][k]['mean']:.4f}" for _, d in data)
                 + f" | {ref['real_reference'][k]['mean']:.4f} |")

out = "\n".join(lines)
print(out)
open("results/results_table.md", "w").write(out + "\n")