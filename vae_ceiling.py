import numpy as np, torch
from utils import get_device, load_stats, load_vae, decode_latents

device = get_device()
mean, std = load_stats("data/latent_stats.pt", device)
vae = load_vae("stabilityai/sd-vae-ft-mse", device)
lat = np.load("data/train_latents.npy")[:1000]
out = []
for i in range(0, len(lat), 50):
    z = (torch.from_numpy(lat[i:i+50]).float().to(device) - mean) / std
    out.append(decode_latents(vae, z, mean, std))
np.save("samples/vae_recon_1000.npy", np.concatenate(out))
print("saved")