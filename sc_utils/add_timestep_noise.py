import torch
from PIL import Image
import torchvision.transforms as T
import argparse

def get_timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.arange(half, dtype=torch.float32, device=t.device)
    freqs = 10000 ** (-freqs / half)   # logarithmic spacing
    angles = t[:, None] * freqs[None, :]  # outer product
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    return emb  # shape [batch, dim]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-T", type=int, default = 1000)
    p.add_argument("imgs", nargs = argparse.REMAINDER)
    p.add_argument("--batch_size", type=int, default = 3)
    args = p.parse_args()

    to_tensor = T.PILToTensor()
    ext = args.imgs[-1].split(".")[-1]
    img_tensor = to_tensor(Image.open(args.imgs[-1])).permute(1, 2, 0)
    img_stack = torch.stack([img_tensor] * args.batch_size).to(torch.float32) / 255.0
    epsilon = torch.rand_like(img_stack)

    t = torch.randint(1, args.T + 1, (args.batch_size,))

    betas = torch.linspace(1e-4, 0.02, args.T)

    alphabar = torch.cumprod(1 - betas, dim=0)

    sqrt_alpha_bar = torch.sqrt(alphabar[t-1])
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alphabar[t-1])

    x_t = sqrt_alpha_bar.unsqueeze(1).unsqueeze(2).unsqueeze(3) * img_stack + \
        sqrt_one_minus_alpha_bar.unsqueeze(1).unsqueeze(2).unsqueeze(3) * epsilon

    noisy_img = (x_t * 255).to(torch.uint8)
    noisy_img = [x.squeeze(0) if x.ndim == 4 else x for x in noisy_img]
    noisy_img = torch.cat(noisy_img, dim=1)
    print("Shape:", noisy_img.shape)
    noisy_img = Image.fromarray(noisy_img.numpy())

    print("Timestep Embedding:", get_timestep_embedding(t, 5))

    noisy_img.save(f"noisy.{ext}")