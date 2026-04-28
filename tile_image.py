import torch
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F

class TileImage:
    def __init__(self, tile_size):
        self.tile_size = tile_size
    
    def __call__(self, x):
        c, _, _ = x.shape

        x = x.unsqueeze(0)
        
        patches = F.unfold(x, kernel_size=self.tile_size, stride=self.tile_size)
        patches = patches.view(1, c, self.tile_size, self.tile_size, -1)
        patches = patches.permute(0, 4, 1, 2, 3).squeeze(0)

        return patches

def tile_image(img, tile_size):
    transform = T.PILToTensor()

    img = transform(img).float()
    c, _, _ = img.shape
    print("Image Shape:", img.shape)
    img = img.unsqueeze(0)

    patches = F.unfold(img, kernel_size=args.tile_size, stride=args.tile_size)
    patches = patches.view(1, c, args.tile_size, args.tile_size, -1)
    patches = patches.permute(0, 4, 1, 2, 3).squeeze(0)
    print(f"Number of Tiles: {patches.shape[0]}")
    print(patches.shape)

    return patches

if __name__ == "__main__":
    import argparse
    import random

    p = argparse.ArgumentParser()

    p.add_argument("--tile_size", type=int, required=True)
    p.add_argument("paths", nargs=argparse.REMAINDER)

    args = p.parse_args()

    img = Image.open(args.paths[-1])
    transform = T.PILToTensor()
    to_tiles = TileImage(args.tile_size)
    # patches = tile_image(img, args.tile_size)
    img = transform(img).float()
    patches = to_tiles(img)

    ind = random.randint(0, patches.shape[0])
    patch = patches[ind].to(torch.uint8).permute(1, 2, 0)
    print(patch.shape)
    patch = Image.fromarray(patch.numpy())

    patch.save("patch.jpg")