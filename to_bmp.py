from PIL import Image
import argparse
import os
from sc_utils.processing.image_processing import set_longest_length, set_shortest_length
import numpy as np
p = argparse.ArgumentParser()

p.add_argument("--path", type=str, required=True)
p.add_argument("--size", nargs="+", type=int, default=None)
p.add_argument("--crop", choices={"vertical", "horizontal"}, default=None)
p.add_argument("--crop_maximum", type=int, default=None)
p.add_argument("--set_length", choices={"longest", "shortest"}, default=None)

args = p.parse_args()

if args.set_length is not None:
    set_length = eval(f"set_{args.set_length}_length")
else:
    set_length = None

if args.crop is not None:
    crop = eval(f"center_crop_{args.crop}")
else:
    crop = None

def center_crop_horizontal(array: np.ndarray, maximum: int):
    H, W, C = array.shape

    start_ind = (W - maximum) // 2

    return array[:, start_ind:start_ind+maximum,:]


def center_crop_vertical(array: np.ndarray, maximum: int):
    H, W, C = array.shape

    start_ind = (H - maximum) // 2

    return array[start_ind:start_ind + maximum, :, :]

with Image.open(args.path) as img:

    path, ext = os.path.splitext(args.path)

    # if args.size is not None:
    if set_length is not None:
        img = set_length(img, args.size[1])
        # img = img.resize(tuple(args.size))
        array = np.array(img)

        if crop is not None:
            array = crop(array, args.maximum)

        img = Image.fromarray(array)


    if args.size is not None:
        img = img.resize(tuple(args.size))

    img.save(f"{path}.bmp")