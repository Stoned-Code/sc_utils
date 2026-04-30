from PIL import Image
import numpy as np
import imagehash
from typing import List, Tuple
import hashlib
from scipy.ndimage import gaussian_filter
import torch
import tqdm


def add_noise(img, density=0.1, strength=0.05, mode='additive', seed=None):
    """
    Improved noise addition function.
    - density: Fraction of pixels affected (0-1).
    - strength: Noise intensity (e.g., std dev for Gaussian).
    - mode: 'multiplicative' (scale pixels) or 'additive' (add/subtract values).
    - seed: For reproducibility.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Convert to array and normalize to [0,1] for easier noise application
    img_array = np.array(img).astype(np.float32) / 255.0
    height, width = img_array.shape[:2]
    channels = 1 if len(img_array.shape) == 2 else img_array.shape[2]
    
    # Generate Gaussian noise (mean=0, std=strength) for more natural distribution
    noise = np.random.normal(0, strength, (height, width))
    
    # Expand to match channels
    if channels > 1:
        noise = np.stack([noise] * channels, axis=-1)
    else:
        noise = noise[..., np.newaxis]  # For grayscale
    
    # Apply density: Randomly mask noise to affect only ~density% of pixels
    mask = np.random.rand(height, width) < density
    if channels > 1:
        mask = np.stack([mask] * channels, axis=-1)
    else:
        mask = mask[..., np.newaxis]
    noise = noise * mask  # Zero out non-affected areas
    
    # Apply noise based on mode
    if mode == 'multiplicative':
        # Multiplicative: img * (1 + noise), clipped to [0,1]
        noisy_array = img_array * (1 + noise)
    elif mode == 'additive':
        # Additive: img + noise, clipped to [0,1]
        noisy_array = img_array + noise
    else:
        raise ValueError("Mode must be 'multiplicative' or 'additive'")
    
    noisy_array = np.clip(noisy_array, 0, 1)
    
    # Scale back to uint8 and return PIL Image
    return Image.fromarray((noisy_array * 255).astype(np.uint8))

def closeness_to_black(img: Image.Image) -> float:
    """
    Return a number in [0.0, 1.0] where 1.0 means the image is completely black,
    and 0.0 means it is completely white (in terms of average luminosity).
    """
    #print("Closeness Type:", type(img))
    try:
        if type(img) == np.ndarray:
            avg = img.mean()
        else:
            gray = img.convert("L")
            arr = np.asarray(gray, dtype=np.float32)
        
            avg = arr.mean()               # in [0,255]


        return 1.0 - (avg / 255.0)
    except OSError as ex:
        return 1.0

def closeness_to_black_from_path(img_path):
    img = Image.open(img_path)
    closeness = closeness_to_black(img)
    return closeness

def closeness_to_white(img: Image.Image) -> float:
    """
    Return a number in [0.0, 1.0] where 1.0 means the image is completely white,
    and 0.0 means it is completely black.
    """
    try:
        if type(img) == np.ndarray:
            avg = img.mean()
        else:
            gray = img.convert("L")
            arr = np.asarray(gray, dtype=np.float32)
            avg = arr.mean()               # in [0,255]

            
        return avg / 255.0
    except OSError as ex:
        return 1.0

def closeness_to_white_from_path(img_path):
    img = Image.open(img_path)
    closeness = closeness_to_white(img)
    return closeness


def __get_image_hash__(image):
    """
    Calculate the SHA-256 hash of a Pillow Image object.

    This function converts the image to RGBA mode, gets its pixel data as bytes,
    and computes the SHA-256 hash of that data.

    Parameters:
        image (PIL.Image.Image): The image object to hash.

    Returns:
        str: The SHA-256 hash of the image's pixel data as a hexadecimal string.
    """
    rgba_image = image.convert('RGBA')
    image_bytes = rgba_image.tobytes()
    hash_obj = hashlib.sha256()
    hash_obj.update(image_bytes)
    return hash_obj.hexdigest()


def get_hash(img):
    img_hash = None
    width = 0
    height = 0
    ratio = 0.0

    img_hash = __get_image_hash__(img)
    width = img.width
    height = img.height

    if width > height:
        ratio = width / height
    else:
        ratio = height / width
    
    return str(img_hash), width, height, ratio


def get_sim_hash(img, hash_size = 32):
    #print("Image Type:", type(img))
    sim_hash = imagehash.phash(img, hash_size=hash_size)
    return sim_hash


def get_hash_from_path(path):
    img_hash = None
    width = 0
    height = 0
    ratio = 0.0

    with Image.open(path) as img:
        # img_hash = imagehash.phash(img)
        # width = img.width
        # height = img.height
        # if width > height:
        #     ratio = width / height
        # else:
        #     ratio = height / width
        width = img.width
        height = img.height
        ratio = max([width, height]) / min([width, height])

        #img_hash, width, height, ratio = get_hash(img)
    hash_obj = hashlib.sha256()  # Initialize SHA-256 hash object
    chunk_size = 4096  # Set chunk size to 4KB for efficient reading

    with open(path, 'rb') as file:  # Open file in binary read mode
        while True:
            try:
                chunk = file.read(chunk_size)  # Read a chunk of the file
                if not chunk:  # If no more data is read, break the loop
                    break

                hash_obj.update(chunk)  # Update hash object with the chunk
            except PermissionError as ex:
                print("Messed up on", path)
                raise ex
    return hash_obj.hexdigest() , width, height, ratio


# def get_hash_from_path(file_path):
#     """
#     Calculate the SHA-256 hash of an image file.

#     This function reads an image file in binary mode, processes it in chunks,
#     and computes its SHA-256 hash, which is returned as a hexadecimal string.

#     Parameters:
#     file_path (str): The path to the image file.

#     Returns:
#     str: The SHA-256 hash of the image file as a hexadecimal string.

#     Raises:
#     FileNotFoundError: If the file does not exist.
#     PermissionError: If the file cannot be read due to permission issues.
#     """
#     hash_obj = hashlib.sha256()  # Initialize SHA-256 hash object
#     chunk_size = 4096  # Set chunk size to 4KB for efficient reading

#     with open(file_path, 'rb') as file:  # Open file in binary read mode
#         while True:
#             chunk = file.read(chunk_size)  # Read a chunk of the file
#             if not chunk:  # If no more data is read, break the loop
#                 break
#             hash_obj.update(chunk)  # Update hash object with the chunk

#     return hash_obj.hexdigest()  # Return the hash as a hexadecimal string


def is_solid_color_tensor(img: torch.Tensor) -> bool:
    """Check if a torch tensor image (C,H,W) is a solid single color."""
    if not isinstance(img, torch.Tensor) or img.dim() != 3:
        return False
    
    c, h, w = img.shape
    if h == 0 or w == 0:
        return False
    
    # Compare everything against the first pixel
    first_pixel = img[:, 0, 0]          # shape (C,)
    
    return bool(torch.all(img == first_pixel.view(c, 1, 1)))


def is_solid_color(image):
    """Check if the given PIL Image is a solid color.
    
    Args:
        image (PIL.Image): The image to check.
        
    Returns:
        bool: True if the image is a solid color, False otherwise.
    """
    # Check if the image is empty (zero width or height)
    if type(image) == np.ndarray:
        # if len(image.shape) == 2:
        #     image = image.reshape(*image.shape, 1)
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        image = Image.fromarray(image)
    if image.size[0] == 0 or image.size[1] == 0:
        return False
    
    # Get the color value of the first pixel
    first_pixel = image.getpixel((0, 0))
    
    # Check if all pixels match the first pixel's color
    return all(pixel == first_pixel for pixel in image.getdata())


def multi_square_crop(img: Image.Image, count: int) -> List[Image.Image]:
    """
    Crop `count` squares from img.
      - If img is wide (width > height), crops run left→right.
      - Otherwise (tall or square), crops run top→bottom.
    Special case: if count == 1, return the single centered square.
    """
    if count < 1:
        raise ValueError("count must be at least 1")

    W, H = img.width, img.height
    S = min(W, H)            # side length of each square
    total_offset = (W - S) if W > H else (H - S)

    # Special case: single centered crop
    if count == 1:
        if W > H:
            left = (W - S) // 2
            upper = 0
        else:
            left = 0
            upper = (H - S) // 2
        return [img.crop((left, upper, left + S, upper + S))]

    # Otherwise, evenly space count windows from 0 to total_offset
    step = total_offset / (count - 1)
    crops = []
    for i in range(count):
        offset = int(round(i * step))
        if W > H:
            left, upper = offset, 0
        else:
            left, upper = 0, offset
        crops.append(img.crop((left, upper, left + S, upper + S)))

    return crops


def set_shortest_length(img: Image, length: int):
    width = img.width
    height = img.height

    new_width = 0
    new_height = 0

    if width > height:
        new_height = length
        new_width = width * (length / height)

    elif height > width:
        new_width = length
        new_height = height * (length / width)
    
    else:
        new_width = length
        new_height = length
        
    return img.resize((int(new_width), int(new_height)))


def square_padding(img, use_grayscale = False):
    if not isinstance(img, np.ndarray):
        if use_grayscale:
            img = img.convert("L")
        img = np.array(img)
    
    shape = img.shape
    
    if len(shape) == 2:
        img = img.reshape((img.shape[0], img.shape[1], 1))
        img = np.concat([img, img, img], axis=2)
        shape = img.shape

    start_X = 0
    start_y = 0

    width = img.shape[1]
    height = img.shape[0]

    if width > height:
        padding = width - height

        padding_top  = int(padding * 0.5)
        padding_bottom = padding - padding_top

        start_X = 0
        start_y = padding_top + 1

        padding_top = np.zeros((padding_top, width, shape[2]))
        padding_bottom = np.zeros((padding_bottom, width, shape[2]))

        concated = np.concat([padding_top, img, padding_bottom], axis=0)
        shape = img.shape
        if not use_grayscale:
            return concated.astype(np.uint8), start_X, start_y, width, height
        else:

            if concated.shape[-1] != 1:
                concated = np.array(Image.fromarray(concated.astype(np.uint8)).convert("L"))

            return concated.reshape((concated.shape[0], concated.shape[1])).astype(np.uint8), start_X, start_y, width, height

    elif height > width:
        padding = height - width
        padding_left = int(padding * 0.5)
        padding_right = padding - padding_left

        start_X = padding_left + 1
        start_y = 0

        padding_left = np.zeros((height, padding_left, shape[2]))
        padding_right = np.zeros((height, padding_right, shape[2]))

        concated = np.concat([padding_left, img, padding_right], axis = 1)

        if not use_grayscale:
            return concated.astype(np.uint8), start_X, start_y, width, height
        else:

            if concated.shape[-1] != 1:
                concated = np.array(Image.fromarray(concated.astype(np.uint8)).convert("L"))
            return concated.reshape((concated.shape[0], concated.shape[1])).astype(np.uint8), start_X, start_y, width, height

    elif height == width:
        
        if not use_grayscale:
            return img.astype(np.uint8), start_X, start_y, width, height
        else:
            if img.shape[-1] != 1:
                img = np.array(Image.fromarray(img.astype(np.uint8)).convert("L"))
            return img.reshape((img.shape[0], img.shape[1])).astype(np.uint8), start_X, start_y, width, height


def ssim(img1, img2, C1=0.01**2, C2=0.03**2):
    """Structural Similarity Index Measure"""
    if type(img1) != np.ndarray:
        img1 = np.array(img1)
    if type(img2) != np.ndarray:
        img2 = np.array(img2)
        
    img1 = img1.astype(float) / 255
    img2 = img2.astype(float) / 255
    mu1 = gaussian_filter(img1, 1.5)
    mu2 = gaussian_filter(img2, 1.5)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2
    sigma1_sq = gaussian_filter(img1 ** 2, 1.5) - mu1_sq
    sigma2_sq = gaussian_filter(img2 ** 2, 1.5) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, 1.5) - mu12
    num = (2 * mu12 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    return np.mean(num / den)


def ms_ssim(img1, img2, weights=None, C1=0.01**2, C2=0.03**2, levels=5):
    if weights is None:
        weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    
    if type(img1) != np.ndarray:
        img1 = np.array(img1)
    if type(img2) != np.ndarray:
        img2 = np.array(img2)
    
    img1 = img1.astype(float) / 255.0
    img2 = img2.astype(float) / 255.0
    
    msssim = 1.0
    for i in range(levels):
        if i > 0:  # downsample
            img1 = img1[::2, ::2]
            img2 = img2[::2, ::2]
        
        mu1 = gaussian_filter(img1, 1.5)
        mu2 = gaussian_filter(img2, 1.5)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu12 = mu1 * mu2
        
        sigma1_sq = gaussian_filter(img1 ** 2, 1.5) - mu1_sq
        sigma2_sq = gaussian_filter(img2 ** 2, 1.5) - mu2_sq
        sigma12 = gaussian_filter(img1 * img2, 1.5) - mu12
        
        l  = (2 * mu12 + C1) / (mu1_sq + mu2_sq + C1)
        cs = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        
        stage = cs if i == levels - 1 else l * cs
        msssim *= stage ** weights[i]
    
    return msssim


def process_image(img, scale, use_grayscale=False, invert=False, pad_to_square = True, normalize = False):
    if type(img) == str:
        img = Image.open(img)

    if use_grayscale:
        img = img.convert("L")
    else:
        img = img.convert("RGB")
    width = img.width
    height = img.height 
    if pad_to_square:
        # img = image_processing.pad_to_square(img, use_grayscale)

        new_scale = None
        if width > height:
            ratio = height / width
            new_scale = (scale[0], int(scale[1] * ratio))

        elif height > width:
            ratio = width / height
            #print((int(scale[0] * ratio), scale[1]))
            new_scale = (int(scale[0] * ratio), scale[1])
        if new_scale != None:
            img = img.resize(new_scale)

        #print(img.width, img.height)
        img, start_X, start_y, width, height = square_padding(img, use_grayscale)
        # print(img.shape)
        img = Image.fromarray(img)
        #print(img.shape)
        img = img.resize(scale)
        
        img = np.array(img)
        #print(img.shape)
        #print(img.shape)
    else:
        img = img.resize(scale)
        img = np.array(img).astype(np.uint8)

        #img = img
        # img = img.astype(np.uint8)
    # img = Image.fromarray(img)

    #print(img.shape)
    if width != scale[0]:
        width = scale[0]
    
    if height != scale[1]:
        height = scale[1]

    if not use_grayscale:
        # print(type(img))
        # print(img.shape)
        img = img[:, :, :3]
   


    if invert:
        img = 255 - img
    if normalize:
        img = img / 255
    #img = (img - 127.5) / 127.5
    # print(img.dtype)
    #print(img.shape)
    if pad_to_square:
        return img, start_X, start_y, width, height
    
    return img, None, None, width, height
    

def split_image(image, N):
    """
    Split a PIL Image into N segments along its longest side.

    Args:
        image (PIL.Image): The input image to split.
        N (int): The number of segments to create.

    Returns:
        list of PIL.Image: A list containing N image segments.
    """
    # Get image dimensions
    width, height = image.size

    if width >= height:
        # Split along width into vertical strips
        base_size = width // N  # Base width of each segment
        remainder = width % N   # Remainder pixels to distribute
        segments = []
        current_x = 0  # Starting x-coordinate

        for i in range(N):
            # First 'remainder' segments get an extra pixel
            segment_width = base_size + (1 if i < remainder else 0)
            next_x = current_x + segment_width
            # Crop box: (left, upper, right, lower)
            box = (current_x, 0, next_x, height)
            segment = image.crop(box)
            segments.append(segment)
            current_x = next_x  # Update starting position
    else:
        # Split along height into horizontal strips
        base_size = height // N  # Base height of each segment
        remainder = height % N   # Remainder pixels to distribute
        segments = []
        current_y = 0  # Starting y-coordinate

        for i in range(N):
            # First 'remainder' segments get an extra pixel
            segment_height = base_size + (1 if i < remainder else 0)
            next_y = current_y + segment_height
            # Crop box: (left, upper, right, lower)
            box = (0, current_y, width, next_y)
            segment = image.crop(box)
            segments.append(segment)
            current_y = next_y  # Update starting position

    return segments

def triple_square_crop(img: Image):
    width = img.width
    height = img.height

    if width > height:
        size = (height, height)
        center_pos = int(width/2) - int(height / 2)
        left = img.crop((0, 0, *size))
        center = img.crop((center_pos, 0, center_pos + size[0], size[1]))
        right = img.crop((width - height, 0, width, height))
        return left, center, right
    else:
        size = (width, width)
        center_pos = int(height / 2) - int(width / 2)
        top = img.crop(0, 0, *size)
        center = img.crop((0, center_pos, size[0], center_pos + size[1]))
        bottom = img.crop((0, height - width, width, height))
        return top, center, bottom


if __name__ == "__main__":
    import sys
    import glob
    import os
    import gc
    import pandas as pd
    import argparse
    import shutil
    import pathlib
    p = argparse.ArgumentParser("Process Images in various ways")
    p.add_argument("--delete_whites", type=float, default=-1)
    p.add_argument("--delete_blacks", type=float, default = -1)
    p.add_argument("--delete_solids", action="store_true")
    p.add_argument("--rename_to_hash", action="store_true")

    p.add_argument("paths", nargs=argparse.REMAINDER)
    

    args = p.parse_args()

    paths = args.paths

    paths_df = pd.DataFrame({"full_path": paths})

    delete_blacks = args.delete_blacks#"--delete_blacks" in sys.argv

    if args.delete_solids:
        for i, row in tqdm.tqdm(paths_df.iterrows(), total=len(paths_df), desc="Deleting Solids"):
            img = Image.open(row["full_path"])
            is_solid = is_solid_color(img)
            
            if is_solid:
                img.close()
                os.remove(row["ful_path"])

    if delete_blacks> 0 and delete_blacks <= 1:
        #paths_df["blackness"] = paths_df["full_path"].apply(lambda p: closeness_to_black_from_path(p))
        #delete_thresh = float(sys.argv[sys.argv.index("--delete_blacks") + 1])
        overall_amt = len(paths_df)
        print("Checking blacks...")
        for i, row in tqdm.tqdm(paths_df.iterrows(), total=overall_amt, desc="Deleting Black"):
            blackness = closeness_to_black_from_path(row["full_path"])
            # print(f"Deleting Black: {i + 1}/{overall_amt}", end="\r")
            if blackness >= delete_blacks:
                os.remove(row["full_path"])
                print("Deleted Black:", row["full_path"])

        print("Finished deleting blacks!")
        gc.collect()

    delete_whites = args.delete_whites


    if delete_whites > 0 and delete_whites <= 1:
        #paths_df["whiteness"] = paths_df["full_path"].apply(lambda p: closeness_to_white_from_path(p))
        #delete_thresh = float(sys.argv[sys.argv.index("--delete_whites") + 1])
        overall_amt = len(paths_df)
        print("Deleting Whites...")
        for i, row in tqdm.tqdm(paths_df.iterrows(), total=overall_amt, desc="Checking Whites"):
            print(f"Checking White: {i + 1}/{overall_amt}", end="\r")
            if not os.path.exists(row["full_path"]):
                continue

            whiteness = closeness_to_white_from_path(row["full_path"])
            if whiteness > delete_blacks:
                os.remove(row["full_path"])
                print("Deleted White:", row["full_path"])
        
        #del delete_thresh

        gc.collect()
        print("Finsihed deleting whites!")
    
    #rename_to_hash = args.rename_to_hash# "--rename_to_hash" in sys.argv


