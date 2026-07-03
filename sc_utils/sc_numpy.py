import hashlib
import numpy as np
from PIL import Image

def numpy_hash(arr):
    """
    Hashes a numpy array using hashlib.

    Parameters:
    arr (numpy.ndarray): The numpy array to hash.

    Returns:
    str: The hexadecimal representation of the hash.
    """
    # Ensure the array is C-contiguous
    arr = np.ascontiguousarray(arr)
    
    # Create a hash object
    h = hashlib.md5()
    
    # Update the hash object with the array's data
    h.update(arr)
    
    # Return the hexadecimal representation of the hash
    return h.hexdigest()


def compare_images(a, b, mode=None):
    """
    Compare two images on a pixel-by-pixel basis and return a value between 0 and 1.
    
    Parameters:
    a, b: Either file paths (str) or NumPy arrays (float32) representing images with pixel values in [0, 255].
    mode: Optional; if specified, convert images to this mode (e.g., 'RGB', 'L') before comparison.
    
    Returns:
    A float between 0 and 1, where 0 means identical images and 1 means maximally different images.
    """
    # Load and convert image a if it's a file path
    if isinstance(a, str):
        img_a = Image.open(a)
        if mode:
            img_a = img_a.convert(mode)
        a = np.array(img_a).astype(np.float32)
    elif isinstance(a, np.ndarray):
        assert a.dtype == np.float32, "Input array a must be of type float32"
    else:
        raise ValueError("Input a must be a string (file path) or a NumPy array")
    
    # Load and convert image b if it's a file path
    if isinstance(b, str):
        img_b = Image.open(b)
        if mode:
            img_b = img_b.convert(mode)
        b = np.array(img_b).astype(np.float32)
    elif isinstance(b, np.ndarray):
        assert b.dtype == np.float32, "Input array b must be of type float32"
    else:
        raise ValueError("Input b must be a string (file path) or a NumPy array")
    
    # Ensure images have the same size
    assert a.shape == b.shape, "Images must have the same shape"
    
    # Compute the normalized average absolute difference
    diff = np.abs(a - b) / 255.0
    return diff.mean()


# Example usage
if __name__ == "__main__":
    # arr = np.arange(101)
    # print(numpy_hash(arr))
    import argparse

    p = argparse.ArgumentParser()
    
    p.add_argument("images", nargs=argparse.REMAINDER)

    args = p.parse_args()

    print(compare_images(args.images[-2], args.images[-1]))