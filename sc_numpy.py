import hashlib
import numpy as np

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


# Example usage
if __name__ == "__main__":
    arr = np.arange(101)
    print(numpy_hash(arr))